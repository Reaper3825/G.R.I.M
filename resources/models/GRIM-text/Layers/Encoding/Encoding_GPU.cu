//======================================================//
//  Encoding_GPU.cu
//  PRODUCTION-READY Transformer Encoder Layer
//  
//  This implementation:
//    - Uses Flash Attention directly (NOT GPUMultiHeadAttention)
//    - Uses modular QKV projection from Layers/Attention/
//    - Uses RMSNorm (NOT LayerNorm - no beta parameter!)
//    - Validates EVERYTHING and screams on errors
//    - GQA-native from the start
//======================================================//

#include "Encoding_GPU.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../FlashAttention/Flash_Attention_Kernal.hpp"
#include "../FeedForward/Feed_Forward_GPU.hpp"
#include "../../Shared/PBM/PositionalBiasMethod.hpp"
#include "../../Shared/StreamController/StreamController_GPU.hpp"
#include "../../Shared/TensorConversion/TensorConversion.hpp"
#include "../../Shared/LogRecorder/BatchLogTape.hpp"  // Centralized equation logging (Rule 21)
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>  // Rule 21 diagnostic: std::min_element, std::max_element
#include <cfloat>     // FLT_MAX
#include <cstdio>     // fprintf, snprintf
#include "../../Shared/CudaAllocUtils.hpp"

using GRIM::CudaAlloc::cudaMallocOrThrow;


namespace {
    constexpr bool kEnableEncoderStepLogs = false;  // Set true to enable [EncoderFwd] step logs
    
    // Use centralized tape for [*_EQUATION] diagnostic logs (Rule 21)
    inline bool isEquationLoggingEnabled() {
        auto* tape = GRIM::Logging::getGlobalTape();
        return tape && tape->accepts(GRIM::Logging::LogLevel::Debug);
    }
    

    // QKV debug logging (GRIM_DEBUG_QKV).
    int qkvDebugLevel() {
        static int level = []() {
            const char* raw = std::getenv("GRIM_DEBUG_QKV");
            return (raw && *raw) ? std::atoi(raw) : 0;
        }();
        return level;
    }

    struct NonFiniteStats {
        int nan_count;
        int inf_count;
        int first_nan_idx;
        int first_inf_idx;
        float first_nan_val;
        float first_inf_val;
    };

    __global__ void scanNonFiniteKernel(const float* data, int count, NonFiniteStats* stats) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) {
            return;
        }
        const float v = data[idx];
        if (isnan(v)) {
            atomicAdd(&stats->nan_count, 1);
            const int old = atomicCAS(&stats->first_nan_idx, -1, idx);
            if (old == -1) {
                stats->first_nan_val = v;
            }
        } else if (isinf(v)) {
            atomicAdd(&stats->inf_count, 1);
            const int old = atomicCAS(&stats->first_inf_idx, -1, idx);
            if (old == -1) {
                stats->first_inf_val = v;
            }
        }
    }

    void logNonFiniteStats(const char* tag,
                           const float* data,
                           int count,
                           cudaStream_t stream,
                           bool always_log) {
        if (!data || count <= 0) {
            fprintf(stderr, "[QKV_DEBUG] %s invalid (ptr=%p count=%d)\n",
                    tag ? tag : "<null>", static_cast<const void*>(data), count);
            return;
        }

        NonFiniteStats init{};
        init.first_nan_idx = -1;
        init.first_inf_idx = -1;

        NonFiniteStats* d_stats = nullptr;
        cudaMallocOrThrow(reinterpret_cast<void**>(&d_stats), sizeof(NonFiniteStats), "QKV_debug_stats");

        cudaError_t err = cudaMemcpyAsync(d_stats, &init, sizeof(init), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "[QKV_DEBUG] %s cudaMemcpyAsync H2D failed: %s\n",
                    tag ? tag : "<null>", cudaGetErrorString(err));
            cudaFree(d_stats);
            return;
        }

        constexpr int kThreads = 256;
        const int blocks = (count + kThreads - 1) / kThreads;
        scanNonFiniteKernel<<<blocks, kThreads, 0, stream>>>(data, count, d_stats);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[QKV_DEBUG] %s scanNonFiniteKernel launch failed: %s\n",
                    tag ? tag : "<null>", cudaGetErrorString(err));
            cudaFree(d_stats);
            return;
        }

        NonFiniteStats out{};
        err = cudaMemcpyAsync(&out, d_stats, sizeof(out), cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "[QKV_DEBUG] %s cudaMemcpyAsync D2H failed: %s\n",
                    tag ? tag : "<null>", cudaGetErrorString(err));
            cudaFree(d_stats);
            return;
        }

        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "[QKV_DEBUG] %s cudaStreamSynchronize failed: %s\n",
                    tag ? tag : "<null>", cudaGetErrorString(err));
            cudaFree(d_stats);
            return;
        }

        cudaFree(d_stats);

        if (!always_log && out.nan_count == 0 && out.inf_count == 0) {
            return;
        }

        fprintf(stderr, "[QKV_DEBUG] %s n=%d nan=%d inf=%d",
                tag ? tag : "<null>", count, out.nan_count, out.inf_count);
        if (out.nan_count > 0) {
            fprintf(stderr, " first_nan_idx=%d first_nan_val=%g",
                    out.first_nan_idx, out.first_nan_val);
        }
        if (out.inf_count > 0) {
            fprintf(stderr, " first_inf_idx=%d first_inf_val=%g",
                    out.first_inf_idx, out.first_inf_val);
        }
        fprintf(stderr, "\n");
    }

    void logTensorNonFinite(const char* tag,
                            const GRIM::Tensor& tensor,
                            cudaStream_t stream,
                            bool always_log) {
        if (!tensor.data) {
            fprintf(stderr, "[QKV_DEBUG] %s invalid (ptr=%p)\n",
                    tag ? tag : "<null>", static_cast<const void*>(tensor.data));
            return;
        }
        const int count = static_cast<int>(tensor.numel());
        logNonFiniteStats(tag, tensor.data, count, stream, always_log);
    }
    
}
//======================================================// 


namespace GRIM {


static_assert(GRIM::HyperParameters::SOFTMAX_TEMPERATURE == 1.0f,
              "FlashAttention v2 forward requires softmax_temperature=1.0f.");

//======================================================//
//  CUDA Error Checking
//======================================================//

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        char msg[512]; \
        snprintf(msg, sizeof(msg), "CUDA ERROR at %s:%d - %s: %s", \
                 __FILE__, __LINE__, #call, cudaGetErrorString(err)); \
        throw std::runtime_error(msg); \
    } \
} while(0)



//======================================================//
//  Local kernels (not duplicated elsewhere)
//======================================================//

// NOTE: fillOnesKernel restored for Pattern B allocateWeights() — fills GPU buffer with 1.0f.
static __global__ void kernel_encoding_fill_ones(float* data, int count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        data[idx] = 1.0f;
    }
}

static __global__ void kernel_encoding_fill_value(float* data, int count, float value) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        data[idx] = value;
    }
}


//======================================================//
//  RMSNorm Forward/Backward (calls into RMSNorm_Kernel_GPU.cu)
//======================================================//

// NOTE: RMSNorm forward/backward are declared in RMSNorm_Kernel_GPU.hpp
// Include that header if needed. The extern declarations below are REMOVED
// as backward pass is now handled via TensorView-based RMSNormBackwardParams.

//======================================================//
//  EncodingLayer Implementation
//======================================================//

// ═══════════════════════════════════════════════════════════════════════════
//  Self-allocating constructor
//  Layer allocates and Xavier-inits its own weights.
//  Optimizer sees them via the existing public accessors (rms1Gamma(), attnWqkv(), etc.)
// ═══════════════════════════════════════════════════════════════════════════
 EncodingLayer::EncodingLayer(const EncodingConfig& cfg, uint64_t seed, cudaStream_t init_stream)
    : config_(cfg) {
    config_.validate("EncodingLayer::EncodingLayer");
    allocateWeights(seed, init_stream);
}

void EncodingLayer::allocateWeights(uint64_t seed, cudaStream_t init_stream) {
    if (weights_ready_) {
        throw std::runtime_error("EncodingLayer::allocateWeights: weights already initialized! "
                                 "Cannot allocate twice.");
    }
    if (!init_stream) {
        throw std::runtime_error("EncodingLayer::allocateWeights: init_stream is NULL");
    }
    config_.validate("EncodingLayer::allocateWeights");
    
    const auto& hp = config_.hp;
    const float residual_projection_init_gain = hp.residual_projection_init_gain;
    const int d_model   = hp.d_model;
    const int qkv_out_dim = hp.qkv_dim;
    cudaStream_t stream = init_stream;
    
    // ── Helper: fill a gamma tensor with 1.0 ──
    auto fillOnes = [stream](Tensor& t) {
        const int count = static_cast<int>(t.numel());
        const int threads = 256;
        const int blocks = (count + threads - 1) / threads;
        kernel_encoding_fill_ones<<<blocks, threads, 0, stream>>>(t.data, count);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error("EncodingLayer::allocateWeights fillOnes: CUDA error: "
                                     + std::string(cudaGetErrorString(err)));
        }
    };

    auto fillValue = [stream](Tensor& t, float value, const char* context) {
        const int count = static_cast<int>(t.numel());
        const int threads = 256;
        const int blocks = (count + threads - 1) / threads;
        kernel_encoding_fill_value<<<blocks, threads, 0, stream>>>(t.data, count, value);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string(context) + ": CUDA error: "
                                     + std::string(cudaGetErrorString(err)));
        }
    };
    
    //==================================================//
    //  RMSNorm gammas (4x) — initialized to 1.0
    //==================================================//
    rms1_gamma_ = Tensor::zeros({d_model}, stream, "enc_rms1_gamma_own");
    rms1_gamma_.requires_grad_();
    rms1_gamma_.ensure_grad();
    fillOnes(rms1_gamma_);
    
    rms2_gamma_ = Tensor::zeros({d_model}, stream, "enc_rms2_gamma_own");
    rms2_gamma_.requires_grad_();
    rms2_gamma_.ensure_grad();
    fillOnes(rms2_gamma_);

    //==================================================//
    //  Attention QKV projection [total_qkv_dim, d_model]
    //==================================================//
    W_qkv_ = Tensor::zeros({qkv_out_dim, d_model}, stream, "enc_W_qkv_own");
    W_qkv_.requires_grad_();
    W_qkv_.ensure_grad();
    Tensor::xavier_uniform_(W_qkv_, seed + 0, stream);

    if (hp.use_bias) {
        b_qkv_ = Tensor::zeros({qkv_out_dim}, stream, "enc_b_qkv_own");
        b_qkv_.requires_grad_();
        b_qkv_.ensure_grad();
        // Biases stay at zero init
    }
    
    //==================================================//
    //  Attention output projection [d_model, d_model]
    //  Residual projection startup init: Xavier with explicit depth gain
    //==================================================//
    W_o_ = Tensor::zeros({d_model, d_model}, stream, "enc_W_o_own");
    W_o_.requires_grad_();
    W_o_.ensure_grad();
    Tensor::xavier_uniform_with_gain_(W_o_, seed + 1, residual_projection_init_gain, stream);
    
    if (hp.use_bias) {
        b_o_ = Tensor::zeros({d_model}, stream, "enc_b_o_own");
        b_o_.requires_grad_();
        b_o_.ensure_grad();
    }
    
    //==================================================//
    //  FFN — uses the new self-allocating constructor
    //  Seed offsets +2/+3 for W1/W2 (matches layer convention)
    //==================================================//
    {
        const HyperParameters::FeedForwardLayerConstructionHP ffn_hp =
            HyperParameters::feedForwardLayerConstructionHP(hp);
        
        // seed+2 is base for FFN (W1 gets seed+2, W2 gets seed+3 inside FFN ctor)
        ffn_ = std::make_unique<FeedForwardLayer>(ffn_hp, seed + 2, stream);
    }
    
    //==================================================//
    //  LayerScale (Issue #109) — learnable per-channel gamma vectors [1, d_model]
    //==================================================//
    if (hp.use_layer_scale) {
        layer_scale1_ = Tensor::zeros({d_model}, stream, "enc_layer_scale1_own");
        layer_scale1_.requires_grad_();
        layer_scale1_.ensure_grad();
        fillValue(layer_scale1_, hp.layer_scale_init, "EncodingLayer::allocateWeights fill LayerScale1");
        
        layer_scale2_ = Tensor::zeros({d_model}, stream, "enc_layer_scale2_own");
        layer_scale2_.requires_grad_();
        layer_scale2_.ensure_grad();
        fillValue(layer_scale2_, hp.layer_scale_init, "EncodingLayer::allocateWeights fill LayerScale2");
    }
    
    weights_ready_ = true;
    
    fprintf(stderr, "[EncodingLayer] Self-allocated weights (Pattern B)\n");
}

EncodingLayer::~EncodingLayer() {
    freeWeights();
}

EncodingLayer::EncodingLayer(EncodingLayer&& other) noexcept
    : config_(other.config_)
    , weights_ready_(other.weights_ready_)
    , rms1_gamma_(std::move(other.rms1_gamma_))
    , rms2_gamma_(std::move(other.rms2_gamma_))
    , W_qkv_(std::move(other.W_qkv_))
    , b_qkv_(std::move(other.b_qkv_))
    , W_o_(std::move(other.W_o_))
    , b_o_(std::move(other.b_o_))
    , ffn_(std::move(other.ffn_))
    , layer_scale1_(std::move(other.layer_scale1_))
    , layer_scale2_(std::move(other.layer_scale2_))
{
    // Null out the moved-from object
    other.weights_ready_ = false;
}

EncodingLayer& EncodingLayer::operator=(EncodingLayer&& other) noexcept {
    if (this != &other) {
        freeWeights();
        
        config_ = other.config_;
        weights_ready_ = other.weights_ready_;
        rms1_gamma_ = std::move(other.rms1_gamma_);
        rms2_gamma_ = std::move(other.rms2_gamma_);
        W_qkv_ = std::move(other.W_qkv_);
        b_qkv_ = std::move(other.b_qkv_);
        W_o_ = std::move(other.W_o_);
        b_o_ = std::move(other.b_o_);
        ffn_ = std::move(other.ffn_);
        layer_scale1_ = std::move(other.layer_scale1_);
        layer_scale2_ = std::move(other.layer_scale2_);
        
        other.weights_ready_ = false;
    }
    return *this;
}

void EncodingLayer::freeWeights() {
    // Tensor handles its own memory cleanup via destructor (owns_data=true).
    rms1_gamma_ = Tensor();
    rms2_gamma_ = Tensor();
    W_qkv_ = Tensor();
    b_qkv_ = Tensor();
    W_o_ = Tensor();
    b_o_ = Tensor();
    layer_scale1_ = Tensor();
    layer_scale2_ = Tensor();
    ffn_.reset();
    weights_ready_ = false;
}

void EncodingLayer::validateReady(const char* context) const {
    if (!weights_ready_) {
        throw std::runtime_error(std::string(context) +
            ": weights not initialized! Call allocateWeights() first.");
    }
    if (!ffn_) {
        throw std::runtime_error(std::string(context) +
            ": FFN sublayer not initialized! Call allocateWeights() first.");
    }
    config_.validate(context);
}

// NOTE: requiredWorkspaceBytes() DELETED (Rule 20/26)
// REASON: encoder_workspace was allocated by InitTrainingState.cu based on this method's output,
// but NOTHING in the forward/backward path ever consumed it. The autograd forward pass creates
// its own intermediate Tensors. This was orphaned GPU memory.

//======================================================//
//  Forward Pass - Autograd Implementation with ForwardIntermediates (Issue #56 Fix)
//======================================================//

Tensor EncodingLayer::forward(const Tensor& input, const BatchPayload& payload,
                               const int* d_sequence_lengths, cudaStream_t stream, cublasHandle_t cublas_handle,
                               ForwardIntermediates& intermediates,
                               uint64_t training_step,
                               bool dropout_enabled,
                               int layer_idx) {
    const int seq_len = payload.max_seq_len;
    if constexpr (kEnableEncoderStepLogs) {
        fprintf(stderr, "[EncoderFwd] START total_tokens=%d seq_len=%d\n", 
                input.shape.flat.rows, seq_len);
    }
    validateReady("EncodingLayer::forward");
    const auto& hp = config_.hp;
    
    // Validate input
    if (!input.data) {
        throw std::runtime_error("EncodingLayer::forward: input.data is NULL");
    }
    if (!stream) {
        throw std::runtime_error("EncodingLayer::forward: stream is NULL");
    }
    if (!cublas_handle) {
        throw std::runtime_error("EncodingLayer::forward: cublas_handle is NULL");
    }
    
    // CRITICAL: Set autograd cuBLAS handle before any autograd::matmul calls.
    // The handle is supplied by the caller's forward payload/request and is thread_local.
    autograd::set_autograd_cublas_handle(cublas_handle);
    if constexpr (kEnableEncoderStepLogs) {
        fprintf(stderr, "[EncoderFwd] autograd cuBLAS handle set: %p\n", (void*)cublas_handle);
    }
    
    const int total_tokens = input.shape.flat.rows;
    const int d_model = hp.d_model;
    
    if (input.shape.flat.cols != d_model) {
        throw std::runtime_error("EncodingLayer::forward: input d_model mismatch. "
                                 "Expected " + std::to_string(d_model) + 
                                 ", got " + std::to_string(input.shape.flat.cols));
    }
    if (total_tokens != payload.batch_size * seq_len) {
        throw std::runtime_error("EncodingLayer::forward: total_tokens (" + std::to_string(total_tokens) +
                                 ") != payload.batch_size * seq_len (" + std::to_string(payload.batch_size) +
                                 " * " + std::to_string(seq_len) + ")");
    }
    
    const int batch_size = payload.batch_size;
    if (hp.center_encoder_residuals) {
        if (!d_sequence_lengths) {
            throw std::runtime_error("EncodingLayer::forward: center_encoder_residuals requires non-null d_sequence_lengths");
        }
        if (static_cast<int>(payload.seq_lengths.size()) != batch_size) {
            throw std::runtime_error("EncodingLayer::forward: payload.seq_lengths size (" +
                                     std::to_string(payload.seq_lengths.size()) +
                                     ") != batch_size (" + std::to_string(batch_size) + ")");
        }
        for (int b = 0; b < batch_size; ++b) {
            const int row_len = payload.seq_lengths[static_cast<size_t>(b)];
            if (row_len <= 1 || row_len > seq_len) {
                throw std::runtime_error("EncodingLayer::forward: center_encoder_residuals invalid seq_lengths[" +
                                         std::to_string(b) + "]=" + std::to_string(row_len) +
                                         " for padded seq_len=" + std::to_string(seq_len));
            }
        }
    }
    const int num_heads = hp.num_heads;
    const int num_kv_heads = hp.num_kv_heads;
    const int head_dim = hp.head_dim;
    const TensorContract::GQADims gqa{num_heads, num_kv_heads, head_dim};
    const int qkv_debug = qkvDebugLevel();
    if constexpr (kEnableEncoderStepLogs) {
        fprintf(stderr, "[EncoderFwd] validated: batch=%d heads=%d kv_heads=%d head_dim=%d\n", 
                batch_size, num_heads, num_kv_heads, head_dim);
    }
    
    //--------------------------------------------------
    // 1. RMSNorm1: input -> ln1_out
    // Issue #56: Store in intermediates to keep autograd graph alive
    //--------------------------------------------------
    intermediates.ln1_out = autograd::rms_norm(input, rms1_gamma_, hp.rms_epsilon, stream);
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 1: RMSNorm1 DONE\n");
    
    
    if (qkv_debug >= 3) {
        const bool always_log = (qkv_debug >= 2);
        logTensorNonFinite("AutogradQKV:ln1_out", intermediates.ln1_out, stream, always_log);
        logTensorNonFinite("AutogradQKV:W_qkv", W_qkv_, stream, always_log);
        logTensorNonFinite("AutogradQKV:b_qkv", b_qkv_, stream, always_log);
    }
    
    //--------------------------------------------------
    // 2. QKV Projection: ln1_out @ W_qkv^T + b_qkv
    //    W_qkv is [total_qkv_dim, d_model] so we compute ln1_out @ W_qkv^T
    // Issue #56: Store in intermediates to keep autograd graph alive
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) {
        fprintf(stderr, "[EncoderFwd] Step 2: QKV matmul...\n");
        fprintf(stderr, "[EncoderFwd] Step 2: ln1_out.data=%p shape=[%d,%d] W_qkv.data=%p shape=[%d,%d]\n",
                (void*)intermediates.ln1_out.data, intermediates.ln1_out.shape.flat.rows, intermediates.ln1_out.shape.flat.cols,
                (void*)W_qkv_.data, W_qkv_.shape.flat.rows, W_qkv_.shape.flat.cols);
        fflush(stderr);
    }
    // Use transpose_b=true since W_qkv is [qkv_dim, d_model] and we need [tokens, d_model] @ [d_model, qkv_dim]
    // Contract: pass explicit A cache for grad_B (W_qkv) path.
    if (!intermediates.ln1_out.data) {
        throw std::runtime_error("EncodingLayer::forward: ln1_out.data is NULL before QKV matmul (cannot supply required a_cache for W_qkv grad)");
    }
    intermediates.qkv_out = autograd::matmul(intermediates.ln1_out, W_qkv_, stream,
                                            intermediates.ln1_out.data, nullptr, true);
    if (qkv_debug > 0) {
        const bool always_log = (qkv_debug >= 2);
        logTensorNonFinite("AutogradQKV:qkv_out_prebias", intermediates.qkv_out, stream, always_log);
    }
    
    // ========================================================================
    // [QKV_EQUATION] DIAGNOSTIC (Rule 21) - Equation-based logging
    // Formula: qkv_out = ln1_out @ W_qkv^T + b_qkv
    // Skipped on non-initial accumulation slots (same weights → duplicate output)
    // ========================================================================
    if (isEquationLoggingEnabled() && !(GRIM::Logging::getGlobalTape() && GRIM::Logging::getGlobalTape()->skipThisPass())) {
        const int qkv_dim_local = hp.qkv_dim;
        const int d_model_local = hp.d_model;
        const int num_heads_local = hp.num_heads;
        const int head_dim_local = hp.head_dim;
        
        // Sample first N tokens for statistics (to avoid huge GPU->CPU transfers)
        const int n_sample = std::min(64, total_tokens);
        
        // Allocate host buffers
        std::vector<float> h_ln1_sample(static_cast<size_t>(n_sample) * d_model_local);
        const int w_sample_cols = std::min(64, d_model_local);
        std::vector<float> h_wqkv_sample(static_cast<size_t>(qkv_dim_local) * w_sample_cols);  // First N cols per row
        std::vector<float> h_qkv_sample(static_cast<size_t>(n_sample) * qkv_dim_local);
        std::vector<float> h_bias_sample(qkv_dim_local);
        
        // Copy data from GPU  
        cudaMemcpyAsync(h_ln1_sample.data(), intermediates.ln1_out.data,
                        h_ln1_sample.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
        // Copy first `w_sample_cols` columns for EVERY W_qkv row (2D copy, row-major [qkv_dim, d_model]).
        cudaMemcpy2DAsync(
            h_wqkv_sample.data(),                                   // dst
            static_cast<size_t>(w_sample_cols) * sizeof(float),     // dst pitch
            W_qkv_.data,                                            // src
            static_cast<size_t>(d_model_local) * sizeof(float),     // src pitch
            static_cast<size_t>(w_sample_cols) * sizeof(float),     // row bytes
            qkv_dim_local,                                          // rows
            cudaMemcpyDeviceToHost,
            stream);
        cudaMemcpyAsync(h_qkv_sample.data(), intermediates.qkv_out.data,
                        h_qkv_sample.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
        if (b_qkv_.data) {
            cudaMemcpyAsync(h_bias_sample.data(), b_qkv_.data,
                            h_bias_sample.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
        }
        cudaStreamSynchronize(stream);
        
        // Compute ln1_out statistics
        float ln1_min = FLT_MAX, ln1_max = -FLT_MAX;
        double ln1_sum_sq = 0.0;
        double ln1_row_rms_sum = 0.0;
        for (int t = 0; t < n_sample; ++t) {
            double row_sum_sq = 0.0;
            for (int d = 0; d < d_model_local; ++d) {
                float v = h_ln1_sample[t * d_model_local + d];
                ln1_min = fminf(ln1_min, v);
                ln1_max = fmaxf(ln1_max, v);
                ln1_sum_sq += v * v;
                row_sum_sq += v * v;
            }
            ln1_row_rms_sum += sqrtf(static_cast<float>(row_sum_sq / d_model_local));
        }
        float ln1_rms = sqrtf(static_cast<float>(ln1_sum_sq / (n_sample * d_model_local)));
        float ln1_row_rms_mean = static_cast<float>(ln1_row_rms_sum / n_sample);
        
        // Compute W_qkv statistics (first 64 columns)
        float wqkv_min = FLT_MAX, wqkv_max = -FLT_MAX;
        double wqkv_sum_sq = 0.0;
        double wqkv_row_rms_sum = 0.0;
        double wq_row_rms_sum = 0.0;
        for (int row = 0; row < qkv_dim_local; ++row) {
            double row_sum_sq = 0.0;
            for (int col = 0; col < w_sample_cols; ++col) {
                float v = h_wqkv_sample[row * w_sample_cols + col];
                wqkv_min = fminf(wqkv_min, v);
                wqkv_max = fmaxf(wqkv_max, v);
                wqkv_sum_sq += v * v;
                row_sum_sq += v * v;
            }
            // Scale row RMS to full d_model (assuming similar variance)
            // RMS of full row = sqrt(sum_sq_full / d_model) = sqrt(row_sum_sq * d_model / w_sample_cols / d_model) = sqrt(row_sum_sq / w_sample_cols)
            const float row_rms_scaled = sqrtf(static_cast<float>(row_sum_sq / w_sample_cols));
            wqkv_row_rms_sum += row_rms_scaled;
            if (row < d_model_local) {
                // Q rows occupy [0, d_model) in unified W_qkv layout.
                wq_row_rms_sum += row_rms_scaled;
            }
        }
        float wqkv_rms = sqrtf(static_cast<float>(wqkv_sum_sq / (qkv_dim_local * w_sample_cols)));
        float wqkv_row_rms_mean = static_cast<float>(wqkv_row_rms_sum / qkv_dim_local);
        float wq_row_rms_mean = static_cast<float>(wq_row_rms_sum / d_model_local);
        
        // Compute qkv_out statistics (Q portion only - first d_model columns)
        float qkv_min = FLT_MAX, qkv_max = -FLT_MAX;
        double qkv_sum_sq = 0.0;
        double qkv_row_rms_sum = 0.0;
        double qkv_head_row_rms_sum = 0.0;
        for (int t = 0; t < n_sample; ++t) {
            double row_sum_sq = 0.0;
            for (int d = 0; d < d_model_local; ++d) {  // Q portion only
                float v = h_qkv_sample[t * qkv_dim_local + d];
                qkv_min = fminf(qkv_min, v);
                qkv_max = fmaxf(qkv_max, v);
                qkv_sum_sq += v * v;
                row_sum_sq += v * v;
            }
            qkv_row_rms_sum += sqrtf(static_cast<float>(row_sum_sq / d_model_local));
            // Measure true per-head Q RMS directly (no balanced-head assumption).
            for (int h = 0; h < num_heads_local; ++h) {
                double head_sum_sq = 0.0;
                const int head_base = t * qkv_dim_local + h * head_dim_local;
                for (int d = 0; d < head_dim_local; ++d) {
                    const float v = h_qkv_sample[head_base + d];
                    head_sum_sq += v * v;
                }
                qkv_head_row_rms_sum += sqrtf(static_cast<float>(head_sum_sq / head_dim_local));
            }
        }
        float qkv_rms = sqrtf(static_cast<float>(qkv_sum_sq / (n_sample * d_model_local)));
        float qkv_row_rms_mean = static_cast<float>(qkv_row_rms_sum / n_sample);
        float qkv_head_row_rms_mean = static_cast<float>(qkv_head_row_rms_sum / (n_sample * num_heads_local));
        
        // Compute bias statistics
        float bias_min = 0.0f, bias_max = 0.0f, bias_rms = 0.0f;
        if (b_qkv_.data) {
            double bias_sum_sq = 0.0;
            for (int i = 0; i < qkv_dim_local; ++i) {
                float v = h_bias_sample[i];
                bias_min = fminf(bias_min, v);
                bias_max = fmaxf(bias_max, v);
                bias_sum_sq += v * v;
            }
            bias_rms = sqrtf(static_cast<float>(bias_sum_sq / qkv_dim_local));
        }
        
        // Compute EXPECTED Q magnitude in CONSISTENT units.
        // GEMM: Y = X @ W^T where X is [N, d_model], W is [qkv_dim, d_model]
        // For iid elements: elem_rms(Y) = sqrt(d_model) * rms(X_row) * rms(W_row)
        const float expected_q_elem_rms =
            ln1_row_rms_mean * wq_row_rms_mean * sqrtf(static_cast<float>(d_model_local));
        // RMS doesn't scale with dimension (unlike L2 norm), so row_rms = elem_rms for iid elements
        const float expected_q_full_row_rms = expected_q_elem_rms;
        const float expected_q_head_row_rms = expected_q_elem_rms;

        const float actual_q_full_row_rms = qkv_row_rms_mean;
        const float actual_q_head_row_rms = qkv_head_row_rms_mean;

        // Healthy-attention targets: for attention scores ~1, Q/K elements should have RMS ~1.0
        const float target_q_head_row_rms = 1.0f;
        const float target_q_full_row_rms = 1.0f;
        
        const int layer_idx_local = layer_idx;
        
        // Print [QKV_EQUATION] diagnostic
        if (isEquationLoggingEnabled()) {
            fprintf(stderr, "\n[QKV_EQUATION] ENCODER_LAYER_%d: qkv_out = ln1_out @ W_qkv^T + b_qkv\n", layer_idx_local);
            fprintf(stderr, "  ln1_out (sample %d tokens): shape=[%d,%d] min=%.10f max=%.10f rms=%.10f\n",
                    n_sample, n_sample, d_model_local, ln1_min, ln1_max, ln1_rms);
            fprintf(stderr, "  ln1_out row_rms: mean=%.10f\n", ln1_row_rms_mean);
            fprintf(stderr, "  W_qkv (sample %d cols): shape=[%d,%d] min=%.10f max=%.10f rms=%.10f\n",
                    w_sample_cols, qkv_dim_local, d_model_local, wqkv_min, wqkv_max, wqkv_rms);
            fprintf(stderr, "  W_q row_rms (scaled to d_model): mean=%.10f\n", wq_row_rms_mean);
            fprintf(stderr, "  W_qkv row_rms (all rows, scaled): mean=%.10f\n", wqkv_row_rms_mean);
            if (b_qkv_.data) {
                fprintf(stderr, "  b_qkv: shape=[%d] min=%.10f max=%.10f rms=%.10f\n",
                        qkv_dim_local, bias_min, bias_max, bias_rms);
            } else {
                fprintf(stderr, "  b_qkv: [nullptr]\n");
            }
            fprintf(stderr, "  EXPECTED qkv_elem_rms = ln1_row_rms * wq_row_rms * sqrt(d_model)\n");
            fprintf(stderr, "                        = %.4f * %.4f * sqrt(%d) = %.4f\n",
                    ln1_row_rms_mean, wq_row_rms_mean, d_model_local, expected_q_elem_rms);
            fprintf(stderr, "  ACTUAL qkv_out (Q portion): min=%.10f max=%.10f rms=%.10f\n",
                    qkv_min, qkv_max, qkv_rms);
            fprintf(stderr, "  EXPECTED Q row_rms (full/head): %.4f / %.4f\n",
                    expected_q_full_row_rms, expected_q_head_row_rms);
            fprintf(stderr, "  ACTUAL   Q row_rms (full/head): %.4f / %.4f\n",
                    actual_q_full_row_rms, actual_q_head_row_rms);
            fprintf(stderr, "  TARGET   Q row_rms (full/head): %.4f / %.4f (healthy attention: elem_rms≈1.0)\n",
                    target_q_full_row_rms, target_q_head_row_rms);
            fprintf(stderr, "  INFLATION(full/head): %.4fx / %.4fx\n",
                    actual_q_full_row_rms / target_q_full_row_rms,
                    actual_q_head_row_rms / target_q_head_row_rms);

            {
                std::ostringstream eq;
                eq << "[QKV_PROJECTION_EQUATION] qkv_out = ln1_out @ W_qkv^T + b_qkv\n";
                eq << "  INPUT (ln1_out): rms=" << ln1_rms << " row_rms=" << ln1_row_rms_mean << "\n";
                eq << "  WEIGHT (W_qkv): rms=" << wqkv_rms << " q_row_rms=" << wq_row_rms_mean << "\n";
                eq << "  EXPECTED qkv_elem_rms = ln1_row_rms * wq_row_rms * sqrt(d_model)\n";
                eq << "                         = " << ln1_row_rms_mean << " * " << wq_row_rms_mean 
                   << " * sqrt(" << d_model_local << ")\n";
                eq << "                         = " << expected_q_elem_rms << "\n";
                eq << "  ACTUAL Q row_rms (full/head): " << actual_q_full_row_rms << " / " << actual_q_head_row_rms << "\n";
                eq << "  TARGET Q row_rms (full/head): " << target_q_full_row_rms << " / " << target_q_head_row_rms << "\n";
                const float inflation_ratio_eq = actual_q_head_row_rms / target_q_head_row_rms;
                eq << "  INFLATION (full/head): " << (actual_q_full_row_rms / target_q_full_row_rms) 
                   << "x / " << inflation_ratio_eq << "x\n";
                if (inflation_ratio_eq > 5.0f) {
                    eq << "  [ANOMALY] per-head q_row_rms=" << actual_q_head_row_rms 
                       << " is " << inflation_ratio_eq << "x larger than target=" << target_q_head_row_rms << "\n";
                }
                if (ln1_row_rms_mean > 50.0f) {
                    eq << "  [ANOMALY] ln1_out row_rms=" << ln1_row_rms_mean << " >> expected ~1.0\n";
                }
                if (wq_row_rms_mean > 5.0f) {
                    eq << "  [ANOMALY] W_q row_rms=" << wq_row_rms_mean << " >> expected ~0.036\n";
                }
                EQ_LOG(GRIM::Logging::getGlobalTape(), GRIM::Logging::LogGroup::Attention, GRIM::Logging::LogPhase::QKV_PROJECTION, layer_idx_local, "QKV_PROJECTION_EQUATION", eq.str().c_str());
            }
        }
    }
    
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 2: QKV matmul DONE, adding bias...\n");
    // ISSUE #97 FIX: Use autograd::broadcast_add for proper gradient tracking
    // Previously: launchFFNBiasAdd bypassed autograd, so b_qkv never received gradients
    // Now: autograd::broadcast_add creates BiasAddGradFn which computes grad_bias = sum(grad_output)
    if (hp.use_bias && b_qkv_.data) {
        intermediates.qkv_out = autograd::broadcast_add(intermediates.qkv_out, b_qkv_, stream);
    }
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 2: QKV bias DONE\n");
    if (qkv_debug > 0) {
        const bool always_log = (qkv_debug >= 2);
        logTensorNonFinite("AutogradQKV:qkv_out", intermediates.qkv_out, stream, always_log);
    }
    
    //--------------------------------------------------
    // 3. Split QKV and reshape to BHSD for attention
    //    qkv_out is [total_tokens, d_model + 2*kv_dim]
    //    Q: [total_tokens, 0:d_model]
    //    K: [total_tokens, d_model:d_model+kv_dim]  
    //    V: [total_tokens, d_model+kv_dim:end]
     //
    // ISSUE #61 FIX: Use autograd::split_and_reshape_qkv() to maintain gradient chain
    // Previous code used Tensor::empty() + cudaMemcpy2D which broke autograd (Q/K/V had no grad_fn)
    // This caused W_qkv gradients to be ZERO since ScaledDotProductAttentionGradFn couldn't
    // continue the chain through Q/K/V with null grad_fn.
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 3: Split QKV with autograd tracking...\n");
    
    // ISSUE #61: This properly tracks gradients through the split/reshape operation
    auto [Q_bhsd_tmp, K_bhsd_tmp, V_bhsd_tmp] = autograd::split_and_reshape_qkv(
        intermediates.qkv_out,
        payload, gqa,
        stream);
    intermediates.Q_bhsd = std::move(Q_bhsd_tmp);
    intermediates.K_bhsd = std::move(K_bhsd_tmp);
    intermediates.V_bhsd = std::move(V_bhsd_tmp);
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 3: Split QKV DONE (autograd tracked)\n");
    if (qkv_debug > 0) {
        const bool always_log = (qkv_debug >= 2);
        logTensorNonFinite("AutogradQKV:Q_bhsd", intermediates.Q_bhsd, stream, always_log);
        logTensorNonFinite("AutogradQKV:K_bhsd", intermediates.K_bhsd, stream, always_log);
        logTensorNonFinite("AutogradQKV:V_bhsd", intermediates.V_bhsd, stream, always_log);
    }
    
    //--------------------------------------------------
    // 3b. Apply RoPE rotation to Q and K
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 3c: RoPE rotation...\n");
    if (config_.pos_encoding && config_.pos_encoding->valid && 
        config_.pos_encoding->rope_inv_freq != nullptr && config_.pos_encoding->rotary_dim > 0) {
        // ISSUE #119 FIX: Use autograd::rope_rotation() instead of raw kernel call
        // The raw PBM::launchRoPERotationGQA() bypassed autograd - dQ/dK gradients
        // were never inverse-rotated in backward pass, causing gradient corruption.
        // Returns new tensors — inputs are never mutated.
        auto [Q_rot, K_rot] = autograd::rope_rotation(
            intermediates.Q_bhsd, intermediates.K_bhsd,
            config_.pos_encoding->rope_inv_freq,
            payload, gqa,
            config_.pos_encoding->rotary_dim, stream);
        intermediates.Q_bhsd = std::move(Q_rot);
        intermediates.K_bhsd = std::move(K_rot);
    } else {
        throw std::runtime_error("EncodingLayer::forward: RoPE not initialized");
    }
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 3c: RoPE DONE\n");
    if (qkv_debug > 0) {
        const bool always_log = (qkv_debug >= 2);
        logTensorNonFinite("AutogradSDPA:Q_rope", intermediates.Q_bhsd, stream, always_log);
        logTensorNonFinite("AutogradSDPA:K_rope", intermediates.K_bhsd, stream, always_log);
        logTensorNonFinite("AutogradSDPA:V_rope", intermediates.V_bhsd, stream, always_log);
    }
    
    //--------------------------------------------------
    // 4. Flash Attention: Q, K, V -> attn_out
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 4: Flash Attention...\n");
    
    // RULE 20: Fail loud if pos_encoding is NULL - GRIM requires hybrid PBM
    if (!config_.pos_encoding) {
        throw std::runtime_error(
            "EncodingLayer::forward: config_.pos_encoding is NULL - "
            "GRIM requires PBM (ALiBi+RoPE) for positional encoding");
    }
    if (!config_.pos_encoding->alibi_slopes) {
        throw std::runtime_error(
            "EncodingLayer::forward: config_.pos_encoding->alibi_slopes is NULL - "
            "PBM ALiBi slopes not initialized");
    }
    
    // [ROPE_ALIBI_INTERACTION] Log configuration once at first call (no stream blocking)
    static bool logged_rope_alibi_config = false;
    if (!logged_rope_alibi_config && config_.pos_encoding->alibi_slopes) {
        const int to_copy = std::min(num_heads, 12);
        float slopes_host[12] = {0};
        cudaError_t copy_err = cudaMemcpy(slopes_host, config_.pos_encoding->alibi_slopes,
                                          to_copy * sizeof(float), cudaMemcpyDeviceToHost);
        if (copy_err != cudaSuccess) {
            fprintf(stderr, "[ROPE_ALIBI_INTERACTION] cudaMemcpy failed: %s\n", cudaGetErrorString(copy_err));
        } else {
            fprintf(stderr, "[ROPE_ALIBI_INTERACTION] Hybrid positional encoding active:\n");
            fprintf(stderr, "  RoPE: rotary_dim=%d (rotates Q/K by position-dependent angle, preserves norm)\n",
                    config_.pos_encoding->rotary_dim);
            if (to_copy >= 1) {
                if (to_copy >= 12) {
                    fprintf(stderr, "  ALiBi: slopes=[%.6f, %.6f, ... %.6f] (adds distance penalty to attention scores)\n",
                            slopes_host[0], slopes_host[5], slopes_host[11]);
                } else if (to_copy >= 6) {
                    fprintf(stderr, "  ALiBi: slopes=[%.6f, %.6f, ...] (adds distance penalty to attention scores)\n",
                            slopes_host[0], slopes_host[5]);
                } else if (to_copy >= 2) {
                    fprintf(stderr, "  ALiBi: slopes=[%.6f, %.6f, ...] (adds distance penalty to attention scores)\n",
                            slopes_host[0], slopes_host[1]);
                } else {
                    fprintf(stderr, "  ALiBi: slopes=[%.6f] (adds distance penalty to attention scores)\n",
                            slopes_host[0]);
                }
            }
            fprintf(stderr, "  Combined: RoPE encodes relative position directionally, ALiBi provides distance bias\n");
        }
        logged_rope_alibi_config = true;
    }
    
    // Issue #56: Store attention output in intermediates
    // Compute per-step per-layer dropout seed using Knuth multiplicative hash.
    // Dropout mode is explicit: training_step is seed material only, never a mode gate.
    const int layer_idx_for_seed = layer_idx;
    const float attention_dropout_p = dropout_enabled ? hp.attention_dropout : 0.0f;
    const uint64_t attn_dropout_seed = (attention_dropout_p > 0.0f)
        ? (training_step * 2654435761ULL + 42 + 1000 * static_cast<uint64_t>(layer_idx_for_seed))
        : 0;
    intermediates.attn_out_bhsd = autograd::scaled_dot_product_attention(
        intermediates.Q_bhsd, intermediates.K_bhsd, intermediates.V_bhsd, 
        config_.pos_encoding->alibi_slopes, 0.0f, stream, true,
        attention_dropout_p, attn_dropout_seed);
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 4: Flash Attention DONE\n");
    if (qkv_debug > 0) {
        const bool always_log = (qkv_debug >= 2);
        logTensorNonFinite("AutogradSDPA:attn_out_bhsd", intermediates.attn_out_bhsd, stream, always_log);
    }
    
    //--------------------------------------------------
    // 5. Reshape attention output: BHSD -> [tokens, d_model]
    // ISSUE #62 FIX: Use autograd::reshape_bhsd_to_flat() to maintain gradient chain
    // Previous code used Tensor::empty() + launchReshapeFromBHSD which broke autograd
    // (attn_out had no grad_fn, causing W_o gradients to not flow through attention backward)
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 5: Reshape from BHSD with autograd tracking...\n");
    intermediates.attn_out = autograd::reshape_bhsd_to_flat(
        intermediates.attn_out_bhsd, payload, gqa, stream);
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 5: Reshape DONE (autograd tracked)\n");
    
    //--------------------------------------------------
    // 6. Output projection: attn_out @ W_o^T + b_o
    // Issue #56: Store in intermediates
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 6: Output projection...\n");
    // W_o is [d_model, d_model], so W_o^T is also [d_model, d_model]
    // Use transpose_b=true to compute attn_out @ W_o^T
    // Contract: pass explicit A cache for grad_B (W_o) path.
    if (!intermediates.attn_out.data) {
        throw std::runtime_error("EncodingLayer::forward: attn_out.data is NULL before output projection matmul (cannot supply required a_cache for W_o grad)");
    }
    intermediates.proj_out = autograd::matmul(intermediates.attn_out, W_o_, stream,
                                              intermediates.attn_out.data, nullptr, true);
    // ISSUE #97 FIX: Use autograd::broadcast_add for proper gradient tracking on b_o
    if (hp.use_bias && b_o_.data) {
        intermediates.proj_out = autograd::broadcast_add(intermediates.proj_out, b_o_, stream);
    }
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 6: Output projection DONE\n");
    
    //--------------------------------------------------
    // 6b. Post-attention sublayer dropout
    //     Standard transformer: residual = input + dropout(sublayer(norm(input)))
    //     Dropout applied BEFORE LayerScale and residual add.
    //--------------------------------------------------
    if (hp.dropout_rate > 0.0f && dropout_enabled) {
        const uint64_t attn_proj_dropout_seed = training_step * 2654435761ULL + 100 + layer_idx;
        const uint64_t attn_proj_dropout_mask_stream = 0x0001000000000000ULL + static_cast<uint64_t>(layer_idx);
        intermediates.proj_out = autograd::dropout(intermediates.proj_out, hp.dropout_rate,
                                                   attn_proj_dropout_seed, true, stream,
                                                   attn_proj_dropout_mask_stream);
    }
    
    //--------------------------------------------------
    // 7. Residual1: input + proj_out -> residual1
    // Issue #56: Store in intermediates
    // Issue #109: Apply LayerScale to proj_out before residual addition
    // Note: Standard pre-norm architecture (Issue #148: Sandwich Norm removed).
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 7: Residual1...\n");
    
    // Issue #109: LayerScale gating for attention sublayer
    Tensor scaled_proj;
    const Tensor* proj_for_residual = &intermediates.proj_out;
    if (hp.use_layer_scale) {
        if (!layer_scale1_.data) {
            throw std::runtime_error("EncodingLayer::forward: use_layer_scale=true but layer_scale1_ is NULL");
        }
        layer_scale1_.shape.require("EncodingLayer::forward layer_scale1_");
        if (!layer_scale1_.shape.is_2d_layout()) {
            throw std::runtime_error("EncodingLayer::forward: layer_scale1_ must be a 2D [1,d_model] gamma vector");
        }
        const auto ls1_dims = layer_scale1_.shape.as_2d();
        if (ls1_dims.rows != 1 || ls1_dims.cols != d_model) {
            throw std::runtime_error("EncodingLayer::forward: layer_scale1_ must have shape [1,d_model]. expected=[1," +
                                     std::to_string(d_model) + "] got=[" + std::to_string(ls1_dims.rows) + "," +
                                     std::to_string(ls1_dims.cols) + "]");
        }
        scaled_proj = autograd::layer_scale(intermediates.proj_out, layer_scale1_, stream);
        proj_for_residual = &scaled_proj;
    }
    
    // ========================================================================
    // STANDARD PRE-NORM RESIDUAL (Issue #148: Sandwich Norm REMOVED)
    //
    // Architecture: residual1 = input + LayerScale(attn_out)
    //
    // Standard pre-norm transformer residual connection. Hidden state norms
    // are FREE to vary across tokens, providing an additional degree of freedom
    // that prevents premature cosine alignment (mode collapse).
    //
    // Why Sandwich Norm was removed:
    //   1. Forced ||h[t]|| = sqrt(D) for ALL tokens → removed magnitude diversity
    //   2. All tokens on a hypersphere → cosine alignment is the ONLY axis
    //   3. Initial rho(0)=0.21 (vs PyTorch 0.05-0.08) → started halfway to collapse
    //   4. Combined with causal attention prefix averaging → mode collapse by batch 3
    // ========================================================================
    intermediates.residual1 = autograd::add(input, *proj_for_residual, stream);
    
    // ========================================================================
    // RESIDUAL CENTERING (Issue #118 / Mode Collapse Fix)
    //
    // WHY: Causal attention creates a shared output component from prefix tokens.
    //   Residual accumulation + RMSNorm direction preservation amplifies this
    //   shared direction through layers: ρ grows +0.01-0.04 per layer.
    //   Over 12 layers: ρ(emb)=0.05 → ρ(final)=0.44 → mode collapse.
    //
    // WHAT: center_columns_by_sequence_lengths subtracts the cross-position mean
    //   over VALID (unpadded) tokens for each feature WITHIN EACH BATCH ROW:
    //   h[b,t,d] -= mean_{u < seq_lengths[b]}(h[b,u,d])   for valid t
    //   h[b,t,d] = 0                                      for padded t
    //   This removes the rank-1 shared direction at each layer without making
    //   sample A's hidden state depend on sample B's hidden state or on PAD rows.
    //
    // GRADIENT COST: The centering projection P = I - 11^T/n has backward
    //   grad_input = P * grad_output within each sequence. Only the per-sequence
    //   constant mode is projected out; non-constant token modes are preserved.
    // ========================================================================
    if (hp.center_encoder_residuals) {
        if (seq_len <= 1) {
            throw std::runtime_error("EncodingLayer::forward: center_encoder_residuals requires payload.max_seq_len > 1; single-row column centering would erase the residual stream");
        }
        intermediates.residual1 = autograd::center_columns_by_sequence_lengths(
            intermediates.residual1, d_sequence_lengths, batch_size, seq_len, stream);
    }
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 7: Residual1 (pre-norm, no sandwich) DONE\n");
    
    
    //--------------------------------------------------
    // 8. RMSNorm2: residual1 -> ln2_out
    // Issue #56: Store in intermediates
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 8: RMSNorm2...\n");
    intermediates.ln2_out = autograd::rms_norm(intermediates.residual1, rms2_gamma_, hp.rms_epsilon, stream);
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 8: RMSNorm2 DONE\n");
    
    //--------------------------------------------------
    // 9. FFN: ln2_out -> ffn_out (already using autograd)
    // Issue #56: FFN also stores its intermediates in this same ForwardIntermediates
    // (ffn_gate_out, ffn_silu_out, ffn_linear1_out, ffn_swiglu_out are written by SwiGLU forward)
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 9: FFN...\n");
    intermediates.ffn_out = ffn_->forward(intermediates.ln2_out, intermediates,
                                          stream, cublas_handle,
                                          training_step, dropout_enabled, layer_idx);
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 9: FFN DONE\n");
    
    //--------------------------------------------------
    // 9b. Post-FFN sublayer dropout
    //     Standard transformer: residual = input + dropout(sublayer(norm(input)))
    //     Dropout applied BEFORE LayerScale and residual add.
    //--------------------------------------------------
    if (hp.dropout_rate > 0.0f && dropout_enabled) {
        const uint64_t ffn_dropout_seed = training_step * 2654435761ULL + 200 + layer_idx;
        const uint64_t ffn_dropout_mask_stream = 0x0002000000000000ULL + static_cast<uint64_t>(layer_idx);
        intermediates.ffn_out = autograd::dropout(intermediates.ffn_out, hp.dropout_rate,
                                                  ffn_dropout_seed, true, stream,
                                                  ffn_dropout_mask_stream);
    }
    
    //--------------------------------------------------
    // 10. Residual2: residual1 + ffn_out -> output
    // Issue #56: The final output IS stored in intermediates too
    // for consistency, but we also return it
    // Issue #109: Apply LayerScale to ffn_out before residual addition
    // Issue #118: Apply centering to remove common direction before residual add
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 10: Residual2...\n");
    
    // Issue #109: LayerScale gating for FFN sublayer
    Tensor scaled_ffn;
    const Tensor* ffn_for_residual = &intermediates.ffn_out;
    if (hp.use_layer_scale) {
        if (!layer_scale2_.data) {
            throw std::runtime_error("EncodingLayer::forward: use_layer_scale=true but layer_scale2_ is NULL");
        }
        layer_scale2_.shape.require("EncodingLayer::forward layer_scale2_");
        if (!layer_scale2_.shape.is_2d_layout()) {
            throw std::runtime_error("EncodingLayer::forward: layer_scale2_ must be a 2D [1,d_model] gamma vector");
        }
        const auto ls2_dims = layer_scale2_.shape.as_2d();
        if (ls2_dims.rows != 1 || ls2_dims.cols != d_model) {
            throw std::runtime_error("EncodingLayer::forward: layer_scale2_ must have shape [1,d_model]. expected=[1," +
                                     std::to_string(d_model) + "] got=[" + std::to_string(ls2_dims.rows) + "," +
                                     std::to_string(ls2_dims.cols) + "]");
        }
        scaled_ffn = autograd::layer_scale(intermediates.ffn_out, layer_scale2_, stream);
        ffn_for_residual = &scaled_ffn;
    }
    
    // ========================================================================
    // STANDARD PRE-NORM RESIDUAL (Issue #148: Sandwich Norm REMOVED)
    //
    // Architecture: output = residual1 + LayerScale(ffn_out)
    //
    // No post-residual normalization. Matches standard PyTorch GPT pre-norm.
    // ========================================================================
    intermediates.output = autograd::add(intermediates.residual1, *ffn_for_residual, stream);
    
    // Issue #155: Post-FFN centering REMOVED from here — moved to AutogradTraining.cu
    // so it happens AFTER all layer-output modifications (including crossAttentionRead).
    // Post-attention centering remains here (between sublayers, no external modification).
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 10: Residual2 (pre-norm, no sandwich) DONE - layer COMPLETE\n");
    
    
    // ═══════════════════════════════════════════════════════════════════════════
    // RULE 21 DIAGNOSTIC: Per-Layer Cosine Similarity (correlation tracking)
    //
    // EQUATION (Pre-Norm): output = input + LS2*FFN(RMSNorm(input + LS1*Attn(RMSNorm(input))))
    //
    // Issue #148: Sandwich Norm removed. Standard pre-norm residual connections.
    // Hidden state norms are now FREE to vary (not clamped to sqrt(D)).
    //
    // Interpretation:
    //   |avg_cos| → 1.0 = mode collapse (all vectors aligned) = BAD
    //   |avg_cos| near 0 = diverse representations = generally healthy
    // Skipped on non-initial accumulation slots (same weights → duplicate output)
    // ═══════════════════════════════════════════════════════════════════════════
    // Log ALL layers so per-layer ρ trajectory is visible.
    // NOTE: intermediates.output is the PRE-centering output — the post-layer center_columns
    // in AutogradTraining.cu hasn't run yet. Logged ρ values will be slightly higher than
    // what the NEXT layer actually receives as input.
    if (isEquationLoggingEnabled() && !(GRIM::Logging::getGlobalTape() && GRIM::Logging::getGlobalTape()->skipThisPass())) {
        cudaStreamSynchronize(stream);  // Ensure data is ready
        
        // Copy layer output to host for analysis
        const int output_size = total_tokens * d_model;
        std::vector<float> h_output(output_size);
        cudaMemcpy(h_output.data(), intermediates.output.data, output_size * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Compute row RMS
        std::vector<float> row_rms(total_tokens);
        for (int t = 0; t < total_tokens; t++) {
            double norm_sq = 0.0;
            for (int d = 0; d < d_model; d++) {
                float v = h_output[t * d_model + d];
                norm_sq += v * v;
            }
            row_rms[t] = sqrtf(norm_sq / d_model);
        }
        
        // Sample pairwise cosine similarity
        const int sample_pairs = std::min(30, total_tokens / 2);
        const int stride = std::max(1, total_tokens / sample_pairs);
        double cos_sum = 0.0;
        int num_pairs = 0;
        
        for (int i = 0; i < total_tokens && num_pairs < sample_pairs; i += stride) {
            int j = (i + total_tokens / 2) % total_tokens;
            if (i == j || row_rms[i] < 1e-8f || row_rms[j] < 1e-8f) continue;
            
            double dot = 0.0;
            for (int d = 0; d < d_model; d++) {
                dot += h_output[i * d_model + d] * h_output[j * d_model + d];
            }
            cos_sum += dot / (static_cast<double>(row_rms[i]) * row_rms[j] * d_model);
            num_pairs++;
        }
        
        const double avg_cos = (num_pairs > 0) ? cos_sum / num_pairs : 0.0;
        const float rms_min = *std::min_element(row_rms.begin(), row_rms.end());
        const float rms_max = *std::max_element(row_rms.begin(), row_rms.end());
        
        struct LayerScaleDiagStats {
            float min = 0.0f;
            float max = 0.0f;
            float mean = 0.0f;
            float rms = 0.0f;
        };
        auto layerScaleStats = [d_model](Tensor& gamma, const char* name) -> LayerScaleDiagStats {
            if (!gamma.data) {
                throw std::runtime_error(std::string("EncodingLayer::forward diagnostics: ") + name + " is NULL while LayerScale is enabled");
            }
            const std::string shape_context = std::string("EncodingLayer::forward diagnostics ") + name;
            gamma.shape.require(shape_context.c_str());
            if (!gamma.shape.is_2d_layout()) {
                throw std::runtime_error(std::string("EncodingLayer::forward diagnostics: ") + name + " must be a 2D [1,d_model] gamma vector");
            }
            const auto gamma_dims = gamma.shape.as_2d();
            if (gamma_dims.rows != 1 || gamma_dims.cols != d_model) {
                throw std::runtime_error(std::string("EncodingLayer::forward diagnostics: ") + name + " must have shape [1,d_model]. expected=[1," +
                                         std::to_string(d_model) + "] got=[" + std::to_string(gamma_dims.rows) + "," +
                                         std::to_string(gamma_dims.cols) + "]");
            }
            std::vector<float> h_gamma(static_cast<size_t>(d_model));
            cudaMemcpy(h_gamma.data(), gamma.data, h_gamma.size() * sizeof(float), cudaMemcpyDeviceToHost);
            LayerScaleDiagStats stats{};
            stats.min = FLT_MAX;
            stats.max = -FLT_MAX;
            double sum = 0.0;
            double sum_sq = 0.0;
            for (float v : h_gamma) {
                stats.min = fminf(stats.min, v);
                stats.max = fmaxf(stats.max, v);
                sum += v;
                sum_sq += static_cast<double>(v) * static_cast<double>(v);
            }
            stats.mean = static_cast<float>(sum / d_model);
            stats.rms = sqrtf(static_cast<float>(sum_sq / d_model));
            return stats;
        };
        LayerScaleDiagStats ls1_stats{};
        LayerScaleDiagStats ls2_stats{};
        if (hp.use_layer_scale) {
            ls1_stats = layerScaleStats(layer_scale1_, "layer_scale1_");
            ls2_stats = layerScaleStats(layer_scale2_, "layer_scale2_");
        }
        
        std::ostringstream eq;
        eq << "[LAYER_COSINE_EQUATION] layer=" << layer_idx 
           << ": residual1[t,d] = input[t,d] + gamma1[d] * attn[t,d]; output[t,d] = residual1[t,d] + gamma2[d] * ffn[t,d]\n";
        eq << "  OUTPUT h_L" << layer_idx << ": shape=[" << total_tokens << ", " << d_model 
           << "] row_rms_range=[" << rms_min << ", " << rms_max << "]\n";
        if (hp.use_layer_scale) {
            eq << "  LAYERSCALE gamma vectors: shape=[1," << d_model << "]"
               << " LS1[min=" << ls1_stats.min << " max=" << ls1_stats.max
               << " mean=" << ls1_stats.mean << " rms=" << ls1_stats.rms << "]"
               << " LS2[min=" << ls2_stats.min << " max=" << ls2_stats.max
               << " mean=" << ls2_stats.mean << " rms=" << ls2_stats.rms << "]\n";
        } else {
            eq << "  LAYERSCALE: disabled\n";
        }
        eq << "  ACTUAL avg_cos=" << avg_cos << " (pairs=" << num_pairs 
           << ") [|avg_cos|->1 = collapse, near 0 = diverse]\n";
        if (fabs(avg_cos) > 0.8) {
            eq << "  [ANOMALY] Layer " << layer_idx << " |avg_cos|=" << fabs(avg_cos) 
               << " HIGH - possible mode collapse!\n";
        }
        EQ_LOG(GRIM::Logging::getGlobalTape(), GRIM::Logging::LogGroup::Attention, GRIM::Logging::LogPhase::RESIDUAL_POST_ATTN, layer_idx, "LAYER_COSINE_EQUATION", eq.str().c_str());
    }
    
    // Return a non-owning view of the output
    // The actual Tensor lives in intermediates and stays alive until backward completes
    Tensor result = Tensor::from_ptr(
        intermediates.output.data,
        intermediates.output.shape,
        false,  // doesn't own data - intermediates.output owns it
        true,    // requires_grad
        "enc_layer_output"
    );
    result.is_leaf = false;
    result.grad_fn = intermediates.output.grad_fn;
    result.stream = stream;
    
    return result;
}

} // namespace GRIM
