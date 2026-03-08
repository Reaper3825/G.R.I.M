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
#include "../../Shared/EquationLogging/EquationLogging.hpp"  // Centralized equation logging (Rule 21)
#include "../../Shared/VerboseLogging.hpp"  // Compile-time diagnostic guards (Issue #151)
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


namespace {
    constexpr bool kEnableEncoderStepLogs = false;  // Set true to enable [EncoderFwd] step logs
    
    // Use centralized EquationLogger for [*_EQUATION] diagnostic logs (Rule 21)
    inline bool isEquationLoggingEnabled() {
        return GRIM::getEquationLogger().isEnabled();
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
        cudaError_t err = cudaMalloc(&d_stats, sizeof(NonFiniteStats));
        if (err != cudaSuccess) {
            fprintf(stderr, "[QKV_DEBUG] %s cudaMalloc failed: %s\n",
                    tag ? tag : "<null>", cudaGetErrorString(err));
            return;
        }

        err = cudaMemcpyAsync(d_stats, &init, sizeof(init), cudaMemcpyHostToDevice, stream);
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

// Issue #142: scale tensor data in-place (for W_o residual scaling)
static __global__ void kernel_encoding_scale_inplace(float* data, size_t count, float scale) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        data[idx] *= scale;
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
EncodingLayer::EncodingLayer(const EncodingConfig& cfg, uint64_t seed,
                             float residual_scale, float layer_scale_init) {
    setConfig(cfg);
    allocateWeights(seed, residual_scale, layer_scale_init);
}

void EncodingLayer::allocateWeights(uint64_t seed, float residual_scale, float layer_scale_init) {
    if (weights_ready_) {
        throw std::runtime_error("EncodingLayer::allocateWeights: weights already initialized! "
                                 "Cannot allocate twice.");
    }
    config_.validate("EncodingLayer::allocateWeights");
    
    const int d_model   = config_.d_model;
    const int kv_dim    = config_.kvDim();
    const int d_ff      = config_.d_ff;
    const int qkv_out_dim = d_model + 2 * kv_dim;
    cudaStream_t stream = config_.stream;
    
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
    
    // ── Helper: in-place scale for Issue #142 residual projection init ──
    auto scaleInplace = [stream](Tensor& t, float scale) {
        if (std::abs(scale - 1.0f) < 1e-8f) return;  // Skip if scale≈1
        const size_t count = t.numel();
        const int threads = 256;
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        kernel_encoding_scale_inplace<<<blocks, threads, 0, stream>>>(t.data, count, scale);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error("EncodingLayer::allocateWeights scaleInplace: CUDA error: "
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
    
    // Issue #148: Sandwich Norm REMOVED — rms_post_attn_gamma_ and rms_post_ffn_gamma_ 
    // are no longer allocated. Standard pre-norm architecture does not use post-residual
    // normalization. This allows hidden state norms to vary freely across tokens.
    
    //==================================================//
    //  Attention QKV projection [total_qkv_dim, d_model]
    //==================================================//
    W_qkv_ = Tensor::zeros({qkv_out_dim, d_model}, stream, "enc_W_qkv_own");
    W_qkv_.requires_grad_();
    W_qkv_.ensure_grad();
    Tensor::xavier_uniform_(W_qkv_, seed + 0, stream);

    // Issue #106: Scale QKV weights by 1/sqrt(d_model) to prevent initial LSE explosion.
    // Without this, coherent summation causes attention scores to start saturated.
    // Expected output norm ~8, actual without scaling ~125.
    scaleInplace(W_qkv_, 1.0f / std::sqrt(static_cast<float>(d_model)));
    
    if (config_.use_bias) {
        b_qkv_ = Tensor::zeros({qkv_out_dim}, stream, "enc_b_qkv_own");
        b_qkv_.requires_grad_();
        b_qkv_.ensure_grad();
        // Biases stay at zero init
    }
    
    //==================================================//
    //  Attention output projection [d_model, d_model]
    //  Issue #142: scaled by residual_scale after Xavier
    //==================================================//
    W_o_ = Tensor::zeros({d_model, d_model}, stream, "enc_W_o_own");
    W_o_.requires_grad_();
    W_o_.ensure_grad();
    Tensor::xavier_uniform_(W_o_, seed + 1, stream);
    scaleInplace(W_o_, residual_scale);
    
    if (config_.use_bias) {
        b_o_ = Tensor::zeros({d_model}, stream, "enc_b_o_own");
        b_o_.requires_grad_();
        b_o_.ensure_grad();
    }
    
    //==================================================//
    //  FFN — uses the new self-allocating constructor
    //  Seed offsets +2/+3 for W1/W2 (matches layer convention)
    //==================================================//
    {
        FeedForwardConfig ffn_cfg;
        ffn_cfg.d_model        = d_model;
        ffn_cfg.d_ff           = d_ff;
        ffn_cfg.stream         = stream;
        ffn_cfg.cublas_handle  = config_.cublas_handle;
        ffn_cfg.use_bias       = config_.use_bias;
        ffn_cfg.dropout_rate   = config_.dropout_rate;
        
        // seed+2 is base for FFN (W1 gets seed+2, W2 gets seed+3 inside FFN ctor)
        ffn_ = std::make_unique<FeedForwardLayer>(ffn_cfg, seed + 2, residual_scale);
    }
    
    //==================================================//
    //  LayerScale (Issue #109) — learnable [1] scalars
    //==================================================//
    if (config_.use_layer_scale) {
        layer_scale1_ = Tensor::zeros({1}, stream, "enc_layer_scale1_own");
        layer_scale1_.requires_grad_();
        layer_scale1_.ensure_grad();
        cudaMemcpyAsync(layer_scale1_.data, &layer_scale_init, sizeof(float),
                        cudaMemcpyHostToDevice, stream);
        
        layer_scale2_ = Tensor::zeros({1}, stream, "enc_layer_scale2_own");
        layer_scale2_.requires_grad_();
        layer_scale2_.ensure_grad();
        cudaMemcpyAsync(layer_scale2_.data, &layer_scale_init, sizeof(float),
                        cudaMemcpyHostToDevice, stream);
    }
    
    //==================================================//
    //  QK-Norm per-head alpha (Dehghani et al. 2023)
    //  Initialized to 1.0 so initial behavior = standard RMSNorm
    //==================================================//
    if (config_.qk_norm_enabled) {
        const int effective_kv_heads = config_.effectiveKVHeads();
        
        alpha_q_ = Tensor::zeros({config_.num_heads}, stream, "enc_alpha_q_own");
        alpha_q_.requires_grad_();
        alpha_q_.ensure_grad();
        fillOnes(alpha_q_);
        
        alpha_k_ = Tensor::zeros({effective_kv_heads}, stream, "enc_alpha_k_own");
        alpha_k_.requires_grad_();
        alpha_k_.ensure_grad();
        fillOnes(alpha_k_);
    }
    
    weights_ready_ = true;
    
    fprintf(stderr, "[EncodingLayer] Self-allocated weights (Pattern B): "
            "qkv=[%d,%d] W_o=[%d,%d] FFN=[%d,%d→%d,%d] residual_scale=%.6f%s\n",
            qkv_out_dim, d_model, d_model, d_model,
            d_model, d_ff, d_ff, d_model, residual_scale,
            config_.use_layer_scale ? " +LayerScale" : "");
}

EncodingLayer::~EncodingLayer() {
    freeWeights();
    // NOTE: config_.cublas_handle is NOT owned by EncodingLayer (Rule 22)
    // TrainingState owns it - do NOT destroy!
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
    , alpha_q_(std::move(other.alpha_q_))
    , alpha_k_(std::move(other.alpha_k_))
{
    // Null out the moved-from object
    other.config_.cublas_handle = nullptr;
    other.weights_ready_ = false;
}

EncodingLayer& EncodingLayer::operator=(EncodingLayer&& other) noexcept {
    if (this != &other) {
        freeWeights();
        // NOTE: config_.cublas_handle is NOT owned - do NOT destroy (Rule 22)
        
        config_ = other.config_;
        weights_ready_ = other.weights_ready_;
        config_.cublas_handle = other.config_.cublas_handle;
        rms1_gamma_ = std::move(other.rms1_gamma_);
        rms2_gamma_ = std::move(other.rms2_gamma_);
        W_qkv_ = std::move(other.W_qkv_);
        b_qkv_ = std::move(other.b_qkv_);
        W_o_ = std::move(other.W_o_);
        b_o_ = std::move(other.b_o_);
        ffn_ = std::move(other.ffn_);
        layer_scale1_ = std::move(other.layer_scale1_);
        layer_scale2_ = std::move(other.layer_scale2_);
        alpha_q_ = std::move(other.alpha_q_);
        alpha_k_ = std::move(other.alpha_k_);
        
        other.config_.cublas_handle = nullptr;
        other.weights_ready_ = false;
    }
    return *this;
}

void EncodingLayer::setConfig(const EncodingConfig& cfg) {
    cfg.validate("EncodingLayer::setConfig");
    config_ = cfg;
}

void EncodingLayer::freeWeights() {
    // Tensor handles its own memory cleanup via destructor (owns_data=true).
    rms1_gamma_ = Tensor();
    rms2_gamma_ = Tensor();
    W_qkv_ = Tensor();
    b_qkv_ = Tensor();
    W_o_ = Tensor();
    b_o_ = Tensor();
    ffn_.reset();
    weights_ready_ = false;
}

void EncodingLayer::validateReady(const char* context) const {
    if (!weights_ready_) {
        throw std::runtime_error(std::string(context) +
            ": weights not initialized! Call allocateWeights() first.");
    }
    if (!config_.cublas_handle) {
        throw std::runtime_error(std::string(context) +
            ": cuBLAS handle not initialized in config!");
    }
    if (!ffn_) {
        throw std::runtime_error(std::string(context) +
            ": FFN sublayer not initialized! Call allocateWeights() first.");
    }
    if (config_.d_model % config_.num_heads != 0) {
        throw std::runtime_error(std::string(context) +
            ": d_model must be divisible by num_heads (head_dim must be integral)");
    }
}

// NOTE: requiredWorkspaceBytes() DELETED (Rule 20/26)
// REASON: encoder_workspace was allocated by InitTrainingState.cu based on this method's output,
// but NOTHING in the forward/backward path ever consumed it. The autograd forward pass creates
// its own intermediate Tensors. This was orphaned GPU memory.

//======================================================//
//  Forward Pass - Autograd Implementation with ForwardIntermediates (Issue #56 Fix)
//======================================================//

Tensor EncodingLayer::forward(const Tensor& input, int seq_len, cudaStream_t stream,
                               ForwardIntermediates& intermediates,
                               uint64_t training_step,
                               int layer_idx) {
    if constexpr (kEnableEncoderStepLogs) {
        fprintf(stderr, "[EncoderFwd] START total_tokens=%d seq_len=%d\n", 
                input.shape.flat.rows, seq_len);
    }
    validateReady("EncodingLayer::forward");
    
    // CRITICAL: Set autograd cuBLAS handle before any autograd::matmul calls
    // The handle must be set per-call since it's thread_local
    autograd::set_autograd_cublas_handle(config_.cublas_handle);
    if constexpr (kEnableEncoderStepLogs) {
        fprintf(stderr, "[EncoderFwd] autograd cuBLAS handle set: %p\n", (void*)config_.cublas_handle);
    }
    
    // Validate input
    if (!input.data) {
        throw std::runtime_error("EncodingLayer::forward: input.data is NULL");
    }
    if (!stream) {
        throw std::runtime_error("EncodingLayer::forward: stream is NULL");
    }
    
    const int total_tokens = input.shape.flat.rows;
    const int d_model = config_.d_model;
    
    if (input.shape.flat.cols != d_model) {
        throw std::runtime_error("EncodingLayer::forward: input d_model mismatch. "
                                 "Expected " + std::to_string(d_model) + 
                                 ", got " + std::to_string(input.shape.flat.cols));
    }
    if (total_tokens % seq_len != 0) {
        throw std::runtime_error("EncodingLayer::forward: total_tokens not divisible by seq_len");
    }
    
    const int batch_size = total_tokens / seq_len;
    const int num_heads = config_.num_heads;
    const int num_kv_heads = config_.effectiveKVHeads();
    const int head_dim = config_.headDim();
    const int qkv_debug = qkvDebugLevel();
    if constexpr (kEnableEncoderStepLogs) {
        fprintf(stderr, "[EncoderFwd] validated: batch=%d heads=%d kv_heads=%d head_dim=%d\n", 
                batch_size, num_heads, num_kv_heads, head_dim);
    }
    
    //--------------------------------------------------
    // 1. RMSNorm1: input -> ln1_out
    // Issue #56: Store in intermediates to keep autograd graph alive
    //--------------------------------------------------
    intermediates.ln1_out = autograd::rms_norm(input, rms1_gamma_, config_.rms_epsilon, stream);
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
    intermediates.qkv_out = autograd::matmul(intermediates.ln1_out, W_qkv_, stream, nullptr, nullptr, true);
    if (qkv_debug > 0) {
        const bool always_log = (qkv_debug >= 2);
        logTensorNonFinite("AutogradQKV:qkv_out_prebias", intermediates.qkv_out, stream, always_log);
    }
    
    // ========================================================================
    // [QKV_EQUATION] DIAGNOSTIC (Rule 21) - Equation-based logging
    // Formula: qkv_out = ln1_out @ W_qkv^T + b_qkv
    // ========================================================================
    if constexpr (GRIM::VerboseLogging::ENABLE_TRAINING_SIGNAL_DIAGNOSTICS) {
    if (isEquationLoggingEnabled()) {
        const int qkv_dim_local = config_.d_model + 2 * config_.kvDim();
        const int d_model_local = config_.d_model;
        const int num_heads_local = config_.num_heads;
        const int head_dim_local = d_model_local / config_.num_heads;
        
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

        // Actual Q magnitudes in matching units.
        const float actual_q_elem_rms = qkv_rms;
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

            // Best-available IDs in this layer scope.
            // NOTE: Global optimizer step is not available here.
            const int batch_idx_local = 0;  // Not available in layer scope
            const int step_idx_local = 0;  // Not available in layer scope
            
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
                EQ_LOG("QKV_PROJECTION_EQUATION", eq.str(), batch_idx_local, layer_idx_local, step_idx_local, 
                       GRIM::EquationPhase::QKV_PROJECTION);
            }
        }
    }
    } // if constexpr ENABLE_TRAINING_SIGNAL_DIAGNOSTICS (QKV_EQUATION)
    
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 2: QKV matmul DONE, adding bias...\n");
    // ISSUE #97 FIX: Use autograd::broadcast_add for proper gradient tracking
    // Previously: launchFFNBiasAdd bypassed autograd, so b_qkv never received gradients
    // Now: autograd::broadcast_add creates BiasAddGradFn which computes grad_bias = sum(grad_output)
    if (config_.use_bias && b_qkv_.data) {
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
        batch_size, seq_len, num_heads, num_kv_heads, head_dim,
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
    // 3a. QK-Norm: Per-head RMSNorm on Q and K (Dehghani et al. 2023)
    //     Stabilizes attention logit magnitude across training.
    //     Applied BEFORE RoPE so rotation operates on unit-RMS vectors.
    //--------------------------------------------------
    if (config_.qk_norm_enabled && alpha_q_.data && alpha_k_.data) {
        if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 3a: QK-Norm...\n");
        intermediates.Q_bhsd = autograd::qk_rms_norm(
            intermediates.Q_bhsd, alpha_q_, num_heads, seq_len, head_dim, 1e-6f, stream);
        intermediates.K_bhsd = autograd::qk_rms_norm(
            intermediates.K_bhsd, alpha_k_, num_kv_heads, seq_len, head_dim, 1e-6f, stream);
        if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 3a: QK-Norm DONE\n");
        if (qkv_debug > 0) {
            const bool always_log = (qkv_debug >= 2);
            logTensorNonFinite("AutogradQKV:Q_qknorm", intermediates.Q_bhsd, stream, always_log);
            logTensorNonFinite("AutogradQKV:K_qknorm", intermediates.K_bhsd, stream, always_log);
        }
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
        autograd::rope_rotation(
            intermediates.Q_bhsd, intermediates.K_bhsd,  // Tensor refs, not .data
            config_.pos_encoding->rope_inv_freq,
            batch_size, num_heads, num_kv_heads, seq_len, head_dim,
            config_.pos_encoding->rotary_dim, stream);
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
    // Compute per-step per-layer dropout seed using Knuth multiplicative hash
    // training_step=0 + attention_dropout=0.0 means no dropout (inference mode)
    const int layer_idx_for_seed = layer_idx;
    const uint64_t attn_dropout_seed = (training_step > 0 && config_.attention_dropout > 0.0f)
        ? (training_step * 2654435761ULL + 42 + 1000 * static_cast<uint64_t>(layer_idx_for_seed))
        : 0;
    intermediates.attn_out_bhsd = autograd::scaled_dot_product_attention(
        intermediates.Q_bhsd, intermediates.K_bhsd, intermediates.V_bhsd, 
        config_.pos_encoding->alibi_slopes, 0.0f, stream, true,
        config_.attention_dropout, attn_dropout_seed);
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
        intermediates.attn_out_bhsd, batch_size, seq_len, num_heads, head_dim, stream);
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 5: Reshape DONE (autograd tracked)\n");
    
    //--------------------------------------------------
    // 6. Output projection: attn_out @ W_o^T + b_o
    // Issue #56: Store in intermediates
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 6: Output projection...\n");
    // W_o is [d_model, d_model], so W_o^T is also [d_model, d_model]
    // Use transpose_b=true to compute attn_out @ W_o^T
    intermediates.proj_out = autograd::matmul(intermediates.attn_out, W_o_, stream, nullptr, nullptr, true);
    // ISSUE #97 FIX: Use autograd::broadcast_add for proper gradient tracking on b_o
    if (config_.use_bias && b_o_.data) {
        intermediates.proj_out = autograd::broadcast_add(intermediates.proj_out, b_o_, stream);
    }
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 6: Output projection DONE\n");
    
    //--------------------------------------------------
    // 6b. Post-attention sublayer dropout
    //     Standard transformer: residual = input + dropout(sublayer(norm(input)))
    //     Dropout applied BEFORE LayerScale and residual add.
    //--------------------------------------------------
    if (config_.dropout_rate > 0.0f && training_step > 0) {
        const uint64_t attn_proj_dropout_seed = training_step * 2654435761ULL + 100 + layer_idx;
        intermediates.proj_out = autograd::dropout(intermediates.proj_out, config_.dropout_rate,
                                                   attn_proj_dropout_seed, true, stream);
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
    const Tensor& proj_for_residual = (config_.use_layer_scale && layer_scale1_.data)
        ? (scaled_proj = autograd::layer_scale(intermediates.proj_out, layer_scale1_, stream), scaled_proj)
        : intermediates.proj_out;
    
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
    intermediates.residual1 = autograd::add(input, proj_for_residual, stream);
    
    // ========================================================================
    // RESIDUAL CENTERING (Issue #118 / Mode Collapse Fix)
    //
    // WHY: Causal attention creates a shared output component from prefix tokens.
    //   Residual accumulation + RMSNorm direction preservation amplifies this
    //   shared direction through layers: ρ grows +0.01-0.04 per layer.
    //   Over 12 layers: ρ(emb)=0.05 → ρ(final)=0.44 → mode collapse.
    //
    // WHAT: center_columns subtracts the cross-position mean for each feature:
    //   h[t,d] -= mean_t(h[t,d])   for each feature d
    //   This removes the rank-1 shared direction at each layer.
    //
    // GRADIENT COST: The centering projection P = I - 11^T/n has backward
    //   grad_input = P * grad_output. Per-layer attenuation = (1 - 1/n_tokens).
    //   For n_tokens ≈ 6000: (1 - 1/6000)^24 ≈ 0.996. Negligible.
    // ========================================================================
    if (config_.center_encoder_residuals) {
        intermediates.residual1 = autograd::center_columns(intermediates.residual1, stream);
    }
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 7: Residual1 (pre-norm, no sandwich) DONE\n");
    
    
    //--------------------------------------------------
    // 8. RMSNorm2: residual1 -> ln2_out
    // Issue #56: Store in intermediates
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 8: RMSNorm2...\n");
    intermediates.ln2_out = autograd::rms_norm(intermediates.residual1, rms2_gamma_, config_.rms_epsilon, stream);
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 8: RMSNorm2 DONE\n");
    
    //--------------------------------------------------
    // 9. FFN: ln2_out -> ffn_out (already using autograd)
    // Issue #56: FFN also stores its intermediates in this same ForwardIntermediates
    // (ffn_gate_up_out, ffn_swiglu_out are written by SwiGLU FFN forward)
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 9: FFN...\n");
    intermediates.ffn_out = ffn_->forward(intermediates.ln2_out, intermediates, training_step, layer_idx);
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 9: FFN DONE\n");
    
    //--------------------------------------------------
    // 9b. Post-FFN sublayer dropout
    //     Standard transformer: residual = input + dropout(sublayer(norm(input)))
    //     Dropout applied BEFORE LayerScale and residual add.
    //--------------------------------------------------
    if (config_.dropout_rate > 0.0f && training_step > 0) {
        const uint64_t ffn_dropout_seed = training_step * 2654435761ULL + 200 + layer_idx;
        intermediates.ffn_out = autograd::dropout(intermediates.ffn_out, config_.dropout_rate,
                                                  ffn_dropout_seed, true, stream);
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
    const Tensor& ffn_for_residual = (config_.use_layer_scale && layer_scale2_.data)
        ? (scaled_ffn = autograd::layer_scale(intermediates.ffn_out, layer_scale2_, stream), scaled_ffn)
        : intermediates.ffn_out;
    
    // ========================================================================
    // STANDARD PRE-NORM RESIDUAL (Issue #148: Sandwich Norm REMOVED)
    //
    // Architecture: output = residual1 + LayerScale(ffn_out)
    //
    // No post-residual normalization. Matches standard PyTorch GPT pre-norm.
    // ========================================================================
    intermediates.output = autograd::add(intermediates.residual1, ffn_for_residual, stream);
    
    // Residual centering after FFN sublayer (same rationale as post-attention above)
    if (config_.center_encoder_residuals) {
        intermediates.output = autograd::center_columns(intermediates.output, stream);
    }
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
    // ═══════════════════════════════════════════════════════════════════════════
    if constexpr (GRIM::VerboseLogging::ENABLE_TRAINING_SIGNAL_DIAGNOSTICS) {
    if (isEquationLoggingEnabled() && (layer_idx == 0 || layer_idx == 11)) {
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
        
        // Log LayerScale values if available
        float ls1_val = 0.0f, ls2_val = 0.0f;
        if (config_.use_layer_scale && layer_scale1_.data) {
            cudaMemcpy(&ls1_val, layer_scale1_.data, sizeof(float), cudaMemcpyDeviceToHost);
        }
        if (config_.use_layer_scale && layer_scale2_.data) {
            cudaMemcpy(&ls2_val, layer_scale2_.data, sizeof(float), cudaMemcpyDeviceToHost);
        }
        
        std::ostringstream eq;
        eq << "[LAYER_COSINE_EQUATION] layer=" << layer_idx 
           << ": output = RMSNorm(RMSNorm(input + LS1*attn) + LS2*ffn)\n";
        eq << "  OUTPUT h_L" << layer_idx << ": shape=[" << total_tokens << ", " << d_model 
           << "] row_rms_range=[" << rms_min << ", " << rms_max << "]\n";
        eq << "  LAYERSCALE: LS1=" << ls1_val << " LS2=" << ls2_val << "\n";
        eq << "  ACTUAL avg_cos=" << avg_cos << " (pairs=" << num_pairs 
           << ") [|avg_cos|->1 = collapse, near 0 = diverse]\n";
        if (fabs(avg_cos) > 0.8) {
            eq << "  [ANOMALY] Layer " << layer_idx << " |avg_cos|=" << fabs(avg_cos) 
               << " HIGH - possible mode collapse!\n";
        }
        EQ_LOG("LAYER_COSINE_EQUATION", eq.str(), 0, layer_idx, 0, GRIM::EquationPhase::RESIDUAL_ADD);
    }
    } // if constexpr ENABLE_TRAINING_SIGNAL_DIAGNOSTICS (LAYER_COSINE_EQUATION)
    
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
