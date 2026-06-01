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
#include "EncoderDiagnostics.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../FlashAttention/EncoderSelfAttention_GPU.hpp"
#include "../FeedForward/Feed_Forward_GPU.hpp"
#include "../../Shared/PBM/PositionalBiasMethod.hpp"
#include "../../Shared/StreamController/StreamController_GPU.hpp"
#include <cmath>
#include <cuda_runtime.h>
#include <cstring>
#include <stdexcept>
#include <string>
#include <cstdio>     // fprintf, snprintf


namespace {
    constexpr bool kEnableEncoderStepLogs = false;  // Set true to enable [EncoderFwd] step logs
}
//======================================================// 


namespace GRIM {

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

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = (call); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        char msg[512]; \
        snprintf(msg, sizeof(msg), "cuBLAS ERROR at %s:%d - %s: status=%d", \
                 __FILE__, __LINE__, #call, static_cast<int>(status)); \
        throw std::runtime_error(msg); \
    } \
} while(0)

namespace {

uint64_t mixEncoderForwardSeed(uint64_t batch_idx, uint64_t layer_nonce) {
    uint64_t x = (batch_idx + 1ULL) * 0x9E3779B97F4A7C15ULL;
    x ^= layer_nonce + 0xBF58476D1CE4E5B9ULL + (x << 6) + (x >> 2);
    return x;
}

void validateLayerScaleGamma(const Tensor& gamma, const char* name, int d_model, const char* context) {
    if (!gamma.data) {
        throw std::runtime_error(std::string(context) + ": " + name + " is NULL while LayerScale is enabled");
    }
    const std::string shape_context = std::string(context) + " " + name;
    gamma.shape.require(shape_context.c_str());
    if (!gamma.shape.is_2d_layout()) {
        throw std::runtime_error(std::string(context) + ": " + name + " must be a 2D [1,d_model] gamma vector");
    }
    const auto dims = gamma.shape.as_2d();
    if (dims.rows != 1 || dims.cols != d_model) {
        throw std::runtime_error(std::string(context) + ": " + name + " must have shape [1,d_model]. expected=[1," +
                                 std::to_string(d_model) + "] got=[" + std::to_string(dims.rows) + "," +
                                 std::to_string(dims.cols) + "]");
    }
}

}  // anonymous namespace



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
 EncodingLayer::EncodingLayer(const HyperParameters::EncoderLayerConstructionHP& hp_snapshot,
                                        uint64_t seed,
                                        cudaStream_t init_stream)
     : hp_(hp_snapshot) {
     validateConstructionSnapshot("EncodingLayer::EncodingLayer");
    allocateWeights(seed, init_stream);
}

void EncodingLayer::allocateWeights(uint64_t seed,
                                    cudaStream_t init_stream) {
    if (weights_ready_) {
        throw std::runtime_error("EncodingLayer::allocateWeights: weights already initialized! "
                                 "Cannot allocate twice.");
    }
    if (!init_stream) {
        throw std::runtime_error("EncodingLayer::allocateWeights: init_stream is NULL");
    }
    validateConstructionSnapshot("EncodingLayer::allocateWeights");
    
    const auto& hp = hp_;
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
        CUDA_CHECK(cudaPeekAtLastError());
    };

    auto fillValue = [stream](Tensor& t, float value, const char* context) {
        const int count = static_cast<int>(t.numel());
        const int threads = 256;
        const int blocks = (count + threads - 1) / threads;
        kernel_encoding_fill_value<<<blocks, threads, 0, stream>>>(t.data, count, value);
        (void)context;
        CUDA_CHECK(cudaPeekAtLastError());
    };
    
    //==================================================//
    //  RMSNorm gammas (2x) — initialized to 1.0
    //==================================================//
    rms1_gamma_ = Tensor::zeros({d_model}, stream, "enc_rms1_gamma_own");
    if (!hp.freeze_learned_rms_gammas) {
        rms1_gamma_.requires_grad_();
        rms1_gamma_.ensure_grad();
    }
    fillOnes(rms1_gamma_);
    
    rms2_gamma_ = Tensor::zeros({d_model}, stream, "enc_rms2_gamma_own");
    if (!hp.freeze_learned_rms_gammas) {
        rms2_gamma_.requires_grad_();
        rms2_gamma_.ensure_grad();
    }
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
    //  FFN — compute layer borrows registry-owned tensors
    //==================================================//
    {
        const HyperParameters::FeedForwardLayerConstructionHP ffn_hp =
            HyperParameters::feedForwardLayerConstructionHP(hp);
        ffn_ = std::make_unique<FeedForwardLayer>(ffn_hp);
    }
    
    //==================================================//
    //  LayerScale (Issue #109) — learnable per-channel gamma vectors [1, d_model]
    //==================================================//
    if (hp.use_layer_scale) {
        layer_scale1_ = Tensor::zeros({1, d_model}, stream, "enc_layer_scale1_own");
        layer_scale1_.requires_grad_();
        layer_scale1_.ensure_grad();
        fillValue(layer_scale1_, hp.layer_scale_init, "EncodingLayer::allocateWeights fill LayerScale1");
        
        layer_scale2_ = Tensor::zeros({1, d_model}, stream, "enc_layer_scale2_own");
        layer_scale2_.requires_grad_();
        layer_scale2_.ensure_grad();
        fillValue(layer_scale2_, hp.layer_scale_init, "EncodingLayer::allocateWeights fill LayerScale2");
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    weights_ready_ = true;
    
    fprintf(stderr, "[EncodingLayer] Initialized encoder-owned tensors and bound registry-owned FFN tensors\n");
}

EncodingLayer::~EncodingLayer() {
    freeWeights();
}

EncodingLayer::EncodingLayer(EncodingLayer&& other) noexcept
    : hp_(other.hp_)
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
    other.weights_ready_ = false;
}

EncodingLayer& EncodingLayer::operator=(EncodingLayer&& other) noexcept {
    if (this != &other) {
        freeWeights();
        
        hp_ = other.hp_;
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
    validateConstructionSnapshot(context);
}

void EncodingLayer::validateConstructionSnapshot(const char* context) const {
    if (hp_.d_model <= 0 || hp_.num_layers <= 0 || hp_.qkv_dim <= 0) {
        throw std::invalid_argument(std::string(context) +
            ": invalid encoder construction snapshot d_model=" + std::to_string(hp_.d_model) +
            " num_layers=" + std::to_string(hp_.num_layers) +
            " qkv_dim=" + std::to_string(hp_.qkv_dim));
    }
}

// NOTE: requiredWorkspaceBytes() DELETED (Rule 20/26)
// REASON: encoder_workspace was allocated by InitTrainingState.cu based on this method's output,
// but NOTHING in the forward/backward path ever consumed it. The autograd forward pass creates
// its own intermediate Tensors. This was orphaned GPU memory.

//======================================================//
//  Forward Pass - Autograd Implementation writing into ModelForwardOutputs
//======================================================//

void EncodingLayer::forward(const Tensor& input, const BatchPayload& payload,
                            const PBM::PBMState& pos_encoding,
                            cudaStream_t stream, cublasHandle_t cublas_handle,
                            Forward::ModelForwardOutputs& forward_outputs,
                            uint64_t batch_idx,
                            bool dropout_enabled,
                            int layer_idx,
                            const EncodingLayerParameterViews* parameter_views) {
    validateReady("EncodingLayer::forward");
    if (layer_idx < 0) {
        throw std::runtime_error("EncodingLayer::forward: layer_idx must be >= 0, got " + std::to_string(layer_idx));
    }
    const size_t layer_slot = static_cast<size_t>(layer_idx);
    forward_outputs.validateLayerIndex(layer_slot, "EncodingLayer::forward");
    Tensor& ln1_out = forward_outputs.ln1_out_per_layer[layer_slot];
    Tensor& residual1 = forward_outputs.residual1_per_layer[layer_slot];
    Tensor& ln2_out = forward_outputs.ln2_out_per_layer[layer_slot];
    Tensor& scaled_proj = forward_outputs.scaled_proj_per_layer[layer_slot];
    Tensor& scaled_ffn = forward_outputs.scaled_ffn_per_layer[layer_slot];
    Tensor& proj_out = forward_outputs.proj_out_per_layer[layer_slot];
    Tensor& ffn_out = forward_outputs.ffn_out_per_layer[layer_slot];
    Tensor& output = forward_outputs.output_per_layer[layer_slot];
    const auto& hp = hp_;
    const float residual_scale = 1.0f / std::sqrt(2.0f * static_cast<float>(hp.num_layers));
    const Tensor& rms1_gamma = (parameter_views && parameter_views->rms1_gamma) ? *parameter_views->rms1_gamma : rms1_gamma_;
    const Tensor& rms2_gamma = (parameter_views && parameter_views->rms2_gamma) ? *parameter_views->rms2_gamma : rms2_gamma_;
    const Tensor& W_qkv = (parameter_views && parameter_views->W_qkv) ? *parameter_views->W_qkv : W_qkv_;
    const Tensor& b_qkv = (parameter_views && parameter_views->b_qkv) ? *parameter_views->b_qkv : b_qkv_;
    const Tensor& W_o = (parameter_views && parameter_views->W_o) ? *parameter_views->W_o : W_o_;
    const Tensor& b_o = (parameter_views && parameter_views->b_o) ? *parameter_views->b_o : b_o_;
    const Tensor& layer_scale1 = (parameter_views && parameter_views->layer_scale1) ? *parameter_views->layer_scale1 : layer_scale1_;
    const Tensor& layer_scale2 = (parameter_views && parameter_views->layer_scale2) ? *parameter_views->layer_scale2 : layer_scale2_;
    Tensor& layer_scale1_for_op = (parameter_views && parameter_views->layer_scale1)
        ? *const_cast<Tensor*>(parameter_views->layer_scale1)
        : layer_scale1_;
    Tensor& layer_scale2_for_op = (parameter_views && parameter_views->layer_scale2)
        ? *const_cast<Tensor*>(parameter_views->layer_scale2)
        : layer_scale2_;
    
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
    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
    input.shape.require("EncodingLayer::forward input");
    if (!input.shape.is_2d_layout()) {
        throw std::runtime_error("EncodingLayer::forward: input must be a 2D [total_tokens,d_model] tensor");
    }
    const auto& input_shape = input.shape.as_2d();
    if constexpr (kEnableEncoderStepLogs) {
        fprintf(stderr, "[EncoderFwd] START total_tokens=%d seq_len=%d\n", 
                input_shape.rows, payload.max_seq_len);
    }
    
    // CRITICAL: Set autograd cuBLAS handle before any autograd::matmul calls.
    // The handle is supplied by the caller's forward payload/request and is thread_local.
    autograd::set_autograd_cublas_handle(cublas_handle);
    if constexpr (kEnableEncoderStepLogs) {
        fprintf(stderr, "[EncoderFwd] autograd cuBLAS handle set: %p\n", (void*)cublas_handle);
    }
    
    if (input_shape.cols != hp.d_model) {
        throw std::runtime_error("EncodingLayer::forward: input d_model mismatch. "
                                 "Expected " + std::to_string(hp.d_model) + 
                                 ", got " + std::to_string(input_shape.cols));
    }
    if (input_shape.rows != payload.total_tokens) {
        throw std::runtime_error("EncodingLayer::forward: input rows (" + std::to_string(input_shape.rows) +
                                 ") != BatchPayload.total_tokens (" + std::to_string(payload.total_tokens) + ")");
    }
    
    if (hp.center_encoder_residuals) {
        if (static_cast<int>(payload.seq_lengths.size()) != payload.batch_size) {
            throw std::runtime_error("EncodingLayer::forward: payload.seq_lengths size (" +
                                     std::to_string(payload.seq_lengths.size()) +
                                     ") != payload.batch_size (" + std::to_string(payload.batch_size) + ")");
        }
        for (int b = 0; b < payload.batch_size; ++b) {
            const int row_len = payload.seq_lengths[static_cast<size_t>(b)];
            if (row_len <= 1 || row_len > payload.max_seq_len) {
                throw std::runtime_error("EncodingLayer::forward: center_encoder_residuals invalid seq_lengths[" +
                                         std::to_string(b) + "]=" + std::to_string(row_len) +
                                         " for payload.max_seq_len=" + std::to_string(payload.max_seq_len));
            }
        }
    }
    if constexpr (kEnableEncoderStepLogs) {
        fprintf(stderr, "[EncoderFwd] validated: batch=%d heads=%d kv_heads=%d head_dim=%d\n", 
                payload.batch_size, hp.num_heads, hp.num_kv_heads, hp.head_dim);
    }
    const uint64_t dropout_batch_seed = mixEncoderForwardSeed(batch_idx, static_cast<uint64_t>(layer_idx) + 1ULL);
    
    //--------------------------------------------------
    // 1. RMSNorm1: input -> ln1_out
    // Retained in ModelForwardOutputs.
    //--------------------------------------------------
    ln1_out = autograd::rms_norm(input, rms1_gamma, hp.rms_epsilon, stream);
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 1: RMSNorm1 DONE\n");
    
    //--------------------------------------------------
    // 2. Attention sublayer: ln1_out -> proj_out
    // Attention owns QKV projection, RoPE/ALiBi, SDPA, diagnostics, dropout seed,
    // BHSD flattening, and output projection. Encoder only supplies the PBM spec.
    //--------------------------------------------------
    const HyperParameters::EncoderSelfAttentionHP attention_hp =
        HyperParameters::encoderSelfAttentionHP(hp, dropout_enabled);
    Attention::EncoderSelfAttentionForwardRequest attention_request{
        payload,
        attention_hp,
        stream,
        cublas_handle,
        dropout_batch_seed,
        layer_idx
    };
    Attention::encoderSelfAttentionForward(
        ln1_out,
        W_qkv,
        b_qkv,
        W_o,
        b_o,
        pos_encoding,
        attention_request,
        forward_outputs);
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 2: Attention facade DONE\n");
    
    //--------------------------------------------------
    // 6b. Post-attention sublayer dropout
    //     Standard transformer: residual = input + dropout(sublayer(norm(input)))
    //     Dropout applied BEFORE LayerScale and residual add.
    //--------------------------------------------------
    if (hp.dropout_rate > 0.0f && dropout_enabled) {
        const uint64_t attn_proj_dropout_seed = dropout_batch_seed * 2654435761ULL + 100 + layer_idx;
        const uint64_t attn_proj_dropout_mask_stream = 0x0001000000000000ULL + static_cast<uint64_t>(layer_idx);
        proj_out = autograd::dropout(proj_out, hp.dropout_rate,
                                     attn_proj_dropout_seed, stream,
                                     attn_proj_dropout_mask_stream);
    }
    proj_out = autograd::mul_scalar(proj_out, residual_scale, stream);
    
    //--------------------------------------------------
    // 7. Residual1: input + proj_out -> residual1
    // Issue #56: Store on ModelForwardOutputs
    // Issue #109: Apply LayerScale to proj_out before residual addition
    // Note: Standard pre-norm architecture (Issue #148: Sandwich Norm removed).
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 7: Residual1...\n");
    
    // Issue #109: LayerScale gating for attention sublayer
    const Tensor* proj_for_residual = &proj_out;
    if (hp.use_layer_scale) {
        validateLayerScaleGamma(layer_scale1, "layer_scale1_", hp.d_model, "EncodingLayer::forward");
        scaled_proj = autograd::layer_scale(proj_out, layer_scale1_for_op, stream);
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
    residual1 = autograd::add(input, *proj_for_residual, stream);
    
    // ========================================================================
    // RESIDUAL CENTERING (Issue #118 / Mode Collapse Fix)
    //
    // WHY: Causal attention creates a shared output component from prefix tokens.
    //   Residual accumulation + RMSNorm direction preservation amplifies this
    //   shared direction through layers: ρ grows +0.01-0.04 per layer.
    //   Over 12 layers: ρ(emb)=0.05 → ρ(final)=0.44 → mode collapse.
    //
    // WHAT: center_columns_by_causal_prefix_lengths subtracts the strict-past
    //   prefix mean over VALID (unpadded) tokens for each feature WITHIN EACH
    //   BATCH ROW:
    //   h[b,0,d] = h[b,0,d]
    //   h[b,t,d] -= mean_{u < t}(h[b,u,d])   for valid t > 0
    //   h[b,t,d] = 0                         for padded t
    //   This removes the running shared direction at each layer without making
    //   sample A depend on sample B, including PAD rows, or leaking future
    //   positions into token t. Strict-past also preserves the first token
    //   instead of erasing it with mean_{u <= 0}.
    //
    // GRADIENT COST: The centering projection is lower-triangular. Backward
    //   applies its transpose within each sequence; no gradient path crosses
    //   from future inputs into earlier forward positions.
    // ========================================================================
    if (hp.center_encoder_residuals) {
        if (payload.max_seq_len <= 1) {
            throw std::runtime_error("EncodingLayer::forward: center_encoder_residuals requires payload.max_seq_len > 1; single-row column centering would erase the residual stream");
        }
        residual1 = autograd::center_columns_by_causal_prefix_lengths(
            residual1, payload.seq_lengths, payload.batch_size, payload.max_seq_len, stream);
    }
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 7: Residual1 (pre-norm, no sandwich) DONE\n");
    
    
    //--------------------------------------------------
    // 8. RMSNorm2: residual1 -> ln2_out
    // Retained in ModelForwardOutputs.
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 8: RMSNorm2...\n");
    ln2_out = autograd::rms_norm(residual1, rms2_gamma, hp.rms_epsilon, stream);
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 8: RMSNorm2 DONE\n");
    
    //--------------------------------------------------
    // 9. FFN: ln2_out -> ffn_out (already using autograd)
    // Retained in ModelForwardOutputs.
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 9: FFN...\n");
    if (!parameter_views) {
        throw std::runtime_error("EncodingLayer::forward: parameter_views is NULL - caller must pass registry-derived FFN parameter views");
    }
    ffn_->forward(ln2_out,
                  stream, cublas_handle,
                  forward_outputs,
                  dropout_batch_seed, dropout_enabled, layer_idx,
                  parameter_views->ffn);
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 9: FFN DONE\n");
    
    //--------------------------------------------------
    // 9b. Post-FFN sublayer dropout
    //     Standard transformer: residual = input + dropout(sublayer(norm(input)))
    //     Dropout applied BEFORE LayerScale and residual add.
    //--------------------------------------------------
    if (hp.dropout_rate > 0.0f && dropout_enabled) {
        const uint64_t ffn_dropout_seed = dropout_batch_seed * 2654435761ULL + 200 + layer_idx;
        const uint64_t ffn_dropout_mask_stream = 0x0002000000000000ULL + static_cast<uint64_t>(layer_idx);
        ffn_out = autograd::dropout(ffn_out, hp.dropout_rate,
                                    ffn_dropout_seed, stream,
                                    ffn_dropout_mask_stream);
    }
    ffn_out = autograd::mul_scalar(ffn_out, residual_scale, stream);
    
    //--------------------------------------------------
    // 10. Residual2: residual1 + ffn_out -> output
    // Issue #56: The final output IS stored in intermediates too
    // for consistency, but we also return it
    // Issue #109: Apply LayerScale to ffn_out before residual addition
    // Issue #118: Apply centering to remove common direction before residual add
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 10: Residual2...\n");
    
    // Issue #109: LayerScale gating for FFN sublayer
    const Tensor* ffn_for_residual = &ffn_out;
    if (hp.use_layer_scale) {
        validateLayerScaleGamma(layer_scale2, "layer_scale2_", hp.d_model, "EncodingLayer::forward");
        scaled_ffn = autograd::layer_scale(ffn_out, layer_scale2_for_op, stream);
        ffn_for_residual = &scaled_ffn;
    }
    
    // ========================================================================
    // STANDARD PRE-NORM RESIDUAL (Issue #148: Sandwich Norm REMOVED)
    //
    // Architecture: output = residual1 + LayerScale(ffn_out)
    //
    // No post-residual normalization. Matches standard PyTorch GPT pre-norm.
    // ========================================================================
    output = autograd::add(residual1, *ffn_for_residual, stream);
    
    // Issue #155: Post-FFN centering REMOVED from here — moved to AutogradTraining.cu
    // so it happens AFTER all layer-output modifications (including crossAttentionRead).
    // Post-attention centering remains here (between sublayers, no external modification).
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 10: Residual2 (pre-norm, no sandwich) DONE - layer COMPLETE\n");
    bool emitLayerResidualDiag = false;
    if (emitLayerResidualDiag){
    EncoderDiagnostics::emitLayerResidualDiagnostic({
        input,
        proj_out,
        scaled_proj,
        residual1,
        ffn_out,
        *ffn_for_residual,
        output,
        hp.use_layer_scale ? &layer_scale1 : nullptr,
        hp.use_layer_scale ? &layer_scale2 : nullptr,
        hp,
        payload,
        stream,
        layer_idx,
        emitLayerResidualDiag
    });
    }
    if (!output.data) {
        throw std::runtime_error("EncodingLayer::forward: result.output.data is NULL before return");
    }
} 

} // namespace GRIM
