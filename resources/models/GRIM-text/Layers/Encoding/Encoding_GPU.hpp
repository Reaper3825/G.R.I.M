//======================================================//
//  Encoding_GPU.hpp
//  PRODUCTION-READY Transformer Encoder Layer
//  
//  Architecture: Sandwich Norm with RMSNorm (NOT LayerNorm)
//    Input -> RMSNorm -> Attention -> Residual -> RMSNorm(sandwich) -> RMSNorm -> FFN -> Residual -> RMSNorm(sandwich) -> Output
//  
//  Features:
//    - GQA (Grouped Query Attention) native support
//    - Flash Attention for memory efficiency
//    - Modular components from Layers/
//    - Strict validation with clear error messages
//    - Zero tolerance for misconfiguration
//  
//  WARNING: This file does NOT use grim_transformer_gpu.hpp
//           All attention is done via Flash Attention kernels
//======================================================//

#pragma once

#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <memory>

#include "../grim_layer_gpu.hpp"
#include "../LayernNorm/RMSNorm_GPU.hpp"
#include "../FeedForward/Feed_Forward_GPU.hpp"
#include "../FlashAttention/Flash_Attention_Kernal.hpp"
#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"
#include "../../Shared/PBM/PositionalBiasMethod.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../Shared/TensorContract/ForwardIntermediates.hpp"

namespace GRIM {

//======================================================//
//  Configuration - MUST be fully specified
//======================================================//

struct EncodingConfig {
    // Model dimensions - ALL REQUIRED (no defaults that could hide bugs)
    int d_model = 0;          // Hidden dimension (MUST be > 0)
    int num_heads = 0;        // Number of Q heads (MUST be > 0)
    int num_kv_heads = 0;     // Number of KV heads for GQA (0 = use num_heads for MHA)
    int d_ff = 0;             // FFN hidden dimension (MUST be > 0)
    
    // Flash Attention
    bool use_flash_attention = true;   // Use Flash Attention 2 for memory efficiency
    int min_seq_len_for_flash = 0;     // REQUIRED - set from hyperparameters (no defaults)
    
    // Normalization
    float rms_epsilon = 1e-5f;
    
    // LayerScale (Issue #109 fix for input row correlation)
    // When enabled, residuals become: residual = input + layer_scale * sublayer_output
    // This reduces the correlation buildup through layers by dampening sublayer contributions
    bool use_layer_scale = true;
    float layer_scale_init = 0.1f;  // Initial scale (CaiT uses 0.1, can go lower for more layers)
    
    // Per-layer residual centering
    // When true: output = center_columns(input + branch) — prevents mode collapse but 24 gradient projections
    // When false: output = input + branch — standard pre-norm, better gradient flow
    bool center_encoder_residuals = false;
    
    // Bias control - when false, skip bias addition in attention projections (b_qkv, b_o)
    bool use_bias = true;
    
    // Attention
    bool causal_mask = true;
    float softmax_temperature = 1.0f;
    bool qk_norm_enabled = false;
    float qk_norm_scale = 8.0f;
    float dropout_rate = 0.0f;        // Sublayer dropout DROP rate (0.0 = disabled). Applied after attention projection and FFN output.
    float attention_dropout = 0.0f;   // Attention dropout DROP rate (0.0 = disabled, 0.15 = 15% dropped)
    
    // Positional encoding (ALiBi+RoPE hybrid) - pointer to shared state
    // WARNING: If nullptr, attention sees no positional info - all positions equivalent!
    const PBM::PBMSpec* pos_encoding = nullptr;
    
    // Execution (Rule 22: Use centralized handles from TrainingState)
    cudaStream_t stream = nullptr;
    cublasHandle_t cublas_handle = nullptr;  // MUST be training_state.cublas_handle
    
    // Validation helper - throws if invalid
    void validate(const char* context) const {
        if (d_model <= 0) {
            throw std::invalid_argument(std::string(context) + 
                ": d_model MUST be > 0, got " + std::to_string(d_model));
        }
        if (num_heads <= 0) {
            throw std::invalid_argument(std::string(context) + 
                ": num_heads MUST be > 0, got " + std::to_string(num_heads));
        }
        if (d_ff <= 0) {
            throw std::invalid_argument(std::string(context) + 
                ": d_ff MUST be > 0, got " + std::to_string(d_ff));
        }
        if (d_model % num_heads != 0) {
            throw std::invalid_argument(std::string(context) + 
                ": d_model (" + std::to_string(d_model) + 
                ") must be divisible by num_heads (" + std::to_string(num_heads) + ")");
        }
        
        // GQA validation
        const int effective_kv_heads = (num_kv_heads > 0) ? num_kv_heads : num_heads;
        if (num_heads % effective_kv_heads != 0) {
            throw std::invalid_argument(std::string(context) + 
                ": num_heads (" + std::to_string(num_heads) + 
                ") must be divisible by num_kv_heads (" + std::to_string(effective_kv_heads) + ")");
        }
    }
    
    // Computed properties
    int headDim() const { return (num_heads > 0) ? (d_model / num_heads) : 0; }
    int effectiveKVHeads() const { return (num_kv_heads > 0) ? num_kv_heads : num_heads; }
    int kvDim() const { return effectiveKVHeads() * headDim(); }
    bool isGQA() const { return num_kv_heads > 0 && num_kv_heads < num_heads; }
};

//======================================================//
//  NOTE: EncodingForwardArgs DELETED per Rule 20
//  
//  Autograd forward takes Tensor directly:
//    Tensor forward(const Tensor& input, int seq_len, cudaStream_t stream);
//  
//  Autograd tape handles all caching for backward pass automatically.
//======================================================//

//======================================================//
//  NOTE: EncodingBackwardArgs DELETED per Rule 20
//  
//  Training backward pass uses BackwardPhase2_Encoder.cu
//  which has its own argument handling via BackwardContext.
//  This struct was dead code - never called by production training.
//======================================================//

//======================================================//
//  EncodingLayer - Clean Implementation (Forward-Only)
//======================================================//

class EncodingLayer final : public Layer<EncodingLayer, float> {
public:
    static constexpr LayerType layer_type = LayerType::kEncoding;
    
    //--------------------------------------------------
    // Construction
    //--------------------------------------------------
    EncodingLayer() = default;
    
    /// Self-allocating constructor — layer owns its weights
    /// @param cfg       Fully-populated EncodingConfig (d_model, d_ff, num_heads, etc.)
    /// @param seed      Base PRNG seed.  Offsets: +0 W_qkv, +1 W_o, +2 FFN W1, +3 FFN W2
    /// @param residual_scale  Issue #142: GPT-2 init scaling for W_o (1/sqrt(2*num_layers))
    /// @param layer_scale_init Issue #109: CaiT LayerScale initial value (only when config.use_layer_scale)
    EncodingLayer(const EncodingConfig& cfg, uint64_t seed,
                  float residual_scale = 1.0f,
                  float layer_scale_init = 1.0f);
    
    ~EncodingLayer();
    
    // No copy (cuBLAS handle)
    EncodingLayer(const EncodingLayer&) = delete;
    EncodingLayer& operator=(const EncodingLayer&) = delete;
    
    // Move OK
    EncodingLayer(EncodingLayer&& other) noexcept;
    EncodingLayer& operator=(EncodingLayer&& other) noexcept;
    
    //--------------------------------------------------
    // Configuration
    //--------------------------------------------------
    void setConfig(const EncodingConfig& cfg);
    const EncodingConfig& config() const noexcept { return config_; }
    
    //--------------------------------------------------
    // Forward Pass - Pure Autograd with ForwardIntermediates
    // backward() handled automatically via output.backward()
    //--------------------------------------------------
    /**
     * Encoder forward with autograd tracking (Issue #56 Fix)
     * 
     * Architecture:
     *   ln1 = RMSNorm(input)
     *   attn_out = Attention(ln1) + input  (residual)
     *   ln2 = RMSNorm(attn_out)
     *   output = FFN(ln2) + attn_out  (residual)
     * 
     * CRITICAL: Caller MUST provide ForwardIntermediates storage for this layer.
     * All intermediate tensors are moved to the storage to keep the autograd
     * graph alive until backward completes. Without this, grad_fn objects are
     * destroyed when this function returns, causing use-after-free in backward.
     * 
     * @param input [total_tokens, d_model] - encoder input (from embedding or prev layer)
     * @param seq_len sequence length (for attention masking)
     * @param stream CUDA stream for execution
     * @param intermediates Storage for this layer's intermediate tensors (REQUIRED for autograd)
     * @param training_step Current training step for per-step dropout seed generation (0 = no dropout)
     * @param layer_idx Layer index within encoder stack (for equation logging and dropout seed)
     * @return output [total_tokens, d_model] with grad_fn attached
     */
    Tensor forward(const Tensor& input, int seq_len, cudaStream_t stream,
                   struct ForwardIntermediates& intermediates,
                   uint64_t training_step = 0,
                   int layer_idx = 0);
    
    //--------------------------------------------------
    // Weight Management (Pattern B: self-allocated)
    //--------------------------------------------------
    bool weightsReady() const noexcept { return weights_ready_; }
    
    //--------------------------------------------------
    // Tensor Accessors (use these for ALL access)
    //--------------------------------------------------
    
    // RMSNorm (pre-norm)
    Tensor& rms1Gamma() { return rms1_gamma_; }
    Tensor& rms2Gamma() { return rms2_gamma_; }
    
    // Sandwich norm (post-residual)
    Tensor& rmsPostAttnGamma() { return rms_post_attn_gamma_; }
    Tensor& rmsPostFfnGamma() { return rms_post_ffn_gamma_; }
    
    // Attention weights/biases
    Tensor& attnWqkv() { return W_qkv_; }
    Tensor& attnBqkv() { return b_qkv_; }
    Tensor& attnWo() { return W_o_; }
    Tensor& attnBo() { return b_o_; }
    
    // FFN weights/biases (delegates to FeedForwardLayer)
    // Rule 20: ffn_ MUST be initialized - crash if null
    Tensor& ffnW1() { return ffn_->W1(); }
    Tensor& ffnB1() { return ffn_->b1(); }
    Tensor& ffnW2() { return ffn_->W2(); }
    Tensor& ffnB2() { return ffn_->b2(); }
    
    // Direct access to FFN layer (for autograd forward)
    FeedForwardLayer* getFfnLayer() { return ffn_.get(); }
    
    // LayerScale (Issue #109)
    Tensor& layerScale1() { return layer_scale1_; }
    Tensor& layerScale2() { return layer_scale2_; }
    
    //--------------------------------------------------
    // Flash Attention Control
    //--------------------------------------------------
    void setFlashAttention(bool enable, int min_seq_len) {
        if (min_seq_len <= 0) {
            throw std::runtime_error("setFlashAttention: min_seq_len must be > 0 (configured from hyperparameters)");
        }
        config_.use_flash_attention = enable;
        config_.min_seq_len_for_flash = min_seq_len;
    }
    
    //--------------------------------------------------
    // Workspace Budget — DELETED (Rule 20/26)
    // The autograd forward creates its own Tensors; encoder_workspace was never consumed.
    //--------------------------------------------------
    
    // NOTE: CRTP interface (forwardImpl, backwardImpl, applyGradientsImpl) DELETED per Rule 20
    // Use Tensor forward(const Tensor& input, int seq_len, cudaStream_t) directly.
    // Training backward uses BackwardPhase2_Encoder.cu.
    
private:
    void freeWeights();
    void validateReady(const char* context) const;
    
    /// Pattern B: self-allocate and Xavier-init all weights + create FFN
    void allocateWeights(uint64_t seed, float residual_scale, float layer_scale_init);
    
    EncodingConfig config_{};
    bool weights_ready_ = false;  // Set by allocateWeights()
    
    // NOTE: cuBLAS handle is in config_.cublas_handle (NOT owned - Rule 22)
    
    // RMSNorm weights (Tensor with requires_grad=true)
    Tensor rms1_gamma_;    // [d_model] - pre-attention norm
    Tensor rms2_gamma_;    // [d_model] - pre-FFN norm
    
    // Sandwich norm weights (post-residual normalization)
    Tensor rms_post_attn_gamma_;  // [d_model] - after attention residual
    Tensor rms_post_ffn_gamma_;   // [d_model] - after FFN residual
    
    // Attention weights (Tensor with requires_grad=true)
    // W_qkv layout: [W_q: d_model x d_model][W_k: kv_dim x d_model][W_v: kv_dim x d_model]
    Tensor W_qkv_;         // [(d_model + 2*kv_dim), d_model]
    Tensor b_qkv_;         // [d_model + 2*kv_dim]
    Tensor W_o_;           // [d_model, d_model]
    Tensor b_o_;           // [d_model]
    
    // FFN layer (owns its own weights as Tensors)
    std::unique_ptr<FeedForwardLayer> ffn_;
    
    // LayerScale parameters (Issue #109: reduces correlation buildup)
    // These are learnable scalars applied to sublayer outputs before residual addition:
    //   residual1 = input + layer_scale1 * attn_output
    //   residual2 = residual1 + layer_scale2 * ffn_output
    Tensor layer_scale1_;  // [1] scalar for attention
    Tensor layer_scale2_;  // [1] scalar for FFN
};

} // namespace GRIM
