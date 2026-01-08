//======================================================//
//  Encoding_GPU.hpp
//  PRODUCTION-READY Transformer Encoder Layer
//  
//  Architecture: Pre-norm with RMSNorm (NOT LayerNorm)
//    Input -> RMSNorm -> Attention -> Residual -> RMSNorm -> FFN -> Residual -> Output
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
#include "../Attention/QKV_Projector.hpp"
#include "../FlashAttention/Flash_Attention_Kernal.hpp"
#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"
#include "../../Shared/PBM/PositionalBiasMethod.hpp"

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
    int min_seq_len_for_flash = 128;   // Minimum seq len to activate Flash Attention
    
    // Normalization
    float rms_epsilon = 1e-5f;
    
    // Attention
    bool causal_mask = true;
    float softmax_temperature = 1.0f;
    bool qk_norm_enabled = false;
    float qk_norm_scale = 8.0f;
    
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
//  Forward Arguments
//======================================================//

struct EncodingForwardArgs {
    // Required I/O - NULL pointers will throw
    const float* input = nullptr;   // [total_tokens, d_model]
    float* output = nullptr;        // [total_tokens, d_model]
    
    // Dimensions - MUST be specified
    int total_tokens = 0;           // batch * seq_len
    int seq_len = 0;                // Sequence length (batch = total_tokens / seq_len)
    
    // Execution - REQUIRED (Rule 20: no fallbacks)
    cudaStream_t stream = nullptr;  // Caller MUST provide valid stream, nullptr will throw
    
    // Optional entropy output for attention diagnostics
    // Size: [batch_size * num_heads] floats (batch_size = total_tokens / seq_len)
    // Flash Attention forward kernel computes mean entropy per (batch, head)
    float* entropy_output = nullptr;  // nullptr = don't compute entropy (zero overhead)

    // Flash Attention BF16 scratch buffers (required for FA v2)
    __nv_bfloat16* fa_q_bf16 = nullptr;
    __nv_bfloat16* fa_k_bf16 = nullptr;
    __nv_bfloat16* fa_v_bf16 = nullptr;
    __nv_bfloat16* fa_out_bf16 = nullptr;
    size_t fa_q_bf16_elems = 0;
    size_t fa_kv_bf16_elems = 0;
    
    // Optional caches for backward pass (all nullptr = inference mode)
    // These names MUST match EncoderLayerCache in grim_language_model_cuda.hpp!
    
    // RMSNorm caches
    float* cache_ln1_input = nullptr;    // [total_tokens, d_model] - input to RMS1 (layer input)
    float* cache_ln1_out = nullptr;       // [total_tokens, d_model] - RMS1 output (matches ln1_output)
    float* cache_attn_input = nullptr;    // [total_tokens, d_model] - input to attention (same as ln1_out)
    float* cache_ln2_input = nullptr;     // [total_tokens, d_model] - input to RMS2 (residual1)
    float* cache_ln2_out = nullptr;       // [total_tokens, d_model] - RMS2 output (matches ln2_output)
    
    // Attention caches (BHSD format for Flash Attention backward)
    float* cache_q = nullptr;             // [batch, num_heads, seq, head_dim]
    float* cache_k = nullptr;             // [batch, num_kv_heads, seq, head_dim]
    float* cache_v = nullptr;             // [batch, num_kv_heads, seq, head_dim]
    float* cache_attn_bhsd = nullptr;     // [batch, num_heads, seq, head_dim] - attn out before W_o
    float* cache_softmax_lse = nullptr;   // [batch, num_heads, seq] - FP32 dense LSE
    float* cache_attn_out = nullptr;      // [batch, num_heads, seq, head_dim] - alias for cache_attn_bhsd
    float* cache_attn_output = nullptr;   // [total_tokens, d_model] - after W_o projection
    
    // FFN caches
    float* cache_ffn_input = nullptr;     // [total_tokens, d_model] - FFN input (same as ln2_out)
    float* cache_ffn_pre_gelu = nullptr;  // [total_tokens, d_ff]
    float* cache_ffn_output = nullptr;    // [total_tokens, d_ff]
    
    // Residual caches
    float* cache_residual1 = nullptr;     // [total_tokens, d_model]
    float* cache_layer_output = nullptr;  // [total_tokens, d_model] - final layer output
    
    // Validation
    void validate(const char* context) const {
        if (!input) {
            throw std::invalid_argument(std::string(context) + ": input MUST NOT be null");
        }
        if (!output) {
            throw std::invalid_argument(std::string(context) + ": output MUST NOT be null");
        }
        if (total_tokens <= 0) {
            throw std::invalid_argument(std::string(context) + 
                ": total_tokens MUST be > 0, got " + std::to_string(total_tokens));
        }
        if (seq_len <= 0) {
            throw std::invalid_argument(std::string(context) + 
                ": seq_len MUST be > 0, got " + std::to_string(seq_len));
        }
        if (total_tokens % seq_len != 0) {
            throw std::invalid_argument(std::string(context) + 
                ": total_tokens (" + std::to_string(total_tokens) + 
                ") must be divisible by seq_len (" + std::to_string(seq_len) + ")");
        }
    }
    
    int batchSize() const { return (seq_len > 0) ? (total_tokens / seq_len) : 0; }
};

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
    explicit EncodingLayer(const EncodingConfig& cfg);
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
    // Forward Pass
    // NOTE: backward() DELETED per Rule 20 - use BackwardPhase2_Encoder.cu
    //--------------------------------------------------
    void forward(const EncodingForwardArgs& args);
    
    //--------------------------------------------------
    // Weight Management
    //--------------------------------------------------
    void allocateWeights();  // Call once after setConfig
    void ensureWeightStorage() { allocateWeights(); }  // Alias for compatibility
    bool weightsAllocated() const noexcept { return weights_allocated_; }
    
    // RMSNorm weights (gamma only - NO BETA, this is RMSNorm not LayerNorm!)
    float* getRMS1Gamma() { return rms1_gamma_; }
    float* getRMS2Gamma() { return rms2_gamma_; }
    float* getRMS1GammaGrad() { return rms1_gamma_grad_; }
    float* getRMS2GammaGrad() { return rms2_gamma_grad_; }
    
    // Attention weights
    // W_qkv layout: [W_q: d_model x d_model][W_k: kv_dim x d_model][W_v: kv_dim x d_model]
    // Total size: (d_model + 2*kv_dim) * d_model
    float* getAttnWqkv() { return W_qkv_; }
    float* getAttnBqkv() { return b_qkv_; }
    float* getAttnWo() { return W_o_; }
    float* getAttnBo() { return b_o_; }
    
    // Attention weight gradients
    float* getAttnWqkvGrad() { return W_qkv_grad_; }
    float* getAttnBqkvGrad() { return b_qkv_grad_; }
    float* getAttnWoGrad() { return W_o_grad_; }
    float* getAttnBoGrad() { return b_o_grad_; }
    
    // FFN weights (owned by FeedForwardLayer)
    float* getFFNW1() { return ffn_ ? ffn_->getW1() : nullptr; }
    float* getFFNB1() { return ffn_ ? ffn_->getB1() : nullptr; }
    float* getFFNW2() { return ffn_ ? ffn_->getW2() : nullptr; }
    float* getFFNB2() { return ffn_ ? ffn_->getB2() : nullptr; }
    
    // NOTE: FFN gradient buffers are managed by TrainingState, not EncodingLayer
    // Use TrainingState::ffn_w1_grads[layer], ffn_b1_grads[layer], etc.
    
    //--------------------------------------------------
    // Flash Attention Control
    //--------------------------------------------------
    void setFlashAttention(bool enable, int min_seq_len = 128) {
        config_.use_flash_attention = enable;
        config_.min_seq_len_for_flash = min_seq_len;
    }
    
    //--------------------------------------------------
    // Workspace
    //--------------------------------------------------
    std::size_t requiredWorkspaceBytes(int total_tokens, int seq_len) const;
    void setWorkspace(float* workspace, std::size_t bytes);
    
    // NOTE: CRTP interface (forwardImpl, backwardImpl, applyGradientsImpl) DELETED per Rule 20
    // Use forward(EncodingForwardArgs&) directly. Training uses BackwardPhase2_Encoder.cu.
    
private:
    void freeWeights();
    void validateReady(const char* context) const;
    
    EncodingConfig config_{};
    bool weights_allocated_ = false;
    
    // NOTE: cuBLAS handle is in config_.cublas_handle (NOT owned - Rule 22)
    
    // RMSNorm weights (gamma only!)
    float* rms1_gamma_ = nullptr;
    float* rms2_gamma_ = nullptr;
    float* rms1_gamma_grad_ = nullptr;
    float* rms2_gamma_grad_ = nullptr;
    
    // Attention weights
    float* W_qkv_ = nullptr;        // [(d_model + 2*kv_dim), d_model]
    float* b_qkv_ = nullptr;        // [d_model + 2*kv_dim]
    float* W_o_ = nullptr;          // [d_model, d_model]
    float* b_o_ = nullptr;          // [d_model]
    
    // Attention weight gradients
    float* W_qkv_grad_ = nullptr;
    float* b_qkv_grad_ = nullptr;
    float* W_o_grad_ = nullptr;
    float* b_o_grad_ = nullptr;
    
    // FFN layer (owns its own weights)
    std::unique_ptr<FeedForwardLayer> ffn_;
    
    // Workspace (NOT owned - provided by caller)
    float* workspace_ = nullptr;
    std::size_t workspace_bytes_ = 0;
};

} // namespace GRIM
