//======================================================//
//  TrainingTensors.hpp
//  Autograd-enabled tensor storage replacing raw float* buffers
//======================================================//

#pragma once

#include "../TensorContract/TensorContract_GPU.hpp"
#include "../HyperParameters/HyperParameters_GPU.hpp"  // Issue #96: For PositionalEncodingType
#include <vector>
#include <cstddef>

#ifdef USE_CUDA
#include <cuda_runtime.h>

namespace GRIM {

/**
 * TrainingTensors - Replaces raw float* gradient buffers with GRIM::Tensor
 * 
 * Benefits over raw pointers:
 * - Automatic gradient computation via autograd
 * - Shape information preserved
 * - Proper memory ownership semantics
 * - Zero-grad, accumulate_grad, backward() built-in
 */
struct TrainingTensors {
    TrainingTensors() = default;
    ~TrainingTensors() = default;
    
    // Non-copyable, non-movable (owns GPU memory)
    TrainingTensors(const TrainingTensors&) = delete;
    TrainingTensors& operator=(const TrainingTensors&) = delete;
    TrainingTensors(TrainingTensors&&) = delete;
    TrainingTensors& operator=(TrainingTensors&&) = delete;
    
    //======================================================//
    //  PARAMETER TENSORS (weights that get optimized)
    //======================================================//
    
    // Embedding weights [vocab_size, d_model]
    Tensor embedding_weights;
    
    // Position embedding weights [max_seq_len, d_model]
    Tensor position_embedding_weights;
    
    // LM head weights [vocab_size, d_model]
    // NOTE: When tie_embeddings=true, this shares data with embedding_weights
    Tensor lm_head_weights;
    Tensor lm_head_bias;  // [vocab_size] - optional
    
    // Numeric head for number prediction
    Tensor numeric_head_weights;  // [d_model]
    Tensor numeric_head_bias;     // [1]
    
    // Final RMSNorm before LM head
    Tensor final_rms_gamma;  // [d_model]
    
    // Per-layer encoder parameters
    struct EncoderLayerParams {
        // Layer normalization (pre-norm)
        Tensor rms1_gamma;  // [d_model] - pre-attention norm
        Tensor rms2_gamma;  // [d_model] - pre-FFN norm
        
        // Sandwich norm (post-residual normalization)
        // Controls residual stream magnitude after each residual add
        Tensor rms_post_attn_gamma;  // [d_model] - post-attention residual norm
        Tensor rms_post_ffn_gamma;   // [d_model] - post-FFN residual norm
        
        // Attention
        Tensor attn_qkv_weight;  // [total_qkv_dim, d_model] where total_qkv_dim = d_model + 2*kv_dim
        Tensor attn_qkv_bias;    // [total_qkv_dim] - optional
        Tensor attn_out_weight;  // [d_model, d_model]
        Tensor attn_out_bias;    // [d_model] - optional
        
        // QK-norm scales (nGPT-style)
        Tensor alpha_q;  // [num_heads]
        Tensor alpha_k;  // [num_kv_heads]
        
        // FFN
        Tensor ffn_w1;  // [d_ff, d_model]
        Tensor ffn_b1;  // [d_ff] - optional
        Tensor ffn_w2;  // [d_model, d_ff]
        Tensor ffn_b2;  // [d_model] - optional
        
        // LayerScale (Issue #109: reduces correlation buildup)
        // Single scalar per residual connection, initialized to 0.1
        Tensor layer_scale1;  // [1] - scales attention output before residual
        Tensor layer_scale2;  // [1] - scales FFN output before residual
    };
    std::vector<EncoderLayerParams> encoder_layers;
    
    //======================================================//
    //  ACTIVATION CACHES (for backward pass)
    //======================================================//
    
    // Input layer
    Tensor cached_embeddings;  // [batch*seq, d_model]
    
    // Per-layer caches
    struct EncoderLayerCache {
        Tensor ln1_output;        // After first layer norm
        Tensor attn_output;       // After attention (before residual)
        Tensor residual1_output;  // After first residual
        Tensor ln2_output;        // After second layer norm
        Tensor ffn_pre_gelu;      // Before GELU activation
        Tensor ffn_output;        // After FFN (before residual)
        Tensor layer_output;      // Final output (after residual2)
        
        // QKV cache for attention backward (BHSD format)
        Tensor Q;  // [batch, num_heads, seq, head_dim]
        Tensor K;  // [batch, num_kv_heads, seq, head_dim]
        Tensor V;  // [batch, num_kv_heads, seq, head_dim]
        Tensor attn_input;   // Input to attention (after LN1)
        Tensor attn_bhsd;    // Attention output before W_o projection
        Tensor softmax_lse;  // [batch, num_heads, seq] FP32
    };
    std::vector<EncoderLayerCache> encoder_caches;
    
    // Output layer
    Tensor cached_encoder_output;  // Final encoder output
    Tensor cached_logits;          // [batch*seq, vocab_size]
    Tensor cached_final_rms_input; // Input to final RMSNorm
    
    //======================================================//
    //  INTERMEDIATE GRADIENT BUFFERS
    //======================================================//
    
    Tensor grad_logits;       // [batch*seq, vocab_size]
    Tensor grad_encoder_out;  // [batch*seq, d_model]
    
    // Reusable backward temporaries
    Tensor grad_ffn_input;
    Tensor grad_ffn_hidden;
    Tensor grad_attn_input;
    Tensor grad_attn_out_before_proj;
    Tensor grad_attn_out_reshaped;
    Tensor grad_q;
    Tensor grad_k;
    Tensor grad_v;
    Tensor grad_qkv_concat;
    Tensor grad_qkv_input;
    Tensor grad_attn_bsm_scratch;
    
    // Issue #43 centering scratch
    Tensor centered_activation_scratch;
    
    //======================================================//
    //  CONFIGURATION
    //======================================================//
    
    int num_layers = 0;
    int d_model = 0;
    int d_ff = 0;
    int num_heads = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    int vocab_size = 0;
    int max_seq_len = 0;
    bool tie_embeddings = true;
    bool use_bias = true;
    bool use_layer_scale = false;  // Issue #109: LayerScale to reduce correlation buildup
    float layer_scale_init = 0.1f;  // Initial value (CaiT paper uses 0.1)
    HyperParameters::PositionalEncodingType positional_encoding_type = HyperParameters::PositionalEncodingType::NONE;  // Stored for downstream queries
    
    //======================================================//
    //  INITIALIZATION
    //======================================================//
    
    /**
     * Initialize all parameter tensors with Xavier uniform initialization
     * @param config Model configuration
     * @param stream CUDA stream for async operations
     * @param positional_encoding Positional encoding type - only LEARNED modes need position embeddings
     * @param use_layer_scale Issue #109: Enable LayerScale to reduce correlation buildup
     * @param layer_scale_init_val Initial value for layer_scale (default 0.1 per CaiT)
     */
    void initializeParams(int vocab_size, int d_model, int d_ff, 
                          int num_layers, int num_heads, int num_kv_heads,
                          int max_seq_len, bool tie_embeddings, bool use_bias,
                          HyperParameters::PositionalEncodingType positional_encoding,
                          bool numeric_head_enabled = false,
                          bool use_layer_scale_flag = false,
                          float layer_scale_init_val = 0.1f,
                          uint64_t seed = 0,
                          cudaStream_t stream = nullptr);
    
    /**
     * Pre-allocate all gradient buffers
     * Called automatically by initializeParams() - needed so GradAccumulationController
     * can register the buffers at startup (before first backward pass)
     */
    void allocateAllGradients();
    
    /**
     * Allocate activation cache tensors for given batch/sequence dimensions
     */
    void allocateCaches(int batch_size, int seq_len, cudaStream_t stream = nullptr);
    
    /**
     * Check if tensors are initialized
     */
    bool isInitialized() const { return initialized_; }
    
private:
    bool initialized_ = false;
};

}  // namespace GRIM

#endif  // USE_CUDA
