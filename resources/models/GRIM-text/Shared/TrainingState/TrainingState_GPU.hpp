//======================================================//
//  TrainingState_GPU.hpp
//  Standalone TrainingState declaration for GPU training
//  
//  Rule 20 MIGRATION: All raw float* converted to GRIM::Tensor
//  - Tensor provides: shape info, automatic cleanup, autograd support
//  - Access raw pointer via tensor.data when needed for CUDA kernels
//======================================================//

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <memory>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>

#include "../TeacherLogits/TeacherLogits_GPU.hpp"
#include "../ScratchBlock/ScratchBlock_GPU.hpp"
#include "../StreamController/StreamController_GPU.hpp"
#include "../GradNorm/GradNormGPU.hpp"
#include "../PBM/PositionalBiasMethod.hpp"
#include "../TensorContract/TensorContract_GPU.hpp"

// Forward declaration for autograd tensor system
namespace GRIM {
    struct TrainingTensors;
    namespace Autograd {
        struct AutogradContext;  // Issue #47: Forward context persists for backward
    }
    
    struct FlashAttentionBF16Scratch;
}

namespace GRIM {

//======================================================//
//  Per-Layer Encoder Cache (Tensor-based)
//======================================================//  
struct EncoderLayerCacheTensors {
    Tensor ln1_output;        // [batch*seq, d_model] After first layer norm 
    Tensor attn_input;        // [batch*seq, d_model] Input to attention (after LN1)
    Tensor attn_bhsd;         // [batch, heads, seq, head_dim] Attention output before W_o
    Tensor softmax_lse;       // [batch, num_heads, seq] Log-sum-exp for FA backward
    Tensor attn_output;       // [batch*seq, d_model] After attention
    Tensor residual1;         // [batch*seq, d_model] After first residual
    Tensor ln2_output;        // [batch*seq, d_model] After second layer norm
    Tensor ffn_pre_gelu;      // [batch*seq, d_ff] Before GELU activation
    Tensor ffn_output;        // [batch*seq, d_ff] After FFN
    Tensor layer_output;      // [batch*seq, d_model] Final layer output
    
    // QKV cache for attention backward (BHSD format)
    Tensor Q;                 // [batch, num_heads, seq, head_dim]
    Tensor K;                 // [batch, num_kv_heads, seq, head_dim] 
    Tensor V;                 // [batch, num_kv_heads, seq, head_dim]
};

struct TrainingState {
    TrainingState();
    ~TrainingState();

    TrainingState(const TrainingState&) = delete;
    TrainingState& operator=(const TrainingState&) = delete;
    TrainingState(TrainingState&&) = delete;
    TrainingState& operator=(TrainingState&&) = delete;

    //======================================================//
    //  PARAMETER TENSORS (weights + gradients via autograd)
    //======================================================//
    // Rule 20: NO raw float* for gradients - use GRIM::Tensor with autograd
    
    // Embedding weights [vocab_size, d_model]
    // NOTE: When tie_embeddings=true, lm_head_weights shares data with embedding_weights
    Tensor embedding_weights;
    
    // Position embedding weights [max_seq_len, d_model]
    // Issue #36 FIX: Position embeddings MUST be trainable to match PyTorch baseline
    Tensor position_embedding_weights;

    // LM head weights [vocab_size, d_model] and optional bias [vocab_size]
    Tensor lm_head_weights;
    Tensor lm_head_bias;  // [vocab_size] - optional, only if use_bias=true

    // Numeric head for number prediction
    Tensor numeric_head_weights;  // [d_model]
    Tensor numeric_head_bias;     // [1]
    
    // Learned loss weighting (homoscedastic uncertainty)
    // log_var = log(σ²), loss = L / (2*exp(log_var)) + 0.5*log_var
    Tensor log_var_text;     // [1] - learned log-variance for text CE loss
    Tensor log_var_numeric;  // [1] - learned log-variance for numeric loss

    // Final RMSNorm before LM head (Issue #33 fix)
    Tensor final_rms_gamma;  // [d_model]
    
    //======================================================//
    //  QK-NORM LEARNED SCALES (nGPT-style)
    //======================================================//
    std::vector<Tensor> attn_alpha_q;       // [num_heads] per layer
    std::vector<Tensor> attn_alpha_k;       // [num_kv_heads] per layer

    // GQA configuration (stored for cache sizing)
    int num_heads = 0;           // Q heads
    int num_kv_heads = 0;        // K,V heads (GQA: num_kv_heads < num_heads)

    //======================================================//
    //  ACTIVATION CACHES (Tensor-managed for backward pass)
    //======================================================//
    
    // Input layer embedding cache
    Tensor cached_embeddings_tensor;    // [max_tokens, d_model]
    
    // Per-layer activation caches (Tensor-based)
    std::vector<EncoderLayerCacheTensors> encoder_layer_caches;

    // Output layer caches
    Tensor cached_encoder_output;       // [max_tokens, d_model] Final encoder output
    Tensor cached_final_rms_input;      // [max_tokens, d_model] Issue #33: Input to final RMSNorm
    Tensor cached_logits_tensor;        // [max_tokens, vocab_size]
    Tensor cached_numeric_predictions;  // [max_tokens]
    
    // Target/input ID caches (int typed)
    Tensor cached_targets_tensor;       // [max_tokens] int32
    Tensor cached_token_ids_tensor;     // [max_tokens] int32
    Tensor cached_token_numeric_values; // [max_tokens] float
    Tensor cached_token_numeric_mask;   // [max_tokens] uint8
    
    // GRMT v4: text features for ScratchBlock
    Tensor cached_token_text_features;  // [max_tokens * kTextFeatureDim] FP16
    Tensor cached_token_text_mask;      // [max_tokens] uint8
    
    int cached_batch_size = 0;
    int cached_seq_len = 0;
    int cached_valid_tokens = 0;
    
    //======================================================//
    //  INCREMENTAL KV CACHE STATE (autoregressive generation)
    //======================================================//
    int kv_cache_len = 0;           // Number of tokens with valid K,V in cache
    int kv_cache_capacity = 0;      // Maximum tokens the cache can hold
    
    // Single-token buffers for incremental generation
    Tensor single_token_logits;      // [vocab_size]
    Tensor single_token_hidden;      // [d_model]
    Tensor single_token_embedding;   // [d_model]
    
    int cached_num_layers = 0;
    
    int max_cached_batch = 0;
    int max_cached_seq_len = 0;
    size_t max_cached_tokens = 0;
    size_t max_logit_tokens = 0;
    TeacherLogits::Buffer teacher_logits;
    TeacherLogits::Buffer reference_logits;
    Tensor sequence_weights_tensor;    // [max_sequences]
    int sequence_weight_count = 0;
    int sequence_weight_capacity = 0;
    
    //======================================================//
    //  ISSUE #38: Token weighting (DEPRECATED - not used)
    //======================================================//
    Tensor token_weights_tensor;       // [vocab_size] inverse frequency weights
    int token_weights_count = 0;

    //======================================================//
    //  AUTOGRAD LOSS TENSORS (Issue #46 FIX)
    //======================================================//
    Tensor loss_tensor;                   // Scalar [1] - loss value + grad_fn
    Tensor logits_tensor;                 // [total_tokens, vocab_size]
    float cached_loss_value = 0.0f;
    float cached_text_loss = 0.0f;
    float cached_numeric_loss = 0.0f;
    
    // Issue #47: Full computation graph for backward
    std::unique_ptr<Autograd::AutogradContext> autograd_ctx;

    //======================================================//
    //  INTERMEDIATE GRADIENT TENSORS (Issue #45 FIX)
    //======================================================//
    Tensor grad_logits_tensor;            // [max_logit_tokens, vocab_size]
    Tensor grad_numeric_tensor;           // [max_logit_tokens]
    Tensor grad_encoder_tensor;           // [max_tokens, d_model]
    Tensor grad_ffn_input_tensor;         // [max_tokens, d_model]
    Tensor grad_ffn_hidden_tensor;        // [max_tokens, d_ff]
    Tensor grad_attn_input_tensor;        // [max_tokens, d_model]
    Tensor grad_attn_out_proj_tensor;     // [max_tokens, d_model]
    Tensor grad_attn_out_bhsd_tensor;     // [max_tokens, d_model]
    Tensor grad_q_tensor;                 // [batch, heads, seq, head_dim]
    Tensor grad_k_tensor;                 // [batch, kv_heads, seq, head_dim]
    Tensor grad_v_tensor;                 // [batch, kv_heads, seq, head_dim]
    Tensor grad_qkv_concat_tensor;        // [max_tokens, total_qkv_dim]
    Tensor grad_qkv_input_tensor;         // [max_tokens, d_model]
    Tensor grad_attn_bsm_tensor;          // [max_tokens, d_model]
    
    /// Zero all intermediate gradient tensors
    void zeroIntermediateGrads(cudaStream_t stream);
    
    //======================================================//
    //  FLASH ATTENTION WORKSPACE
    //======================================================//
    Tensor fa_dq_accum;         // [batch, heads, seq, head_dim] FP32
    Tensor fa_dsoftmax_sum;     // [batch, heads, seq] FP32
    
    // BF16 conversion buffers for FA v2 (use bf16 typed Tensor)
    Tensor fa_q_bf16;           // BSHD layout
    Tensor fa_k_bf16;
    Tensor fa_v_bf16;
    Tensor fa_out_bf16;
    Tensor fa_dout_bf16;
    Tensor fa_dq_bf16;
    Tensor fa_dk_bf16;
    Tensor fa_dv_bf16;
    size_t fa_q_bf16_elems = 0;
    size_t fa_kv_bf16_elems = 0;
    
    //======================================================//
    //  Issue #43 FIX: Centering Scratch Buffer
    //======================================================//
    Tensor centering_scratch_tensor;
    size_t centering_scratch_elems() const { return centering_scratch_tensor.numel(); }
    float* centered_activation_scratch() const { return centering_scratch_tensor.data; }

    //======================================================//
    //  LOSS COMPUTATION SCRATCH
    //======================================================//
    Tensor d_loss_scratch;         // Per-token losses
    Tensor d_loss_sum_scratch;     // Reduced loss sum (scalar)
    Tensor d_numeric_loss_sum;     // Numeric loss sum (scalar)
    Tensor d_numeric_loss_count;   // Numeric loss count (scalar int)
    
    // Attention entropy output
    Tensor d_entropy_output;       // [batch_size * num_heads]

    // Encoder workspace for GPU-native forward pass
    Tensor encoder_workspace;
    size_t encoder_workspace_size = 0;

    //======================================================//
    //  STREAM & GRADIENT MANAGEMENT
    //======================================================//
    StreamController stream_ctrl;
    GradNorm::GradNormController gradnorm_ctrl;
    cublasHandle_t cublas_handle = nullptr;

    //======================================================//
    //  POSITIONAL ENCODING STATE
    //======================================================//
    PBM::PBMState pbm_state;
    PBM::PBMSpec pbm_spec;
    bool pbm_initialized = false;

    // Scratch block pool
    ScratchBlock::ScratchBlockPool* scratch_pool = nullptr;
    bool scratch_enabled = true;

    // ScratchBlock reasoning layer caches (Tensor-based)
    Tensor cached_scratch_block_embeddings;  // [max_atoms, atom_embedding_dim]
    Tensor cached_scratch_block_positions;   // [max_atoms] int32
    Tensor cached_scratch_block_types;       // [max_atoms] int32
    Tensor cached_scratch_block_num_atoms;   // [1] int32

    //======================================================//
    //  AUTOGRAD SYSTEM FLAGS
    //======================================================//
    std::unique_ptr<TrainingTensors> tensors_;  // DEPRECATED - always nullptr
    bool use_autograd_tensors = false;
    
    void initializeAutogradTensors(int vocab_size, int d_model, int d_ff,
                                   int num_layers, int num_heads, int num_kv_heads,
                                   int max_seq_len, bool tie_embeddings, bool use_bias,
                                   HyperParameters::PositionalEncodingType positional_encoding,
                                   bool use_layer_scale = false,
                                   float layer_scale_init = 0.1f,
                                   uint64_t seed = 0,
                                   cudaStream_t stream = nullptr);

    //======================================================//
    //  OPTIMIZER STATE BUFFERS
    //======================================================//
    std::vector<Tensor> optimizer_m_states;  // First moment per param group
    std::vector<Tensor> optimizer_v_states;  // Second moment per param group
    bool optimizer_states_allocated = false;
    
    void allocateOptimizerStates(const std::vector<size_t>& sizes, cudaStream_t stream = nullptr);
    void freeOptimizerStates();

    bool initialized = false;
    uint64_t architecture_config_hash = 0;
    
    //======================================================//
    //  DEBUG GRADIENT ATTRIBUTION (Issue #60)
    //======================================================//
    bool debug_gradient_attribution = false;
    Tensor debug_lm_head_only_grad;
    Tensor debug_embedding_only_grad;
    
    void allocateDebugGradBuffers(int vocab_size, int d_model, cudaStream_t stream);
    void freeDebugGradBuffers();
    void logGradientAttribution(int batch_idx, cudaStream_t stream);

    //======================================================//
    //  ISSUE #60 FIX: PCGRAD BUFFER FOR TIED WEIGHTS
    //======================================================//
    Tensor pcgrad_temp_buffer;
    
    void allocatePCGradBuffer(int vocab_size, int d_model, cudaStream_t stream);
    void freePCGradBuffer();
    
    //======================================================//
    //  GUESS CACHE BUFFERS (GRIM-TS - typed buffers, NOT Tensors)
    //======================================================//
    struct GuessCacheBuffers {
        // These are typed buffers - NOT float Tensors, so stay as raw pointers
        void* records = nullptr;            // GuessRecord array
        uint64_t* keys = nullptr;           // Hash keys
        unsigned int* size = nullptr;       // Current size counter
        unsigned int* evict_cursor = nullptr;
        uint32_t* diversity_bloom = nullptr;
        float* calibration_offset = nullptr;
        void* single_meta_buffer = nullptr;
        float* single_reward_buffer = nullptr;
        
        // Pinned host memory (host side - not GPU Tensor)
        void* pinned_meta = nullptr;
        float* pinned_rewards = nullptr;
        size_t pinned_capacity = 0;
        
        size_t capacity = 0;
        size_t bloom_words = 0;
        bool allocated = false;
    };
    GuessCacheBuffers guess_cache_buffers;
    
    bool allocateGuessCacheBuffers(size_t capacity, bool enable_diversity, 
                                   size_t diversity_bloom_bits, size_t pinned_buffer_size);
    void freeGuessCacheBuffers();
    
    //======================================================//
    //  BATCH PREPARATION BUFFERS (CPU-side, not GPU Tensors)
    //======================================================//
    std::vector<int> batch_prep_input_ids;
    std::vector<int> batch_prep_target_ids;
    std::vector<float> batch_prep_numeric_values;
    std::vector<uint8_t> batch_prep_numeric_mask;
    std::vector<uint16_t> batch_prep_text_features;
    std::vector<uint8_t> batch_prep_text_mask;
    std::vector<int> batch_prep_valid_target_counts;
    size_t batch_prep_capacity = 0;
    
    //======================================================//
    //  CACHE ALLOCATION
    //======================================================//
    
    /// Allocate all activation caches for given dimensions
    void allocateActivationCaches(int max_batch, int max_seq_len, int num_layers,
                                  int d_model, int d_ff, int num_heads, int num_kv_heads,
                                  int vocab_size, cudaStream_t stream = nullptr);
    
    /// Allocate Flash Attention BF16 scratch buffers
    void allocateFlashAttentionBuffers(int max_batch, int max_seq_len, 
                                       int num_heads, int num_kv_heads, int head_dim,
                                       cudaStream_t stream = nullptr);
    
};

} // namespace GRIM

#else

namespace GRIM {
struct TrainingState {
    TrainingState() = default;
    ~TrainingState() = default;
};
} // namespace GRIM

#endif  // USE_CUDA

