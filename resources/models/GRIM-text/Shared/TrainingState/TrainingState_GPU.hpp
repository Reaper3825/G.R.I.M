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
    struct FlashAttentionBF16Scratch;
}

// AutogradIntermediates: owns all intermediate tensors during forward→backward
#include "../../training/Autograd/AutogradIntermediates.hpp"

namespace GRIM {

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
    //
    // All weight tensors live in TrainingTensors (tensors_->), for BOTH training and inference.
    // Training: TrainingTensors::initializeParams() allocates with gradients.
    // Inference: InitinferenceState.cu allocates without gradients.
    // Use these accessors for uniform access — crash if tensors_ is null:
    Tensor& getEmbeddingWeights();
    const Tensor& getEmbeddingWeights() const;
    Tensor& getPositionEmbeddingWeights();
    const Tensor& getPositionEmbeddingWeights() const;
    Tensor& getLmHeadWeights();
    const Tensor& getLmHeadWeights() const;
    
    // NOTE: lm_head_bias, numeric_head_weights, numeric_head_bias, final_rms_gamma
    // are owned by TrainingTensors (tensors_->). Access via tensors_->X.
    
    // Learned loss weighting (homoscedastic uncertainty)
    // log_var = log(σ²), loss = L / (2*exp(log_var)) + 0.5*log_var
    Tensor log_var_text;     // [1] - learned log-variance for text CE loss
    Tensor log_var_numeric;  // [1] - learned log-variance for numeric loss
    
    //======================================================//
    //  QK-NORM LEARNED SCALES (nGPT-style)
    //======================================================//
    std::vector<Tensor> attn_alpha_q;       // [num_heads] per layer
    std::vector<Tensor> attn_alpha_k;       // [num_kv_heads] per layer

    // GQA configuration (stored for cache sizing)
    int num_heads = 0;           // Q heads
    int num_kv_heads = 0;        // K,V heads (GQA: num_kv_heads < num_heads)

    //======================================================//
    //  SCRATCH BUFFERS (pre-allocated GPU memory for forward/backward)
    //  NOTE: Autograd Tensors (with grad_fn) live in autograd_intermediates.
    //  These are just raw pre-allocated buffers that autograd wraps via from_ptr.
    //======================================================//
    Tensor cached_encoder_output;       // [max_tokens, d_model] Pre-allocated scratch
    Tensor cached_logits_tensor;        // [max_tokens, vocab_size] Pre-allocated scratch
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
    //  AUTOGRAD STATE
    //======================================================//
    float cached_loss_value = 0.0f;
    float cached_text_loss = 0.0f;
    float cached_numeric_loss = 0.0f;
    int   cached_numeric_count = 0;     // Issue #137: atom count for weight grad normalization
    
    // Owns ALL intermediate tensors during forward→backward cycle
    // Replaces old autograd_ctx (which mixed input args with tensor storage)
    Autograd::AutogradIntermediates autograd_intermediates;

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
    
    // DELETED: FA bf16/dq_accum/dsoftmax_sum buffers — FlashAttentionLayer::ensureScratch() self-manages.
    // Autograd ScaledDotProductAttentionGradFn also self-allocates backward buffers.
    
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
    //  AUTOGRAD SYSTEM STATE
    //======================================================//
    std::unique_ptr<TrainingTensors> tensors_;
    bool use_autograd_tensors = false;
    
    void initializeAutogradTensors(int vocab_size, int d_model, int d_ff,
                                   int num_layers, int num_heads, int num_kv_heads,
                                   int max_seq_len, bool tie_embeddings, bool use_bias,
                                   HyperParameters::PositionalEncodingType positional_encoding,
                                   bool numeric_head_enabled = false,
                                   bool use_layer_scale = false,
                                   float layer_scale_init = 1.0f,
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
    //  ISSUE #141 FIX: SCRATCHBLOCK GRADIENT TAP BUFFER
    //  Captures encoder input gradient before dropout consumes it.
    //  Used by ScratchBlock backward to compute parameter gradients.
    //======================================================//
    Tensor scratchblock_grad_tap;
    
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
    
    void allocateGuessCacheBuffers(size_t capacity, bool enable_diversity, 
                                   size_t diversity_bloom_bits, size_t pinned_buffer_size);
    void freeGuessCacheBuffers();
    
    // DELETED: batch_prep_* vectors (Rule 20) — replaced by BatchPayload struct
    
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

