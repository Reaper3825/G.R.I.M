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
#include "../ScratchBlock/ScratchBlockPool_GPU.hpp"
#include "../StreamController/StreamController_GPU.hpp"
#include "../GradNorm/GradNormGPU.hpp"
#include "../PBM/PositionalBiasMethod.hpp"
#include "../TensorContract/TensorContract_GPU.hpp"

// Forward declaration for autograd tensor system
namespace GRIM {
    class EmbeddingLayer;
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
    // ALL weight tensors are owned by Pattern B layers (self-managing):
    //   - Embedding: LanguageModel::getEmbeddingLayer()->tokenWeights() / positionWeights()
    //   - LM Head: LanguageModel::getLmHeadLayer()->weights() / bias() / finalRmsGamma()
    //   - Encoder: Each EncodingLayer self-allocates in constructor
    //   - ScratchBlock: ScratchBlockLayer self-allocates in constructor
    //
    // Session 7: TrainingTensors deleted — zero weight parameters remain in god object.
    // weight_init_seed stored directly on TrainingState for Pattern B layer construction.
    uint64_t weight_init_seed = 0;
    
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
    
    // Target/input ID caches (int typed)
    Tensor cached_targets_tensor;       // [max_tokens] int32
    Tensor mtp_shifted_targets_tensor;   // [max_logit_tokens] int32 — MTP shifted targets (one buffer, reused per head)
    Tensor cached_token_ids_tensor;     // [max_tokens] int32
    Tensor cached_token_numeric_values; // [max_tokens] float
    
    // Unified atom side-channel
    Tensor cached_token_text_features;  // [max_tokens * kTextFeatureDim] FP16
    Tensor cached_token_atom_mask;      // [max_tokens] uint8 (1 = atom token)
    Tensor cached_token_atom_flags;     // [max_tokens] uint32 (type-specific metadata from AtomTable)
    Tensor cached_token_to_slot_map;    // [max_tokens] int32  (-1 = non-state-bearing; >=0 = valid slot_id)
    
    int cached_batch_size = 0;
    int cached_seq_len = 0;
    int cached_valid_tokens = 0;
    
    //======================================================//
    //  INCREMENTAL KV CACHE STATE (autoregressive generation)
    //======================================================//
    int kv_cache_len = 0;           // Number of tokens with valid K,V in cache
    int kv_cache_capacity = 0;      // Maximum tokens the cache can hold
    float cached_numeric_pred[2] = {0.0f, 0.0f};  // Last-token numeric head output (host-side)

    // Per-layer KV cache tensors (BF16, BSHD layout for FlashAttention)
    // Shape per entry: [1, num_kv_heads, kv_cache_capacity, head_dim]
    // Allocated in initInferenceState, sized by num_layers.
    std::vector<void*> kv_cache_k;    // BF16 device pointers, one per encoder layer
    std::vector<void*> kv_cache_v;    // BF16 device pointers, one per encoder layer
    float* kv_cache_softmax_lse = nullptr;  // [num_layers, num_heads, kv_cache_capacity] FP32

    // Decode scratch buffers (tiny, pre-allocated for single-token KV cache decode)
    void* decode_q_bf16 = nullptr;        // [num_heads * head_dim] BF16
    void* decode_kv_bf16 = nullptr;       // [num_kv_heads * head_dim] BF16 (reused for K then V)
    void* decode_attn_out_bf16 = nullptr; // [num_heads * head_dim] BF16
    float* decode_attn_out_fp32 = nullptr;// [num_heads * head_dim] FP32 (= [d_model])

    //======================================================//
    //  PERSISTENT EXECUTION TRACE (per-forward lifecycle)
    //  Reset at the start of every forward that runs ExecutionBlock.
    //  execution_trace_by_row: host-side record log per batch row.
    //  trace_state_by_row:    device [1, d_model] running sum of
    //                         step_emb per row — autograd Tensor,
    //                         updated via autograd::add in executeStep.
    //  Do NOT recompute trace_state from execution_trace_by_row.
    //======================================================//
    std::vector<std::vector<ExecutionRecord>> execution_trace_by_row;
    std::vector<Tensor> trace_state_by_row;

    //======================================================//
    //  PERSISTENT EXECUTION MEMORY (Step Z — generation)
    //  Survives across forwardStep calls during autoregressive decode.
    //  Reset only on resetKVCache (session boundary).
    //  After each forward, the last exec_memories[0] is saved here so
    //  the generation loop can read slot values for <NUM> binding.
    //======================================================//
    ExecutionMemory inference_exec_memory;
    bool has_inference_exec_memory = false;
    int  inference_exec_last_write_slot = -1;
    
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

    // MTP diagnostics (filled by computeAutogradLoss when MTP enabled; logged by Phase2)
    struct MTPDiagnostics {
        std::vector<float> head_loss;
        std::vector<float> head_acc;
        float L0_main = 0.0f;       // Main (next-token) loss before adding MTP terms
        float alpha_effective = 0.0f;
        float L_total = 0.0f;
        bool valid = false;
    };
    MTPDiagnostics mtp_diagnostics;
    
    // Owns ALL intermediate tensors during forward→backward cycle
    // Replaces old autograd_ctx (which mixed input args with tensor storage)
    Autograd::AutogradIntermediates autograd_intermediates;

    //======================================================//
    //  INTERMEDIATE GRADIENT TENSORS (Issue #45 FIX)
    //======================================================//
    Tensor grad_logits_tensor;            // [max_logit_tokens, vocab_size]
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
    
    // NOTE: encoder_workspace DELETED (Rule 20/26)
    // Autograd forward creates its own intermediate Tensors — nothing consumed the workspace.

    //======================================================//
    //  STREAM & GRADIENT MANAGEMENT
    //======================================================//
    StreamController stream_ctrl;
    GradNorm::GradNormScratch* grad_norm_scratch = nullptr;  // Allocated in Phase1, freed in ~TrainingState
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

    //======================================================//
    //  AUTOGRAD SYSTEM STATE
    //======================================================//
    // Sentinel: initializeAutogradSeed() must be called before initGPU().
    // Guards against initialization order bugs (Rule 20).
    bool seed_initialized_ = false;
    
    /// Store the weight init seed and mark autograd as initialized.
    /// Called by Phase1_Startup step 2.75 before initGPU().
    void initializeAutogradSeed(uint64_t seed);

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
    //  CLASS-BALANCED LOSS WEIGHTS (owned GPU buffer)
    //======================================================//
    float* d_class_weights = nullptr;     // [vocab_size] on GPU, w_v = 1/freq(v)^β
    int class_weights_vocab_size = 0;     // For validation
    
    //======================================================//
    //  DEBUG GRADIENT ATTRIBUTION (Issue #60)
    //======================================================//
    bool debug_gradient_attribution = false;
    Tensor debug_lm_head_only_grad;
    Tensor debug_embedding_only_grad;
    
    void allocateDebugGradBuffers(int vocab_size, int d_model, cudaStream_t stream);
    void freeDebugGradBuffers();
    void logGradientAttribution(int batch_idx, cudaStream_t stream, const EmbeddingLayer* embedding_layer);
    
    // Dynamic collapse token tracking — set by Phase2 argmax detection each diagnostic interval.
    // -1 means no collapse token detected yet.
    int tracked_collapse_token = -1;

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

