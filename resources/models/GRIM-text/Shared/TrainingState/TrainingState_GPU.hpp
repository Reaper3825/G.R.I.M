//======================================================//
//  TrainingState_GPU.hpp
//  Standalone TrainingState declaration for GPU training
//  
//  Rule 20 MIGRATION: Training tensors use GRIM::Tensor ownership.
//  - Tensor members provide shape info, automatic cleanup, autograd support.
//  - Access raw pointer via tensor.data when needed for CUDA kernels.
//  - Non-Tensor resources remain explicit TrainingState-owned fields when
//    they are typed buffers, BF16 caches, pinned host pools, helper-owned
//    buffers, or external library handles.
//======================================================//

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <memory>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "CublasHandleOwner_GPU.hpp"
#include "../TeacherLogits/TeacherLogits_GPU.hpp"
#include "../ScratchBlock/ScratchBlockPool_GPU.hpp"
#include "../StreamController/StreamController_GPU.hpp"
#include "../GradNorm/GradNormGPU.hpp"
#include "../TensorContract/TensorContract_GPU.hpp"

// Forward declaration for autograd tensor system
namespace GRIM {
    class EmbeddingLayer;
    struct FlashAttentionBF16Scratch;
    enum class SlotSelectionStatus : uint8_t;
    struct SlotSelectionResult;
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
    //   - Embedding: LanguageModel::getEmbeddingLayer()->tokenWeights()
    //   - LM Head: LanguageModel::getLmHeadLayer()->weights() / bias() / finalRmsGamma()
    //   - Encoder: Each EncodingLayer self-allocates in constructor
    //   - ScratchBlock: ScratchBlockLayer self-allocates in constructor
    //
    // Session 7: TrainingTensors deleted — zero weight parameters remain in god object.
    // Weight init seed is passed directly from Phase1 RNG into LanguageModel::initGPU().
    
    // GQA configuration (stored for cache sizing)
    int num_heads = 0;           // Q heads
    int num_kv_heads = 0;        // K,V heads (GQA: num_kv_heads < num_heads)

    //======================================================//
    //  STEP DEVICE WORKSPACES / SNAPSHOTS (Category 3)
    //======================================================//
    // Capacity is authored upstream by RunCapacity -> LanguageModelConfig.
    // Allocated Tensor shapes are the only TrainingState-local capacity record.
    // BatchPayload remains the host-side source of truth for batch geometry
    // and token semantics.
    //
    // Training/eval upload copies BatchPayload host arrays into these reusable
    // device buffers and returns BatchDeviceBindings as the canonical per-step
    // device view. Forward/loss code should consume BatchDeviceBindings, not
    // rediscover the "current batch" by reading these fields directly.
    //
    // Autograd Tensors with grad_fn live in autograd_intermediates, not here.
    // Output snapshots written by executeAutogradForward(). They are NOT graph
    // owners; they preserve reduced step outputs for diagnostics/inference after
    // AutogradIntermediates is cleared at the forward/backward boundary.
    Tensor cached_encoder_output;       // [max_tokens, d_model] LM-head input snapshot for diagnostics/inference selector
    Tensor cached_logits_tensor;        // [max_tokens, vocab_size] logits snapshot for diagnostics/inference return
    
    // Device mirrors of BatchPayload arrays. uploadBatchToDevice() is the only
    // writer for training/eval and returns BatchDeviceBindings as the only
    // forward/loss reader-facing view. Inference writes these as its single-row
    // decode/prefill cache because there is no host BatchPayload upload step.
    Tensor cached_targets_tensor;       // [max_tokens] int32
    Tensor cached_token_ids_tensor;     // [max_tokens] int32
    Tensor cached_token_numeric_values; // [max_tokens] float
    
    // Unified atom side-channel
    Tensor cached_token_text_features;  // [max_tokens * kTextFeatureDim] FP16
    Tensor cached_token_atom_mask;      // [max_tokens] uint8 (1 = atom token)
    Tensor cached_token_atom_flags;     // [max_tokens] uint32 (type-specific metadata from AtomTable)
    Tensor cached_token_to_slot_map;    // [max_tokens] int32  (-1 = non-state-bearing; >=0 = valid slot_id)
    
    // Authoritative training step counter for autograd forward passes.
    // Sourced from TrainingContext::global_step (checkpointed); set by
    // autogradTrainingStep ONLY on train calls. Eval never mutates this.
    // Controls: dropout PRNG seeds, MTP alpha warmup schedule.
    uint64_t autograd_step = 0;
    
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
    //  PERSISTENT EXECUTION MEMORY (generation)
    //  Survives across forwardStep calls during autoregressive decode.
    //  Reset only on resetKVCache (session boundary).
    //  Preserved so decode-time ExecutionBlock state can continue across
    //  autoregressive steps. Decode-time <NUM> selection is NOT inferred
    //  from this state; that requires an explicit selector workstream.
    //======================================================//
    ExecutionMemory inference_exec_memory;
    bool has_inference_exec_memory = false;

    //======================================================//
    //  DECODE-TIME SLOT SELECTOR RESULT
    //  Written by executeDecodeForward_ when selector is active.
    //  Consumed by generateSequenceGPU to decide <NUM> admissibility.
    //  Reset to invalid each step; valid only after executeDecodeForward_.
    //======================================================//
    bool decode_selector_valid = false;
    int32_t decode_selected_slot = -1;       // Real slot index when Selected
    float decode_selected_value = 0.0f;      // Numeric value from selected slot
    uint8_t decode_selector_status = 0;      // Cast of SlotSelectionStatus
    
    // Single-token buffers for incremental generation
    Tensor single_token_logits;      // [vocab_size]
    Tensor single_token_hidden;      // [d_model]
    Tensor single_token_embedding;   // [d_model]
    
    int cached_num_layers = 0;
    
    TeacherLogits::Buffer teacher_logits;
    TeacherLogits::Buffer reference_logits;
    Tensor sequence_weights_tensor;    // [max_sequences]
    int sequence_weight_count = 0;
    int sequence_weight_capacity = 0;

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
    //  CROSS-ATTENTION READ-GATE TELEMETRY (Rule 20 ownership taxonomy)
    //======================================================//
    // read_gate_accum_tensor: Category 3 (workspace). [2] device buffer
    //   = [sum_of_gate_values, total_token_count]. Reusable across batches;
    //   contents are stale across the autograd boundary — must be re-zeroed
    //   before each forward and snapshotted before backward consumes the tape.
    // h_read_gate_mean: Category 2 (durable telemetry scalar). Survives the
    //   autograd boundary by design — consumed by TelemetryUpdate after clear().
    Tensor read_gate_accum_tensor;
    float  h_read_gate_mean  = 0.0f;
    // NOTE: encoder_workspace DELETED (Rule 20/26)
    // Autograd forward creates its own intermediate Tensors — nothing consumed the workspace.

    //======================================================//
    //  STREAM & GRADIENT MANAGEMENT
    //======================================================//
    StreamController stream_ctrl;
    std::unique_ptr<GradNorm::GradNormScratch> grad_norm_scratch;  // Owned by TrainingState; allocated/validated by GradClip
    CublasHandleOwner cublas_handle;

    // Scratch block pool
    std::unique_ptr<ScratchBlock::ScratchBlockPool> scratch_pool;
    bool scratch_enabled = true;

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
    //  CLASS-BALANCED LOSS WEIGHTS (Tensor-owned GPU buffer)
    //======================================================//
    Tensor class_weights_tensor;          // [1, vocab_size] on GPU, w_v = 1/freq(v)^β
    int class_weights_vocab_size = 0;     // For validation

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

        GuessCacheBuffers() = default;
        ~GuessCacheBuffers();
        GuessCacheBuffers(const GuessCacheBuffers&) = delete;
        GuessCacheBuffers& operator=(const GuessCacheBuffers&) = delete;
        GuessCacheBuffers(GuessCacheBuffers&&) = delete;
        GuessCacheBuffers& operator=(GuessCacheBuffers&&) = delete;

        void release();
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

