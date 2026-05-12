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
#include "../StreamController/StreamController_GPU.hpp"
#include "../GradNorm/GradNormGPU.hpp"
#include "../HyperParameters/HyperParameters_GPU.hpp"
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

    void allocateStepDeviceWorkspaces(
        const HyperParameters::LanguageModelConfig& config,
        cudaStream_t stream);

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
    // Weight init seed is passed directly from Phase1 RNG into Startup/Model GPU assembly.
    
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
    Tensor cached_encoder_output;       // [max_tokens, d_model] LM-head input snapshot for diagnostics only
    Tensor cached_logits_tensor;        // [max_tokens, vocab_size] logits snapshot for diagnostics/inference return
    
    // Device mirrors of BatchPayload arrays. uploadBatchToDevice() is the only
    // writer for training/eval and returns BatchDeviceBindings as the only
    // forward/loss reader-facing view. Inference writes these as its single-row
    // decode/prefill cache because there is no host BatchPayload upload step.
    Tensor cached_targets_tensor;       // [max_tokens] int32
    Tensor cached_token_ids_tensor;     // [max_tokens] int32
    Tensor cached_seq_lengths_tensor;   // [max_sequences] int32 real token count per padded row
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
    //  TRAINING EXECUTION TRACE (per-forward lifecycle)
    //  Reset at the start of every training/eval forward that runs ExecutionBlock.
    //  Inference/session trace state lives in GenerationState.
    //======================================================//
    std::vector<std::vector<ExecutionRecord>> execution_trace_by_row;
    std::vector<Tensor> trace_state_by_row;
    
    TeacherLogits::Buffer teacher_logits;
    TeacherLogits::Buffer reference_logits;
    Tensor sequence_weights_tensor;    // [max_sequences]
    int sequence_weight_count = 0;
    int sequence_weight_capacity = 0;

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

    bool initialized = false;

    //======================================================//
    //  CLASS-BALANCED LOSS WEIGHTS (Tensor-owned GPU buffer)
    //======================================================//
    Tensor class_weights_tensor;          // [1, vocab_size] on GPU, w_v = 1/freq(v)^β
    int class_weights_vocab_size = 0;     // For validation

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

