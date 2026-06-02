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
#include "../Forward/ModelForwardExecutionRuntime.hpp"
#include "../TensorContract/TensorContract_GPU.hpp"

namespace GRIM {

struct TrainingState {
    TrainingState();
    ~TrainingState();

    TrainingState(const TrainingState&) = delete;
    TrainingState& operator=(const TrainingState&) = delete;
    TrainingState(TrainingState&&) = delete;
    TrainingState& operator=(TrainingState&&) = delete;

    void allocateReadGateWorkspace(cudaStream_t stream);

    //======================================================//
    //  PARAMETER TENSORS (weights + gradients via autograd)
    //======================================================//
    // Rule 20: NO raw float* for gradients - use GRIM::Tensor with autograd
    //
    // ALL weight tensors are owned by startup-registry or layer boundaries:
    //   - Embedding: TrainingContext::parameter_registry.getEmbeddingParameters()->token_weights
    //   - LM Head: TrainingContext::parameter_registry.getLmHeadParameters()->weights / bias / final_rms_gamma
    //   - Encoder: Each EncodingLayer self-allocates in constructor
    //
    // Session 7: TrainingTensors deleted — zero weight parameters remain in god object.
    // Weight init seed is passed directly from Phase1 RNG into Startup/Model GPU assembly.
    
    // Authoritative batch index for autograd forward passes.
    // Sourced from runEpoch's active batch_idx; set by autogradTrainingStep
    // ONLY on train calls. Eval never mutates this.
    // Controls forward-time stochastic kernels such as dropout PRNG seeds.
    uint64_t autograd_batch_idx = 0;
    
    //======================================================//
    //  TRAINING EXECUTION TRACE (per-forward lifecycle)
    //  Reset at the start of every training/eval forward that runs ExecutionBlock.
    //  Inference/session trace state lives in GenerationState.
    //======================================================//
    Forward::ModelForwardExecutionRuntime execution_runtime;
    
    TeacherLogits::Buffer teacher_logits;
    TeacherLogits::Buffer reference_logits;

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

