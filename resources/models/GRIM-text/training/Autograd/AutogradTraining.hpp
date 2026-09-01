//======================================================//
//  AutogradTraining.hpp
//  Full autograd-based training flow using TensorContract
//  
//  REFACTORED: AutogradContext is now a THIN INPUT STRUCT.
//  Retained forward tensors and the scalar loss root are passed through
//  explicit step-state owners instead of routed through TrainingState. This
//  eliminates:
//  - Dual-buffer sync (cached_embeddings_tensor is DELETED)
//  - Fragile clearIntermediates() lifecycle
//  - 30+ field bloated context struct
//  
//  Rule 20: Current autograd path only. Old AutogradContext
//  with intermediate tensor storage is DELETED.
//======================================================//

#pragma once

#include "AutogradIntermediates.hpp"
#include "../Phases/Startup/Model/ParameterRegistry.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../Shared/TrainingState/TrainingState_GPU.hpp"
#include "../../Shared/Loss/ComputeLoss/AutogradLoss.hpp"
#include "../../Shared/Batching/BatchPayload.hpp"
#include "../../Shared/Batching/BatchDeviceBindings.hpp"
// MUST include full definitions for types used in AutogradContext
#include "../../GRIM/grim_language_model_cuda.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <string>
#include <memory>
#include <stdexcept>

namespace GRIM {

namespace Autograd {

using ::GRIM::GPUGrimEncoder;

/**
 * Result of autograd loss computation
 * Contains decomposed loss components for logging and gradient weighting.
 */
struct LossResult {
    float loss_value = 0.0f;         // Ground-truth: D2H read of loss_tensor AFTER all autograd::add()
    float text_loss = 0.0f;          // Primary task loss (next-token CE or atom BCE)
    float selector_loss = 0.0f;      // Scaled arg/option selector CE (α_sel * CE)
    float local_atom_retrieval_loss_raw = 0.0f; // Mean causal retrieval CE before weighting
    float local_atom_retrieval_loss = 0.0f;     // Weighted contribution present in loss_value
    float weight_text = 1.0f;
    float weight_local_atom_retrieval = 0.0f;
    int valid_tokens = 0;
    int local_atom_retrieval_queries = 0;
    int local_atom_reference_targets = 0;
    bool success = false;
    std::string error_message;
};

/**
 * Result of autograd backward pass
 */
struct BackwardResult {
    bool success = false;
    float grad_rms = 0.0f;         // Total gradient RMS
    std::string error_message;
};

/**
 * AutogradContext - THIN INPUT STRUCT
 * 
 * Contains ONLY what the forward/backward functions need as INPUT.
 * Retained forward tensors and the scalar loss root are provided through
 * explicit step-state owners for the active batch.
 * 
 * Rule 20: No tensor ownership here. If you need to store a Tensor for
 * backward lifetime, put it on the explicit caller-owned step-state owners.
 */
struct AutogradContext {
    // ═══════════════════════════════════════════════════════════════════════════
    // MODEL REFERENCES (non-owning pointers)
    // ═══════════════════════════════════════════════════════════════════════════
    const Config::AiConfigSnapshot* config = nullptr;
    TrainingState* training_state = nullptr;
    Forward::ModelForwardOutputs* forward_outputs = nullptr;
    AutogradLossState* loss_state = nullptr;
    GPUGrimEncoder* gpu_encoder = nullptr;
    cublasHandle_t cublas_handle = nullptr;
    cudaStream_t stream = nullptr;
    
    // ═══════════════════════════════════════════════════════════════════════════
    // REQUIRED COMPONENTS (startup-owned durable topology + registry-owned params)
    // ═══════════════════════════════════════════════════════════════════════════
    // OPTIONAL COMPONENTS (nullptr if disabled)
    // ═══════════════════════════════════════════════════════════════════════════

    bool execution_block_enabled = false;
    ::ParameterRegistry::StartupParameterRegistry* parameter_registry = nullptr;

    // ═══════════════════════════════════════════════════════════════════════════
    // BATCH PAYLOAD VIEW
    // Training only: payload points to the caller-owned BatchPayload runtime
    // datum. Fixed-shape training geometry is config-owned via
    // HyperparameterGroupings/LanguageModelConfig and enforced at upload; the
    // payload carries the realized supervision, token stats, and row layout.
    // Inference MUST NOT enter AutogradContext; use
    // Shared/Forward/ModelForward_GPU.hpp with a caller-authored read-only
    // ModelForwardGraphPolicy instead.
    // payload is NEVER null after the training initAutogradContext overload.
    //
    // device_bindings carries the device pointers for THIS step (slot map,
    // atom mask, etc.). Replaces the old `mutable d_*` fields on BatchPayload.
    // - Training path: filled by Batching::uploadBatchToDevice().
    // Always non-null before Phase2/shared-forward and loss code reads device pointers.
    // ═══════════════════════════════════════════════════════════════════════════
    const Batching::BatchPayload* payload = nullptr;
    const Batching::BatchDeviceBindings* device_bindings = nullptr;
    uint64_t batch_idx = 0;
    uint64_t global_step = 0;  ///< Phase2-authored optimizer step for persistent verifier telemetry
    /** When true, skip duplicate equation logging on non-initial accumulation slots. */
    bool skip_equation_logging = false;
    
    // ═══════════════════════════════════════════════════════════════════════════
    // LOSS RUNTIME INPUTS
    // d_class_weights is runtime TrainingState, kept separate from the
    // Phase1-authored LossConfigHP grouping passed directly to loss assembly.
    // ═══════════════════════════════════════════════════════════════════════════
    const float* d_class_weights = nullptr;
    

    
    // ═══════════════════════════════════════════════════════════════════════════
    // VALIDATION (Rule 20: Fail loud)
    // ═══════════════════════════════════════════════════════════════════════════
    void validate(const char* caller) const {
        if (!config) throw std::runtime_error(std::string(caller) + ": config is NULL");
        if (!training_state) throw std::runtime_error(std::string(caller) + ": training_state is NULL");
        if (!forward_outputs) throw std::runtime_error(std::string(caller) + ": forward_outputs is NULL");
        if (!loss_state) throw std::runtime_error(std::string(caller) + ": loss_state is NULL");
        if (!gpu_encoder) throw std::runtime_error(std::string(caller) + ": gpu_encoder is NULL");
        if (!parameter_registry) throw std::runtime_error(std::string(caller) + ": parameter_registry is NULL");
        (void)parameter_registry->requireEmbeddingParameters(caller);
        if (!cublas_handle) throw std::runtime_error(std::string(caller) + ": cublas_handle is NULL");
        if (!payload) throw std::runtime_error(std::string(caller) + ": payload is NULL");
        if (!device_bindings) throw std::runtime_error(std::string(caller) + ": device_bindings is NULL");
        payload->validate(caller);
        const bool supervision_uploaded = payload->EnableAtomIdentification
            ? device_bindings->d_atom_insertion_gap_targets &&
                device_bindings->d_atom_insertion_valid_gap_mask
            : device_bindings->d_target_ids != nullptr;
        if (!device_bindings->d_input_ids || !supervision_uploaded ||
            !device_bindings->d_token_to_slot_index_map) {
            throw std::runtime_error(std::string(caller) + ": BatchDeviceBindings has NULL device pointers");
        }
    }
};

/**
 * Initialize autograd context for TRAINING. `payload` is the realized runtime
 * batch datum; fixed-shape training geometry is config-owned and validated at
 * Batching::uploadBatchToDevice(). `bindings` must point at device memory already
 * populated by that upload boundary.
 */
AutogradContext initAutogradContext(
    const Config::AiConfigSnapshot* config,
    TrainingState* training_state,
    Forward::ModelForwardOutputs& forward_outputs,
    AutogradLossState& loss_state,
    GPUGrimEncoder* gpu_encoder,
    bool execution_block_enabled,
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    cublasHandle_t cublas_handle,
    cudaStream_t stream,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    uint64_t batch_idx,
    uint64_t global_step
);

/**
 * Compute loss with autograd
 * 
 * Single entry point for ALL loss computation. Creates loss Tensor with
 * CrossEntropyGradFn attached, stores that Tensor directly in
 * the caller-owned AutogradLossState for backward, and returns
 * the host-side LossResult consumed by Phase2.
 * 
 * Computes the configured primary task loss (text CE or atom insertion BCE),
 * adds the weighted local-atom retrieval CE for eligible ordinary-LM batches,
 * leaves the single canonical total-loss Tensor on AutogradLossState for
 * backward, and returns decomposed host-side scalars to Phase2.
 * 
 * @param ctx     Autograd context with logits populated by shared forward
 * @param payload Phase1-authored BatchPayload for this explicit loss boundary
 * @param loss_config Caller-derived loss hyperparameter grouping
 */
LossResult computeAutogradLoss(
    AutogradContext& ctx,
    const Batching::BatchPayload& payload,
    const HyperParameters::LossConfigHP& loss_config
);

/**
 * Execute backward pass with autograd
 * 
 * Calls loss.backward() to propagate gradients through entire graph.
 */
BackwardResult executeAutogradBackward(
    AutogradContext& ctx,
    bool accumulate
);

/**
 * Verify that gradients are properly wired up and accessible to optimizer
 * (Diagnostic function - does not copy, checks connectivity)
 */
bool verifyGradientsAreConnected(AutogradContext& ctx);

}  // namespace Autograd
}  // namespace GRIM
