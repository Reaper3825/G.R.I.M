//======================================================//
//  AutogradTraining.hpp
//  Full autograd-based training flow using TensorContract
//  
//  REFACTORED: AutogradContext is now a THIN INPUT STRUCT.
//  All intermediate tensor ownership lives in TrainingState
//  (via its autograd_intermediates member). This eliminates:
//  - Dual-buffer sync (cached_embeddings_tensor is DELETED)
//  - Fragile clearIntermediates() lifecycle
//  - 30+ field bloated context struct
//  
//  Rule 20: No backwards compatibility. Old AutogradContext
//  with intermediate tensor storage is DELETED.
//======================================================//

#pragma once

#include "AutogradIntermediates.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../Shared/TensorContract/ForwardIntermediates.hpp"
#include "../../Shared/TrainingState/TrainingState_GPU.hpp"
#include "../../Shared/Loss/ComputeLoss/AutogradLoss.hpp"
#include "../../Shared/Batching/BatchPayload.hpp"
#include "../../Shared/Batching/BatchDeviceBindings.hpp"
// MUST include full definitions for types used in AutogradContext
#include "../../GRIM/grim_language_model_cuda.hpp"
// ScratchBlock for autograd forward path
#include "../../Layers/ScratchBlock/ScratchBlockReasoning_GPU.hpp"
// ReasoningHead for structured reasoning
#include "../../Layers/ReasoningHead/reasoning_head_GPU.hpp"
// ExecutionBlock for internal numeric reasoning
#include "../../Layers/ExecutionBlock/execution_block_GPU.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <string>
#include <memory>

namespace GRIM {

namespace Autograd {

using ::GRIM::GPUGrimEncoder;
using ::GRIM::HyperParameters::LanguageModelConfig;
using ::GRIM::LanguageModel;

/**
 * Result of autograd forward pass
 */
struct ForwardResult {
    float* encoder_output = nullptr;  // Raw pointer to encoder output data
    int total_tokens = 0;
    int vocab_size = 0;
    bool success = false;
    std::string error_message;
};

/**
 * Result of autograd loss computation
 * Contains decomposed loss components for logging and gradient weighting.
 */
struct LossResult {
    float loss_value = 0.0f;         // Ground-truth: D2H read of loss_tensor AFTER all autograd::add()
    float text_loss = 0.0f;          // Text CE + MTP only (snapshot before exec/selector additions)
    float numeric_loss = 0.0f;       // Reserved (legacy); always 0 — no value head
    float selector_loss = 0.0f;      // Decode-time selector supervision loss (host scalar)
    float aux_loss = 0.0f;           // loss_value - text_loss (all non-text auxiliary terms)
    float weight_text = 1.0f;
    int valid_tokens = 0;
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
 * All intermediate tensors are stored in TrainingState::autograd_intermediates.
 * 
 * Rule 20: No tensor ownership here. If you need to store a Tensor for
 * backward, put it in AutogradIntermediates (owned by TrainingState).
 */
struct AutogradContext {
    // ═══════════════════════════════════════════════════════════════════════════
    // MODEL REFERENCES (non-owning pointers)
    // ═══════════════════════════════════════════════════════════════════════════
    const LanguageModelConfig* config = nullptr;
    TrainingState* training_state = nullptr;
    GPUGrimEncoder* gpu_encoder = nullptr;
    cublasHandle_t cublas_handle = nullptr;
    cudaStream_t stream = nullptr;
    
    // ═══════════════════════════════════════════════════════════════════════════
    // REQUIRED COMPONENTS (Pattern B: persistent, self-allocating layers)
    // ═══════════════════════════════════════════════════════════════════════════
    EmbeddingLayer* embedding_layer = nullptr;
    LMHeadLayer* lm_head = nullptr;

    // ═══════════════════════════════════════════════════════════════════════════
    // OPTIONAL COMPONENTS (nullptr if disabled)
    // ═══════════════════════════════════════════════════════════════════════════
    ScratchBlockLayer* scratch_block = nullptr;
    ReasoningHeadLayer* reasoning_head = nullptr;
    ExecutionBlockLayer* execution_block = nullptr;

    /** Model pointer for MTP head access in computeAutogradLoss; set by autogradTrainingStep. */
    LanguageModel* model = nullptr;
    
    // ═══════════════════════════════════════════════════════════════════════════
    // BATCH PARAMETERS
    // Training: payload points to the caller-owned BatchPayload (single source of truth).
    // Inference: payload points to inference_payload_ below (geometry-only, vectors empty).
    // payload is NEVER null after initAutogradContext.
    //
    // device_bindings carries the device pointers for THIS step (slot map,
    // atom mask, etc.). Replaces the old `mutable d_*` fields on BatchPayload.
    // - Training/eval path: filled by LanguageModel::uploadBatchToDevice() and
    //   passed to initAutogradContext; non-null.
    // - Inference geometry-only path: null (decode constructs its own row-local
    //   bindings at the call site).
    // ═══════════════════════════════════════════════════════════════════════════
    const Batching::BatchPayload* payload = nullptr;
    const Batching::BatchDeviceBindings* device_bindings = nullptr;
    int batch_size = 0;
    int seq_len = 0;
    float grad_scale = 1.0f;
    uint64_t step = 0;
    bool is_training = true;
    /** When true, encoder layers skip QKV_EQUATION D2H + fprintf (gradient accumulation micro-batches) */
    bool skip_equation_logging = false;

    // FOR INFERENCE ONLY — geometry-only BatchPayload (vectors empty).
    // payload points here when initialized via the inference overload.
    Batching::BatchPayload inference_payload_;
    
    // ═══════════════════════════════════════════════════════════════════════════
    // LOSS CONFIGURATION
    // Populated by caller from model config (LossOptions → autograd::LossConfig)
    // Used by computeAutogradLoss() → unified_loss()
    // ═══════════════════════════════════════════════════════════════════════════
    autograd::LossConfig loss_config;
    

    
    // ═══════════════════════════════════════════════════════════════════════════
    // VALIDATION (Rule 20: Fail loud)
    // ═══════════════════════════════════════════════════════════════════════════
    void validate(const char* caller) const {
        if (!config) throw std::runtime_error(std::string(caller) + ": config is NULL");
        if (!training_state) throw std::runtime_error(std::string(caller) + ": training_state is NULL");
        if (!gpu_encoder) throw std::runtime_error(std::string(caller) + ": gpu_encoder is NULL");
        if (!embedding_layer) throw std::runtime_error(std::string(caller) + ": embedding_layer is NULL");
        if (!lm_head) throw std::runtime_error(std::string(caller) + ": lm_head is NULL");
        if (!cublas_handle) throw std::runtime_error(std::string(caller) + ": cublas_handle is NULL");
        if (batch_size <= 0) throw std::runtime_error(std::string(caller) + ": batch_size <= 0");
        if (seq_len <= 0) throw std::runtime_error(std::string(caller) + ": seq_len <= 0");
    }
};

/**
 * Initialize autograd context for TRAINING (derives batch_size/seq_len from payload).
 * `bindings` must describe the same batch as `payload` (geometry-checked) and
 * must point at device memory already populated by uploadBatchToDevice().
 */
AutogradContext initAutogradContext(
    const LanguageModelConfig* config,
    TrainingState* training_state,
    GPUGrimEncoder* gpu_encoder,
    EmbeddingLayer* embedding_layer,
    LMHeadLayer* lm_head,
    ScratchBlockLayer* scratch_block,
    ReasoningHeadLayer* reasoning_head,
    ExecutionBlockLayer* execution_block,
    cublasHandle_t cublas_handle,
    cudaStream_t stream,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    float grad_scale,
    uint64_t step,
    bool is_training = true
);

/**
 * Initialize autograd context for INFERENCE (geometry-only payload, vectors empty).
 * payload pointer re-seated to inference_payload_ by executeAutogradForward.
 */
AutogradContext initAutogradContext(
    const LanguageModelConfig* config,
    TrainingState* training_state,
    GPUGrimEncoder* gpu_encoder,
    EmbeddingLayer* embedding_layer,
    LMHeadLayer* lm_head,
    ScratchBlockLayer* scratch_block,
    ReasoningHeadLayer* reasoning_head,
    ExecutionBlockLayer* execution_block,
    cublasHandle_t cublas_handle,
    cudaStream_t stream,
    int batch_size,
    int seq_len,
    float grad_scale,
    uint64_t step,
    bool is_training = true
);

/**
 * Execute forward pass with autograd
 * 
 * Builds computation graph via grad_fn nodes.
 * Intermediate tensors stored in training_state->autograd_intermediates.
 * 
 * @param ctx Thin input context
 * @return Forward result (success/error only, tensors in TrainingState)
 */
ForwardResult executeAutogradForward(AutogradContext& ctx);

/**
 * Build autograd::LossConfig from model LossOptions (single conversion point).
 * Eliminates duplicate LossOptions → LossConfig conversions across callsites.
 */
autograd::LossConfig buildLossConfig(const LossContext::LossOptions& opts, const float* d_class_weights = nullptr);

/**
 * Compute loss with autograd
 * 
 * Single entry point for ALL loss computation. Creates loss Tensor with
 * CrossEntropyGradFn attached. Loss tensor stored in
 * training_state->autograd_intermediates.loss_tensor.
 * 
 * Handles:
 *   1. Text CE via autograd::unified_loss()
 *   2. Caches loss value in training_state for backward pass
 * 
 * @param ctx     Autograd context (must have logits populated by executeAutogradForward,
 *                 and ctx.payload set to a valid BatchPayload)
 */
LossResult computeAutogradLoss(
    AutogradContext& ctx
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

// computeGradientNorm() DELETED — redundant with Phase2's computeGradNorm()

/**
 * Full autograd training step: forward → loss → backward.
 *
 * Caller is responsible for the H2D upload via
 * LanguageModel::uploadBatchToDevice(payload) and must pass the resulting
 * BatchDeviceBindings as `bindings`. autogradTrainingStep does not write any
 * device pointers back through `payload`.
 *
 * @param model          LanguageModel (provides config, encoder, loss options)
 * @param training_state TrainingState (GPU buffers, optimizer state)
 * @param payload        BatchPayload (host-only single source of truth)
 * @param bindings       BatchDeviceBindings produced by uploadBatchToDevice(payload)
 * @param accumulate     Whether to accumulate gradients (true for micro-batches > 0)
 * @param grad_scale     Gradient scaling factor
 * @param step           Global training step counter
 * @return LossResult with decomposed loss components and success/error status.
 *         If loss is non-finite, backward is SKIPPED and success=false.
 */
LossResult autogradTrainingStep(
    LanguageModel& model,
    TrainingState& training_state,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    bool accumulate,
    float grad_scale,
    uint64_t step
);

/**
 * AutogradStepScope — RAII single-owner of AutogradIntermediates::clear().
 *
 * Rule 20 ownership taxonomy enforcement: clearing of Category 1 (graph-owned,
 * transient) state MUST happen at exactly one site per autograd step. Wrap the
 * entire step (forward + loss + backward + post-step diagnostics that consume
 * intermediate .data) in this scope. The destructor calls clear() unconditionally
 * — including on early return / exception — so explicit clear() calls in error
 * paths are no longer required and MUST NOT be added back.
 *
 * Lifetime contract: the scope MUST outlive every reader of any field in
 * training_state.autograd_intermediates. After the scope ends, those fields
 * are reset to default-constructed Tensors — reading them is undefined.
 */
class AutogradStepScope {
public:
    explicit AutogradStepScope(TrainingState& ts) : ts_(ts) {}
    ~AutogradStepScope() { ts_.autograd_intermediates.clear(); }
    AutogradStepScope(const AutogradStepScope&) = delete;
    AutogradStepScope& operator=(const AutogradStepScope&) = delete;
    AutogradStepScope(AutogradStepScope&&) = delete;
    AutogradStepScope& operator=(AutogradStepScope&&) = delete;
private:
    TrainingState& ts_;
};

}  // namespace Autograd
}  // namespace GRIM
