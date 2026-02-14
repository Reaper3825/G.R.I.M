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
// MUST include full definitions for types used in AutogradContext
#include "../../GRIM/grim_language_model_cuda.hpp"
// ScratchBlock and NumericHead for autograd forward path
#include "../../Layers/ScratchBlock/ScratchBlock_GPU.hpp"
#include "../../Layers/NumericHead/numeric_head_GPU.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <string>
#include <memory>

namespace GRIM {

namespace Autograd {

using ::GRIM::GPUGrimEncoder;
using ::GRIM::LanguageModelConfig;
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
    float loss_value = 0.0f;         // Combined weighted loss (text + numeric + regularization)
    float text_loss = 0.0f;          // Raw text cross-entropy loss
    float numeric_loss = 0.0f;       // Raw numeric regression loss (0 if disabled)
    int numeric_count = 0;           // Number of numeric tokens in batch
    float weight_text = 1.0f;        // Learned text weight (1.0 if no learned weighting)
    float weight_numeric = 1.0f;     // Learned numeric weight (config default if no learned weighting)
    int valid_tokens = 0;            // Number of valid (non-padding) tokens
    bool success = false;
    std::string error_message;
};

/**
 * Result of autograd backward pass
 */
struct BackwardResult {
    bool success = false;
    float grad_norm = 0.0f;         // Total gradient norm
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
    // OPTIONAL COMPONENTS (nullptr if disabled)
    // ═══════════════════════════════════════════════════════════════════════════
    ScratchBlockLayer* scratch_block = nullptr;
    
    // ScratchBlock forward inputs (raw pointers into TrainingState buffers)
    const float* token_numeric_values = nullptr;
    const uint8_t* token_numeric_mask = nullptr;
    const uint16_t* token_text_features = nullptr;
    const uint8_t* token_text_mask = nullptr;
    
    // ═══════════════════════════════════════════════════════════════════════════
    // BATCH PARAMETERS
    // For training: payload is the single source of truth; batch_size/seq_len
    // are derived from it by initAutogradContext(const BatchPayload&, ...).
    // For inference: payload is nullptr; batch_size/seq_len set directly.
    // ═══════════════════════════════════════════════════════════════════════════
    const Batching::BatchPayload* payload = nullptr;  // Non-owning. Set for training, nullptr for inference.
    int batch_size = 0;
    int seq_len = 0;
    float grad_scale = 1.0f;
    uint64_t step = 0;
    bool is_training = true;
    
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
        if (!cublas_handle) throw std::runtime_error(std::string(caller) + ": cublas_handle is NULL");
        if (batch_size <= 0) throw std::runtime_error(std::string(caller) + ": batch_size <= 0");
        if (seq_len <= 0) throw std::runtime_error(std::string(caller) + ": seq_len <= 0");
    }
};

/**
 * Initialize autograd context for TRAINING (derives batch_size/seq_len from payload)
 */
AutogradContext initAutogradContext(
    const LanguageModelConfig* config,
    TrainingState* training_state,
    GPUGrimEncoder* gpu_encoder,
    ScratchBlockLayer* scratch_block,
    cublasHandle_t cublas_handle,
    cudaStream_t stream,
    const Batching::BatchPayload& payload,
    float grad_scale,
    uint64_t step,
    bool is_training = true
);

/**
 * Initialize autograd context for INFERENCE (no payload, set batch_size/seq_len directly)
 */
AutogradContext initAutogradContext(
    const LanguageModelConfig* config,
    TrainingState* training_state,
    GPUGrimEncoder* gpu_encoder,
    ScratchBlockLayer* scratch_block,
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
autograd::LossConfig buildLossConfig(const LossContext::LossOptions& opts);

/**
 * Compute loss with autograd
 * 
 * Single entry point for ALL loss computation. Creates loss Tensor with
 * CrossEntropyGradFn attached. Loss tensor stored in
 * training_state->autograd_intermediates.loss_tensor.
 * 
 * Handles:
 *   1. Text CE via autograd::unified_loss()
 *   2. Numeric regression loss via launchNumericLoss() (if enabled)
 *   3. Learned loss weighting (homoscedastic uncertainty, if log_var tensors exist)
 *   4. Caches all loss values in training_state for backward pass
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

/**
 * Compute gradient norm from all parameter gradients
 */
float computeGradientNorm(const AutogradContext& ctx);

/**
 * Full autograd training step: GPU copies → forward → loss → backward
 * 
 * This is the SINGLE entry point for a complete training iteration.
 * Replaces the scattered computeLossBatch() + backward() two-call pattern.
 * 
 * @param model          LanguageModel (provides config, encoder, loss options)
 * @param training_state TrainingState (GPU buffers, optimizer state)
 * @param payload        BatchPayload (single source of truth for batch data)
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
    bool accumulate,
    float grad_scale,
    uint64_t step
);

}  // namespace Autograd
}  // namespace GRIM
