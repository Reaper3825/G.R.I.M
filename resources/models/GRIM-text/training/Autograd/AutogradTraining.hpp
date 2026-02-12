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
 */
struct LossResult {
    float loss_value = 0.0f;         // Host-side loss value
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
    // ═══════════════════════════════════════════════════════════════════════════
    int batch_size = 0;
    int seq_len = 0;
    float grad_scale = 1.0f;
    uint64_t step = 0;
    bool is_training = true;
    
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
 * Initialize autograd context (thin input struct only)
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
 * Compute loss with autograd
 * 
 * Creates loss Tensor with CrossEntropyGradFn attached.
 * Loss tensor stored in training_state->autograd_intermediates.loss_tensor.
 */
LossResult computeAutogradLoss(
    AutogradContext& ctx,
    const int* targets,
    const float* valid_mask = nullptr
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
 * Full autograd training step: forward → loss → backward
 */
float autogradTrainingStep(
    LanguageModel& model,
    TrainingState& training_state,
    int batch_size,
    int seq_len,
    bool accumulate,
    float grad_scale,
    uint64_t step
);

}  // namespace Autograd
}  // namespace GRIM
