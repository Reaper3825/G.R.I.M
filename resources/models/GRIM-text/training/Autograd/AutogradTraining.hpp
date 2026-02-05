//======================================================//
//  AutogradTraining.hpp
//  Full autograd-based training flow using TensorContract
//  
//  This module provides autograd-enabled forward+backward that
//  properly builds the computation graph and propagates gradients.
//  
//  REPLACES the legacy 3-phase backward system which reads from
//  cached_* buffers that are never populated by the autograd forward.
//  
//  ISSUE #56 FIX: Uses ForwardIntermediates to keep all intermediate
//  tensors alive during forward-backward cycle. Without this, grad_fn
//  objects are destroyed when intermediate tensors go out of scope.
//======================================================//

#pragma once

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

// LanguageModel, LanguageModelConfig, GPUGrimEncoder are now fully defined from included header
namespace Autograd {

// Use GRIM:: prefix to refer to outer namespace types
using ::GRIM::GPUGrimEncoder;
using ::GRIM::LanguageModelConfig;
using ::GRIM::LanguageModel;

/**
 * Result of autograd forward pass
 * Contains the logits Tensor with grad_fn attached for backward propagation
 */
struct ForwardResult {
    Tensor logits;           // [total_tokens, vocab_size] with requires_grad=true
    float* encoder_output;   // Raw pointer for compatibility (data lives in Tensor)
    int total_tokens;
    int vocab_size;
    bool success;
    std::string error_message;
};

/**
 * Result of autograd loss computation
 * Contains the scalar loss Tensor with grad_fn attached
 */
struct LossResult {
    Tensor loss;             // Scalar loss with grad_fn chain
    float loss_value;        // Host-side loss value
    int valid_tokens;        // Number of valid (non-padding) tokens
    bool success;
    std::string error_message;
};

/**
 * Result of autograd backward pass
 */
struct BackwardResult {
    bool success;
    float grad_norm;         // Total gradient norm
    std::string error_message;
};

// NOTE: GPUGrimEncoder is brought in via 'using' declaration above

/**
 * Context for autograd training operations
 * 
 * PRODUCTION-READY (Issue #56 Fix): Stores ALL intermediate Tensors from 
 * forward pass so the autograd graph stays alive for backward propagation.
 * 
 * The AllLayerIntermediates struct keeps each encoder layer's intermediate
 * tensors alive. Without this, when EncodingLayer::forward() returns, all
 * local tensors are destroyed, cascading the grad_fn destructor chain and
 * invalidating the autograd graph before backward() runs.
 */
struct AutogradContext {
    // Model and config
    const LanguageModelConfig* config;
    TrainingState* training_state;
    GPUGrimEncoder* gpu_encoder;      // Encoder layers for autograd forward
    cublasHandle_t cublas_handle;
    cudaStream_t stream;
    
    // ═══════════════════════════════════════════════════════════════════════════
    // OPTIONAL COMPONENTS (ScratchBlock, NumericHead)
    // These can be nullptr if not enabled in config
    // ═══════════════════════════════════════════════════════════════════════════
    ScratchBlockLayer* scratch_block = nullptr;    // Numeric/code processing layer
    
    // ScratchBlock forward inputs (from DataLoader)
    // These are raw pointers matching TrainingState cached buffers
    const float* token_numeric_values = nullptr;   // [total_tokens] numeric values
    const uint8_t* token_numeric_mask = nullptr;   // [total_tokens] 1=has_value, 0=no_value
    const uint16_t* token_text_features = nullptr; // [total_tokens, TEXT_FEATURE_DIM] text features
    const uint8_t* token_text_mask = nullptr;      // [total_tokens] 1=has_text_feature
    
    // NumericHead outputs - stored for backward
    Tensor numeric_head_output;        // [total_tokens, 1] numeric predictions
    
    // Batch info
    int batch_size;
    int seq_len;
    
    // ═══════════════════════════════════════════════════════════════════════════
    // ISSUE #56 FIX: ForwardIntermediates storage
    // Keeps ALL intermediate tensors alive during forward-backward cycle.
    // This is the PRIMARY mechanism for autograd graph preservation.
    // ═══════════════════════════════════════════════════════════════════════════
    AllLayerIntermediates layer_intermediates;
    
    // ═══════════════════════════════════════════════════════════════════════════
    // INTERMEDIATE TENSORS (kept alive during forward-backward cycle)
    // These preserve the autograd graph so backward() propagates through all ops
    // ═══════════════════════════════════════════════════════════════════════════
    
    // Embedding outputs
    Tensor embedding_tensor;     // [total_tokens, d_model] - output of embedding lookup
    
    // Encoder layer outputs (one per layer)
    // Each Tensor has grad_fn pointing to the operations that created it
    // NOTE: These are non-owning views - actual data lives in layer_intermediates
    std::vector<Tensor> encoder_layer_outputs;
    
    // Final encoder output (after final RMSNorm)
    Tensor encoder_output_tensor;  // [total_tokens, d_model]
    
    // Centered encoder output (if centering enabled) - MUST persist until backward!
    // ISSUE #127 FIX: CenterColumnsGradFn takes ownership of encoder_output_tensor's grad_fn,
    // so this tensor must live until backward completes to keep the grad_fn chain alive.
    Tensor centered_encoder_output;  // [total_tokens, d_model]
    
    // LM head input (view of encoder output with grad_fn linked)
    // MUST persist until backward completes - grad_fn stores pointer to this
    Tensor lm_input_tensor;        // [total_tokens, d_model] - input to LM head matmul
    
    // LM head outputs
    Tensor logits_tensor;        // [total_tokens, vocab_size] - output of forward, input to loss
    Tensor loss_tensor;          // Scalar loss - output of loss, drives backward
    
    // Gradient scale (for gradient accumulation)
    float grad_scale;
    
    // Step counter
    uint64_t step;
    
    // Error tracking
    std::string error_message;
    int error_layer;
    
    // Validation
    bool isValid() const {
        return config != nullptr && training_state != nullptr 
            && cublas_handle != nullptr && batch_size > 0 && seq_len > 0;
    }
    
    // Clear all intermediate tensors (call after backward to free memory)
    // Issue #56: MUST clear layer_intermediates - this is where grad_fn ownership lives
    void clearIntermediates() {
        // Issue #56: Clear layer intermediates FIRST - they own the grad_fn objects
        layer_intermediates.clear();
        
        embedding_tensor = Tensor();
        encoder_layer_outputs.clear();
        encoder_output_tensor = Tensor();
        centered_encoder_output = Tensor();  // Issue #127: Clear centering tensor
        lm_input_tensor = Tensor();  // Issue #48: Must clear this too
        logits_tensor = Tensor();
        loss_tensor = Tensor();
        numeric_head_output = Tensor();  // Clear NumericHead output
    }
};

// NOTE: linkEncoderWeightsToTrainingState was removed.
// Encoder owns its weights internally; optimizer accesses gradients via
// enc->getAttnWqkvGrad(), enc->getFFNW1Grad(), etc.

/**
 * Initialize autograd context
 * 
 * @param gpu_encoder Encoder for running layer forward passes
 */
AutogradContext initAutogradContext(
    const LanguageModelConfig* config,
    TrainingState* training_state,
    GPUGrimEncoder* gpu_encoder,
    ScratchBlockLayer* scratch_block,   // Optional: nullptr if disabled
    cublasHandle_t cublas_handle,
    cudaStream_t stream,
    int batch_size,
    int seq_len,
    float grad_scale,
    uint64_t step
);

/**
 * Execute forward pass with autograd
 * 
 * This differs from legacy forward by:
 * 1. Building computation graph via grad_fn nodes
 * 2. Storing intermediate Tensors (not just raw pointers)
 * 3. Returning logits as Tensor for loss computation
 * 
 * @param ctx Autograd context (modified in-place to store tensors)
 * @return Forward result with logits Tensor
 */
ForwardResult executeAutogradForward(AutogradContext& ctx);

/**
 * Compute loss with autograd
 * 
 * Creates loss Tensor with CrossEntropyGradFn attached.
 * When backward() is called on loss, gradients propagate to all inputs.
 * 
 * @param ctx Autograd context (must have valid logits_tensor from forward)
 * @param targets Device pointer to target token IDs [total_tokens]
 * @param valid_mask Device pointer to validity mask [total_tokens] (optional)
 * @return Loss result with loss Tensor
 */
LossResult computeAutogradLoss(
    AutogradContext& ctx,
    const int* targets,
    const float* valid_mask = nullptr
);

/**
 * Execute backward pass with autograd
 * 
 * Calls loss.backward() to propagate gradients through the entire graph.
 * Then copies gradients from Tensor.grad to TrainingState buffers for optimizer.
 * 
 * @param ctx Autograd context (must have valid loss_tensor from computeLoss)
 * @param accumulate If true, accumulates gradients instead of overwriting
 * @return Backward result with success status and gradient norm
 */
BackwardResult executeAutogradBackward(
    AutogradContext& ctx,
    bool accumulate
);

/**
 * Copy gradients from Tensor.grad to TrainingState raw buffers
 * 
 * The optimizer expects gradients in specific TrainingState buffers.
 * This function copies from autograd Tensor.grad fields.
 * 
 * @param ctx Autograd context with computed gradients
 * @return true on success
 */
bool copyGradientsToTrainingState(AutogradContext& ctx);

/**
 * Compute gradient norm from all parameter gradients
 */
float computeGradientNorm(const AutogradContext& ctx);

/**
 * Full autograd training step: forward → loss → backward
 * 
 * This is the main entry point for autograd-based training.
 * It combines forward, loss, and backward into a single call.
 * 
 * @param model Language model
 * @param training_state Training state with cached inputs/targets
 * @param batch_size Batch size
 * @param seq_len Sequence length
 * @param accumulate Whether to accumulate gradients
 * @param grad_scale Gradient scaling factor
 * @param step Current training step
 * @return Loss value (negative on error)
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
