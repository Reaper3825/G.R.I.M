//======================================================//
//  AutogradTraining.hpp
//  Full autograd-based training flow using TensorContract
//  
//  This module provides autograd-enabled forward+backward that
//  properly builds the computation graph and propagates gradients.
//  
//  REPLACES the legacy 3-phase backward system which reads from
//  cached_* buffers that are never populated by the autograd forward.
//======================================================//

#pragma once

#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../Shared/TrainingState/TrainingState_GPU.hpp"
#include "../../Shared/Loss/ComputeLoss/AutogradLoss.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <string>
#include <memory>

namespace GRIM {

// Forward declarations
class LanguageModel;
struct LanguageModelConfig;
class GPUGrimEncoder;

namespace Autograd {

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

// Forward declare encoder class
class GPUGrimEncoder;

/**
 * Context for autograd training operations
 * 
 * PRODUCTION-READY: Stores ALL intermediate Tensors from forward pass
 * so the autograd graph stays alive for backward propagation.
 */
struct AutogradContext {
    // Model and config
    const LanguageModelConfig* config;
    TrainingState* training_state;
    GPUGrimEncoder* gpu_encoder;      // Encoder layers for autograd forward
    cublasHandle_t cublas_handle;
    cudaStream_t stream;
    
    // Batch info
    int batch_size;
    int seq_len;
    
    // ═══════════════════════════════════════════════════════════════════════════
    // INTERMEDIATE TENSORS (kept alive during forward-backward cycle)
    // These preserve the autograd graph so backward() propagates through all ops
    // ═══════════════════════════════════════════════════════════════════════════
    
    // Embedding outputs
    Tensor embedding_tensor;     // [total_tokens, d_model] - output of embedding lookup
    
    // Encoder layer outputs (one per layer)
    // Each Tensor has grad_fn pointing to the operations that created it
    std::vector<Tensor> encoder_layer_outputs;
    
    // Final encoder output (after final RMSNorm)
    Tensor encoder_output_tensor;  // [total_tokens, d_model]
    
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
    void clearIntermediates() {
        embedding_tensor = Tensor();
        encoder_layer_outputs.clear();
        encoder_output_tensor = Tensor();
        logits_tensor = Tensor();
        loss_tensor = Tensor();
    }
};

/**
 * Link encoder layer weights to TrainingState Tensors
 * 
 * CRITICAL: This must be called ONCE before any autograd training to ensure
 * that gradients computed by autograd go directly to TrainingState buffers
 * (which the optimizer reads from).
 * 
 * Without this, encoder layers would have their own weight Tensors, and
 * autograd gradients would never reach the optimizer.
 * 
 * @param gpu_encoder The encoder whose layers need weight linking
 * @param training_state The TrainingState that owns the actual weight/grad buffers
 */
void linkEncoderWeightsToTrainingState(
    GPUGrimEncoder* gpu_encoder,
    TrainingState* training_state
);

/**
 * Initialize autograd context
 * 
 * @param gpu_encoder Encoder for running layer forward passes
 */
AutogradContext initAutogradContext(
    const LanguageModelConfig* config,
    TrainingState* training_state,
    GPUGrimEncoder* gpu_encoder,
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
