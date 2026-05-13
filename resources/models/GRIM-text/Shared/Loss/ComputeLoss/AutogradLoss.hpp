//======================================================//
//  AutogradLoss.hpp
//  Unified autograd-enabled loss: Focal + Label Smoothing + Cross Entropy + Entropy Reg
//
//  This is the ONLY public loss computation path.
//  Cross-entropy / NLL internals live in CrossEntropyNLL.hpp/.cu.
//======================================================//

#pragma once

#include "../../HyperParameters/HyperparameterGroupings.hpp"
#include "../../TensorContract/TensorContract_GPU.hpp"
#include "../../Batching/BatchPayload.hpp"
#include "../../Batching/BatchDeviceBindings.hpp"
#include <cuda_runtime.h>
#include <cstdint>

namespace GRIM {
namespace autograd {

//=============================================================================
// MAIN API
//=============================================================================

/**
 * Compute unified loss with autograd support
 * 
 * Architecture (PyTorch gold standard):
 *   logits → autograd::log_softmax() → log_probs → NLL loss → scalar loss
 *
 * Internally composes:
 *   1. log_softmax(logits) — numerically stable log(softmax(x)) = x - logsumexp(x)
 *   2. NLL loss on log_probs — -log_probs[target] with focal/smoothing/entropy
 *
 * Backward chain:
 *   NLLLossGradFn → LogSoftmaxGradFn → upstream (MatMulGradFn)
 *   Produces: grad_logits[j] = (p_j - q_j) / N  (standard CE gradient)
 *
 * Loss formula:
 *   When focal_enabled:  L = α * (1-p_t)^γ * CE_smooth + λ * Σ p*log(p)
 *   When focal disabled: L = CE_smooth + λ * Σ p*log(p)
 *   (focal_alpha is ONLY applied when focal_enabled=true)
 * 
 * @param logits      [num_tokens, vocab_size] - raw logits from LM head
 * @param payload     Host-side batch metadata: geometry, vocab, valid-token counts
 * @param bindings    Device-side batch view containing uploaded d_target_ids
 * @param config      Durable loss grouping from HyperparameterGroupings.hpp
 * @param d_class_weights Optional class-balanced weights owned by TrainingState
 * @param stream      CUDA stream
 * @return Scalar loss tensor with grad_fn attached (if logits.requires_grad)
 */
Tensor unified_loss(
    Tensor& logits,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    const HyperParameters::LossConfigHP& config,
    const float* d_class_weights,
    cudaStream_t stream
);

/**
 * Compute unified loss from an already-uploaded target buffer.
 *
 * This is for auxiliary heads whose targets are derived from BatchPayload but
 * are not the payload's primary d_target_ids buffer, e.g. MTP shifted targets.
 */
Tensor unified_loss_from_target_buffer(
    Tensor& logits,
    const int* targets,
    int num_tokens,
    int vocab_size,
    const HyperParameters::LossConfigHP& config,
    const float* d_class_weights,
    cudaStream_t stream
);

// Cross-entropy / NLL implementation details live in CrossEntropyNLL.hpp/.cu.
// AutogradLoss.hpp exposes the payload/bindings unified_loss() entry point and
// the explicit shifted-target-buffer helper for auxiliary heads.

//=============================================================================
// TOKEN 277 DIAGNOSTIC LOGGING (Rule 21 Equation-Based)
//=============================================================================

/**
 * Launch collapse token diagnostic with ACTUAL loss computation.
 * Computes real loss (focal + smoothing + entropy) per-token to identify
 * mode collapse and gradient issues for the tracked token.
 */
void launchToken277DiagnosticActual(
    const float* log_probs,
    const float* logits,
    const int* targets,
    const float* grad_log_probs,
    int num_tokens,
    int vocab_size,
    float focal_alpha,
    float focal_gamma,
    float smoothing_epsilon,
    float entropy_reg_lambda,
    int batch_idx,
    int tracked_token,
    cudaStream_t stream
);

/**
 * MTP kernels moved to Shared/MTP/MTP_GPU.hpp
 * (launchMTPAccuracyKernel, computeMTPAuxiliaryLosses)
 */

// Issue #142: cross_entropy_loss() DELETED (Rule 26: dead code).
// Was a thin wrapper calling the loss path with hardcoded plain CE config.
// Production callers use unified_loss() with BatchPayload + BatchDeviceBindings
// and LossConfigHP from HyperparameterGroupings.hpp.

}  // namespace autograd
}  // namespace GRIM
