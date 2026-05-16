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
 * Compute unified loss for one MTP auxiliary head.
 *
 * The shifted targets are Phase1-authored in BatchPayload and uploaded by
 * LanguageModel::uploadBatchToDevice() into BatchDeviceBindings. The loss path
 * must not allocate or upload target buffers during loss assembly.
 */
Tensor unified_loss_for_mtp_head(
    Tensor& logits,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    int head_idx,
    const HyperParameters::LossConfigHP& config,
    const float* d_class_weights,
    cudaStream_t stream
);

// Cross-entropy / NLL implementation details live in CrossEntropyNLL.hpp/.cu.
// AutogradLoss.hpp exposes payload/bindings entry points for primary CE and MTP.

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
