/**
 * @file BackwardPhase1_OutputLayer.hpp
 * @brief Phase 1: Output Layer Backward Pass
 *
 * This phase computes gradients for:
 * 1. Cross-entropy loss gradient (dL/d_logits)
 * 2. LM Head backward (grad_hidden, grad_W_lm_head, grad_b_lm_head)
 *
 * OUTPUT: ctx.current_grad = grad_encoder_out (ready for Phase 2)
 *
 * TENSOR CONTRACTS:
 * - grad_logits: [total_tokens, vocab_size] BSM
 * - grad_encoder_out: [total_tokens, d_model] BSM
 * - lm_head_weight_grads: [vocab_size, d_model]
 */

#pragma once

#include "BackwardContext.hpp"

namespace GRIM {
namespace Backward {

/**
 * @brief Execute Phase 1: Output Layer Backward
 *
 * Steps:
 * 1. Compute cross-entropy gradient: dL/d_logits = softmax(logits) - one_hot(targets)
 * 2. Scale gradients by grad_scale (token normalization)
 * 3. LM Head backward: grad_encoder_out = grad_logits @ W_lm_head^T
 * 4. LM Head weight gradient: grad_W_lm_head = grad_logits^T @ encoder_output
 *
 * @param ctx Backward context (must be validated)
 * @return BackwardStatus::SUCCESS or error code
 *
 * FAIL LOUD: Any error returns immediately with detailed message in ctx.error_message
 */
BackwardStatus executePhase1_OutputLayer(BackwardContext& ctx);

/**
 * @brief Compute cross-entropy gradient
 *
 * dL/d_logits[i,j] = softmax(logits)[i,j] - (j == target[i] ? 1 : 0)
 *
 * @param ctx Backward context
 * @return BackwardStatus
 */
BackwardStatus computeCrossEntropyGradient(BackwardContext& ctx);

/**
 * @brief Compute LM Head backward pass
 *
 * grad_encoder_out = grad_logits @ W_lm_head^T
 * grad_W_lm_head = grad_logits^T @ encoder_output
 * grad_b_lm_head = sum(grad_logits, axis=0) [if bias enabled]
 *
 * @param ctx Backward context
 * @return BackwardStatus
 */
BackwardStatus computeLMHeadBackward(BackwardContext& ctx);

} // namespace Backward
} // namespace GRIM
