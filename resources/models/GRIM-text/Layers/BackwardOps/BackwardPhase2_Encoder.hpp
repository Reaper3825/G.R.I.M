/**
 * @file BackwardPhase2_Encoder.hpp
 * @brief Phase 2: Encoder Backward Pass Declaration
 *
 * This phase handles the main backward loop through all encoder layers (N-1 → 0).
 * It is the most complex phase and where gradient explosion typically occurs.
 *
 * Operations per layer (in reverse order):
 * 1. RMSNorm2 backward (FFN pre-norm)
 * 2. FFN backward (W2 → GELU → W1)
 * 3. Residual connection (FFN path)
 * 4. RMSNorm1 backward (Attention pre-norm)
 * 5. Attention W_o backward
 * 6. Flash Attention backward (GQA-aware)
 * 7. QKV projection backward
 * 8. Residual connection (Attention path)
 *
 * CRITICAL: This phase uses GQA (Grouped Query Attention) with num_kv_heads < num_heads.
 * Flash Attention v2 backward uses BF16 BSHD inputs/outputs with workspace buffers.
 *
 * @note All tensors are row-major (BSM format for sequences, BHSD for attention)
 */

#pragma once

#include "BackwardContext.hpp"

namespace GRIM {
namespace Backward {

//======================================================//
//  Phase 2 Entry Point
//======================================================//

/**
 * @brief Execute encoder backward pass (all layers)
 *
 * @param ctx Backward context (input: current_grad from Phase 1)
 * @return BackwardStatus::SUCCESS or appropriate error code
 *
 * @pre ctx.current_grad points to grad_encoder_out from Phase 1
 * @pre ctx.gpu_encoder is valid with initialized layers
 * @post ctx.current_grad points to gradient ready for Phase 3
 *
 * FAIL LOUD: Any error will halt training with detailed diagnostics
 */
BackwardStatus executePhase2_Encoder(BackwardContext& ctx);

//======================================================//
//  Per-Layer Backward
//======================================================//

/**
 * @brief Execute backward pass for a single encoder layer
 *
 * @param ctx Backward context
 * @param layer Layer index (N-1 to 0)
 * @return BackwardStatus
 *
 * This function contains the core backward logic for one transformer layer.
 * It is called in a loop from executePhase2_Encoder().
 *
 * GRADIENT FLOW:
 *   input_grad → RMSNorm2 → FFN → residual → RMSNorm1 → Attention → QKV → residual → output_grad
 */
BackwardStatus executeLayerBackward(BackwardContext& ctx, int layer);

//======================================================//
//  Component Backward Functions
//======================================================//

/**
 * @brief RMSNorm backward pass
 *
 * Computes gradients for gamma and propagates gradient to input.
 * Uses cached RMS values from forward pass for efficiency.
 *
 * @param ctx Backward context
 * @param layer Layer index
 * @param grad_output Gradient from upstream (input)
 * @param cached_input Cached input from forward pass
 * @param gamma Current gamma weights
 * @param grad_gamma Output: gradient for gamma (accumulated)
 * @param grad_input Output: gradient w.r.t. input
 * @param size Number of elements
 * @param norm_type 1 for RMSNorm1 (attention), 2 for RMSNorm2 (FFN)
 */
BackwardStatus computeRMSNormBackward(
    BackwardContext& ctx,
    int layer,
    const float* grad_output,
    const float* cached_input,
    const float* gamma,
    float* grad_gamma,
    float* grad_input,
    size_t size,
    int norm_type);

/**
 * @brief FFN backward pass
 *
 * Computes gradients for W1, W2, b1, b2 and propagates gradient.
 * FFN structure: hidden = GELU(x @ W1^T + b1), output = hidden @ W2^T + b2
 *
 * @param ctx Backward context
 * @param layer Layer index
 * @param grad_ffn_output Gradient from upstream (d_ff dimensional)
 * @param cached_ffn_input Cached input to FFN (d_model dimensional)
 * @param cached_ffn_hidden Cached GELU output (d_ff dimensional)
 * @return BackwardStatus
 */
BackwardStatus computeFFNBackward(
    BackwardContext& ctx,
    int layer,
    const float* grad_ffn_output,
    const float* cached_ffn_input,
    const float* cached_ffn_hidden);

/**
 * @brief Attention backward pass (W_o + Flash Attention + QKV)
 *
 * Full attention backward including:
 * 1. W_o projection backward
 * 2. Flash Attention backward (GQA-aware)
 * 3. QKV projection backward
 *
 * @param ctx Backward context
 * @param layer Layer index
 * @param grad_attn_output Gradient from upstream (d_model dimensional)
 * @param cached_ln1_output Cached RMSNorm1 output (input to attention)
 * @return BackwardStatus
 *
 * CRITICAL: Flash Attention v2 backward requires BF16 scratch buffers and softmax_lse
 * from the matching forward pass (no legacy accumulation semantics).
 */
BackwardStatus computeAttentionBackward(
    BackwardContext& ctx,
    int layer,
    const float* grad_attn_output,
    const float* cached_ln1_output);

} // namespace Backward
} // namespace GRIM
