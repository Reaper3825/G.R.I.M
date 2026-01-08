/**
 * @file BackwardPhase3_InputLayer.hpp
 * @brief Phase 3: Input Layer Backward Pass Declaration
 *
 * This phase handles the final backward steps:
 * 1. ScratchBlock backward (if enabled) - atom embedding injection
 * 2. Embedding backward - token gradient accumulation
 *
 * This is the simplest phase but has important considerations:
 * - ScratchBlock uses atom tables for structural reasoning
 * - Embedding gradients accumulate into vocab-sized buffer
 * - When tie_embeddings=true, embedding grads merge with LM head grads
 *
 * @note After this phase, all gradients are computed and ready for optimizer step
 */

#pragma once

#include "BackwardContext.hpp"

namespace GRIM {
namespace Backward {

//======================================================//
//  Phase 3 Entry Point
//======================================================//

/**
 * @brief Execute input layer backward pass
 *
 * @param ctx Backward context (input: current_grad from Phase 2)
 * @return BackwardStatus::SUCCESS or appropriate error code
 *
 * @pre ctx.current_grad points to gradient ready for input layers
 * @post All gradients computed, ready for optimizer step
 *
 * OPERATIONS:
 * 1. ScratchBlock backward (conditional)
 * 2. Embedding backward (scatter-add to vocab-sized buffer)
 * 3. Tie embeddings (merge with LM head if enabled)
 */
BackwardStatus executePhase3_InputLayer(BackwardContext& ctx);

//======================================================//
//  Component Backward Functions
//======================================================//

/**
 * @brief ScratchBlock backward pass (if enabled)
 *
 * Propagates gradients through atom embedding injection.
 * Uses cached atom embeddings and positions from forward pass.
 *
 * @param ctx Backward context
 * @return BackwardStatus
 */
BackwardStatus computeScratchBlockBackward(BackwardContext& ctx);

/**
 * @brief Embedding backward pass
 *
 * Accumulates gradients from sequence positions back to token embeddings.
 * Uses scatter-add pattern: embedding_grads[token_id] += grad for each position.
 *
 * @param ctx Backward context
 * @return BackwardStatus
 *
 * @note When tie_embeddings=true, embedding gradients are added to LM head weight gradients
 */
BackwardStatus computeEmbeddingBackward(BackwardContext& ctx);

} // namespace Backward
} // namespace GRIM
