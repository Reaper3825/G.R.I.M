/**
 * @file VerboseLogging.hpp
 * @brief Global compile-time flags for verbose logging control
 * 
 * Set these to false to disable spammy per-operation logs.
 * Controlled at compile time for zero runtime overhead.
 */

#pragma once

namespace GRIM {
namespace VerboseLogging {
 
// Forward pass logging - DISABLED FOR PRODUCTION (enable for debugging only)
constexpr bool ENABLE_FORWARD_FLASH_ATTN_LOGS = true; ///< "[FORWARD] Flash Attention complete"
constexpr bool ENABLE_FORWARD_CACHE_LOGS = true;      ///< "[FORWARD] Cached attn_bhsd"
constexpr bool ENABLE_ORDER_LOGS = true;              ///< "[ORDER] ForwardPhase*" orchestration logs
constexpr bool ENABLE_VOCAB_TIMING_LOGS = true;       ///< "[VOCAB_TIMING]" embedding/LM head timing
constexpr bool ENABLE_FORWARD_DIAG_LOGS = true;       ///< "[ForwardDiag]" pre-loss logit diagnostics
constexpr bool ENABLE_GPU_COPY_LOGS = true;           ///< "[GPU_COPY]" H2D copy progress logs
constexpr bool ENABLE_AUTOGRAD_TRAINING_LOGS = true;   ///< "[AutogradTraining]" forward/backward step info

// Backward pass logging - DISABLED FOR PRODUCTION
constexpr bool ENABLE_BACKWARD_LAYER_LOGS = true;     ///< "[BACKWARD] Starting layer=X"
constexpr bool ENABLE_BACKWARD_GRADIENT_LOGS = true;  ///< "[BACKWARD layer=X] W_o weight gradient computed"
constexpr bool ENABLE_BACKWARD_FLASH_ATTN_LOGS = true;///< "[BACKWARD layer=X] Flash Attention backward complete"

// Loss computation logging - DISABLED FOR PRODUCTION
constexpr bool ENABLE_LOSS_ORDER_LOGS = true;         ///< "[ORDER] computeLossBatch.*"
constexpr bool ENABLE_LOSS_BACKWARD_SAMPLING = true;  ///< NLLLossGradFn diagnostic sampling

// Expensive diagnostics (D2H copies for analysis - disable for production training)
constexpr bool ENABLE_EXPENSIVE_DIAGNOSTICS = false;   ///< Rule 21 argmax analysis, embedding cosine, etc.

// FlashAttention equation diagnostics — 5 sync D2H copies per layer × 12 layers = 60 pipeline
// drains per batch PLUS O(seqlen²) host-side attention score computation. These were critical
// during Issue #76/#84 debugging but are catastrophic for training throughput.
constexpr bool ENABLE_FA_EQUATION_DIAGNOSTICS = false;  ///< [FA-FWD-*], [ATTN_SCORE_EQUATION]

} // namespace VerboseLogging
} // namespace GRIM
