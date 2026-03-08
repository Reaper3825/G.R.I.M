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
constexpr bool ENABLE_FORWARD_FLASH_ATTN_LOGS = false; ///< "[FORWARD] Flash Attention complete"
constexpr bool ENABLE_FORWARD_CACHE_LOGS = false;      ///< "[FORWARD] Cached attn_bhsd"
constexpr bool ENABLE_ORDER_LOGS = false;              ///< "[ORDER] ForwardPhase*" orchestration logs
constexpr bool ENABLE_VOCAB_TIMING_LOGS = false;       ///< "[VOCAB_TIMING]" embedding/LM head timing
constexpr bool ENABLE_FORWARD_DIAG_LOGS = false;       ///< "[ForwardDiag]" pre-loss logit diagnostics
constexpr bool ENABLE_GPU_COPY_LOGS = false;           ///< "[GPU_COPY]" H2D copy progress logs
constexpr bool ENABLE_AUTOGRAD_TRAINING_LOGS = false;   ///< "[AutogradTraining]" forward/backward step info

// Backward pass logging - DISABLED FOR PRODUCTION
constexpr bool ENABLE_BACKWARD_LAYER_LOGS = false;     ///< "[BACKWARD] Starting layer=X"
constexpr bool ENABLE_BACKWARD_GRADIENT_LOGS = false;  ///< "[BACKWARD layer=X] W_o weight gradient computed"
constexpr bool ENABLE_BACKWARD_FLASH_ATTN_LOGS = false;///< "[BACKWARD layer=X] Flash Attention backward complete"

// Loss computation logging - DISABLED FOR PRODUCTION
constexpr bool ENABLE_LOSS_ORDER_LOGS = false;         ///< "[ORDER] computeLossBatch.*"

// Expensive diagnostics (D2H copies for analysis - disable for production training)
constexpr bool ENABLE_EXPENSIVE_DIAGNOSTICS = false;   ///< Rule 21 argmax analysis, embedding cosine, etc.

// FlashAttention equation diagnostics — 5 sync D2H copies per layer × 12 layers = 60 pipeline
// drains per batch PLUS O(seqlen²) host-side attention score computation. These were critical
// during Issue #76/#84 debugging but are catastrophic for training throughput.
constexpr bool ENABLE_FA_EQUATION_DIAGNOSTICS = false;  ///< [FA-FWD-*], [ATTN_SCORE_EQUATION]

// Training signal diagnostics — sync D2H copies of logits, hidden states, and weight rows
// every batch for argmax distribution, logit statistics, hidden cosine, rho buildup, and
// LM head row norms. Includes 500+ individual cudaMemcpy calls for weight row sampling
// (~50ms) plus ~20MB of logit D2H copies. Total overhead: ~70-80ms per batch.
// Tags: BATCH_PRED_DIST, [LogitSignal], [HiddenCosine], [RHO_BUILDUP_EQUATION], [LMHeadNorm]
constexpr bool ENABLE_TRAINING_SIGNAL_DIAGNOSTICS = false;

// Token 277 (collapse token) weight tracking — two sync D2H copies per optimizer step
// (pre and post) to track ||W[277]|| delta. ~2ms per step.
// Tags: [Token277] PRE-OPT, [Token277] POST-OPT
constexpr bool ENABLE_TOKEN277_TRACKING = false;

// NLL loss backward gradient sampling — 200 individual sizeof(float) cudaMemcpy D2H
// calls in a tight loop to sample grad_log_probs at target columns. Each is an implicit
// pipeline drain. Total overhead: ~20ms per backward pass.
// Tags: [NLL-BWD-OUT]
constexpr bool ENABLE_LOSS_BACKWARD_SAMPLING = false;

} // namespace VerboseLogging
} // namespace GRIM
