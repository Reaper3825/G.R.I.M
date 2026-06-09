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
constexpr bool ENABLE_GPU_COPY_LOGS = false;           ///< "[GPU_COPY]" H2D copy progress logs
constexpr bool ENABLE_AUTOGRAD_TRAINING_LOGS = false;   ///< "[AutogradTraining]" forward/backward step info

// Backward pass logging - DISABLED FOR PRODUCTION
constexpr bool ENABLE_BACKWARD_LAYER_LOGS = false;     ///< "[BACKWARD] Starting layer=X"
constexpr bool ENABLE_BACKWARD_GRADIENT_LOGS = false;  ///< "[BACKWARD layer=X] W_o weight gradient computed"
constexpr bool ENABLE_BACKWARD_FLASH_ATTN_LOGS = false;///< "[BACKWARD layer=X] Flash Attention backward complete"

// Loss computation logging - DISABLED FOR PRODUCTION
constexpr bool ENABLE_LOSS_BACKWARD_SAMPLING = false;  ///< NLLLossGradFn diagnostic sampling

// Expensive diagnostics (D2H copies for analysis - disable for production training)
constexpr bool ENABLE_EXPENSIVE_DIAGNOSTICS = false;   ///< Rule 21 argmax analysis, embedding cosine, etc.

// GPU allocation diagnostics - OFF by default. When enabled, logs allocation requests
// with requested size and current free/total VRAM to pinpoint OOM sources.
constexpr bool ENABLE_GPU_ALLOCATION_LOGS = false;     ///< [GPU_ALLOC] allocation request diagnostics

// FlashAttention equation diagnostics — 5 sync D2H copies per layer × 12 layers = 60 pipeline
// drains per batch PLUS O(seqlen²) host-side attention score computation. These were critical
// during Issue #76/#84 debugging but are catastrophic for training throughput.
constexpr bool ENABLE_FA_EQUATION_DIAGNOSTICS = true;  ///< [FA-FWD-*], [ATTN_SCORE_EQUATION], plus the layer-recorder [ATTN_BREADTH]

// QKV projection equation diagnostics — extremely spammy per-layer/per-batch tape entries.
// Keep off unless actively debugging the QKV projection path.
constexpr bool ENABLE_QKV_PROJECTION_EQUATION_LOGS = false;  ///< [QKV_PROJECTION_EQUATION]

// LM-head GEMM equation diagnostics — targeted at the hidden-state alignment / Issue #132
// investigation path. Gated again by BatchLogTape Debug level and skipThisPass().
constexpr bool ENABLE_GEMM_EQUATION_LOGS = true;  ///< [LM_HEAD_GEMM_EQUATION], [LM_HEAD_GEMM_BACKWARD_EQUATION]

} // namespace VerboseLogging
} // namespace GRIM
