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

// Forward pass logging
constexpr bool ENABLE_FORWARD_FLASH_ATTN_LOGS = false;  ///< "[FORWARD] Flash Attention complete"
constexpr bool ENABLE_FORWARD_CACHE_LOGS = false;       ///< "[FORWARD] Cached attn_bhsd"
constexpr bool ENABLE_ORDER_LOGS = false;               ///< "[ORDER] ForwardPhase*" orchestration logs
constexpr bool ENABLE_VOCAB_TIMING_LOGS = false;        ///< "[VOCAB_TIMING]" embedding/LM head timing

// Backward pass logging
constexpr bool ENABLE_BACKWARD_LAYER_LOGS = false;      ///< "[BACKWARD] Starting layer=X"
constexpr bool ENABLE_BACKWARD_GRADIENT_LOGS = false;   ///< "[BACKWARD layer=X] W_o weight gradient computed"
constexpr bool ENABLE_BACKWARD_FLASH_ATTN_LOGS = false; ///< "[BACKWARD layer=X] Flash Attention backward complete"

// Loss computation logging
constexpr bool ENABLE_LOSS_ORDER_LOGS = false;          ///< "[ORDER] computeLossBatch.*"

} // namespace VerboseLogging
} // namespace GRIM
