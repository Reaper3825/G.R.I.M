#pragma once

/**
 * @file ForwardPhase2_Encoder.hpp
 * @brief Phase 2: Encoder forward pass (full sequence or incremental)
 *
 * This phase runs the transformer encoder:
 * - Full sequence (TrainingFull / Prefill / DecodeFull)
 * - Incremental single-token path (DecodeIncremental)
 *
 * No buffer allocations - uses TrainingState preallocated caches.
 */

#include "ForwardContext.hpp"

namespace GRIM {
namespace Forward {

ForwardStatus executePhase2_Encoder(ForwardContext& ctx);

} // namespace Forward
} // namespace GRIM
