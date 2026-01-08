#pragma once

/**
 * @file ForwardPhase3_InputLayer.hpp
 * @brief Phase 3: Input layer forward (token copy, embedding, ScratchBlock)
 *
 * This phase:
 * 1) Copies/append tokens to GPU buffers (if needed)
 * 2) Runs embedding lookup (full sequence or single-token)
 * 3) Optionally applies ScratchBlock reasoning
 * 4) Optionally applies activation quantization to embeddings
 */

#include "ForwardContext.hpp"

namespace GRIM {
namespace Forward {

ForwardStatus executePhase3_InputLayer(ForwardContext& ctx);

} // namespace Forward
} // namespace GRIM
