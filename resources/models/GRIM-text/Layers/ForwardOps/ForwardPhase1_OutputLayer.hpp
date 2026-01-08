#pragma once

/**
 * @file ForwardPhase1_OutputLayer.hpp
 * @brief Phase 1: Output layer forward (LM head projection)
 */

#include "ForwardContext.hpp"

namespace GRIM {
namespace Forward {

ForwardStatus executePhase1_OutputLayer(ForwardContext& ctx);

} // namespace Forward
} // namespace GRIM
