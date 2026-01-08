#pragma once

#include "Shared/Loss/Loss.hpp"

namespace GRIM::Loss {

// Accumulates RLAIF-style preference KL between student and reference logits.
void accumulatePreferenceKL(const LossContext& ctx,
							  const PreferenceKLConfig& cfg,
							  DeviceBuffers buffers,
							  LossBreakdown& out_loss);

} // namespace GRIM::Loss
