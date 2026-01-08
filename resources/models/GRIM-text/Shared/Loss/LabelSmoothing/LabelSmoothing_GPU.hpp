#pragma once

#include "Shared/Loss/Loss.hpp"

namespace GRIM::Loss {

void applyLabelSmoothing(const LossContext& ctx,
						 const LabelSmoothingConfig& cfg,
						 DeviceBuffers buffers,
						 LossBreakdown& out_loss);

} // namespace GRIM::Loss

