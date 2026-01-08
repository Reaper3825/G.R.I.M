#pragma once

#include "Shared/Loss/Loss.hpp"

namespace GRIM::Loss {

void accumulateDistillationKL(const LossContext& ctx,
							  const DistillationConfig& cfg,
							  DeviceBuffers buffers,
							  LossBreakdown& out_loss);

} // namespace GRIM::Loss

