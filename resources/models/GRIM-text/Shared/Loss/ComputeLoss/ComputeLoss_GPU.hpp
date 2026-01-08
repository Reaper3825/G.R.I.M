//======================================================//
//  ComputeLoss_GPU.hpp
//  UNIFIED LOSS PIPELINE - Production Ready
//  
//  Single kernel: Focal + LabelSmoothing + CrossEntropy
//  Telemetry-ready for hierarchical monitoring
//======================================================//

#pragma once

#include "Shared/Loss/Loss.hpp"
#include "Shared/Loss/UnifiedLoss/UnifiedLoss_GPU.hpp"

namespace GRIM::Loss {

// Runs the unified loss pipeline and returns breakdown.
// FAIL LOUD: Returns zero breakdown on any error (check stderr).
LossBreakdown launchLossPipeline(const LossContext& ctx,
								  const LossConfig& cfg,
								  DeviceBuffers buffers);

// Access telemetry from last loss computation
// Contains: loss stats, gradient stats, focal weights, error counts
const UnifiedLossTelemetry& getLastTelemetry();

// Reduces the per-token loss buffer into a single scalar on the device.
void launchReduceLoss(const float* losses,
					  float* output,
					  int n,
					  cudaStream_t stream);

// Convenience wrapper for the cross-entropy gradient kernel.
// NOTE: Gradients are now computed in launchLossPipeline, this is for legacy use.
void launchCrossEntropyGradient(const LossContext& ctx,
								  DeviceBuffers buffers);

} // namespace GRIM::Loss
