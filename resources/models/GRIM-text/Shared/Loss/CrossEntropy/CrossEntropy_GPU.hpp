#pragma once

#include "Shared/Loss/Loss.hpp"

namespace GRIM::Loss {

void validate(const LossContext& ctx);

void computeCrossEntropyLoss(const LossContext& ctx,
							 DeviceBuffers buffers);

void reduceLossBuffer(const float* losses,
					  float* output,
					  int count,
					  cudaStream_t stream);

void computeCrossEntropyGradient(const LossContext& ctx,
								 DeviceBuffers buffers);

} // namespace GRIM::Loss

