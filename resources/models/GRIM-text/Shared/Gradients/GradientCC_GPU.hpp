//======================================================//
//  GradientCC_GPU.hpp
//  Gradient Clamp + Clip launchers (CUDA)
//======================================================//

#pragma once

#include <cuda_runtime.h>

extern "C" {

// Clamp gradients element-wise into [min_val, max_val] (handles inf/NaN).
void launchClampGradients(
	float* gradients,
	int n,
	float min_val,
	float max_val,
	cudaStream_t stream);

// Scale all gradient elements by a constant factor: gradients[i] *= scale_factor.
void launchScaleGradients(
	float* gradients,
	int n,
	float scale_factor,
	cudaStream_t stream);

}
