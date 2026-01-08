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

// Clip gradients by their global L2 norm.
void launchGradientClipping(
	float* gradients,
	int n,
	float max_norm,
	cudaStream_t stream);

}
