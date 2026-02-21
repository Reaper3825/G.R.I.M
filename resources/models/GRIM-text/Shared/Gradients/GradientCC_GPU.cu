//======================================================//
//  GradientCC_GPU.cu
//  CUDA kernels for gradient clamp + clip utilities
//======================================================//

#include "GradientCC_GPU.hpp"
#include "../HyperParameters/HyperParameters_GPU.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <cstdio>

namespace {

constexpr int kBlockSize = GRIM::HyperParameters::CUDA_BLOCK_SIZE_STANDARD;

__global__ void clampGradientsKernel(
	float* __restrict__ gradients,
	int n,
	float min_val,
	float max_val)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n) {
		return;
	}

	const float val = gradients[idx];

	if (!isfinite(val) || val < min_val) {
		gradients[idx] = min_val;
	} else if (val > max_val) {
		gradients[idx] = max_val;
	}
}

__global__ void scaleGradientsKernel(
	float* __restrict__ gradients,
	float scale_factor,
	int n)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		gradients[idx] *= scale_factor;
	}
}

inline bool validatePointers(const float* gradients, int n)
{
	if (!gradients || n <= 0) {
		fprintf(stderr, "GradientCC: invalid inputs (ptr=%p, n=%d)\n", gradients, n);
		return false;
	}
	return true;
}

inline dim3 computeGrid(int n)
{
	return dim3((n + kBlockSize - 1) / kBlockSize);
}

} // namespace

extern "C" {

void launchClampGradients(
	float* gradients,
	int n,
	float min_val,
	float max_val,
	cudaStream_t stream)
{
	if (!validatePointers(gradients, n)) {
		return;
	}

	if (min_val > max_val) {
		fprintf(stderr, "GradientCC: min_val (%.3f) > max_val (%.3f)\n", min_val, max_val);
		return;
	}

	clampGradientsKernel<<<computeGrid(n), kBlockSize, 0, stream>>>(
		gradients, n, min_val, max_val);

	const cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "GradientCC: launchClampGradients failed - %s\n",
				cudaGetErrorString(err));
	}
}

void launchScaleGradients(
	float* gradients,
	int n,
	float scale_factor,
	cudaStream_t stream)
{
	if (!validatePointers(gradients, n)) {
		return;
	}

	scaleGradientsKernel<<<computeGrid(n), kBlockSize, 0, stream>>>(
		gradients, scale_factor, n);

	const cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "GradientCC: launchScaleGradients failed - %s\n",
				cudaGetErrorString(err));
	}
}

}
