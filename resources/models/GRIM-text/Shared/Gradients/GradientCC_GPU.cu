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

__global__ void gradientSquaredNormKernel(
	const float* __restrict__ gradients,
	float* __restrict__ partial_sums,
	int n)
{
	__shared__ float shared[kBlockSize];

	const int tid = threadIdx.x;
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	float val = 0.0f;
	if (idx < n) {
		const float g = gradients[idx];
		val = g * g;
	}
	shared[tid] = val;
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (tid < stride) {
			shared[tid] += shared[tid + stride];
		}
		__syncthreads();
	}

	if (tid == 0) {
		atomicAdd(partial_sums, shared[0]);
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

void launchGradientClipping(
	float* gradients,
	int n,
	float max_norm,
	cudaStream_t stream)
{
	if (!validatePointers(gradients, n)) {
		return;
	}

	if (max_norm <= 0.0f || !std::isfinite(max_norm)) {
		fprintf(stderr, "GradientCC: invalid max_norm=%.6f\n", max_norm);
		return;
	}

	float* d_squared_norm = nullptr;
	if (cudaMalloc(&d_squared_norm, sizeof(float)) != cudaSuccess) {
		fprintf(stderr, "GradientCC: cudaMalloc failed in launchGradientClipping\n");
		return;
	}

	cudaMemsetAsync(d_squared_norm, 0, sizeof(float), stream);

	gradientSquaredNormKernel<<<computeGrid(n), kBlockSize, 0, stream>>>(
		gradients, d_squared_norm, n);

	float h_squared_norm = 0.0f;
	cudaMemcpyAsync(&h_squared_norm, d_squared_norm, sizeof(float),
					cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

	const float norm = sqrtf(h_squared_norm);

	if (norm > max_norm && norm > 0.0f) {
		const float scale_factor = max_norm / norm;
		scaleGradientsKernel<<<computeGrid(n), kBlockSize, 0, stream>>>(
			gradients, scale_factor, n);

		const cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			fprintf(stderr, "GradientCC: scaleGradientsKernel failed - %s\n",
					cudaGetErrorString(err));
		}
	}

	cudaFree(d_squared_norm);
}

}
