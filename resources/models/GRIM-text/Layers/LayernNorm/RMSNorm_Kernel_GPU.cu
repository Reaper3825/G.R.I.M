#include "RMSNorm_Kernel_GPU.hpp"

#include <cstdio>
#include <cuda_runtime.h>

namespace {

constexpr int kWarpSize = 32;

__device__ __forceinline__ float warpReduceSum(float value) {
	for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
		value += __shfl_down_sync(0xffffffff, value, offset);
	}
	return value;
}

inline void logCudaFailure(cudaError_t err,
						   const char* expr,
						   const char* file,
						   int line) {
	if (err == cudaSuccess) {
		return;
	}
	std::fprintf(stderr,
				 "RMSNorm CUDA error at %s:%d (%s): %s\n",
				 file,
				 line,
				 expr,
				 cudaGetErrorString(err));
}

#define RMSN_CUDA_CHECK(expr) logCudaFailure((expr), #expr, __FILE__, __LINE__)

__global__ void rmsNormKernel(const float* __restrict__ input,
						 float* __restrict__ output,
						 const float* __restrict__ gamma,
						 int batch_size,
						 int hidden_dim,
						 float eps) {
	int idx = blockIdx.x;
	if (idx >= batch_size) {
		return;
	}

	const float* x = input + idx * hidden_dim;
	float* y = output + idx * hidden_dim;

	__shared__ float shared_sum[kWarpSize];
	__shared__ float shared_inv_rms;

	float sum_sq = 0.0f;
	for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
		float value = x[i];
		sum_sq += value * value;
	}

	sum_sq = warpReduceSum(sum_sq);

	const int warp_id = threadIdx.x / kWarpSize;
	const int lane_id = threadIdx.x % kWarpSize;

	if (lane_id == 0 && warp_id < kWarpSize) {
		shared_sum[warp_id] = sum_sq;
	}
	__syncthreads();

	if (threadIdx.x < kWarpSize) {
		float value = (threadIdx.x < (blockDim.x + kWarpSize - 1) / kWarpSize)
						? shared_sum[threadIdx.x]
						: 0.0f;
		value = warpReduceSum(value);
		if (threadIdx.x == 0) {
			float mean_square = value / hidden_dim;
			float inv = rsqrtf(mean_square + eps);
			// Clamp inv_rms to match backward pass and prevent explosion
			shared_inv_rms = fminf(inv, 100.0f);
		}
	}
	__syncthreads();

	const float inv_rms = shared_inv_rms;

	if (gamma) {
		for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
			float normalized = x[i] * inv_rms;
			y[i] = gamma[i] * normalized;
		}
	} else {
		for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
			y[i] = x[i] * inv_rms;
		}
	}
}

} // namespace
namespace GRIM {
void launchRMSNorm(const float* input,
						        float* output,
						        const float* gamma,
						        int batch_size,
						        int hidden_dim,
						        float eps,
						        cudaStream_t stream) {
	if (batch_size <= 0 || hidden_dim <= 0) {
		std::fprintf(stderr,
					 "launchRMSNorm: Invalid dimensions - batch_size=%d hidden_dim=%d\n",
					 batch_size,
					 hidden_dim);
		return;
	}

	int threads = std::min(1024, ((hidden_dim + kWarpSize - 1) / kWarpSize) * kWarpSize);
	cudaError_t prev_err = cudaGetLastError();
	if (prev_err != cudaSuccess) {
		std::fprintf(stderr,
					 "launchRMSNorm: Previous CUDA error detected: %s\n",
					 cudaGetErrorString(prev_err));
	}

	rmsNormKernel<<<batch_size, threads, 0, stream>>>(
		input,
		output,
		gamma,
		batch_size,
		hidden_dim,
		eps);

	RMSN_CUDA_CHECK(cudaGetLastError());
}

__global__ void rmsNormBackwardKernel(const float* __restrict__ input,
                                      const float* __restrict__ grad_output,
                                      const float* __restrict__ gamma,
                                      float* __restrict__ grad_input,
                                      float* __restrict__ grad_gamma,
                                      int hidden_dim,
                                      float eps) {
	int token_idx = blockIdx.x;
	const float* x = input + token_idx * hidden_dim;
	const float* go = grad_output + token_idx * hidden_dim;
	float* gi = grad_input + token_idx * hidden_dim;

	const int tid = threadIdx.x;
	const int warp_id = tid / kWarpSize;
	const int lane_id = tid % kWarpSize;
	const int num_warps = (blockDim.x + kWarpSize - 1) / kWarpSize;

	__shared__ float warp_sums[32];  // Max 32 warps (1024 threads)
	__shared__ float inv_rms;
	__shared__ float dot_shared;

	// Compute mean square with proper multi-warp reduction
	float sum_sq = 0.0f;
	for (int i = tid; i < hidden_dim; i += blockDim.x) {
		float v = x[i];
		sum_sq += v * v;
	}
	sum_sq = warpReduceSum(sum_sq);
	
	// Gather warp results
	if (lane_id == 0 && warp_id < 32) {
		warp_sums[warp_id] = sum_sq;
	}
	__syncthreads();
	
	// Final reduction by thread 0
	if (tid == 0) {
		float total = 0.0f;
		for (int i = 0; i < num_warps && i < 32; ++i) {
			total += warp_sums[i];
		}
		float mean_sq = total / hidden_dim;
		inv_rms = rsqrtf(mean_sq + eps);
		// Clamp inv_rms to prevent explosion when variance is near zero
		inv_rms = fminf(inv_rms, 100.0f);
	}
	__syncthreads();

	float local_inv_rms = inv_rms;

	// Compute grad_gamma and first part of grad_input
	// NOTE: grad_output (go) is already per-token normalized via grad_scale at loss level.
	// We sum contributions here; no additional normalization needed.
	float dot = 0.0f;
	for (int i = tid; i < hidden_dim; i += blockDim.x) {
		float norm = x[i] * local_inv_rms;
		float gscale = gamma ? gamma[i] : 1.0f;
		float g = go[i] * gscale;
		dot += g * norm;
		gi[i] = g * local_inv_rms; // first term, will adjust after reduction

		if (grad_gamma) {
			atomicAdd(&grad_gamma[i], go[i] * norm);
		}
	}
	
	// Proper multi-warp reduction for dot product
	dot = warpReduceSum(dot);
	if (lane_id == 0 && warp_id < 32) {
		warp_sums[warp_id] = dot;
	}
	__syncthreads();
	
	if (tid == 0) {
		float total = 0.0f;
		for (int i = 0; i < num_warps && i < 32; ++i) {
			total += warp_sums[i];
		}
		dot_shared = total / hidden_dim;
	}
	__syncthreads();

	// Apply correction term for grad_input
	for (int i = tid; i < hidden_dim; i += blockDim.x) {
		float norm = x[i] * local_inv_rms;
		gi[i] -= norm * dot_shared;
	}
}

void launchRMSNormBackward(const float* input,
                           const float* grad_output,
                           const float* gamma,
                           float* grad_input,
                           float* grad_gamma,
                           int batch_size,
                           int hidden_dim,
                           float eps,
                           cudaStream_t stream) {
	if (batch_size <= 0 || hidden_dim <= 0) {
		return;
	}
	int threads = std::min(1024, ((hidden_dim + kWarpSize - 1) / kWarpSize) * kWarpSize);
	rmsNormBackwardKernel<<<batch_size, threads, 0, stream>>>(
		input, grad_output, gamma, grad_input, grad_gamma, hidden_dim, eps);
	RMSN_CUDA_CHECK(cudaGetLastError());
}

// Wrapper matching Encoding_GPU.cu naming convention
void launchRMSNormForward(const float* input, const float* gamma,
                          float* output, int tokens, int hidden_dim,
                          float epsilon, cudaStream_t stream) {
    launchRMSNorm(input, output, gamma, tokens, hidden_dim, epsilon, stream);
}

} // namespace GRIM

#undef RMSN_CUDA_CHECK
