#include "RMSNorm_Kernel_GPU.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <algorithm>

namespace {

constexpr int kWarpSize = 32;

__device__ __forceinline__ float warpReduceSum(float value) {
	for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
		value += __shfl_down_sync(0xffffffff, value, offset);
	}
	return value;
}

#define RMSN_CUDA_CHECK(expr)                                                    \
    do {                                                                         \
        cudaError_t err__ = (expr);                                              \
        if (err__ != cudaSuccess) {                                              \
            throw std::runtime_error(std::string("[RMSNorm] CUDA error: ") +     \
                                     cudaGetErrorString(err__));                 \
        }                                                                        \
    } while (0)

__global__ void rmsNormKernel(const float* __restrict__ input,
							  float* __restrict__ output,
							  const float* __restrict__ gamma,
							  int batch_size,
							  int hidden_dim,
							  float eps) {
	const int idx = blockIdx.x;
	if (idx >= batch_size) {
		return;
	}

	const float* x = input + idx * hidden_dim;
	float* y = output + idx * hidden_dim;

	const int tid = threadIdx.x;
	const int warp_id = tid / kWarpSize;
	const int lane_id = tid % kWarpSize;
	const int num_warps = (blockDim.x + kWarpSize - 1) / kWarpSize;

	__shared__ float warp_sums[32];   // max 32 warps (1024 threads)
	__shared__ float shared_inv_rms;

	float sum_sq = 0.0f;
	for (int i = tid; i < hidden_dim; i += blockDim.x) {
		const float v = x[i];
		sum_sq += v * v;
	}

	// Reduce within warp
	sum_sq = warpReduceSum(sum_sq);

	// Write warp partials
	if (lane_id == 0 && warp_id < 32) {
		warp_sums[warp_id] = sum_sq;
	}
	__syncthreads();

	// Final reduction on thread 0
	if (tid == 0) {
		float total = 0.0f;
		for (int i = 0; i < num_warps && i < 32; ++i) {
			total += warp_sums[i];
		}
		const float mean_square = total / hidden_dim;
		// Issue #104: Removed inv_rms clamp - RMSNorm MUST normalize to RMS=1.0
		// The clamp broke normalization when input RMS was small (e.g., embeddings with RMS=0.006)
		shared_inv_rms = rsqrtf(mean_square + eps);
	}
	__syncthreads();

	const float inv_rms = shared_inv_rms;

	if (gamma) {
		for (int i = tid; i < hidden_dim; i += blockDim.x) {
			const float normalized = x[i] * inv_rms;
			y[i] = gamma[i] * normalized;
		}
	} else {
		for (int i = tid; i < hidden_dim; i += blockDim.x) {
			y[i] = x[i] * inv_rms;
		}
	}
}

__global__ void rmsNormBackwardKernel(const float* __restrict__ input,
                                      const float* __restrict__ grad_output,
                                      const float* __restrict__ gamma,
                                      float* __restrict__ grad_input,
                                      float* __restrict__ grad_gamma,
                                      int hidden_dim,
                                      float eps) {
	const int token_idx = blockIdx.x;
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
		const float v = x[i];
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
		const float mean_sq = total / hidden_dim;
		// Issue #104: Removed inv_rms clamp - RMSNorm MUST normalize to RMS=1.0
		inv_rms = rsqrtf(mean_sq + eps);
	}
	__syncthreads();

	const float local_inv_rms = inv_rms;

	// Compute grad_gamma and first part of grad_input
	// NOTE: grad_output (go) is already per-token normalized via grad_scale at loss level.
	// We sum contributions here; no additional normalization needed.
	float dot = 0.0f;
	for (int i = tid; i < hidden_dim; i += blockDim.x) {
		const float norm = x[i] * local_inv_rms;
		const float gscale = gamma ? gamma[i] : 1.0f;
		const float g = go[i] * gscale;

		dot += g * norm;

		// first term, will adjust after reduction
		gi[i] = g * local_inv_rms;

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
	// Correct formula: dx = inv_rms * (g - x * inv_rms^2 * (sum(g*x*inv_rms)/hidden_dim))
	for (int i = tid; i < hidden_dim; i += blockDim.x) {
		const float norm = x[i] * local_inv_rms;
		gi[i] -= norm * local_inv_rms * dot_shared;
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
		throw std::runtime_error("[RMSNorm] Invalid dimensions: batch_size=" +
								 std::to_string(batch_size) + " hidden_dim=" +
								 std::to_string(hidden_dim));
	}

	int threads = std::min(1024, ((hidden_dim + kWarpSize - 1) / kWarpSize) * kWarpSize);

	rmsNormKernel<<<batch_size, threads, 0, stream>>>(
		input,
		output,
		gamma,
		batch_size,
		hidden_dim,
		eps);

	RMSN_CUDA_CHECK(cudaGetLastError());
}


//======================================================//
//  TensorView-based launchers (PREFERRED)
//======================================================//

void launchRMSNormForward(const RMSNormForwardParams& params) {
	// RULE 20: Fail loud validation
	params.validate("launchRMSNormForward");

	const int tokens = params.tokens();
	const int hidden_dim = params.hidden_dim();

	int threads = std::min(1024, ((hidden_dim + kWarpSize - 1) / kWarpSize) * kWarpSize);
	rmsNormKernel<<<tokens, threads, 0, params.stream>>>(
		params.input.ptr,
		params.output.ptr,
		params.gamma.ptr,
		tokens,
		hidden_dim,
		params.epsilon);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		throw std::runtime_error(std::string("[RMSNormForward] Kernel launch failed: ") +
								 cudaGetErrorString(err));
	}
}

void launchRMSNormBackward(const RMSNormBackwardParams& params) {
	// RULE 20: Fail loud validation
	params.validate("launchRMSNormBackward");

	const int tokens = params.tokens();
	const int hidden_dim = params.hidden_dim();

	int threads = std::min(1024, ((hidden_dim + kWarpSize - 1) / kWarpSize) * kWarpSize);
	rmsNormBackwardKernel<<<tokens, threads, 0, params.stream>>>(
		params.input.ptr,
		params.grad_output.ptr,
		params.gamma.ptr,
		params.grad_input.ptr,
		params.grad_gamma.is_valid() ? params.grad_gamma.ptr : nullptr,
		hidden_dim,
		params.epsilon);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		throw std::runtime_error(std::string("[RMSNormBackward] Kernel launch failed: ") +
								 cudaGetErrorString(err));
	}
}

} // namespace GRIM

#undef RMSN_CUDA_CHECK
