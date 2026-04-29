//======================================================//
//  GradientCC_GPU.cu
//  CUDA kernels for gradient clamp + clip utilities
//  + Registry-level clipping (GRIM::GradClip)
//======================================================//

#include "GradientCC_GPU.hpp"
#include "../HyperParameters/HyperParameters_GPU.hpp"
#include "../GradNorm/GradNormGPU.hpp"
#include "../TensorContract/TensorContract_GPU.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

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

//======================================================//
//  Layer 2: Registry-level gradient clipping
//  Operates on ParameterGroup tensors via GradNorm + scale
//======================================================//

namespace GRIM::GradClip {

ClipResult clipGradientNorms(
    ParameterGroup* groups,
    size_t num_groups,
    GradNorm::GradNormScratch* scratch,
    const ClipConfig& config,
    cudaStream_t stream
) {
    if (!groups || num_groups == 0) {
        throw std::runtime_error("[GradClip] clipGradientNorms called with null/empty parameter groups");
    }
    if (!scratch) {
        throw std::runtime_error("[GradClip] clipGradientNorms called with null GradNormScratch");
    }
    if (config.max_rms <= 0.0f) {
        throw std::runtime_error("[GradClip] max_rms must be > 0, got " + std::to_string(config.max_rms));
    }

    // Step 1: Measure gradient norms through the tensor registry
    auto status = GradNorm::measureGradientNorms(groups, num_groups, scratch, stream);
    if (status != GradNorm::GradNormStatus::SUCCESS) {
        throw std::runtime_error("[GradClip] measureGradientNorms failed: " +
                                 std::string(GradNorm::statusToString(status)));
    }
    // measureGradientNorms syncs internally — h_partial_sums is valid.

    // Step 2: Aggregate all finite per-group sums into one global RMS. Use the
    // per-group scratch directly so every registered group type participates.
    float global_sum_sq = 0.0f;
    int64_t global_count = 0;
    for (size_t i = 0; i < num_groups; ++i) {
        if (!groups[i].grads() || groups[i].size() == 0) continue;

        const float sq = scratch->h_partial_sums[i];
        if (!std::isfinite(sq)) continue;

        global_sum_sq += sq;
        global_count += static_cast<int64_t>(groups[i].size());
    }

    const float global_rms = (global_count > 0)
        ? std::sqrt(global_sum_sq / static_cast<float>(global_count))
        : 0.0f;

    ClipResult result;
    result.global_rms_pre = global_rms;

    // Step 3: Clip all registered gradients with one global coefficient
    if (global_rms > config.max_rms) {
        const float coef = config.max_rms / (global_rms + 1e-8f);
        for (size_t i = 0; i < num_groups; ++i) {
            if (!groups[i].grads() || groups[i].size() == 0) continue;
            launchScaleGradients(groups[i].grads(),
                                 static_cast<int>(groups[i].size()),
                                 coef, stream);
        }
        result.clipped = true;
    }

    // Step 4: Compute post-clip RMS
    result.global_rms_post = result.clipped ? config.max_rms : global_rms;

    return result;
}

} // namespace GRIM::GradClip
