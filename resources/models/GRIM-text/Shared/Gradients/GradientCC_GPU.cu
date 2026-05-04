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
#include <cstdint>
#include <limits>
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

void ensureGradNormScratchForClip(
	std::unique_ptr<GRIM::GradNorm::GradNormScratch>& scratch,
	size_t required_groups,
	cudaStream_t stream)
{
	if (required_groups == 0) {
		throw std::runtime_error("[GradClip] cannot allocate GradNormScratch for zero parameter groups");
	}

	if (!scratch) {
		scratch.reset(GRIM::GradNorm::allocateGradNormScratch(required_groups, stream));
		if (!scratch) {
			throw std::runtime_error("[GradClip] allocateGradNormScratch returned NULL");
		}
	}

	if (!scratch->d_partial_sums || !scratch->h_partial_sums || !scratch->h_metrics) {
		throw std::runtime_error("[GradClip] GradNormScratch buffer set is incomplete");
	}
	if (scratch->max_groups < required_groups) {
		throw std::runtime_error("[GradClip] GradNormScratch capacity mismatch required_groups=" +
								 std::to_string(required_groups) +
								 " scratch_max_groups=" + std::to_string(scratch->max_groups));
	}
}

float rmsOrThrow(double sum_sq, uint64_t count, const char* label) {
    if (count == 0) {
		throw std::runtime_error(std::string("[GradClip] ") + label + " count is zero");
	}
	if (!std::isfinite(sum_sq)) {
		throw std::runtime_error(std::string("[GradClip] ") + label + " sum_sq is non-finite");
	}
	return static_cast<float>(std::sqrt(sum_sq / static_cast<double>(count)));
}

float encoderTelemetryRms(const GRIM::GradNorm::GradMetrics& gm) {
	const double sum_sq = gm.attention_sum_sq + gm.ffn_sum_sq + gm.rmsnorm_sum_sq +
		gm.scratchblock_sum_sq + gm.reasoning_head_sum_sq + gm.execution_block_sum_sq;
	const uint64_t count = gm.attention_count + gm.ffn_count +
		gm.rmsnorm_count + gm.scratchblock_count + gm.reasoning_head_count +
		gm.execution_block_count;
	if (count == 0) {
		return std::numeric_limits<float>::quiet_NaN();
	}
	return rmsOrThrow(sum_sq, count, "encoder gradient");
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
	std::unique_ptr<GradNorm::GradNormScratch>& scratch,
    const ClipConfig& config,
    cudaStream_t stream
) {
    if (!groups || num_groups == 0) {
        throw std::runtime_error("[GradClip] clipGradientNorms called with null/empty parameter groups");
    }
    if (config.max_rms <= 0.0f) {
        throw std::runtime_error("[GradClip] max_rms must be > 0, got " + std::to_string(config.max_rms));
    }

	ensureGradNormScratchForClip(scratch, num_groups, stream);

    // Step 1: Measure gradient norms through the tensor registry
	auto status = GradNorm::measureGradientNorms(groups, num_groups, scratch.get(), stream);
    if (status != GradNorm::GradNormStatus::SUCCESS) {
        throw std::runtime_error("[GradClip] measureGradientNorms failed: " +
                                 std::string(GradNorm::statusToString(status)));
    }
    // measureGradientNorms syncs internally — h_partial_sums is valid.
	if (!scratch->h_metrics) {
		throw std::runtime_error("[GradClip] h_metrics is NULL after measureGradientNorms");
	}
	const auto& measured_metrics = *scratch->h_metrics;
	if (measured_metrics.groups_processed != num_groups) {
		throw std::runtime_error("[GradClip] GradNorm processed group count mismatch expected=" +
								 std::to_string(num_groups) +
								 " actual=" + std::to_string(measured_metrics.groups_processed));
	}
	if (measured_metrics.has_nan || measured_metrics.has_inf) {
		throw std::runtime_error("[GradClip] NaN/Inf detected in gradients first_nan_group=" +
								 std::to_string(measured_metrics.first_nan_group) +
								 " first_inf_group=" + std::to_string(measured_metrics.first_inf_group));
	}

    // Step 2: Aggregate all finite per-group sums into one global RMS. Use the
    // per-group scratch directly so every registered group type participates.
	double global_sum_sq = 0.0;
	uint64_t global_count = 0;
    for (size_t i = 0; i < num_groups; ++i) {
        if (!groups[i].grads() || groups[i].size() == 0) continue;

        const float sq = scratch->h_partial_sums[i];
        if (!std::isfinite(sq)) continue;

		global_sum_sq += static_cast<double>(sq);
		global_count += static_cast<uint64_t>(groups[i].size());
    }

	const float global_rms = rmsOrThrow(global_sum_sq, global_count, "registered global gradient");

    ClipResult result;
	result.measured_group_count = num_groups;
    result.global_rms_pre = global_rms;
	result.encoder_rms_pre = encoderTelemetryRms(measured_metrics);
	result.scratchblock_rms_pre = (measured_metrics.scratchblock_count > 0)
		? static_cast<float>(std::sqrt(measured_metrics.scratchblock_sum_sq / static_cast<double>(measured_metrics.scratchblock_count)))
		: std::numeric_limits<float>::quiet_NaN();
	result.metrics = measured_metrics;

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
