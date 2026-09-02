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
#include <utility>

namespace {

constexpr int kBlockSize = GRIM::HyperParameters::CUDA_BLOCK_SIZE_STANDARD;
constexpr float EPSILON_SAFE_DIV = 1e-6f;


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
		gm.execution_block_sum_sq;
	const uint64_t count = gm.attention_count + gm.ffn_count +
		gm.rmsnorm_count + gm.execution_block_count;
	if (count == 0) {
		return std::numeric_limits<float>::quiet_NaN();
	}
	return rmsOrThrow(sum_sq, count, "encoder gradient");
}

void recordTopGroup(
	GRIM::GradClip::ClipResult& result,
	size_t index,
	const GRIM::ParameterGroup& group,
	float sum_sq)
{
	const uint64_t count = static_cast<uint64_t>(group.size());
	if (count == 0 || !std::isfinite(sum_sq)) {
		return;
	}

	const float rms = static_cast<float>(std::sqrt(static_cast<double>(sum_sq) / static_cast<double>(count)));
	GRIM::GradClip::ClipResult::TopGroup candidate{};
	candidate.index = index;
	candidate.rms = rms;
	candidate.sum_sq = sum_sq;
	candidate.count = count;
	candidate.type = static_cast<int>(group.type);
	candidate.layer_index = group.layer_index;
	candidate.valid = true;

	for (auto& top : result.top_groups) {
		if (!top.valid || candidate.rms > top.rms) {
			std::swap(candidate, top);
		}
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

} // namespace

namespace GRIM::GradClip {

ClipResult clipGradientNorms(
    ParameterGroup* groups,
    size_t num_groups,
	std::unique_ptr<GradNorm::GradNormScratch>& scratch,
    const HyperParameters::GradientClippingHP& clipping_hp,
    const HyperParameters::TrainingScheduleHP& schedule_hp,
    cudaStream_t stream
) {
    if (!groups || num_groups == 0) {
        throw std::runtime_error("[GradClip] clipGradientNorms called with null/empty parameter groups");
    }
    const float accumulation_scale = schedule_hp.accumulation_normalization_scale;
    if (!std::isfinite(accumulation_scale) || accumulation_scale <= 0.0f) {
        throw std::runtime_error("[GradClip] accumulation_normalization_scale must be finite and > 0, got " +
                                 std::to_string(accumulation_scale));
    }
    if (clipping_hp.enabled &&
        (!std::isfinite(clipping_hp.effective_per_token_limit) ||
         clipping_hp.effective_per_token_limit <= 0.0f)) {
        throw std::runtime_error("[GradClip] invalid effective_per_token_limit: " +
                                 std::to_string(clipping_hp.effective_per_token_limit));
    }

    // Step 1: Apply accumulation normalization (1/accum_steps) before norm measurement.
    if (accumulation_scale != 1.0f) {
        for (size_t i = 0; i < num_groups; ++i) {
            if (!groups[i].grads() || groups[i].size() == 0) continue;
            launchScaleGradients(groups[i].grads(),
                                 static_cast<int>(groups[i].size()),
                                 accumulation_scale, stream);
        }
    }

	ensureGradNormScratchForClip(scratch, num_groups, stream);

    // Step 2: Measure gradient norms through the tensor registry
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

    // Step 3: Aggregate all finite per-group sums into one global RMS. Use the
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
	result.scratchblock_rms_pre = std::numeric_limits<float>::quiet_NaN();
	result.metrics = measured_metrics;
	for (size_t i = 0; i < num_groups; ++i) {
		if (!groups[i].grads() || groups[i].size() == 0) continue;
		recordTopGroup(result, i, groups[i], scratch->h_partial_sums[i]);
	}

    // Step 4: Clip all registered gradients with one global coefficient.
    // Skipped when clipping is disabled.
    const float max_rms = clipping_hp.effective_per_token_limit;
    float coef = 1.0f;
    if (clipping_hp.enabled && global_rms > max_rms) {
        coef = max_rms / (global_rms + EPSILON_SAFE_DIV);
        for (size_t i = 0; i < num_groups; ++i) {
            if (!groups[i].grads() || groups[i].size() == 0) continue;
            launchScaleGradients(groups[i].grads(),
                                 static_cast<int>(groups[i].size()),
                                 coef, stream);
        }
        result.clipped = true;
    }

    // Step 5: True post-clip RMS (exact, not clamped to threshold).
    result.global_rms_post = global_rms * coef;

    return result;
}

} // namespace GRIM::GradClip
