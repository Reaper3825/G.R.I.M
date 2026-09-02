//======================================================//
//  GradientCC_GPU.cu
//  CUDA kernels for gradient scaling and registry-level
//  clipping (GRIM::GradClip)
//======================================================//

#include "GradientCC_GPU.hpp"
#include "../HyperParameters/HyperParameters_GPU.hpp"
#include "../StreamController/StreamController_GPU.hpp"
#include "../TensorContract/TensorContract_GPU.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

constexpr int kBlockSize = GRIM::HyperParameters::CUDA_BLOCK_SIZE_STANDARD;
constexpr int kMaxBlocksPerGroup = GRIM::HyperParameters::CUDA_REDUCTION_MAX_BLOCKS;
constexpr int kExpectedParamGroupTypes = 6;
constexpr float EPSILON_SAFE_DIV = 1e-6f;

constexpr bool isPowerOfTwo(int value) {
	return value > 0 && (value & (value - 1)) == 0;
}

static_assert(static_cast<int>(GRIM::ParamGroupType::COUNT) == kExpectedParamGroupTypes,
	"GradClip ParamGroupType cardinality must match accumulateGroupMetrics");
static_assert(kBlockSize >= 64, "GradClip reduction requires kBlockSize >= 64");
static_assert(isPowerOfTwo(kBlockSize), "GradClip reduction requires power-of-two kBlockSize");

__global__ void sumSquaresBlockKernel(
	const float* __restrict__ gradients,
	size_t size,
	float* __restrict__ partial_sum)
{
	__shared__ float shared[kBlockSize];

	const size_t thread_index = threadIdx.x;
	const size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t stride = blockDim.x * gridDim.x;

	float local_sum = 0.0f;
	for (size_t index = global_index; index < size; index += stride) {
		const float value = gradients[index];
		local_sum += value * value;
	}

	shared[thread_index] = local_sum;
	__syncthreads();

	for (int offset = blockDim.x / 2; offset > 32; offset >>= 1) {
		if (thread_index < offset) {
			shared[thread_index] += shared[thread_index + offset];
		}
		__syncthreads();
	}

	if (thread_index < 32) {
		local_sum = shared[thread_index] + shared[thread_index + 32];
		for (int offset = 16; offset > 0; offset >>= 1) {
			local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
		}
	}

	if (thread_index == 0) {
		atomicAdd(partial_sum, local_sum);
	}
}

__global__ void scaleGradientsKernel(
	float* __restrict__ gradients,
	float scale_factor,
	size_t size)
{
	const size_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t stride = blockDim.x * gridDim.x;
	for (size_t index = global_index; index < size; index += stride) {
		gradients[index] *= scale_factor;
	}
}

inline dim3 computeGrid(size_t size)
{
	const size_t blocks = (size + kBlockSize - 1) / kBlockSize;
	return dim3(static_cast<unsigned int>(
		std::min(blocks, static_cast<size_t>(kMaxBlocksPerGroup))));
}

void ensureClipScratch(
	std::unique_ptr<GRIM::GradClip::ClipScratch>& scratch,
	size_t required_groups)
{
	if (required_groups == 0) {
		throw std::runtime_error("[GradClip] cannot allocate ClipScratch for zero parameter groups");
	}

	if (!scratch) {
		auto allocated = std::make_unique<GRIM::GradClip::ClipScratch>();
		allocated->max_groups = required_groups;

		cudaError_t err = cudaMalloc(&allocated->d_partial_sums, required_groups * sizeof(float));
		if (err != cudaSuccess) {
			throw std::runtime_error("[GradClip] cudaMalloc d_partial_sums failed: " +
				std::string(cudaGetErrorString(err)));
		}
		err = cudaMallocHost(&allocated->h_partial_sums, required_groups * sizeof(float));
		if (err != cudaSuccess) {
			throw std::runtime_error("[GradClip] cudaMallocHost h_partial_sums failed: " +
				std::string(cudaGetErrorString(err)));
		}
		err = cudaMallocHost(&allocated->h_metrics, sizeof(GRIM::GradClip::ClipMetrics));
		if (err != cudaSuccess) {
			throw std::runtime_error("[GradClip] cudaMallocHost h_metrics failed: " +
				std::string(cudaGetErrorString(err)));
		}
		*allocated->h_metrics = GRIM::GradClip::ClipMetrics{};
		scratch = std::move(allocated);
	}

	if (!scratch->d_partial_sums || !scratch->h_partial_sums || !scratch->h_metrics) {
		throw std::runtime_error("[GradClip] ClipScratch buffer set is incomplete");
	}
	if (scratch->max_groups < required_groups) {
		throw std::runtime_error("[GradClip] ClipScratch capacity mismatch required_groups=" +
								 std::to_string(required_groups) +
								 " scratch_max_groups=" + std::to_string(scratch->max_groups));
	}
}

void accumulateGroupMetrics(
	GRIM::GradClip::ClipMetrics& metrics,
	GRIM::ParamGroupType type,
	double sum_sq,
	uint64_t count)
{
	switch (type) {
		case GRIM::ParamGroupType::EMBEDDING:
			metrics.embedding_sum_sq += sum_sq;
			metrics.embedding_count += count;
			return;
		case GRIM::ParamGroupType::LM_HEAD:
			metrics.lm_head_sum_sq += sum_sq;
			metrics.lm_head_count += count;
			return;
		case GRIM::ParamGroupType::ATTENTION:
			metrics.attention_sum_sq += sum_sq;
			metrics.attention_count += count;
			return;
		case GRIM::ParamGroupType::FFN:
			metrics.ffn_sum_sq += sum_sq;
			metrics.ffn_count += count;
			return;
		case GRIM::ParamGroupType::RMSNORM:
			metrics.rmsnorm_sum_sq += sum_sq;
			metrics.rmsnorm_count += count;
			return;
		case GRIM::ParamGroupType::ARG_SELECTOR:
			metrics.arg_selector_sum_sq += sum_sq;
			metrics.arg_selector_count += count;
			return;
		case GRIM::ParamGroupType::COUNT:
			break;
	}
	throw std::runtime_error("[GradClip] invalid ParamGroupType=" +
		std::to_string(static_cast<int>(type)));
}

void measureGradientNorms(
	const GRIM::ParameterGroup* groups,
	size_t num_groups,
	GRIM::GradClip::ClipScratch& scratch,
	cudaStream_t stream)
{
	cudaError_t err = cudaMemsetAsync(
		scratch.d_partial_sums, 0, num_groups * sizeof(float), stream);
	if (err != cudaSuccess) {
		throw std::runtime_error("[GradClip] cudaMemsetAsync d_partial_sums failed: " +
			std::string(cudaGetErrorString(err)));
	}

	for (size_t group_index = 0; group_index < num_groups; ++group_index) {
		float* gradients = groups[group_index].grads();
		const size_t size = groups[group_index].size();
		if (!gradients || size == 0) continue;

		int blocks = static_cast<int>((size + kBlockSize - 1) / kBlockSize);
		blocks = std::min(blocks, kMaxBlocksPerGroup);
		sumSquaresBlockKernel<<<blocks, kBlockSize, 0, stream>>>(
			gradients, size, &scratch.d_partial_sums[group_index]);
	}

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		throw std::runtime_error("[GradClip] sumSquaresBlockKernel launch failed: " +
			std::string(cudaGetErrorString(err)));
	}
	err = cudaMemcpyAsync(
		scratch.h_partial_sums,
		scratch.d_partial_sums,
		num_groups * sizeof(float),
		cudaMemcpyDeviceToHost,
		stream);
	if (err != cudaSuccess) {
		throw std::runtime_error("[GradClip] gradient norm D2H copy failed: " +
			std::string(cudaGetErrorString(err)));
	}
	err = cudaStreamSynchronize(stream);
	if (err != cudaSuccess) {
		throw std::runtime_error("[GradClip] gradient norm stream sync failed: " +
			std::string(cudaGetErrorString(err)));
	}

	GRIM::GradClip::ClipMetrics& metrics = *scratch.h_metrics;
	metrics = GRIM::GradClip::ClipMetrics{};
	for (size_t group_index = 0; group_index < num_groups; ++group_index) {
		if (!groups[group_index].grads() || groups[group_index].size() == 0) continue;

		const float sum_sq = scratch.h_partial_sums[group_index];
		if (std::isnan(sum_sq)) {
			if (!metrics.has_nan) {
				metrics.has_nan = 1;
				metrics.first_nan_group = static_cast<int32_t>(group_index);
				metrics.first_nan_value = sum_sq;
			}
			continue;
		}
		if (std::isinf(sum_sq)) {
			if (!metrics.has_inf) {
				metrics.has_inf = 1;
				metrics.first_inf_group = static_cast<int32_t>(group_index);
				metrics.first_inf_value = sum_sq;
			}
			continue;
		}

		accumulateGroupMetrics(
			metrics,
			groups[group_index].type,
			static_cast<double>(sum_sq),
			static_cast<uint64_t>(groups[group_index].size()));
	}
	metrics.groups_processed = static_cast<uint32_t>(num_groups);
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

float encoderTelemetryRms(const GRIM::GradClip::ClipMetrics& gm) {
	const double sum_sq = gm.attention_sum_sq + gm.ffn_sum_sq + gm.rmsnorm_sum_sq;
	const uint64_t count = gm.attention_count + gm.ffn_count +
		gm.rmsnorm_count;
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
	size_t size,
	float scale_factor,
	cudaStream_t stream)
{
	if (!gradients || size == 0) {
		throw std::runtime_error("[GradClip] launchScaleGradients received invalid input");
	}

	scaleGradientsKernel<<<computeGrid(size), kBlockSize, 0, stream>>>(
		gradients, scale_factor, size);

	const cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		throw std::runtime_error("[GradClip] scaleGradientsKernel launch failed: " +
			std::string(cudaGetErrorString(err)));
	}
}

} // namespace

namespace GRIM::GradClip {

ClipScratch::~ClipScratch() {
	if (d_partial_sums) { cudaFree(d_partial_sums); d_partial_sums = nullptr; }
	if (h_partial_sums) { cudaFreeHost(h_partial_sums); h_partial_sums = nullptr; }
	if (h_metrics) { cudaFreeHost(h_metrics); h_metrics = nullptr; }
	max_groups = 0;
}

ClipResult clipGradientNorms(
    ParameterGroup* groups,
    size_t num_groups,
	std::unique_ptr<ClipScratch>& scratch,
    const HyperParameters::GradientClippingHP& clipping_hp,
    const HyperParameters::TrainingScheduleHP& schedule_hp,
    cudaStream_t stream
) {
	StreamController::fatalIfDefaultStream(stream, "clipGradientNorms");
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
								 groups[i].size(),
                                 accumulation_scale, stream);
        }
    }

	ensureClipScratch(scratch, num_groups);

    // Step 2: Measure gradient norms through the tensor registry
	measureGradientNorms(groups, num_groups, *scratch, stream);
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
								 groups[i].size(),
                                 coef, stream);
        }
        result.clipped = true;
    }

    // Step 5: True post-clip RMS (exact, not clamped to threshold).
    result.global_rms_post = global_rms * coef;

    return result;
}

} // namespace GRIM::GradClip
