#define USE_CUDA

#include "NumericLoss_GPU.hpp"
#include "../../HyperParameters/HyperParameters_GPU.hpp"

#include <cmath>
#include <stdexcept>

namespace GRIM {

namespace {

constexpr int kBlockSize = HyperParameters::CUDA_BLOCK_SIZE_STANDARD;

__device__ __forceinline__ float clampLogValue(float value, float max_abs) {
	if (!isfinite(value)) {
		return 0.0f;
	}
	if (value > max_abs) {
		return max_abs;
	}
	if (value < -max_abs) {
		return -max_abs;
	}
	return value;
}

__device__ __forceinline__ float logScaleValue(float value, float max_abs) {
	if (!isfinite(value)) {
		return 0.0f;
	}
	const float sign = (value < 0.0f) ? -1.0f : 1.0f;
	const float scaled = log1pf(fabsf(value));
	return clampLogValue(sign * scaled, max_abs);
}

__global__ void numericLossKernel(const float* __restrict__ predictions,
                                  const float* __restrict__ numeric_values,
                                  const uint8_t* __restrict__ numeric_mask,
                                  const int* __restrict__ targets,
                                  float* __restrict__ loss_sum,
                                  int* __restrict__ count,
                                  float* __restrict__ grad_predictions,
                                  int total_tokens,
                                  int seq_len,
                                  float huber_delta,
                                  bool log_scale,
                                  float log_max,
                                  float loss_weight) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= total_tokens) {
		return;
	}

	float grad = 0.0f;
	float loss = 0.0f;
	bool valid = false;

	const int pos = idx % seq_len;
	if (pos + 1 < seq_len && targets[idx] >= 0) {
		const int target_idx = idx + 1;
		if (numeric_mask[target_idx]) {
			float target_val = numeric_values[target_idx];
			if (isfinite(target_val)) {
				valid = true;
				if (log_scale) {
					target_val = logScaleValue(target_val, log_max);
				}
				const float pred = predictions[idx];
				const float diff = pred - target_val;
				const float abs_diff = fabsf(diff);
				if (abs_diff <= huber_delta) {
					loss = 0.5f * diff * diff;
					grad = diff;
				} else {
					loss = huber_delta * (abs_diff - 0.5f * huber_delta);
					grad = huber_delta * (diff < 0.0f ? -1.0f : 1.0f);
				}
			}
		}
	}

	grad_predictions[idx] = grad * loss_weight;
	if (valid) {
		atomicAdd(loss_sum, loss);
		atomicAdd(count, 1);
	}
}

// ============================================================================
// Issue #74 FIX: Scale numeric gradients by 1/count for proper mean reduction
// ============================================================================
// The numericLossKernel computes gradients as: grad_predictions[i] = grad * loss_weight
// But the loss is averaged: loss_avg = loss_sum / count
// By chain rule: d(loss_avg)/d(pred) = (1/count) * d(loss_sum)/d(pred)
// ============================================================================
__global__ void scaleNumericGradKernel(
	float* __restrict__ grad_predictions,
	int total_tokens,
	int valid_count,
	int valid_text_tokens  // UNUSED - kept for API compatibility
) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= total_tokens) return;
	
	if (valid_count <= 0) return;
	
	// Two-level scaling to match text CE gradient scale:
	// 1. Mean reduction over valid atoms: 1/valid_count
	// 2. Scale to text token count: valid_text_tokens measures the overall batch scale
	//
	// Both text and numeric losses update the same encoder. Without this scaling,
	// numeric gradients dominate because they're only divided by atom count (~400),
	// while text CE is divided by token count (~6000+).
	//
	// Combined scale makes them comparable: grad *= 1 / (valid_count * valid_text_tokens)^0.5
	const float atom_reduction = 1.0f / static_cast<float>(valid_count);
	const float token_scale_factor = 1.0f / sqrtf(static_cast<float>(valid_text_tokens));
	const float scale = atom_reduction * token_scale_factor;
	
	grad_predictions[idx] *= scale;
}

} // namespace

bool launchNumericLoss(const NumericLossInputs& inputs,
                       const NumericLossOutputs& outputs,
                       cudaStream_t stream) {
	if (!inputs.predictions || !inputs.token_numeric_values ||
	    !inputs.token_numeric_mask || !inputs.targets) {
		return false;
	}
	if (!outputs.loss_sum || !outputs.count || !outputs.grad_predictions) {
		return false;
	}
	if (inputs.total_tokens <= 0 || inputs.seq_len <= 0) {
		return false;
	}
	if (!stream) {
		return false;
	}
	
	// RULE 20: FAIL LOUD - valid_text_tokens MUST be provided by caller
	// This is NOT optional. If caller doesn't know valid_text_tokens, that's a BUG in the caller.
	if (inputs.valid_text_tokens <= 0) {
		throw std::runtime_error("launchNumericLoss: valid_text_tokens is " + 
		                         std::to_string(inputs.valid_text_tokens) + 
		                         " but MUST be > 0. Caller MUST provide valid text token count " +
		                         "for proper gradient scaling (Issue #136).");
	}

	cudaMemsetAsync(outputs.loss_sum, 0, sizeof(float), stream);
	cudaMemsetAsync(outputs.count, 0, sizeof(int), stream);

	const int blocks = (inputs.total_tokens + kBlockSize - 1) / kBlockSize;
	numericLossKernel<<<blocks, kBlockSize, 0, stream>>>(
		inputs.predictions,
		inputs.token_numeric_values,
		inputs.token_numeric_mask,
		inputs.targets,
		outputs.loss_sum,
		outputs.count,
		outputs.grad_predictions,
		inputs.total_tokens,
		inputs.seq_len,
		inputs.huber_delta,
		inputs.log_scale,
		inputs.log_max,
		inputs.loss_weight);

	// Issue #74b: Must sync before reading count — atomicAdd in kernel 1 must complete
	// before kernel 2 reads it. Both on same stream so kernel ordering is guaranteed,
	// but we need the count on HOST to pass as a kernel parameter.
	cudaStreamSynchronize(stream);
	
	int h_count = 0;
	cudaMemcpy(&h_count, outputs.count, sizeof(int), cudaMemcpyDeviceToHost);
	
	if (h_count <= 0) {
		// No valid numeric tokens in this batch — gradients are already all 0, nothing to scale
		return cudaGetLastError() == cudaSuccess;
	}
	
	// Scale by 1/h_count — each loss divides by its OWN valid count (not total_tokens)
	// ISSUE #136: Also apply compensation for token count mismatch
	scaleNumericGradKernel<<<blocks, kBlockSize, 0, stream>>>(
		outputs.grad_predictions,
		inputs.total_tokens,
		h_count,
		inputs.valid_text_tokens);

	return cudaGetLastError() == cudaSuccess;
}

} // namespace GRIM
