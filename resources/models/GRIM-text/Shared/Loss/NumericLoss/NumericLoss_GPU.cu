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
// Issue #74 FIX: Scale numeric gradients by 1/count to match mean reduction
// ============================================================================
// The numericLossKernel computes gradients as: grad_predictions[i] = grad * loss_weight
// But the loss is averaged: loss_avg = loss_sum / count
// Issue #85 FIX: Scale gradients by 1/total_tokens to match text loss normalization
// ============================================================================
// Previously (Issue #74), we scaled by 1/numeric_count, but this created a magnitude
// imbalance: text loss divides by ~7000 tokens, numeric loss divided by ~400 atoms.
// Result: numeric gradients were ~18x larger than text gradients!
//
// The correct fix is to scale by 1/total_tokens so both losses have the same 
// gradient magnitude normalization. When loss_weight=1.0, this gives equal influence.
// ============================================================================
__global__ void scaleNumericGradKernel(
	float* __restrict__ grad_predictions,
	int total_tokens
) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= total_tokens) return;
	
	// Scale by 1/total_tokens to match text loss normalization
	grad_predictions[idx] *= (1.0f / static_cast<float>(total_tokens));
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

	// Issue #85 FIX: Scale gradients by 1/total_tokens to match text loss normalization
	// Previously we scaled by 1/numeric_count (Issue #74), but this created imbalance:
	// text loss divides by ~7000 tokens, numeric divided by ~400 atoms → 18x larger!
	// Now we scale by 1/total_tokens for matching normalization.
	// 
	// NOTE: We no longer need to sync/read count since we use total_tokens directly.
	scaleNumericGradKernel<<<blocks, kBlockSize, 0, stream>>>(
		outputs.grad_predictions,
		inputs.total_tokens);

	return cudaGetLastError() == cudaSuccess;
}

} // namespace GRIM
