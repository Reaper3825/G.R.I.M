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
	int valid_text_tokens
) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= total_tokens) return;
	
	if (valid_text_tokens <= 0) return;
	
	// Issue #137 FIX: Scale by 1/valid_text_tokens to match text CE mean reduction.
	// Text CE: grad = (softmax - target) / valid_tokens
	// Numeric: grad = huber_grad * loss_weight / valid_text_tokens
	// Both paths feed into the same encoder, so using the same denominator
	// ensures comparable gradient magnitudes without heuristic sqrt factors.
	const float scale = 1.0f / static_cast<float>(valid_text_tokens);
	
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
	
	if (inputs.valid_text_tokens <= 0) {
		throw std::runtime_error("launchNumericLoss: valid_text_tokens is " +
		                         std::to_string(inputs.valid_text_tokens) +
		                         " but MUST be > 0. Caller MUST provide total valid token count.");
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

	// Scale by 1/valid_text_tokens to match text CE mean reduction denominator.
	// Both text and numeric gradients flow into the same encoder — using the same
	// denominator ensures comparable magnitude without heuristic scaling.
	// No sync needed — valid_text_tokens is host-side, and kernel ordering on
	// the same stream guarantees numericLossKernel completes before this runs.
	scaleNumericGradKernel<<<blocks, kBlockSize, 0, stream>>>(
		outputs.grad_predictions,
		inputs.total_tokens,
		inputs.valid_text_tokens);

	return cudaGetLastError() == cudaSuccess;
}

} // namespace GRIM
