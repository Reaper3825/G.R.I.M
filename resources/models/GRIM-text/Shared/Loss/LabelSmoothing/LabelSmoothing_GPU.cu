#include "Shared/Loss/LabelSmoothing/LabelSmoothing_GPU.hpp"
#include "../../HyperParameters/HyperParameters_GPU.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <cuda_runtime.h>

namespace GRIM::Loss {
namespace {

constexpr int kBlockSize = HyperParameters::CUDA_BLOCK_SIZE_STANDARD;

__device__ inline float safeEpsilon(float value)
{
	return fminf(fmaxf(value, 0.0f), 0.5f);
}

__global__ void labelSmoothingKernel(const float* __restrict__ logits,
									 const int* __restrict__ targets,
									 float* __restrict__ token_losses,
									 float* __restrict__ grad_logits,
									 int total_tokens,
									 int vocab_size,
									 int seq_len,
									 float epsilon,
									 const float* __restrict__ sequence_weights,
									 int weight_count,
									 float* __restrict__ delta_loss)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= total_tokens) {
		return;
	}

	const int target = targets ? targets[idx] : -1;
	if (target < 0 || target >= vocab_size) {
		return;
	}

	const int offset = idx * vocab_size;
	const int seq_index = (seq_len > 0) ? (idx / seq_len) : 0;
	float sample_weight = 1.0f;
	if (sequence_weights && seq_index < weight_count) {
		sample_weight = sequence_weights[seq_index];
	}

	float max_logit = logits[offset];
	for (int i = 1; i < vocab_size; ++i) {
		max_logit = fmaxf(max_logit, logits[offset + i]);
	}

	float sum_exp = 0.0f;
	for (int i = 0; i < vocab_size; ++i) {
		sum_exp += expf(logits[offset + i] - max_logit);
	}
	const float log_sum_exp = logf(sum_exp) + max_logit;
	const float inv_sum_exp = 1.0f / fmaxf(sum_exp, 1e-6f);

	float sum_log_probs = 0.0f;
	float target_log_prob = 0.0f;
	for (int i = 0; i < vocab_size; ++i) {
		const float log_prob = logits[offset + i] - log_sum_exp;
		sum_log_probs += log_prob;
		if (i == target) {
			target_log_prob = log_prob;
		}
	}

	const float normalized_epsilon = safeEpsilon(epsilon);
	float on_target = 1.0f;
	float off_target = 0.0f;
	if (vocab_size > 1) {
		on_target = 1.0f - normalized_epsilon;
		off_target = normalized_epsilon / static_cast<float>(vocab_size - 1);
	}

	if (grad_logits) {
		const float normalization = sample_weight / static_cast<float>(total_tokens);
		for (int i = 0; i < vocab_size; ++i) {
			const float softmax_i = expf(logits[offset + i] - max_logit) * inv_sum_exp;
			const float smooth_target = (i == target) ? on_target : off_target;
			grad_logits[offset + i] = (softmax_i - smooth_target) * normalization;
		}
	}

	if (token_losses) {
		float ce_smoothed = -on_target * target_log_prob;
		if (vocab_size > 1) {
			ce_smoothed -= off_target * (sum_log_probs - target_log_prob);
		}
		const float previous = token_losses[idx];
		token_losses[idx] = ce_smoothed;
		if (delta_loss) {
			atomicAdd(delta_loss, (ce_smoothed - previous) * sample_weight);
		}
	}
}

inline dim3 launchGrid(int total_threads)
{
	return dim3((total_threads + kBlockSize - 1) / kBlockSize);
}

} // namespace

void applyLabelSmoothing(const LossContext& ctx,
						 const LabelSmoothingConfig& cfg,
						 DeviceBuffers buffers,
						 LossBreakdown& out_loss)
{
	if (!cfg.enabled) {
		return;
	}
	if (!ctx.logits || !ctx.targets) {
		return;
	}

	const int total_tokens = ctx.batch_size * ctx.seq_len;
	if (total_tokens <= 0 || ctx.vocab_size <= 1) {
		return;
	}

	float* delta_buffer = nullptr;
	bool owns_buffer = false;
	if (buffers.scratch) {
		delta_buffer = buffers.scratch;
	} else if (cudaMalloc(&delta_buffer, sizeof(float)) == cudaSuccess) {
		owns_buffer = true;
	}

	if (delta_buffer) {
		cudaMemsetAsync(delta_buffer, 0, sizeof(float), ctx.stream);
	}

	labelSmoothingKernel<<<launchGrid(total_tokens), kBlockSize, 0, ctx.stream>>>(
		ctx.logits,
		ctx.targets,
		buffers.token_losses,
		buffers.grad_logits,
		total_tokens,
		ctx.vocab_size,
		ctx.seq_len,
		cfg.epsilon,
		ctx.sequence_weights,
		ctx.sequence_weight_count > 0 ? ctx.sequence_weight_count : ctx.batch_size,
		delta_buffer);

	const cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "LabelSmoothing: kernel launch failed - %s\n", cudaGetErrorString(err));
	}

	if (delta_buffer) {
		float delta = 0.0f;
		cudaMemcpyAsync(&delta, delta_buffer, sizeof(float), cudaMemcpyDeviceToHost, ctx.stream);
		cudaStreamSynchronize(ctx.stream);
		out_loss.label_smoothing += delta;
	}

	if (owns_buffer && delta_buffer) {
		cudaFree(delta_buffer);
	}
}

} // namespace GRIM::Loss

