#include "Shared/Loss/Preference/Preference_KL_GPU.hpp"
#include "../../HyperParameters/HyperParameters_GPU.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

namespace GRIM::Loss {
namespace {

constexpr int kBlockSize = HyperParameters::CUDA_BLOCK_SIZE_STANDARD;
constexpr float kEpsilon = HyperParameters::EPSILON_SAFE_DIV;

inline __device__ float safeLog(float value)
{
	return logf(fmaxf(value, kEpsilon));
}

inline __device__ float sampleWeightForToken(int token_index,
						 int seq_len,
						 const float* sequence_weights,
						 int weight_count)
{
	if (!sequence_weights || seq_len <= 0) {
		return 1.0f;
	}
	const int seq_index = token_index / seq_len;
	if (seq_index >= weight_count) {
		return 1.0f;
	}
	return sequence_weights[seq_index];
}

__global__ void preferenceKernel(const float* __restrict__ student_logits,
									 const float* __restrict__ reference_logits,
									 float* __restrict__ grad_logits,
									 int total_tokens,
									 int vocab_size,
									 int seq_len,
									 const float* __restrict__ sequence_weights,
									 int weight_count,
									 float scale,
									 float* __restrict__ loss_accum)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= total_tokens) {
		return;
	}

	const int offset = idx * vocab_size;
	const float sample_weight = sampleWeightForToken(idx, seq_len, sequence_weights, weight_count);
	const float total_tokens_f = static_cast<float>(total_tokens);
	const float norm = sample_weight / fmaxf(total_tokens_f, 1.0f);

	float student_max = student_logits[offset];
	float reference_max = reference_logits[offset];
	for (int i = 1; i < vocab_size; ++i) {
		student_max = fmaxf(student_max, student_logits[offset + i]);
		reference_max = fmaxf(reference_max, reference_logits[offset + i]);
	}

	float student_sum = 0.0f;
	float reference_sum = 0.0f;
	for (int i = 0; i < vocab_size; ++i) {
		student_sum += expf(student_logits[offset + i] - student_max);
		reference_sum += expf(reference_logits[offset + i] - reference_max);
	}
	const float inv_student_sum = 1.0f / fmaxf(student_sum, kEpsilon);
	const float inv_reference_sum = 1.0f / fmaxf(reference_sum, kEpsilon);

	float loss_token = 0.0f;
	float mean_g = 0.0f;
	for (int i = 0; i < vocab_size; ++i) {
		const float s_prob = expf(student_logits[offset + i] - student_max) * inv_student_sum;
		const float r_prob = expf(reference_logits[offset + i] - reference_max) * inv_reference_sum;
		const float log_ratio = safeLog(s_prob) - safeLog(r_prob);
		const float g = log_ratio + 1.0f;
		loss_token += s_prob * log_ratio;
		mean_g += s_prob * g;
	}

	if (grad_logits) {
		for (int i = 0; i < vocab_size; ++i) {
			const float s_prob = expf(student_logits[offset + i] - student_max) * inv_student_sum;
			const float r_prob = expf(reference_logits[offset + i] - reference_max) * inv_reference_sum;
			const float log_ratio = safeLog(s_prob) - safeLog(r_prob);
			const float g = log_ratio + 1.0f;
			const float grad_delta = s_prob * (g - mean_g);
			grad_logits[offset + i] += scale * norm * grad_delta;
		}
	}

	if (loss_accum) {
		const float scaled_loss = scale * sample_weight * loss_token;
		atomicAdd(loss_accum, scaled_loss);
	}
}

inline dim3 launchGrid(int total_threads)
{
	return dim3((total_threads + kBlockSize - 1) / kBlockSize);
}

inline float* acquireDeviceAccumulator(DeviceBuffers buffers, bool& owns_device)
{
	owns_device = false;
	if (buffers.scratch) {
		return buffers.scratch;
	}
	float* dev = nullptr;
	if (cudaMalloc(&dev, sizeof(float)) == cudaSuccess) {
		owns_device = true;
	}
	return dev;
}

inline void releaseAccumulator(float* buffer, bool owns_buffer)
{
	if (owns_buffer && buffer) {
		cudaFree(buffer);
	}
}

} // namespace

void accumulatePreferenceKL(const LossContext& ctx,
							  const PreferenceKLConfig& cfg,
							  DeviceBuffers buffers,
							  LossBreakdown& out_loss)
{
	if (!cfg.enabled || cfg.beta <= 0.0f) {
		return;
	}
	if (!ctx.logits || !ctx.reference_logits || !buffers.grad_logits) {
		return;
	}

	const int total_tokens = ctx.batch_size * ctx.seq_len;
	if (total_tokens <= 0 || ctx.vocab_size <= 1) {
		return;
	}

	const int weight_count = ctx.sequence_weight_count > 0
				 ? ctx.sequence_weight_count
				 : ctx.batch_size;

	bool owns_buffer = false;
	float* loss_accum = acquireDeviceAccumulator(buffers, owns_buffer);
	if (loss_accum) {
		cudaMemsetAsync(loss_accum, 0, sizeof(float), ctx.stream);
	}

	preferenceKernel<<<launchGrid(total_tokens), kBlockSize, 0, ctx.stream>>>(
		ctx.logits,
		ctx.reference_logits,
		buffers.grad_logits,
		total_tokens,
		ctx.vocab_size,
		ctx.seq_len,
		ctx.sequence_weights,
		weight_count,
		cfg.beta,
		loss_accum);

	const cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "Preference: kernel failed - %s\n",
				cudaGetErrorString(err));
	}

	if (loss_accum) {
		float host_loss = 0.0f;
		cudaMemcpyAsync(&host_loss, loss_accum, sizeof(float), cudaMemcpyDeviceToHost, ctx.stream);
		cudaStreamSynchronize(ctx.stream);
		out_loss.preference_kl += host_loss;
	}

	releaseAccumulator(loss_accum, owns_buffer);
}

} // namespace GRIM::Loss
