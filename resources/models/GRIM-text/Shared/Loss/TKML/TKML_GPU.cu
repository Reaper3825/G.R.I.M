#include "Shared/Loss/TKML/TKML_GPU.hpp"
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

inline __device__ void computeSoftmax(const float* logits,
									  int vocab_size,
									  float temperature,
									  float& max_value,
									  float& sum_exp)
{
	const float inv_temp = 1.0f / temperature;
	max_value = logits[0] * inv_temp;
	for (int i = 1; i < vocab_size; ++i) {
		max_value = fmaxf(max_value, logits[i] * inv_temp);
	}

	sum_exp = 0.0f;
	for (int i = 0; i < vocab_size; ++i) {
		sum_exp += expf(logits[i] * inv_temp - max_value);
	}
}

__global__ void distillationKernel(const float* __restrict__ student_logits,
								   const float* __restrict__ teacher_logits,
								   float* __restrict__ grad_logits,
								   int total_tokens,
								   int vocab_size,
								   int seq_len,
								   const float* __restrict__ sequence_weights,
								   int weight_count,
								   float temperature,
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
	const float temp_sq = temperature * temperature;
	const float grad_scale = scale * temp_sq * norm;

	float student_max = 0.0f;
	float student_sum = 0.0f;
	computeSoftmax(student_logits + offset, vocab_size, temperature, student_max, student_sum);
	const float inv_student_sum = 1.0f / fmaxf(student_sum, kEpsilon);

	float teacher_max = 0.0f;
	float teacher_sum = 0.0f;
	computeSoftmax(teacher_logits + offset, vocab_size, temperature, teacher_max, teacher_sum);
	const float inv_teacher_sum = 1.0f / fmaxf(teacher_sum, kEpsilon);

	float loss_token = 0.0f;
	for (int i = 0; i < vocab_size; ++i) {
		const float t_prob = expf(teacher_logits[offset + i] / temperature - teacher_max) * inv_teacher_sum;
		const float s_prob = expf(student_logits[offset + i] / temperature - student_max) * inv_student_sum;

		loss_token += t_prob * (safeLog(t_prob) - safeLog(s_prob));

		if (grad_logits) {
			const float grad_delta = s_prob - t_prob;
			grad_logits[offset + i] += grad_scale * grad_delta;
		}
	}

	if (loss_accum) {
		const float scaled_loss = scale * temp_sq * sample_weight * loss_token;
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

void accumulateDistillationKL(const LossContext& ctx,
							  const DistillationConfig& cfg,
							  DeviceBuffers buffers,
							  LossBreakdown& out_loss)
{
	if (!cfg.enabled || cfg.lambda <= 0.0f) {
		return;
	}
	if (!ctx.logits || !ctx.teacher_logits || !buffers.grad_logits) {
		return;
	}

	const int total_tokens = ctx.batch_size * ctx.seq_len;
	if (total_tokens <= 0 || ctx.vocab_size <= 1) {
		return;
	}

	const float temperature = fmax(cfg.temperature, 1e-3f);
	const int weight_count = ctx.sequence_weight_count > 0
								 ? ctx.sequence_weight_count
								 : ctx.batch_size;

	bool owns_buffer = false;
	float* loss_accum = acquireDeviceAccumulator(buffers, owns_buffer);
	if (loss_accum) {
		cudaMemsetAsync(loss_accum, 0, sizeof(float), ctx.stream);
	}

	distillationKernel<<<launchGrid(total_tokens), kBlockSize, 0, ctx.stream>>>(
		ctx.logits,
		ctx.teacher_logits,
		buffers.grad_logits,
		total_tokens,
		ctx.vocab_size,
		ctx.seq_len,
		ctx.sequence_weights,
		weight_count,
		temperature,
		cfg.lambda,
		loss_accum);

	const cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "TKML: distillation kernel failed - %s\n",
				cudaGetErrorString(err));
	}

	if (loss_accum) {
		float host_loss = 0.0f;
		cudaMemcpyAsync(&host_loss, loss_accum, sizeof(float), cudaMemcpyDeviceToHost, ctx.stream);
		cudaStreamSynchronize(ctx.stream);
		out_loss.distillation_kl += host_loss;
	}

	releaseAccumulator(loss_accum, owns_buffer);
}

} // namespace GRIM::Loss

