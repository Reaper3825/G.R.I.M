#include "Shared/Loss/CrossEntropy/CrossEntropy_GPU.hpp"
#include "../../HyperParameters/HyperParameters_GPU.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace GRIM::Loss {
namespace {

constexpr int kBlockSize = HyperParameters::CUDA_BLOCK_SIZE_STANDARD;

__global__ void crossEntropyLossKernel(const float* __restrict__ logits,
									   const int* __restrict__ targets,
									   float* __restrict__ losses,
									   int total_tokens,
									   int vocab_size)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= total_tokens) {
		return;
	}

	const int target_class = targets[idx];
	if (target_class < 0 || target_class >= vocab_size) {
		losses[idx] = 0.0f;
		return;
	}

	const int offset = idx * vocab_size;
	float max_logit = logits[offset];
	for (int i = 1; i < vocab_size; ++i) {
		max_logit = fmaxf(max_logit, logits[offset + i]);
	}

	float sum_exp = 0.0f;
	for (int i = 0; i < vocab_size; ++i) {
		sum_exp += expf(logits[offset + i] - max_logit);
	}

	const float log_sum_exp = logf(sum_exp) + max_logit;
	const float target_logit = logits[offset + target_class];
	losses[idx] = log_sum_exp - target_logit;
}

__global__ void reduceLossKernel(const float* __restrict__ losses,
								 float* __restrict__ output,
								 int n)
{
	__shared__ float shared[kBlockSize];

	const int tid = threadIdx.x;
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	shared[tid] = (idx < n) ? losses[idx] : 0.0f;
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (tid < stride) {
			shared[tid] += shared[tid + stride];
		}
		__syncthreads();
	}

	if (tid == 0) {
		atomicAdd(output, shared[0]);
	}
}

__global__ void crossEntropyGradientKernel(const float* __restrict__ logits,
										   const int* __restrict__ targets,
										   float* __restrict__ grad_logits,
										   int total_tokens,
										   int vocab_size,
										   int valid_tokens,
										   int seq_len,
										   const float* __restrict__ sequence_weights,
										   int weight_count)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= total_tokens) {
		return;
	}

	float sample_weight = 1.0f;
	if (sequence_weights && seq_len > 0) {
		const int seq_index = idx / seq_len;
		if (seq_index < weight_count) {
			sample_weight = sequence_weights[seq_index];
		}
	}

	const int target_class = targets[idx];
	const int offset = idx * vocab_size;

	if (target_class < 0 || target_class >= vocab_size) {
		for (int i = 0; i < vocab_size; ++i) {
			grad_logits[offset + i] = 0.0f;
		}
		return;
	}

	float max_logit = logits[offset];
	for (int i = 1; i < vocab_size; ++i) {
		max_logit = fmaxf(max_logit, logits[offset + i]);
	}

	float sum_exp = 0.0f;
	for (int i = 0; i < vocab_size; ++i) {
		sum_exp += expf(logits[offset + i] - max_logit);
	}

	// NOTE: Gradients computed here are NOT normalized by valid_tokens.
	// The caller (backward pass) applies grad_scale = 1/valid_tokens to match
	// the mean-reduced loss: L_avg = sum(loss_i) / valid_tokens
	// This ensures d(L_avg)/d(logit) = (1/N) * d(L_sum)/d(logit)
	// Standard cross-entropy gradient is: softmax - one_hot
	for (int i = 0; i < vocab_size; ++i) {
		const float softmax_i = expf(logits[offset + i] - max_logit) / sum_exp;
		const float one_hot = (i == target_class) ? 1.0f : 0.0f;
		grad_logits[offset + i] = (softmax_i - one_hot) * sample_weight;
	}
}

inline dim3 launchGrid(int total_threads)
{
	return dim3((total_threads + kBlockSize - 1) / kBlockSize);
}

} // namespace

void validate(const LossContext& ctx)
{
	if (!ctx.logits) {
		throw std::runtime_error("[CrossEntropy] ctx.logits is NULL");
	}
	if (!ctx.targets) {
		throw std::runtime_error("[CrossEntropy] ctx.targets is NULL");
	}
	if (ctx.batch_size <= 0 || ctx.seq_len <= 0 || ctx.vocab_size <= 0) {
		throw std::runtime_error("[CrossEntropy] invalid dimensions: batch=" +
			std::to_string(ctx.batch_size) + " seq=" + std::to_string(ctx.seq_len) +
			" vocab=" + std::to_string(ctx.vocab_size));
	}
}

void computeCrossEntropyLoss(const LossContext& ctx,
							 DeviceBuffers buffers)
{
	validate(ctx);  // Throws on failure
	
	if (!buffers.token_losses) {
		throw std::runtime_error("[CrossEntropy::computeLoss] buffers.token_losses is NULL");
	}

	const int total_tokens = ctx.batch_size * ctx.seq_len;
	if (total_tokens <= 0) {
		throw std::runtime_error("[CrossEntropy::computeLoss] total_tokens=" +
			std::to_string(total_tokens) + " (must be > 0)");
	}

	crossEntropyLossKernel<<<launchGrid(total_tokens), kBlockSize, 0, ctx.stream>>>(
		ctx.logits,
		ctx.targets,
		buffers.token_losses,
		total_tokens,
		ctx.vocab_size);

	const cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		throw std::runtime_error(std::string("[CrossEntropy::computeLoss] kernel failed: ") +
			cudaGetErrorString(err));
	}
}

void reduceLossBuffer(const float* losses,
					  float* output,
					  int count,
					  cudaStream_t stream)
{
	if (!losses) {
		throw std::runtime_error("[CrossEntropy::reduceLossBuffer] losses is NULL");
	}
	if (!output) {
		throw std::runtime_error("[CrossEntropy::reduceLossBuffer] output is NULL");
	}
	if (count <= 0) {
		throw std::runtime_error("[CrossEntropy::reduceLossBuffer] count=" +
			std::to_string(count) + " (must be > 0)");
	}

	cudaError_t err = cudaMemsetAsync(output, 0, sizeof(float), stream);
	if (err != cudaSuccess) {
		throw std::runtime_error(std::string("[CrossEntropy::reduceLossBuffer] cudaMemsetAsync failed: ") +
			cudaGetErrorString(err));
	}

	reduceLossKernel<<<launchGrid(count), kBlockSize, 0, stream>>>(losses, output, count);

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		throw std::runtime_error(std::string("[CrossEntropy::reduceLossBuffer] kernel failed: ") +
			cudaGetErrorString(err));
	}
}

void computeCrossEntropyGradient(const LossContext& ctx,
								 DeviceBuffers buffers)
{
	validate(ctx);  // Throws on failure
	
	if (!buffers.grad_logits) {
		throw std::runtime_error("[CrossEntropy::computeGradient] buffers.grad_logits is NULL");
	}

	const int total_tokens = ctx.batch_size * ctx.seq_len;
	if (total_tokens <= 0) {
		throw std::runtime_error("[CrossEntropy::computeGradient] total_tokens=" +
			std::to_string(total_tokens) + " (must be > 0)");
	}

	const int valid_tokens = ctx.valid_tokens > 0 ? ctx.valid_tokens : total_tokens;
	const int weight_count = ctx.sequence_weight_count > 0
								 ? ctx.sequence_weight_count
								 : ctx.batch_size;

	crossEntropyGradientKernel<<<launchGrid(total_tokens), kBlockSize, 0, ctx.stream>>>(
		ctx.logits,
		ctx.targets,
		buffers.grad_logits,
		total_tokens,
		ctx.vocab_size,
		valid_tokens,
		ctx.seq_len,
		ctx.sequence_weights,
		weight_count);

	const cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		throw std::runtime_error(std::string("[CrossEntropy::computeGradient] kernel failed: ") +
			cudaGetErrorString(err));
	}
	
	// === TOKEN 277 GRADIENT DIAGNOSTIC ===
	// Log the gradient values for token 277 across first few positions
	// This shows how the loss is trying to adjust logit[277]
	static int s_grad_diag_count = 0;
	constexpr int kToken277 = 277;
	if (++s_grad_diag_count <= 20 && kToken277 < ctx.vocab_size) {
		cudaStreamSynchronize(ctx.stream);
		
		const int num_sample = std::min(10, total_tokens);
		std::vector<float> grad_277_samples(num_sample);
		std::vector<int> target_samples(num_sample);
		
		// Read grad[t, 277] for first few positions
		for (int t = 0; t < num_sample; ++t) {
			cudaMemcpy(&grad_277_samples[t], 
					   buffers.grad_logits + static_cast<size_t>(t) * ctx.vocab_size + kToken277,
					   sizeof(float), cudaMemcpyDeviceToHost);
		}
		// Read targets
		cudaMemcpy(target_samples.data(), ctx.targets, num_sample * sizeof(int), cudaMemcpyDeviceToHost);
		
		// Compute sum of grad[277] across ALL tokens
		float grad_277_sum = 0.0f;
		std::vector<float> all_grad_277(total_tokens);
		for (int t = 0; t < total_tokens; ++t) {
			cudaMemcpy(&all_grad_277[t],
					   buffers.grad_logits + static_cast<size_t>(t) * ctx.vocab_size + kToken277,
					   sizeof(float), cudaMemcpyDeviceToHost);
			grad_277_sum += all_grad_277[t];
		}
		
		// Count how many positions have target=277
		int target_277_count = 0;
		for (int t = 0; t < total_tokens; ++t) {
			int target;
			cudaMemcpy(&target, ctx.targets + t, sizeof(int), cudaMemcpyDeviceToHost);
			if (target == kToken277) target_277_count++;
		}
		
		fprintf(stderr, "[Token277Grad] call=%d total_tokens=%d target_277_count=%d grad_277_sum=%.6f\n",
				s_grad_diag_count, total_tokens, target_277_count, grad_277_sum);
		fprintf(stderr, "[Token277Grad] first_%d: grad_277=[", num_sample);
		for (int t = 0; t < num_sample; ++t) {
			fprintf(stderr, "%.4f%s", grad_277_samples[t], t < num_sample - 1 ? ", " : "");
		}
		fprintf(stderr, "] targets=[");
		for (int t = 0; t < num_sample; ++t) {
			fprintf(stderr, "%d%s", target_samples[t], t < num_sample - 1 ? ", " : "");
		}
		fprintf(stderr, "]\n");
	}
}

} // namespace GRIM::Loss

