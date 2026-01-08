#include "Shared/Loss/Divergence/Divergence_GPU.hpp"
#include "../../HyperParameters/HyperParameters_GPU.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <cuda_runtime.h>

namespace GRIM::Loss {
namespace {

constexpr int kBlockSize = HyperParameters::CUDA_BLOCK_SIZE_STANDARD;
constexpr float kEpsilon = HyperParameters::EPSILON_SAFE_DIV;

inline int computeGridSize(int work_items)
{
	const int blocks = (work_items + kBlockSize - 1) / kBlockSize;
	return std::max(1, std::min(blocks, 1024));
}

__global__ void reduceSquaresKernel(const float* __restrict__ vec,
									int dim,
									float* __restrict__ output)
{
	__shared__ float shared[kBlockSize];
	float thread_sum = 0.0f;
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < dim; idx += blockDim.x * gridDim.x) {
		const float value = vec[idx];
		thread_sum += value * value;
	}
	shared[threadIdx.x] = thread_sum;
	__syncthreads();
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride) {
			shared[threadIdx.x] += shared[threadIdx.x + stride];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) {
		atomicAdd(output, shared[0]);
	}
}

__global__ void reduceDotKernel(const float* __restrict__ a,
								const float* __restrict__ b,
								int dim,
								float* __restrict__ output)
{
	__shared__ float shared[kBlockSize];
	float thread_sum = 0.0f;
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < dim; idx += blockDim.x * gridDim.x) {
		thread_sum += a[idx] * b[idx];
	}
	shared[threadIdx.x] = thread_sum;
	__syncthreads();
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride) {
			shared[threadIdx.x] += shared[threadIdx.x + stride];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) {
		atomicAdd(output, shared[0]);
	}
}

__global__ void negativeStatsKernel(const float* __restrict__ anchor,
									const float* __restrict__ negatives,
									int feature_dim,
									int negative_count,
									float* __restrict__ out_dots,
									float* __restrict__ out_norms)
{
	const int neg_idx = blockIdx.x;
	if (neg_idx >= negative_count) {
		return;
	}

	extern __shared__ float shared[];
	float* dot_shared = shared;
	float* norm_shared = shared + blockDim.x;

	float dot_sum = 0.0f;
	float norm_sum = 0.0f;
	const float* neg_vec = negatives + neg_idx * feature_dim;
	for (int idx = threadIdx.x; idx < feature_dim; idx += blockDim.x) {
		const float a_val = anchor[idx];
		const float b_val = neg_vec[idx];
		dot_sum += a_val * b_val;
		norm_sum += b_val * b_val;
	}

	dot_shared[threadIdx.x] = dot_sum;
	norm_shared[threadIdx.x] = norm_sum;
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride) {
			dot_shared[threadIdx.x] += dot_shared[threadIdx.x + stride];
			norm_shared[threadIdx.x] += norm_shared[threadIdx.x + stride];
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		out_dots[neg_idx] = dot_shared[0];
		out_norms[neg_idx] = norm_shared[0];
	}
}

__global__ void infoNceFinalizeKernel(const float* __restrict__ anchor_norm_sq,
									  const float* __restrict__ pos_norm_sq,
									  const float* __restrict__ pos_dot,
									  const float* __restrict__ neg_dots,
									  const float* __restrict__ neg_norms,
									  int negative_count,
									  float temperature,
									  float* __restrict__ loss_out)
{
	if (!loss_out) {
		return;
	}

	const float anchor_norm = sqrtf(fmaxf(*anchor_norm_sq, kEpsilon));
	const float pos_norm = sqrtf(fmaxf(*pos_norm_sq, kEpsilon));
	const float denom_pos = fmaxf(anchor_norm * pos_norm, kEpsilon);
	const float pos_sim = (*pos_dot) / denom_pos;
	const float pos_logit = pos_sim / temperature;

	float max_logit = pos_logit;
	if (negative_count > 0 && neg_dots && neg_norms) {
		for (int i = 0; i < negative_count; ++i) {
			const float neg_norm = sqrtf(fmaxf(neg_norms[i], kEpsilon));
			const float denom_neg = fmaxf(anchor_norm * neg_norm, kEpsilon);
			const float sim = neg_dots[i] / denom_neg;
			const float logit = sim / temperature;
			max_logit = fmaxf(max_logit, logit);
		}
	}

	float denom = expf(pos_logit - max_logit);
	if (negative_count > 0 && neg_dots && neg_norms) {
		for (int i = 0; i < negative_count; ++i) {
			const float neg_norm = sqrtf(fmaxf(neg_norms[i], kEpsilon));
			const float denom_neg = fmaxf(anchor_norm * neg_norm, kEpsilon);
			const float sim = neg_dots[i] / denom_neg;
			const float logit = sim / temperature;
			denom += expf(logit - max_logit);
		}
	}

	const float loss = -(pos_logit - (logf(denom) + max_logit));
	*loss_out = loss;
}

__global__ void cosineFinalizeKernel(const float* __restrict__ dot_product,
									 const float* __restrict__ norm_a_sq,
									 const float* __restrict__ norm_b_sq,
									 float margin,
									 float* __restrict__ loss_out,
									 float* __restrict__ grad_out)
{
	const float norm_a = sqrtf(fmaxf(*norm_a_sq, kEpsilon));
	const float norm_b = sqrtf(fmaxf(*norm_b_sq, kEpsilon));
	const float denom = fmaxf(norm_a * norm_b, kEpsilon);
	const float cos_sim = (*dot_product) / denom;
	const float loss = fmaxf(0.0f, margin - cos_sim);

	if (loss_out) {
		*loss_out = loss;
	}
	if (grad_out) {
		*grad_out = (loss > 0.0f) ? -1.0f : 0.0f;
	}
}

__global__ void tokenMaskKernel(const float* __restrict__ mask,
								float* __restrict__ token_losses,
								float* __restrict__ grad_logits,
								int total_tokens,
								int vocab_size,
								float* __restrict__ loss_delta)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= total_tokens) {
		return;
	}

	const float mask_value = mask ? fminf(fmaxf(mask[idx], 0.0f), 1.0f) : 1.0f;
	if (mask_value >= 0.999f) {
		return;
	}

	if (grad_logits) {
		const int offset = idx * vocab_size;
		for (int i = 0; i < vocab_size; ++i) {
			grad_logits[offset + i] *= mask_value;
		}
	}

	if (token_losses) {
		const float previous = token_losses[idx];
		const float updated = previous * mask_value;
		token_losses[idx] = updated;
		if (loss_delta) {
			atomicAdd(loss_delta, previous - updated);
		}
	}
}

inline dim3 launchGrid(int total_threads)
{
	return dim3((total_threads + kBlockSize - 1) / kBlockSize);
}

inline float* acquireAccumulator(DeviceBuffers buffers, bool& owns_device)
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

inline float dotProduct(const float* a, const float* b, int dim)
{
	float sum = 0.0f;
	for (int i = 0; i < dim; ++i) {
		sum += a[i] * b[i];
	}
	return sum;
}

inline float l2Norm(const float* a, int dim)
{
	return std::sqrt(std::max(dotProduct(a, a, dim), kEpsilon));
}

inline void writeScalar(float value, float* destination, cudaStream_t stream)
{
	if (!destination) {
		return;
	}
	cudaError_t err = cudaMemcpyAsync(destination, &value, sizeof(float), cudaMemcpyHostToDevice, stream);
	if (err == cudaSuccess) {
		cudaStreamSynchronize(stream);
	} else if (err == cudaErrorInvalidValue) {
		*destination = value;
	}
}

inline float normalizeConfidence(float confidence, float min_threshold)
{
	const float clamped = std::clamp(confidence, 0.0f, 1.0f);
	if (clamped <= min_threshold) {
		return 0.0f;
	}
	return (clamped - min_threshold) / std::max(1e-3f, 1.0f - min_threshold);
}

inline float rewardInfluence(float reward)
{
	if (reward >= 0.0f) {
		return std::clamp(1.0f + reward, 1.0f, 2.0f);
	}
	return std::clamp(1.0f / (1.0f + std::fabs(reward)), 0.5f, 1.0f);
}

inline void writeScalarResult(float value, float* destination)
{
	if (!destination) {
		return;
	}

	cudaPointerAttributes attr{};
	const cudaError_t attr_err = cudaPointerGetAttributes(&attr, destination);
	if (attr_err == cudaSuccess && attr.type == cudaMemoryTypeDevice) {
		cudaMemcpy(destination, &value, sizeof(float), cudaMemcpyHostToDevice);
	} else {
		*destination = value;
		if (attr_err != cudaErrorInvalidValue) {
			cudaGetLastError();
		}
	}
}

} // namespace

void applyTokenMasking(const LossContext& ctx,
					   const MaskConfig& cfg,
					   DeviceBuffers buffers,
					   LossBreakdown& out_loss)
{
	if (!cfg.enabled || !ctx.token_mask) {
		return;
	}

	const int total_tokens = ctx.batch_size * ctx.seq_len;
	if (total_tokens <= 0 || ctx.vocab_size <= 0) {
		return;
	}

	bool owns_buffer = false;
	float* delta_buffer = acquireAccumulator(buffers, owns_buffer);
	if (delta_buffer) {
		cudaMemsetAsync(delta_buffer, 0, sizeof(float), ctx.stream);
	}

	tokenMaskKernel<<<launchGrid(total_tokens), kBlockSize, 0, ctx.stream>>>(
		ctx.token_mask,
		buffers.token_losses,
		buffers.grad_logits,
		total_tokens,
		ctx.vocab_size,
		delta_buffer);

	const cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "Divergence: token mask kernel failed - %s\n",
				cudaGetErrorString(err));
	}

	if (delta_buffer) {
		float masked = 0.0f;
		cudaMemcpyAsync(&masked, delta_buffer, sizeof(float), cudaMemcpyDeviceToHost, ctx.stream);
		cudaStreamSynchronize(ctx.stream);
		out_loss.masked += masked;
	}

	releaseAccumulator(delta_buffer, owns_buffer);
}

void blendGuessFeedback(const AuxiliaryBatchViews& aux,
						const GuessFeedbackConfig& cfg,
						float* sequence_weights,
						LossBreakdown& out_loss)
{
	if (!cfg.enabled || cfg.lambda <= 0.0f || aux.sample_count <= 0 || !sequence_weights) {
		return;
	}

	const float min_conf = std::clamp(cfg.min_confidence, 0.0f, 0.99f);
	float cumulative_delta = 0.0f;
	for (int i = 0; i < aux.sample_count; ++i) {
		const float confidence = aux.guess_confidence ? aux.guess_confidence[i] : 0.0f;
		const float reward = aux.reward_scores ? aux.reward_scores[i] : 0.0f;
		float weight = 1.0f;

		const float conf_term = normalizeConfidence(confidence, min_conf);
		if (conf_term > 0.0f) {
			weight += cfg.lambda * conf_term * rewardInfluence(reward);
		}

		sequence_weights[i] = std::clamp(weight, 0.5f, 5.0f);
		cumulative_delta += (sequence_weights[i] - 1.0f);
	}

	if (aux.sample_count > 0) {
		out_loss.custom += cumulative_delta / static_cast<float>(aux.sample_count);
	}
}

float computeInfoNCELoss(const float* anchors,
						 const float* positives,
						 const float* negatives,
						 int feature_dim,
						 int negative_count,
						 float temperature,
						 cudaStream_t stream)
{
	if (!anchors || !positives || feature_dim <= 0 || temperature <= 0.0f) {
		return 0.0f;
	}

	const int grid = computeGridSize(feature_dim);
	float result = 0.0f;

	float *anchor_norm_dev = nullptr;
	float *pos_norm_dev = nullptr;
	float *pos_dot_dev = nullptr;
	float *loss_dev = nullptr;
	float *neg_dots_dev = nullptr;
	float *neg_norms_dev = nullptr;

	auto allocAndZero = [&](float** ptr) -> bool {
		if (cudaMalloc(ptr, sizeof(float)) != cudaSuccess) {
			return false;
		}
		cudaMemsetAsync(*ptr, 0, sizeof(float), stream);
		return true;
	};

	if (!allocAndZero(&anchor_norm_dev) ||
		!allocAndZero(&pos_norm_dev) ||
		!allocAndZero(&pos_dot_dev) ||
		cudaMalloc(&loss_dev, sizeof(float)) != cudaSuccess) {
		goto cleanup;
	}

	reduceSquaresKernel<<<grid, kBlockSize, 0, stream>>>(anchors, feature_dim, anchor_norm_dev);
	reduceSquaresKernel<<<grid, kBlockSize, 0, stream>>>(positives, feature_dim, pos_norm_dev);
	reduceDotKernel<<<grid, kBlockSize, 0, stream>>>(anchors, positives, feature_dim, pos_dot_dev);

	if (negative_count > 0 && negatives) {
		if (cudaMalloc(&neg_dots_dev, negative_count * sizeof(float)) != cudaSuccess ||
			cudaMalloc(&neg_norms_dev, negative_count * sizeof(float)) != cudaSuccess) {
			goto cleanup;
		}
		negativeStatsKernel<<<negative_count, kBlockSize, 2 * kBlockSize * sizeof(float), stream>>>(
			anchors,
			negatives,
			feature_dim,
			negative_count,
			neg_dots_dev,
			neg_norms_dev);
	} else {
		negative_count = 0;
	}

	infoNceFinalizeKernel<<<1, 1, 0, stream>>>(
		anchor_norm_dev,
		pos_norm_dev,
		pos_dot_dev,
		neg_dots_dev,
		neg_norms_dev,
		negative_count,
		temperature,
		loss_dev);

	if (cudaGetLastError() != cudaSuccess) {
		goto cleanup;
	}

	cudaMemcpyAsync(&result, loss_dev, sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

cleanup:
	if (loss_dev) cudaFree(loss_dev);
	if (pos_dot_dev) cudaFree(pos_dot_dev);
	if (pos_norm_dev) cudaFree(pos_norm_dev);
	if (anchor_norm_dev) cudaFree(anchor_norm_dev);
	if (neg_dots_dev) cudaFree(neg_dots_dev);
	if (neg_norms_dev) cudaFree(neg_norms_dev);
	return result;
}

void computeCosineSimilarityLoss(const float* a,
								 const float* b,
								 int feature_dim,
								 float margin,
								 cudaStream_t stream,
								 float* out_value,
								 float* out_gradient)
{
	if (!a || !b || feature_dim <= 0) {
		writeScalarResult(0.0f, out_value);
		writeScalarResult(0.0f, out_gradient);
		return;
	}

	const int grid = computeGridSize(feature_dim);
	float result_loss = 0.0f;
	float result_grad = 0.0f;

	float *dot_dev = nullptr;
	float *norm_a_dev = nullptr;
	float *norm_b_dev = nullptr;
	float *loss_dev = nullptr;
	float *grad_dev = nullptr;

	auto allocAndZero = [&](float** ptr) -> bool {
		if (cudaMalloc(ptr, sizeof(float)) != cudaSuccess) {
			return false;
		}
		cudaMemsetAsync(*ptr, 0, sizeof(float), stream);
		return true;
	};

	if (!allocAndZero(&dot_dev) ||
		!allocAndZero(&norm_a_dev) ||
		!allocAndZero(&norm_b_dev)) {
		goto cleanup;
	}

	if (cudaMalloc(&loss_dev, sizeof(float)) != cudaSuccess) {
		goto cleanup;
	}
	if (out_gradient) {
		if (cudaMalloc(&grad_dev, sizeof(float)) != cudaSuccess) {
			goto cleanup;
		}
	}

	reduceDotKernel<<<grid, kBlockSize, 0, stream>>>(a, b, feature_dim, dot_dev);
	reduceSquaresKernel<<<grid, kBlockSize, 0, stream>>>(a, feature_dim, norm_a_dev);
	reduceSquaresKernel<<<grid, kBlockSize, 0, stream>>>(b, feature_dim, norm_b_dev);

	cosineFinalizeKernel<<<1, 1, 0, stream>>>(
		dot_dev,
		norm_a_dev,
		norm_b_dev,
		margin,
		loss_dev,
		grad_dev);

	if (cudaGetLastError() != cudaSuccess) {
		goto cleanup;
	}

	cudaMemcpyAsync(&result_loss, loss_dev, sizeof(float), cudaMemcpyDeviceToHost, stream);
	if (grad_dev) {
		cudaMemcpyAsync(&result_grad, grad_dev, sizeof(float), cudaMemcpyDeviceToHost, stream);
	}
	cudaStreamSynchronize(stream);

	writeScalarResult(result_loss, out_value);
	if (out_gradient) {
		writeScalarResult(result_grad, out_gradient);
	}

cleanup:
	if (grad_dev) cudaFree(grad_dev);
	if (loss_dev) cudaFree(loss_dev);
	if (norm_b_dev) cudaFree(norm_b_dev);
	if (norm_a_dev) cudaFree(norm_a_dev);
	if (dot_dev) cudaFree(dot_dev);
}

} // namespace GRIM::Loss

