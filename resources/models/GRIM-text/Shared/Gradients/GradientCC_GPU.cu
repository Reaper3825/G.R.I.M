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
#include <stdexcept>
#include <string>

namespace {

constexpr int kBlockSize = GRIM::HyperParameters::CUDA_BLOCK_SIZE_STANDARD;

__global__ void clampGradientsKernel(
	float* __restrict__ gradients,
	int n,
	float min_val,
	float max_val)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n) {
		return;
	}

	const float val = gradients[idx];

	if (!isfinite(val) || val < min_val) {
		gradients[idx] = min_val;
	} else if (val > max_val) {
		gradients[idx] = max_val;
	}
}

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

} // namespace

void launchClampGradients(
	float* gradients,
	int n,
	float min_val,
	float max_val,
	cudaStream_t stream)
{
	if (!validatePointers(gradients, n)) {
		return;
	}

	if (min_val > max_val) {
		fprintf(stderr, "GradientCC: min_val (%.3f) > max_val (%.3f)\n", min_val, max_val);
		return;
	}

	clampGradientsKernel<<<computeGrid(n), kBlockSize, 0, stream>>>(
		gradients, n, min_val, max_val);

	const cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "GradientCC: launchClampGradients failed - %s\n",
				cudaGetErrorString(err));
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

//======================================================//
//  Layer 2: Registry-level gradient clipping
//  Operates on ParameterGroup tensors via GradNorm + scale
//======================================================//

namespace GRIM::GradClip {

/// Returns true if this type belongs to the embedding/LM-head clip bucket
static inline bool isEmbBucket(ParamGroupType type, bool tie_embeddings) {
    if (type == ParamGroupType::LM_HEAD) return true;
    if (!tie_embeddings && type == ParamGroupType::EMBEDDING) return true;
    return false;
}

/// Returns true if this type belongs to the encoder clip bucket
static inline bool isEncBucket(ParamGroupType type) {
    switch (type) {
        case ParamGroupType::ATTENTION:
        case ParamGroupType::FFN:
        case ParamGroupType::RMSNORM:
        case ParamGroupType::SCRATCHBLOCK:
        case ParamGroupType::NUMERIC_HEAD:
        case ParamGroupType::MTP:
        case ParamGroupType::REASONING_HEAD:
        case ParamGroupType::EXECUTION_BLOCK:
            return true;
        default:
            return false;
    }
}

ClipResult clipGradientNorms(
    ParameterGroup* groups,
    size_t num_groups,
    GradNorm::GradNormScratch* scratch,
    const ClipConfig& config,
    cudaStream_t stream
) {
    if (!groups || num_groups == 0) {
        throw std::runtime_error("[GradClip] clipGradientNorms called with null/empty parameter groups");
    }
    if (!scratch) {
        throw std::runtime_error("[GradClip] clipGradientNorms called with null GradNormScratch");
    }
    if (config.max_rms <= 0.0f) {
        throw std::runtime_error("[GradClip] max_rms must be > 0, got " + std::to_string(config.max_rms));
    }

    // Step 1: Measure per-type gradient norms through the tensor registry
    auto status = GradNorm::measureGradientNorms(groups, num_groups, scratch, stream);
    if (status != GradNorm::GradNormStatus::SUCCESS) {
        throw std::runtime_error("[GradClip] measureGradientNorms failed: " +
                                 std::string(GradNorm::statusToString(status)));
    }
    // measureGradientNorms syncs internally — h_metrics is valid

    const auto& m = *scratch->h_metrics;

    // Step 2: Aggregate per-type metrics into clip buckets
    float emb_sum_sq = m.lm_head_sum_sq;
    int64_t emb_count = m.lm_head_count;
    if (!config.tie_embeddings) {
        emb_sum_sq += m.embedding_sum_sq;
        emb_count += m.embedding_count;
    }

    const float enc_sum_sq = m.attention_sum_sq + m.ffn_sum_sq
                           + m.rmsnorm_sum_sq + m.scratchblock_sum_sq
                           + m.numeric_head_sum_sq + m.mtp_sum_sq
                           + m.reasoning_head_sum_sq + m.execution_block_sum_sq;
    const int64_t enc_count = m.attention_count + m.ffn_count
                            + m.rmsnorm_count + m.scratchblock_count
                            + m.numeric_head_count + m.mtp_count
                            + m.reasoning_head_count + m.execution_block_count;

    const float emb_rms = (emb_count > 0) ? std::sqrt(emb_sum_sq / static_cast<float>(emb_count)) : 0.0f;
    const float enc_rms = (enc_count > 0) ? std::sqrt(enc_sum_sq / static_cast<float>(enc_count)) : 0.0f;

    ClipResult result;
    result.emb_rms = emb_rms;
    result.enc_rms = enc_rms;
    result.total_rms_pre = std::sqrt(emb_rms * emb_rms + enc_rms * enc_rms);

    // Step 3: Clip each bucket independently through the tensor registry
    if (emb_rms > config.max_rms) {
        const float coef = config.max_rms / (emb_rms + 1e-8f);
        for (size_t i = 0; i < num_groups; ++i) {
            if (!groups[i].grads() || groups[i].size() == 0) continue;
            if (isEmbBucket(groups[i].type, config.tie_embeddings)) {
                launchScaleGradients(groups[i].grads(),
                                     static_cast<int>(groups[i].size()),
                                     coef, stream);
            }
        }
        result.emb_clipped = true;
    }

    if (enc_rms > config.max_rms) {
        const float coef = config.max_rms / (enc_rms + 1e-8f);
        for (size_t i = 0; i < num_groups; ++i) {
            if (!groups[i].grads() || groups[i].size() == 0) continue;
            if (isEncBucket(groups[i].type)) {
                launchScaleGradients(groups[i].grads(),
                                     static_cast<int>(groups[i].size()),
                                     coef, stream);
            }
        }
        result.enc_clipped = true;
    }

    // Step 4: Compute post-clip RMS
    result.emb_rms_post = std::min(emb_rms, config.max_rms);
    result.enc_rms_post = std::min(enc_rms, config.max_rms);
    result.total_rms_post = std::sqrt(result.emb_rms_post * result.emb_rms_post
                                    + result.enc_rms_post * result.enc_rms_post);

    return result;
}

} // namespace GRIM::GradClip
