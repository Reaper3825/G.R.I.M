//======================================================//
//  CrossEntropyNLL.cu
//  Cross-entropy-family NLL kernels for unified autograd loss
//
//  Implements: Cross Entropy + Label Smoothing + Focal Loss + Entropy Regularization
//  Input contract: receives log_probs = log_softmax(logits), NOT raw logits.
//======================================================//

#include "CrossEntropyNLL.hpp"
#include "../../CudaAllocUtils.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <string>
#include <stdexcept>

using GRIM::CudaAlloc::cudaMallocOrThrow;

namespace GRIM {
namespace autograd {
namespace {

struct DeviceFreeGuard {
    void* ptr;

    explicit DeviceFreeGuard(void* ptr_) : ptr(ptr_) {}
    ~DeviceFreeGuard() {
        if (ptr) cudaFree(ptr);
    }

    DeviceFreeGuard(const DeviceFreeGuard&) = delete;
    DeviceFreeGuard& operator=(const DeviceFreeGuard&) = delete;
};

//========================================================================
// CUDA Kernels — NLL Loss on log-probabilities
// These kernels receive log_probs = log_softmax(logits), NOT raw logits.
// Softmax is computed ONCE in autograd::log_softmax() and saved for backward.
//========================================================================

/**
 * NLL Loss Forward kernel — one block per token
 *
 * log_probs already contains log(softmax(logits)), computed by log_softmax.
 * We just pick -log_probs[target] and add focal/smoothing/entropy.
 *
 * Computes:
 *   When focal_enabled:  L = α * (1 - p_t)^γ * CE_smooth + λ * neg_entropy
 *   When focal disabled: L = CE_smooth + λ * neg_entropy
 *   (focal_alpha/gamma are ONLY applied when focal_enabled=true)
 *
 * Where:
 *   p_t  = exp(log_probs[target])          — ONE exp, not 50K
 *   CE_smooth = -(1-ε)*log_probs[target] - ε/(V-1)*Σ_{i≠t} log_probs[i]
 *   neg_entropy = Σ exp(log_probs[i]) * log_probs[i]
 */
__global__ void kernelCrossEntropyNLLForward(
    const float* __restrict__ log_probs,    // [num_tokens, vocab_size] — log-probabilities
    const int* __restrict__ targets,
    float* __restrict__ per_token_loss,
    float* __restrict__ loss_sum,
    int* __restrict__ valid_count,
    float* __restrict__ weight_sum,             // Accumulated class weights (nullable if no class_balanced)
    int num_tokens,
    int vocab_size,
    float focal_alpha,
    float focal_gamma,
    bool focal_enabled,
    float smoothing_epsilon,
    bool smoothing_enabled,
    float entropy_reg_lambda,
    bool entropy_reg_enabled,
    const float* __restrict__ class_weights     // [vocab_size] per-class weights (nullable = all 1.0)
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    const int target = targets[token_idx];

    // Skip masked/padding positions (target == -1)
    if (target == -1) {
        per_token_loss[token_idx] = 0.0f;
        return;
    }

    const float* row = log_probs + static_cast<size_t>(token_idx) * vocab_size;

    // p_t = exp(log_probs[target]) — just ONE exp call
    const float log_p_t = row[target];
    const float p_t = expf(log_p_t);

    // ── Entropy: Σ p_i * log(p_i) = Σ exp(lp_i) * lp_i ──
    __shared__ float s_neg_entropy;
    __shared__ float s_sum_log_off;
    if (threadIdx.x == 0) {
        s_neg_entropy = 0.0f;
        s_sum_log_off = 0.0f;
    }
    __syncthreads();

    float local_neg_entropy = 0.0f;
    float local_sum_log_off = 0.0f;

    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        if (entropy_reg_enabled) {
            const float p_v = expf(row[v]);
            if (p_v > 0.0f) {
                local_neg_entropy += p_v * row[v];  // p * log(p)
            }
        }
        if (smoothing_enabled && v != target) {
            local_sum_log_off += row[v];  // log(p_v) — already in log space!
        }
    }

    // Warp + cross-warp reduction
    for (int off = warpSize / 2; off > 0; off /= 2) {
        local_neg_entropy += __shfl_down_sync(0xffffffff, local_neg_entropy, off);
        local_sum_log_off += __shfl_down_sync(0xffffffff, local_sum_log_off, off);
    }
    if (threadIdx.x % warpSize == 0) {
        atomicAdd(&s_neg_entropy, local_neg_entropy);
        atomicAdd(&s_sum_log_off, local_sum_log_off);
    }
    __syncthreads();

    // ── Thread 0: assemble loss ──
    if (threadIdx.x == 0) {
        // Label-smoothed cross entropy
        float ce_smooth;
        if (smoothing_enabled && vocab_size > 1) {
            const float q_on  = 1.0f - smoothing_epsilon;
            const float q_off = smoothing_epsilon / static_cast<float>(vocab_size - 1);
            ce_smooth = -q_on * log_p_t - q_off * s_sum_log_off;
        } else {
            ce_smooth = -log_p_t;  // Standard CE: -log(p_target)
        }

        // Focal loss weight
        float focal_weight = 1.0f;
        if (focal_enabled) {
            focal_weight = powf(fmaxf(1.0f - p_t, 0.0f), focal_gamma);
        }

        // When focal is disabled, focal_alpha MUST NOT scale the CE term.
        // Otherwise flipping focal_enabled=false silently changes the CE:entropy
        // ratio via a parameter whose name says "focal" — a quality footgun.
        float ce_loss = ce_smooth;
        if (focal_enabled) {
            ce_loss = focal_alpha * focal_weight * ce_smooth;
        }
        float entropy_loss = 0.0f;
        if (entropy_reg_enabled) {
            entropy_loss = entropy_reg_lambda * s_neg_entropy;
        }
        const float total_loss = ce_loss + entropy_loss;

        // Class-balanced weighting: w_{y_t} scales the ENTIRE loss for this position
        // This weights the whole gradient row grad_logits[t, :] = w * (p - q) / W
        float cw = 1.0f;
        if (class_weights != nullptr) {
            cw = class_weights[target];
        }
        const float weighted_loss = cw * total_loss;

        per_token_loss[token_idx] = weighted_loss;
        atomicAdd(loss_sum, weighted_loss);
        atomicAdd(valid_count, 1);
        if (weight_sum != nullptr) {
            atomicAdd(weight_sum, cw);
        }
    }
}

/**
 * NLL Loss Backward kernel — gradient w.r.t. LOG-PROBABILITIES
 *
 * For plain CE:  ∂L/∂(log_p_i) is -1/N at target, 0 elsewhere
 *   But we actually want d/d(log_p) of the full loss, so:
 *
 *   ∂(-log_p_t)/∂(log_p_i) = -δ_{i=t}
 *
 * With label smoothing:
 *   ∂CE_smooth/∂(log_p_i) = -q_i  where q_t = 1-ε, q_i = ε/(V-1) for i≠t
 *
 * With focal:
 *   ∂[α*(1-p_t)^γ * CE_smooth]/∂(log_p_i)
 *     = α * [focal_weight * (-q_i) + (1-p_t)^γ's derivative via p_t = exp(log_p_t)]
 *
 * With entropy reg:
 *   H(p) = Σ p_j * log_p_j where p_j = exp(log_p_j)
 *   ∂H/∂(log_p_i) = ∂(p_i * log_p_i)/∂(log_p_i) = p_i * (log_p_i + 1)
 *   The LogSoftmaxGradFn applies the upstream log-softmax Jacobian.
 *
 * FINAL GRADIENT (same as PyTorch F.nll_loss):
 *   For plain CE:  grad_log_p[i] = -δ_{i=t} / N
 *   For smoothed:  grad_log_p[i] = -q_i / N
 *
 * When this flows into LogSoftmaxGradFn:
 *   grad_logits[j] = grad_log_p[j] - p_j * Σ_i grad_log_p[i]
 *                  = -q_j/N - p_j * (-1/N)
 *                  = (p_j - q_j) / N
 */
__global__ void kernelCrossEntropyNLLBackward(
    const float* __restrict__ log_probs,    // [num_tokens, vocab_size]
    const int* __restrict__ targets,
    float* __restrict__ grad_log_probs,     // [num_tokens, vocab_size] — OUTPUT
    int num_tokens,
    int vocab_size,
    float inv_valid_count,  // 1/N or 1/W (weight_sum) when class_balanced
    float focal_alpha,
    float focal_gamma,
    bool focal_enabled,
    float smoothing_epsilon,
    bool smoothing_enabled,
    float entropy_reg_lambda,
    bool entropy_reg_enabled,
    const float* __restrict__ class_weights     // [vocab_size] per-class weights (nullable = all 1.0)
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    const float* row = log_probs + static_cast<size_t>(token_idx) * vocab_size;
    float* grad_row = grad_log_probs + static_cast<size_t>(token_idx) * vocab_size;
    const int target = targets[token_idx];

    // Skip masked/padding positions (target == -1)
    if (target == -1) {
        for (int v = threadIdx.x; v < vocab_size; v += blockDim.x)
            grad_row[v] = 0.0f;
        return;
    }

    // Label smoothing targets
    float q_on = 1.0f;
    float q_off = 0.0f;
    if (smoothing_enabled && vocab_size > 1) {
        q_on = 1.0f - smoothing_epsilon;
        q_off = smoothing_epsilon / static_cast<float>(vocab_size - 1);
    }

    // Focal precomputation
    const float p_t = expf(row[target]);
    float focal_weight = 1.0f;
    float focal_deriv_factor = 0.0f;
    if (focal_enabled) {
        const float one_minus_pt = fmaxf(1.0f - p_t, 0.0f);
        if (one_minus_pt > 0.0f) {
            focal_weight = powf(one_minus_pt, focal_gamma);
            focal_deriv_factor = focal_gamma * powf(one_minus_pt, focal_gamma - 1.0f);
        } else {
            focal_weight = 0.0f;
            focal_deriv_factor = 0.0f;
        }
    }

    // Entropy centering term (Issue #124)
    // Need neg_entropy = Σ p_v * log_p_v for centering.
    __shared__ float s_neg_entropy;
    if (threadIdx.x == 0) s_neg_entropy = 0.0f;
    __syncthreads();

    float local_ne = 0.0f;
    if (entropy_reg_enabled) {
        for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
            const float p_v = expf(row[v]);
            if (p_v > 0.0f) local_ne += p_v * row[v];
        }
        for (int off = warpSize / 2; off > 0; off /= 2)
            local_ne += __shfl_down_sync(0xffffffff, local_ne, off);
        if (threadIdx.x % warpSize == 0) atomicAdd(&s_neg_entropy, local_ne);
    }
    __syncthreads();
    const float neg_entropy = s_neg_entropy;

    // Precompute sum_log_off if needed for focal derivative with smoothing.
    __shared__ float s_sum_log_off;
    if (threadIdx.x == 0) s_sum_log_off = 0.0f;
    __syncthreads();

    if (focal_enabled && smoothing_enabled) {
        float local_sum_log_off = 0.0f;
        for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
            if (v != target) {
                local_sum_log_off += row[v];
            }
        }
        for (int off = warpSize / 2; off > 0; off /= 2)
            local_sum_log_off += __shfl_down_sync(0xffffffff, local_sum_log_off, off);
        if (threadIdx.x % warpSize == 0) atomicAdd(&s_sum_log_off, local_sum_log_off);
    }
    __syncthreads();
    const float sum_log_off = s_sum_log_off;

    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        float q_v = q_off;
        if (v == target) {
            q_v = q_on;
        }

        // Base: -q_v (plain NLL loss gradient w.r.t. log_probs)
        // When focal is disabled, focal_alpha MUST NOT scale the gradient.
        float grad_v = -q_v;
        if (focal_enabled) {
            grad_v = focal_alpha * focal_weight * (-q_v);
        }

        // Focal derivative: only affects gradient through p_t = exp(log_p_t)
        if (focal_enabled) {
            float ce_smooth = -row[target];
            if (smoothing_enabled) {
                ce_smooth = -(q_on * row[target] + q_off * sum_log_off);
            }
            if (v == target) {
                grad_v += focal_alpha * (-focal_deriv_factor) * p_t * ce_smooth;
            }
        }

        // Entropy regularization gradient w.r.t. log_probs.
        if (entropy_reg_enabled) {
            const float p_v = expf(row[v]);
            if (p_v > 0.0f) {
                grad_v += entropy_reg_lambda * p_v * (row[v] - neg_entropy);
            }
        }

        // Apply mean reduction with class-balanced weighting and upstream grad scaling.
        float cw = 1.0f;
        if (class_weights != nullptr) {
            cw = class_weights[target];
        }
        grad_row[v] = grad_v * cw * inv_valid_count;
    }
}

void launchCrossEntropyNLLForward(
    const float* log_probs,
    const int* targets,
    float* per_token_loss,
    float* loss_sum,
    int* valid_count,
    float* weight_sum,
    int num_tokens,
    int vocab_size,
    const HyperParameters::LossConfigHP& config,
    const float* d_class_weights,
    cudaStream_t stream
) {
    cudaMemsetAsync(loss_sum, 0, sizeof(float), stream);
    cudaMemsetAsync(valid_count, 0, sizeof(int), stream);
    if (weight_sum) cudaMemsetAsync(weight_sum, 0, sizeof(float), stream);

    const int block_size = 256;
    kernelCrossEntropyNLLForward<<<num_tokens, block_size, 0, stream>>>(
        log_probs, targets,
        per_token_loss, loss_sum, valid_count, weight_sum,
        num_tokens, vocab_size,
        config.focal_alpha, config.focal_gamma, config.focal_enabled,
        config.smoothing_epsilon, config.smoothing_enabled,
        config.entropy_reg_lambda, config.entropy_reg_enabled,
        d_class_weights
    );
}

void launchCrossEntropyNLLBackward(
    const float* log_probs,
    const int* targets,
    float* grad_log_probs,
    int num_tokens,
    int vocab_size,
    int valid_count,
    float weight_sum,
    const HyperParameters::LossConfigHP& config,
    const float* d_class_weights,
    float grad_output_scale,
    cudaStream_t stream
) {
    if (valid_count <= 0) {
        throw std::runtime_error("[launchCrossEntropyNLLBackward] valid_count=" + std::to_string(valid_count)
            + " — no valid tokens, caller MUST ensure valid_count > 0");
    }

    float normalization = static_cast<float>(valid_count);
    if (d_class_weights != nullptr && weight_sum > 0.0f) {
        normalization = weight_sum;
    }
    const float inv_valid_count = grad_output_scale / normalization;

    const int block_size = 256;
    kernelCrossEntropyNLLBackward<<<num_tokens, block_size, 0, stream>>>(
        log_probs, targets, grad_log_probs,
        num_tokens, vocab_size, inv_valid_count,
        config.focal_alpha, config.focal_gamma, config.focal_enabled,
        config.smoothing_epsilon, config.smoothing_enabled,
        config.entropy_reg_lambda, config.entropy_reg_enabled,
        d_class_weights
    );
}

}  // namespace

CrossEntropyForwardResult computeCrossEntropyForwardFromLogProbs(
    const float* log_probs,
    const int* targets,
    int num_tokens,
    int vocab_size,
    const HyperParameters::LossConfigHP& config,
    const float* d_class_weights,
    cudaStream_t stream
) {
    if (!log_probs) {
        throw std::runtime_error("[computeCrossEntropyForwardFromLogProbs] log_probs pointer is NULL — caller MUST provide log_softmax output");
    }
    if (!targets) {
        throw std::runtime_error("[computeCrossEntropyForwardFromLogProbs] targets pointer is NULL — caller MUST provide valid target token IDs");
    }
    if (num_tokens <= 0) {
        throw std::runtime_error("[computeCrossEntropyForwardFromLogProbs] num_tokens=" + std::to_string(num_tokens) + " — must be > 0");
    }
    if (vocab_size <= 0) {
        throw std::runtime_error("[computeCrossEntropyForwardFromLogProbs] vocab_size=" + std::to_string(vocab_size) + " — must be > 0");
    }

    float* per_token_loss = nullptr;
    float* d_loss_sum = nullptr;
    int* d_valid_count = nullptr;
    float* d_weight_sum = nullptr;

    cudaMallocOrThrow(reinterpret_cast<void**>(&per_token_loss), num_tokens * sizeof(float), "cross_entropy_per_token_loss");
    DeviceFreeGuard per_token_loss_guard(per_token_loss);

    cudaMallocOrThrow(reinterpret_cast<void**>(&d_loss_sum), sizeof(float), "cross_entropy_d_loss_sum");
    DeviceFreeGuard loss_sum_guard(d_loss_sum);

    cudaMallocOrThrow(reinterpret_cast<void**>(&d_valid_count), sizeof(int), "cross_entropy_d_valid_count");
    DeviceFreeGuard valid_count_guard(d_valid_count);

    if (d_class_weights) {
        cudaMallocOrThrow(reinterpret_cast<void**>(&d_weight_sum), sizeof(float), "cross_entropy_d_weight_sum");
    }
    DeviceFreeGuard weight_sum_guard(d_weight_sum);

    launchCrossEntropyNLLForward(
        log_probs,
        targets,
        per_token_loss,
        d_loss_sum,
        d_valid_count,
        d_weight_sum,
        num_tokens,
        vocab_size,
        config,
        d_class_weights,
        stream
    );

    float h_loss_sum = 0.0f;
    int h_valid_count = 0;
    float h_weight_sum = 0.0f;
    cudaMemcpyAsync(&h_loss_sum, d_loss_sum, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_valid_count, d_valid_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
    if (d_weight_sum) {
        cudaMemcpyAsync(&h_weight_sum, d_weight_sum, sizeof(float), cudaMemcpyDeviceToHost, stream);
    }
    cudaStreamSynchronize(stream);

    if (h_valid_count <= 0) {
        throw std::runtime_error("[computeCrossEntropyForwardFromLogProbs] valid_count=0 — no valid tokens in batch. "
            "Check targets for corruption: all targets are -1.");
    }

    if (!std::isfinite(h_loss_sum)) {
        throw std::runtime_error("[computeCrossEntropyForwardFromLogProbs] h_loss_sum is non-finite (" + std::to_string(h_loss_sum) +
            ") — NLL kernel produced NaN/Inf. valid_count=" + std::to_string(h_valid_count) +
            " weight_sum=" + std::to_string(h_weight_sum) + " focal=" + std::to_string(config.focal_gamma) +
            " smoothing=" + std::to_string(config.smoothing_epsilon) + " entropy_lambda=" + std::to_string(config.entropy_reg_lambda));
    }

    float normalization = static_cast<float>(h_valid_count);
    if (d_class_weights && h_weight_sum > 0.0f) {
        normalization = h_weight_sum;
    }
    if (normalization <= 0.0f || !std::isfinite(normalization)) {
        throw std::runtime_error("[computeCrossEntropyForwardFromLogProbs] normalization invalid (" + std::to_string(normalization) +
            ") — valid_count=" + std::to_string(h_valid_count) + " weight_sum=" + std::to_string(h_weight_sum));
    }

    const float mean_loss = h_loss_sum / normalization;
    if (!std::isfinite(mean_loss)) {
        throw std::runtime_error("[computeCrossEntropyForwardFromLogProbs] mean_loss is non-finite (" + std::to_string(mean_loss) +
            ") after h_loss_sum=" + std::to_string(h_loss_sum) + " / norm=" + std::to_string(normalization));
    }

    return CrossEntropyForwardResult{mean_loss, h_valid_count, h_weight_sum};
}

void computeCrossEntropyBackwardToLogProbs(
    const float* log_probs,
    const int* targets,
    float* grad_log_probs,
    int num_tokens,
    int vocab_size,
    int valid_count,
    float weight_sum,
    const HyperParameters::LossConfigHP& config,
    const float* d_class_weights,
    float grad_output_scale,
    cudaStream_t stream
) {
    if (!log_probs) {
        throw std::runtime_error("[computeCrossEntropyBackwardToLogProbs] log_probs pointer is NULL — caller MUST provide saved log-probabilities");
    }
    if (!targets) {
        throw std::runtime_error("[computeCrossEntropyBackwardToLogProbs] targets pointer is NULL — caller MUST provide valid target token IDs");
    }
    if (!grad_log_probs) {
        throw std::runtime_error("[computeCrossEntropyBackwardToLogProbs] grad_log_probs pointer is NULL — caller MUST provide output buffer");
    }
    if (num_tokens <= 0) {
        throw std::runtime_error("[computeCrossEntropyBackwardToLogProbs] num_tokens=" + std::to_string(num_tokens) + " — must be > 0");
    }
    if (vocab_size <= 0) {
        throw std::runtime_error("[computeCrossEntropyBackwardToLogProbs] vocab_size=" + std::to_string(vocab_size) + " — must be > 0");
    }

    launchCrossEntropyNLLBackward(
        log_probs,
        targets,
        grad_log_probs,
        num_tokens,
        vocab_size,
        valid_count,
        weight_sum,
        config,
        d_class_weights,
        grad_output_scale,
        stream
    );
}

}  // namespace autograd
}  // namespace GRIM
