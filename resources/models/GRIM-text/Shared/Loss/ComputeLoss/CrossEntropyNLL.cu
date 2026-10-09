//======================================================//
//  CrossEntropyNLL.cu
//  Cross-entropy-family NLL kernels for unified autograd loss
//
//  Implements: Cross Entropy + Label Smoothing + Focal Loss + Entropy Regularization
//  Input contract: receives log_probs = log_softmax(logits), NOT raw logits.
//======================================================//

#include "CrossEntropyNLL.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <type_traits>

namespace GRIM {
namespace autograd {
namespace {

static_assert(std::is_trivially_copyable<HyperParameters::LossConfigHP>::value,
              "LossConfigHP must remain device-passable by value");

void checkCudaStatus(cudaError_t err, const char* caller, const char* operation) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("[") + caller + "] " + operation + " failed: " + cudaGetErrorString(err));
    }
}

void checkKernelLaunch(const char* caller, const char* kernel_name) {
    const std::string operation = std::string(kernel_name) + " launch";
    checkCudaStatus(cudaGetLastError(), caller, operation.c_str());
}

void requireValidStream(cudaStream_t stream, const char* caller) {
    if (!stream) {
        throw std::runtime_error(std::string("[") + caller + "] stream is NULL — caller MUST provide a valid CUDA stream");
    }
}

void validateLossConfigForCompute(
    const HyperParameters::LossConfigHP& config,
    const float* d_class_weights,
    const char* caller
) {
    if (config.cuda_block_size < 32 || config.cuda_block_size > 1024 || config.cuda_block_size % 32 != 0) {
        throw std::runtime_error(std::string("[") + caller + "] cuda_block_size must be a multiple of 32 in [32, 1024]");
    }
    if (config.entropy_reg_enabled && config.entropy_reg_lambda == 0.0f) {
        throw std::runtime_error(std::string("[") + caller + "] entropy_reg_enabled=true but entropy_reg_lambda is 0");
    }
    if (!config.entropy_reg_enabled && config.entropy_reg_lambda != 0.0f) {
        throw std::runtime_error(std::string("[") + caller + "] entropy_reg_enabled=false but entropy_reg_lambda=" +
            std::to_string(config.entropy_reg_lambda) + " — disable by setting lambda to 0");
    }
    if (config.class_balanced_enabled && !d_class_weights) {
        throw std::runtime_error(std::string("[") + caller + "] class_balanced_enabled=true but d_class_weights is NULL");
    }
    if (!config.class_balanced_enabled && d_class_weights) {
        throw std::runtime_error(std::string("[") + caller + "] d_class_weights is non-NULL while class_balanced_enabled=false");
    }
}

const char* targetSelectionName(
    const CrossEntropyTargetSelection& target_selection,
    const char* caller
) {
    switch (target_selection.source) {
        case CrossEntropyTargetSource::PrimaryLm:
            return "primary_lm_targets";
    }

    throw std::runtime_error(std::string("[") + caller + "] CrossEntropyTargetSelection.source contains an unknown value");
}

int expectedValidCountForSelection(
    const Batching::BatchPayload& payload,
    const CrossEntropyTargetSelection& target_selection,
    const char* caller
) {
    switch (target_selection.source) {
        case CrossEntropyTargetSource::PrimaryLm:
            if (payload.lm_valid_tokens <= 0) {
                throw std::runtime_error(std::string("[") + caller + "] BatchPayload.lm_valid_tokens=" +
                    std::to_string(payload.lm_valid_tokens) + " — primary loss requires a positive BatchPayload-authored valid count");
            }
            return payload.lm_valid_tokens;
    }

    throw std::runtime_error(std::string("[") + caller + "] CrossEntropyTargetSelection.source contains an unknown value");
}

const int* resolveDeviceTargetsForSelection(
    const Batching::BatchDeviceBindings& bindings,
    const CrossEntropyTargetSelection& target_selection,
    const char* caller
) {
    switch (target_selection.source) {
        case CrossEntropyTargetSource::PrimaryLm:
            if (!bindings.d_target_ids) {
                throw std::runtime_error(std::string("[") + caller + "] BatchDeviceBindings.d_target_ids is NULL — caller MUST upload BatchPayload before primary loss");
            }
            return bindings.d_target_ids;
    }

    throw std::runtime_error(std::string("[") + caller + "] CrossEntropyTargetSelection.source contains an unknown value");
}

void validateForwardWorkspace(
    const CrossEntropyForwardWorkspace& workspace,
    const HyperParameters::LossConfigHP& config,
    cudaStream_t stream,
    const char* caller
) {
    if (!workspace.loss_sum) {
        throw std::runtime_error(std::string("[") + caller + "] workspace.loss_sum is NULL — TextLossGradFn::capture_inputs MUST provide CE reduction scratch");
    }
    if (!workspace.valid_count) {
        throw std::runtime_error(std::string("[") + caller + "] workspace.valid_count is NULL — TextLossGradFn::capture_inputs MUST provide CE reduction scratch");
    }
    if (!workspace.weight_sum) {
        throw std::runtime_error(std::string("[") + caller + "] workspace.weight_sum is NULL — TextLossGradFn::capture_inputs MUST provide CE reduction scratch");
    }
    if (workspace.loss_sum_bytes < sizeof(float)) {
        throw std::runtime_error(std::string("[") + caller + "] workspace.loss_sum_bytes is smaller than sizeof(float)");
    }
    if (workspace.valid_count_bytes < sizeof(int)) {
        throw std::runtime_error(std::string("[") + caller + "] workspace.valid_count_bytes is smaller than sizeof(int)");
    }
    if (workspace.weight_sum_bytes < sizeof(float)) {
        throw std::runtime_error(std::string("[") + caller + "] workspace.weight_sum_bytes is smaller than sizeof(float)");
    }
    if (!workspace.owner_stream) {
        throw std::runtime_error(std::string("[") + caller + "] workspace.owner_stream is NULL");
    }
    if (workspace.owner_stream != stream) {
        throw std::runtime_error(std::string("[") + caller + "] workspace.owner_stream does not match compute stream");
    }

    const auto loss_addr = reinterpret_cast<std::uintptr_t>(workspace.loss_sum);
    const auto count_addr = reinterpret_cast<std::uintptr_t>(workspace.valid_count);
    const auto weight_addr = reinterpret_cast<std::uintptr_t>(workspace.weight_sum);
    if (loss_addr == count_addr || loss_addr == weight_addr || count_addr == weight_addr) {
        throw std::runtime_error(std::string("[") + caller + "] CE workspace scalar buffers alias each other");
    }

    if (config.class_balanced_enabled && !workspace.weight_sum) {
        throw std::runtime_error(std::string("[") + caller + "] class_balanced_enabled=true but workspace.weight_sum is NULL");
    }
}

__device__ __forceinline__ void trapInvalidCrossEntropyInput() {
    asm volatile("trap;");
}

//========================================================================
// CUDA Kernels — NLL Loss on log-probabilities
// These kernels receive log_probs = log_softmax(logits), NOT raw logits.
// The loss owns the single log-softmax forward result used by backward.
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
    float* __restrict__ loss_sum,
    int* __restrict__ valid_count,
    float* __restrict__ weight_sum,             // Accumulated class weights (nullable if no class_balanced)
    int total_tokens,
    int vocab_size,
    HyperParameters::LossConfigHP loss_config,
    const float* __restrict__ class_weights     // [vocab_size] per-class weights (nullable = all 1.0)
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= total_tokens) return;

    const int target = targets[token_idx];

    if (target < -1 || target >= vocab_size) {
        trapInvalidCrossEntropyInput();
        return;
    }

    // Skip masked/padding positions (target == -1)
    if (target == -1) {
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
        if (loss_config.entropy_reg_enabled) {
            const float p_v = expf(row[v]);
            if (p_v > 0.0f) {
                local_neg_entropy += p_v * row[v];  // p * log(p)
            }
        }
        if (loss_config.smoothing_enabled && v != target) {
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
        if (loss_config.smoothing_enabled && vocab_size > 1) {
            const float q_on  = 1.0f - loss_config.smoothing_epsilon;
            const float q_off = loss_config.smoothing_epsilon / static_cast<float>(vocab_size - 1);
            ce_smooth = -q_on * log_p_t - q_off * s_sum_log_off;
        } else {
            ce_smooth = -log_p_t;  // Standard CE: -log(p_target)
        }

        // Focal loss weight
        float focal_weight = 1.0f;
        if (loss_config.focal_enabled) {
            if (loss_config.focal_gamma == 0.0f) {
                focal_weight = 1.0f;
            } else {
                focal_weight = powf(fmaxf(1.0f - p_t, 0.0f), loss_config.focal_gamma);
            }
        }

        // When focal is disabled, focal_alpha MUST NOT scale the CE term.
        // Otherwise flipping focal_enabled=false silently changes the CE:entropy
        // ratio via a parameter whose name says "focal" — a quality footgun.
        float ce_loss = ce_smooth;
        if (loss_config.focal_enabled) {
            ce_loss = loss_config.focal_alpha * focal_weight * ce_smooth;
        }
        float entropy_loss = 0.0f;
        if (loss_config.entropy_reg_enabled) {
            entropy_loss = loss_config.entropy_reg_lambda * s_neg_entropy;
        }
        const float total_loss = ce_loss + entropy_loss;

        // Class-balanced weighting: w_{y_t} scales the ENTIRE loss for this position
        // This weights the whole gradient row grad_logits[t, :] = w * (p - q) / W
        float cw = 1.0f;
        if (class_weights != nullptr) {
            cw = class_weights[target];
            if (!isfinite(cw) || cw <= 0.0f) {
                trapInvalidCrossEntropyInput();
                return;
            }
        }
        const float weighted_loss = cw * total_loss;

        atomicAdd(loss_sum, weighted_loss);
        atomicAdd(valid_count, 1);
        if (weight_sum != nullptr) {
            atomicAdd(weight_sum, cw);
        }
    }
}

// Evaluate the original NLL derivative in registers. Both fused passes use
// identical arithmetic, including scaling before the log-softmax row reduction.
__device__ __forceinline__ float crossEntropyLogProbGradient(
    const float* row, int v, int target, float q_on, float q_off,
    float p_t, float focal_weight, float focal_deriv_factor, float sum_log_off,
    HyperParameters::LossConfigHP loss_config, float cw, float grad_output_inv_norm) {
    float q_v = q_off;
    if (v == target) {
        q_v = q_on;
    }

    // Base: -q_v (plain NLL loss gradient w.r.t. log_probs)
    // When focal is disabled, focal_alpha MUST NOT scale the gradient.
    float grad_v = -q_v;
    if (loss_config.focal_enabled) {
        grad_v = loss_config.focal_alpha * focal_weight * (-q_v);
    }

    // Focal derivative: only affects gradient through p_t = exp(log_p_t)
    if (loss_config.focal_enabled) {
        float ce_smooth = -row[target];
        if (loss_config.smoothing_enabled) {
            ce_smooth = -(q_on * row[target] + q_off * sum_log_off);
        }
        if (v == target) {
            grad_v += loss_config.focal_alpha * (-focal_deriv_factor) * p_t * ce_smooth;
        }
    }

    // Entropy regularization gradient w.r.t. log_probs.
    // The fused second pass applies the log-softmax Jacobian; this is the
    // derivative of Σ p_i log(p_i) w.r.t. log_probs_i.
    if (loss_config.entropy_reg_enabled) {
        const float p_v = expf(row[v]);
        if (p_v > 0.0f) {
            grad_v += loss_config.entropy_reg_lambda * p_v * (row[v] + 1.0f);
        }
    }

    return grad_v * cw * grad_output_inv_norm;
}

// Fused NLL + log-softmax backward: saved log_probs -> accumulated dL/dlogits.
__global__ void kernelCrossEntropyLogitsBackward(
    const float* __restrict__ log_probs,    // [num_tokens, vocab_size]
    const int* __restrict__ targets,
    float* __restrict__ grad_logits,     // [num_tokens, vocab_size] — OUTPUT
    int total_tokens,
    int vocab_size,
    float grad_output_inv_norm,  // grad_output_scale / N, or grad_output_scale / W when class_balanced
    HyperParameters::LossConfigHP loss_config,
    const float* __restrict__ class_weights     // [vocab_size] per-class weights (nullable = all 1.0)
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= total_tokens) return;

    const float* row = log_probs + static_cast<size_t>(token_idx) * vocab_size;
    float* grad_row = grad_logits + static_cast<size_t>(token_idx) * vocab_size;
    const int target = targets[token_idx];

    if (target < -1 || target >= vocab_size) {
        trapInvalidCrossEntropyInput();
        return;
    }

    // Skip masked/padding positions (target == -1)
    if (target == -1) {
        // Zero contribution: preserve any gradient already in the destination.
        return;
    }

    // Label smoothing targets
    float q_on = 1.0f;
    float q_off = 0.0f;
    if (loss_config.smoothing_enabled && vocab_size > 1) {
        q_on = 1.0f - loss_config.smoothing_epsilon;
        q_off = loss_config.smoothing_epsilon / static_cast<float>(vocab_size - 1);
    }

    // Focal precomputation
    const float p_t = expf(row[target]);
    float focal_weight = 1.0f;
    float focal_deriv_factor = 0.0f;
    if (loss_config.focal_enabled) {
        const float one_minus_pt = fmaxf(1.0f - p_t, 0.0f);
        if (loss_config.focal_gamma == 0.0f) {
            focal_weight = 1.0f;
            focal_deriv_factor = 0.0f;
        } else if (one_minus_pt > 0.0f) {
            focal_weight = powf(one_minus_pt, loss_config.focal_gamma);
            focal_deriv_factor = loss_config.focal_gamma * powf(one_minus_pt, loss_config.focal_gamma - 1.0f);
        } else {
            focal_weight = 0.0f;
            focal_deriv_factor = 0.0f;
        }
    }

    // Precompute sum_log_off if needed for focal derivative with smoothing.
    __shared__ float s_sum_log_off;
    if (threadIdx.x == 0) s_sum_log_off = 0.0f;
    __syncthreads();

    if (loss_config.focal_enabled && loss_config.smoothing_enabled) {
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

    // cw depends only on target, not on v — hoist out of the vocab loop.
    float cw = 1.0f;
    if (class_weights != nullptr) {
        cw = class_weights[target];
        if (!isfinite(cw) || cw <= 0.0f) {
            trapInvalidCrossEntropyInput();
            return;
        }
    }

    // First pass: reduce dL/d(log_p) without materializing a vocabulary-sized
    // gradient. One slot per warp supports all validated launch geometries.
    __shared__ float warp_sums[32];
    __shared__ float sum_grad;
    float local_sum = 0.0f;
    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        local_sum += crossEntropyLogProbGradient(row, v, target, q_on, q_off,
            p_t, focal_weight, focal_deriv_factor, sum_log_off, loss_config, cw, grad_output_inv_norm);
    }
    for (int off = warpSize / 2; off > 0; off >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, off);
    const int warp = threadIdx.x / warpSize;
    if (threadIdx.x % warpSize == 0) warp_sums[warp] = local_sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int w = 0; w < blockDim.x / warpSize; ++w) total += warp_sums[w];
        sum_grad = total;
    }
    __syncthreads();

    // Second pass: compose the log-softmax Jacobian and add to the destination.
    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        const float grad_v = crossEntropyLogProbGradient(row, v, target, q_on, q_off,
            p_t, focal_weight, focal_deriv_factor, sum_log_off, loss_config, cw, grad_output_inv_norm);
        grad_row[v] += grad_v - expf(row[v]) * sum_grad;
    }
}

void launchCrossEntropyNLLForward(
    const float* log_probs,
    const int* targets,
    float* loss_sum,
    int* valid_count,
    float* weight_sum,
    const Batching::BatchPayload& payload,
    const HyperParameters::LossConfigHP& config,
    const float* d_class_weights,
    cudaStream_t stream
) {
    const char* caller = "launchCrossEntropyNLLForward";
    requireValidStream(stream, caller);
    if (!log_probs) throw std::runtime_error("[launchCrossEntropyNLLForward] log_probs is NULL");
    if (!targets) throw std::runtime_error("[launchCrossEntropyNLLForward] targets is NULL");
    if (!loss_sum) throw std::runtime_error("[launchCrossEntropyNLLForward] loss_sum is NULL");
    if (!valid_count) throw std::runtime_error("[launchCrossEntropyNLLForward] valid_count is NULL");
    if (config.class_balanced_enabled && !weight_sum) throw std::runtime_error("[launchCrossEntropyNLLForward] class_balanced_enabled=true but weight_sum is NULL");
    if (config.class_balanced_enabled && !d_class_weights) throw std::runtime_error("[launchCrossEntropyNLLForward] class_balanced_enabled=true but d_class_weights is NULL");
    if (!config.class_balanced_enabled && d_class_weights) throw std::runtime_error("[launchCrossEntropyNLLForward] d_class_weights is non-NULL while class_balanced_enabled=false");

    checkCudaStatus(cudaMemsetAsync(loss_sum, 0, sizeof(float), stream), caller, "cudaMemsetAsync(loss_sum)");
    checkCudaStatus(cudaMemsetAsync(valid_count, 0, sizeof(int), stream), caller, "cudaMemsetAsync(valid_count)");
    if (weight_sum) checkCudaStatus(cudaMemsetAsync(weight_sum, 0, sizeof(float), stream), caller, "cudaMemsetAsync(weight_sum)");

    kernelCrossEntropyNLLForward<<<payload.total_tokens, config.cuda_block_size, 0, stream>>>(
        log_probs, targets,
        loss_sum, valid_count, weight_sum,
        payload.total_tokens,
        payload.vocab_size,
        config,
        d_class_weights
    );
    checkKernelLaunch(caller, "kernelCrossEntropyNLLForward");
}

}  // namespace

CrossEntropyForwardResult computeCrossEntropyForwardFromLogProbs(
    const float* log_probs,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    const CrossEntropyTargetSelection& target_selection,
    const CrossEntropyForwardWorkspace& workspace,
    const HyperParameters::LossConfigHP& config,
    const float* d_class_weights,
    cudaStream_t stream
) {
    requireValidStream(stream, "computeCrossEntropyForwardFromLogProbs");
    const char* target_name = targetSelectionName(target_selection, "computeCrossEntropyForwardFromLogProbs");
    const int expected_valid_count = expectedValidCountForSelection(payload, target_selection, "computeCrossEntropyForwardFromLogProbs");
    validateLossConfigForCompute(config, d_class_weights, "computeCrossEntropyForwardFromLogProbs");
    validateForwardWorkspace(workspace, config, stream, "computeCrossEntropyForwardFromLogProbs");

    if (!log_probs) {
        throw std::runtime_error("[computeCrossEntropyForwardFromLogProbs] log_probs pointer is NULL — caller MUST provide log_softmax output");
    }

    float* d_loss_sum = workspace.loss_sum;
    int* d_valid_count = workspace.valid_count;
    float* d_weight_sum = config.class_balanced_enabled ? workspace.weight_sum : nullptr;

    launchCrossEntropyNLLForward(
        log_probs,
        resolveDeviceTargetsForSelection(bindings, target_selection, "computeCrossEntropyForwardFromLogProbs"),
        d_loss_sum,
        d_valid_count,
        d_weight_sum,
        payload,
        config,
        d_class_weights,
        stream
    );

    float h_loss_sum = 0.0f;
    int h_valid_count = 0;
    float h_weight_sum = 0.0f;
    checkCudaStatus(cudaMemcpyAsync(&h_loss_sum, d_loss_sum, sizeof(float), cudaMemcpyDeviceToHost, stream),
        "computeCrossEntropyForwardFromLogProbs", "cudaMemcpyAsync(loss_sum D2H)");
    checkCudaStatus(cudaMemcpyAsync(&h_valid_count, d_valid_count, sizeof(int), cudaMemcpyDeviceToHost, stream),
        "computeCrossEntropyForwardFromLogProbs", "cudaMemcpyAsync(valid_count D2H)");
    if (d_weight_sum) {
        checkCudaStatus(cudaMemcpyAsync(&h_weight_sum, d_weight_sum, sizeof(float), cudaMemcpyDeviceToHost, stream),
            "computeCrossEntropyForwardFromLogProbs", "cudaMemcpyAsync(weight_sum D2H)");
    }
    checkCudaStatus(cudaStreamSynchronize(stream), "computeCrossEntropyForwardFromLogProbs", "cudaStreamSynchronize after CE forward readback");

    if (h_valid_count <= 0) {
        throw std::runtime_error("[computeCrossEntropyForwardFromLogProbs] valid_count=0 — no valid tokens in batch. "
            "Check targets for corruption: all targets are -1.");
    }

    if (h_valid_count != expected_valid_count) {
        throw std::runtime_error("[computeCrossEntropyForwardFromLogProbs] kernel valid_count=" +
            std::to_string(h_valid_count) + " != BatchPayload-authored " + target_name +
            ".expected_valid_count=" + std::to_string(expected_valid_count));
    }

    if (!std::isfinite(h_loss_sum)) {
        throw std::runtime_error("[computeCrossEntropyForwardFromLogProbs] h_loss_sum is non-finite (" + std::to_string(h_loss_sum) +
            ") — NLL kernel produced NaN/Inf. valid_count=" + std::to_string(h_valid_count) +
            " weight_sum=" + std::to_string(h_weight_sum) + " focal=" + std::to_string(config.focal_gamma) +
            " smoothing=" + std::to_string(config.smoothing_epsilon) + " entropy_lambda=" + std::to_string(config.entropy_reg_lambda));
    }

    float normalization = static_cast<float>(h_valid_count);
    if (config.class_balanced_enabled) {
        if (h_weight_sum <= 0.0f || !std::isfinite(h_weight_sum)) {
            throw std::runtime_error("[computeCrossEntropyForwardFromLogProbs] class_balanced_enabled=true but h_weight_sum=" +
                std::to_string(h_weight_sum));
        }
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

void computeCrossEntropyBackwardToLogits(
    const float* log_probs,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    const CrossEntropyTargetSelection& target_selection,
    float* grad_logits,
    int valid_count,
    float weight_sum,
    const HyperParameters::LossConfigHP& config,
    const float* d_class_weights,
    float grad_output_scale,
    cudaStream_t stream
) {
    requireValidStream(stream, "computeCrossEntropyBackwardToLogits");
    const char* target_name = targetSelectionName(target_selection, "computeCrossEntropyBackwardToLogits");
    const int expected_valid_count = expectedValidCountForSelection(payload, target_selection, "computeCrossEntropyBackwardToLogits");
    validateLossConfigForCompute(config, d_class_weights, "computeCrossEntropyBackwardToLogits");

    if (!log_probs) {
        throw std::runtime_error("[computeCrossEntropyBackwardToLogits] log_probs pointer is NULL — caller MUST provide saved log-probabilities");
    }
    if (!grad_logits) {
        throw std::runtime_error("[computeCrossEntropyBackwardToLogits] grad_logits pointer is NULL — caller MUST provide output buffer");
    }
    if (valid_count != expected_valid_count) {
        throw std::runtime_error("[computeCrossEntropyBackwardToLogits] saved valid_count=" +
            std::to_string(valid_count) + " != BatchPayload-authored " + target_name +
            ".expected_valid_count=" + std::to_string(expected_valid_count));
    }
    if (config.class_balanced_enabled && (weight_sum <= 0.0f || !std::isfinite(weight_sum))) {
        throw std::runtime_error("[computeCrossEntropyBackwardToLogits] class_balanced_enabled=true but saved weight_sum=" +
            std::to_string(weight_sum));
    }

    const int* targets = resolveDeviceTargetsForSelection(
        bindings, target_selection, "computeCrossEntropyBackwardToLogits");
    const float normalization = config.class_balanced_enabled
        ? weight_sum : static_cast<float>(valid_count);
    const float grad_output_inv_norm = grad_output_scale / normalization;

    kernelCrossEntropyLogitsBackward<<<payload.total_tokens, config.cuda_block_size, 0, stream>>>(
        log_probs, targets, grad_logits,
        payload.total_tokens,
        payload.vocab_size,
        grad_output_inv_norm,
        config,
        d_class_weights
    );
    checkKernelLaunch("computeCrossEntropyBackwardToLogits", "kernelCrossEntropyLogitsBackward");
}

}  // namespace autograd
}  // namespace GRIM
