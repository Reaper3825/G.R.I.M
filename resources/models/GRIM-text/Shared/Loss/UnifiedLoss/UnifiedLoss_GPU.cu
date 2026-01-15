/**
 * @file UnifiedLoss_GPU.cu
 * @brief Unified loss kernel: Focal + Label Smoothing + Cross Entropy
 *
 * Implementation goals:
 *  - Single softmax exponentiation pass per token
 *  - No recomputed softmax in gradient loop
 *  - Full telemetry with strict error handling
 */

#include "UnifiedLoss_GPU.hpp"
#include "../../HyperParameters/HyperParameters_GPU.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace GRIM::Loss {
namespace {

constexpr int kBlockSize = HyperParameters::CUDA_BLOCK_SIZE_STANDARD;
constexpr float kEpsilon = HyperParameters::EPSILON_LOG_PROB;
constexpr int kDebugSamples = 10;
constexpr int kDebugTokenId = 277;

__device__ __forceinline__ float clampProb(float value) {
    return fminf(fmaxf(value, kEpsilon), 1.0f - kEpsilon);
}

__device__ __forceinline__ void atomicMaxFloat(float* addr, float value) {
    int* addr_as_int = reinterpret_cast<int*>(addr);
    int old = *addr_as_int;
    int assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
                        __float_as_int(fmaxf(value, __int_as_float(assumed))));
    } while (assumed != old);
}

__device__ __forceinline__ void atomicMinFloat(float* addr, float value) {
    int* addr_as_int = reinterpret_cast<int*>(addr);
    int old = *addr_as_int;
    int assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
                        __float_as_int(fminf(value, __int_as_float(assumed))));
    } while (assumed != old);
}

__device__ __forceinline__ void zeroVector(float* data, int count) {
    for (int i = 0; i < count; ++i) {
        data[i] = 0.0f;
    }
}

__global__ void unifiedLossKernelV2(
    const float* __restrict__ logits,
    const int* __restrict__ targets,
    const float* __restrict__ sequence_weights,
    float* __restrict__ token_losses,
    float* __restrict__ grad_logits,
    float* __restrict__ loss_sum,
    DeviceTelemetryAccum* __restrict__ telemetry,
    int total_tokens,
    int vocab_size,
    int seq_len,
    int weight_count,
    float focal_alpha,
    float focal_gamma,
    float smoothing_epsilon,
    bool strict_mode
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_tokens) {
        return;
    }

    const int target = targets[idx];
    const int offset = idx * vocab_size;

    if (target == -1) {
        token_losses[idx] = 0.0f;
        zeroVector(grad_logits + offset, vocab_size);
        atomicAdd(&telemetry->masked_count, 1u);
        return;
    }
    if (target < -1 || target >= vocab_size) {
        token_losses[idx] = 0.0f;
        zeroVector(grad_logits + offset, vocab_size);
        atomicAdd(&telemetry->invalid_target_count, 1u);
        return;
    }

float sample_weight = 1.0f;

if (sequence_weights) {
    const int seq_idx = idx / seq_len;
    if ((unsigned)seq_idx < (unsigned)weight_count) {
        sample_weight = sequence_weights[seq_idx];
    }
}


    const float* token_logits = logits + offset;
    float* token_grads = grad_logits + offset;

    float max_logit = token_logits[0];
for (int i = 1; i < vocab_size; ++i) {
    max_logit = fmaxf(max_logit, token_logits[i]);
}

  


    if (!isfinite(max_logit)) {
        if (strict_mode) {
            atomicAdd(&telemetry->nan_count, 1u);
        }
        atomicAdd(&telemetry->exit_max_logit_nan, 1u);
        token_losses[idx] = 0.0f;
        zeroVector(token_grads, vocab_size);
        return;
    }

    float sum_exp = 0.0f;
    float exp_target = 0.0f;
    float exp_token_277 = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        const float ex = expf(token_logits[i] - max_logit);
        token_grads[i] = ex;  // temporary storage for exp(logit - max)
        sum_exp += ex;
        if (i == target) {
            exp_target = ex;
        }
        if (i == kDebugTokenId) {
            exp_token_277 = ex;
        }
    }

    if (!(sum_exp > kEpsilon) || !isfinite(sum_exp)) {
        if (strict_mode) {
            atomicAdd(&telemetry->inf_count, 1u);
        }
        atomicAdd(&telemetry->exit_sum_exp_zero, 1u);
        token_losses[idx] = 0.0f;
        zeroVector(token_grads, vocab_size);
        return;
    }

    const float inv_sum_exp = 1.0f / sum_exp;
    const float log_sum_exp = logf(sum_exp) + max_logit;
    const float target_logit = token_logits[target];

    float p_t = exp_target * inv_sum_exp;
    p_t = clampProb(p_t);
    float p_277 = -1.0f;
    if (kDebugTokenId >= 0 && kDebugTokenId < vocab_size) {
        p_277 = exp_token_277 * inv_sum_exp;
        p_277 = clampProb(p_277);
    }

    const float log_p_t = target_logit - log_sum_exp;
    const float one_minus_p_t = 1.0f - p_t;

    float q_on = 1.0f;
    float q_off = 0.0f;
    if (vocab_size > 1 && smoothing_epsilon > 0.0f) {
        q_on = 1.0f - smoothing_epsilon;
        q_off = smoothing_epsilon / static_cast<float>(vocab_size - 1);
    }

float ce_smooth = -q_on * log_p_t;
    if (vocab_size > 1 && smoothing_epsilon > 0.0f) {
        float sum_log_p_off = 0.0f;
        for (int i = 0; i < vocab_size; ++i) {
            if (i != target) {
                sum_log_p_off += (token_logits[i] - log_sum_exp);
            }
        }
        ce_smooth -= q_off * sum_log_p_off;
    }

    float focal_weight = 1.0f;
    if (focal_gamma > 0.0f) {
        focal_weight = powf(one_minus_p_t, focal_gamma);
    }

    const float loss = focal_alpha * focal_weight * ce_smooth * sample_weight;
    if (!isfinite(loss)) {
        if (strict_mode) {
            atomicAdd(&telemetry->nan_count, 1u);
        }
        atomicAdd(&telemetry->exit_loss_nan, 1u);
        token_losses[idx] = 0.0f;
        zeroVector(token_grads, vocab_size);
        return;
    }

    const uint32_t debug_slot = atomicAdd(&telemetry->debug_count, 1u);
    if (debug_slot < kDebugSamples) {
        telemetry->debug_p_t_10[debug_slot] = p_t;
        telemetry->debug_ce_smooth_10[debug_slot] = ce_smooth;
    }
    if (debug_slot == 0) {
        telemetry->debug_max_logit = max_logit;
        telemetry->debug_sum_exp = sum_exp;
        telemetry->debug_p_t = p_t;
        telemetry->debug_p_277 = p_277;
        telemetry->debug_ce_smooth = ce_smooth;
        telemetry->debug_focal_weight = focal_weight;
        telemetry->debug_sample_weight = sample_weight;
        telemetry->debug_loss = loss;
        telemetry->debug_focal_alpha = focal_alpha;
        telemetry->debug_target = target + 1;
    }

    token_losses[idx] = loss;
    atomicAdd(loss_sum, loss);
    atomicAdd(&telemetry->exit_success, 1u);


    float grad_norm_sq = 0.0f;
    float focal_base = focal_alpha;
    if (focal_gamma > 0.0f) {
        const float denom = fmaxf(one_minus_p_t, kEpsilon);
        focal_base = focal_alpha * (focal_weight / denom);
    }
    const float focal_ce_term = focal_gamma * p_t * ce_smooth;

    for (int i = 0; i < vocab_size; ++i) {
        const float p_i = token_grads[i] * inv_sum_exp;
        const float q_i = (i == target) ? q_on : q_off;
        const float delta_it = (i == target) ? 1.0f : 0.0f;
        const float ce_grad = p_i - q_i;

        float grad;
        if (focal_gamma > 0.0f) {
            grad = focal_base * (one_minus_p_t * ce_grad + focal_ce_term * (p_i - delta_it));
        } else {
            grad = focal_alpha * ce_grad;
        }
        grad *= sample_weight;
        token_grads[i] = grad;
        grad_norm_sq += grad * grad;
    }

    const float grad_norm = sqrtf(grad_norm_sq);

    // Track grad_logit[277] breakdown by whether this position targets 277
    const float grad_277 = token_grads[kDebugTokenId];
    atomicAdd(&telemetry->grad_277_sum, grad_277);
    if (target == kDebugTokenId) {
        atomicAdd(&telemetry->grad_277_sum_target, grad_277);  // Should be negative (p_277 - 1)
        atomicAdd(&telemetry->target_277_count, 1u);
    } else {
        atomicAdd(&telemetry->grad_277_sum_nontarget, grad_277);  // Will be positive (p_277)
    }

    atomicAdd(&telemetry->loss_sum, loss);
    atomicAdd(&telemetry->loss_sq_sum, loss * loss);
    atomicMaxFloat(&telemetry->loss_max, loss);
    atomicMinFloat(&telemetry->loss_min, loss);
    atomicAdd(&telemetry->grad_norm_sum, grad_norm);
    atomicMaxFloat(&telemetry->grad_norm_max, grad_norm);
    atomicAdd(&telemetry->focal_weight_sum, focal_weight);
    atomicAdd(&telemetry->valid_count, 1u);
    if (p_t < 0.5f) {
        atomicAdd(&telemetry->hard_example_count, 1u);
    }
}

inline dim3 launchGrid(int total_threads) {
    return dim3((total_threads + kBlockSize - 1) / kBlockSize);
}

} // namespace

const char* getErrorMessage(int32_t error_code) {
    switch (error_code) {
        case UnifiedLossTelemetry::OK:               return "Success";
        case UnifiedLossTelemetry::ERR_NULL_LOGITS:  return "FATAL: logits pointer is NULL";
        case UnifiedLossTelemetry::ERR_NULL_TARGETS: return "FATAL: targets pointer is NULL";
        case UnifiedLossTelemetry::ERR_NULL_OUTPUTS: return "FATAL: output buffer is NULL";
        case UnifiedLossTelemetry::ERR_INVALID_DIMS: return "FATAL: invalid dimensions";
        case UnifiedLossTelemetry::ERR_NAN_IN_LOGITS:return "FATAL: NaN detected in logits";
        case UnifiedLossTelemetry::ERR_NAN_IN_LOSS:  return "FATAL: NaN detected in loss";
        case UnifiedLossTelemetry::ERR_INF_IN_LOSS:  return "FATAL: Inf detected in loss";
        case UnifiedLossTelemetry::ERR_KERNEL_LAUNCH:return "FATAL: CUDA kernel launch failed";
        case UnifiedLossTelemetry::ERR_CUDA_SYNC:    return "FATAL: CUDA synchronization failed";
        case UnifiedLossTelemetry::ERR_INVALID_WEIGHTS: return "FATAL: invalid sequence weight configuration";
        case UnifiedLossTelemetry::ERR_INVALID_TARGET:  return "FATAL: invalid target token id";
        default: return "Unknown error";
    }
}

UnifiedLossContext::UnifiedLossContext() {
    allocateGPU();
}

UnifiedLossContext::~UnifiedLossContext() {
    freeGPU();
}

UnifiedLossContext::UnifiedLossContext(UnifiedLossContext&& other) noexcept
    : d_telemetry_(other.d_telemetry_) {
    other.d_telemetry_ = nullptr;
}

UnifiedLossContext& UnifiedLossContext::operator=(UnifiedLossContext&& other) noexcept {
    if (this != &other) {
        freeGPU();
        d_telemetry_ = other.d_telemetry_;
        other.d_telemetry_ = nullptr;
    }
    return *this;
}

void UnifiedLossContext::allocateGPU() {
    if (d_telemetry_) {
        return;
    }
    cudaError_t err = cudaMalloc(&d_telemetry_, sizeof(DeviceTelemetryAccum));
    if (err != cudaSuccess) {
        fprintf(stderr, "[UnifiedLoss] Failed to allocate telemetry buffer: %s\n",
                cudaGetErrorString(err));
        throw std::runtime_error("UnifiedLoss GPU allocation failed");
    }
}

void UnifiedLossContext::freeGPU() {
    if (d_telemetry_) {
        cudaFree(d_telemetry_);
        d_telemetry_ = nullptr;
    }
}

UnifiedLossTelemetry UnifiedLossContext::compute(
    const UnifiedLossConfig& config,
    const UnifiedLossInputs& inputs,
    UnifiedLossOutputs& outputs
) {
    UnifiedLossTelemetry result = {};

    if (!inputs.logits) {
        result.error_code = UnifiedLossTelemetry::ERR_NULL_LOGITS;
        fprintf(stderr, "[UnifiedLoss] %s\n", getErrorMessage(result.error_code));
        return result;
    }
    if (!inputs.targets) {
        result.error_code = UnifiedLossTelemetry::ERR_NULL_TARGETS;
        fprintf(stderr, "[UnifiedLoss] %s\n", getErrorMessage(result.error_code));
        return result;
    }
    if (!outputs.token_losses || !outputs.grad_logits || !outputs.loss_sum) {
        result.error_code = UnifiedLossTelemetry::ERR_NULL_OUTPUTS;
        fprintf(stderr, "[UnifiedLoss] %s (losses=%p, grads=%p, sum=%p)\n",
                getErrorMessage(result.error_code),
                outputs.token_losses, outputs.grad_logits, outputs.loss_sum);
        return result;
    }
    if (inputs.batch_size <= 0 || inputs.seq_len <= 0 || inputs.vocab_size <= 0) {
        result.error_code = UnifiedLossTelemetry::ERR_INVALID_DIMS;
        fprintf(stderr, "[UnifiedLoss] %s (batch=%d, seq=%d, vocab=%d)\n",
                getErrorMessage(result.error_code),
                inputs.batch_size, inputs.seq_len, inputs.vocab_size);
        return result;
    }
    if (inputs.weight_count < 0) {
        result.error_code = UnifiedLossTelemetry::ERR_INVALID_WEIGHTS;
        fprintf(stderr, "[UnifiedLoss] %s (weight_count=%d)\n",
                getErrorMessage(result.error_code),
                inputs.weight_count);
        return result;
    }
    if (inputs.sequence_weights) {
        if (inputs.weight_count <= 0) {
            result.error_code = UnifiedLossTelemetry::ERR_INVALID_WEIGHTS;
            fprintf(stderr, "[UnifiedLoss] %s (sequence_weights set but weight_count=%d)\n",
                    getErrorMessage(result.error_code),
                    inputs.weight_count);
            return result;
        }
        if (inputs.weight_count != inputs.batch_size) {
            result.error_code = UnifiedLossTelemetry::ERR_INVALID_WEIGHTS;
            fprintf(stderr, "[UnifiedLoss] %s (weight_count=%d, batch_size=%d)\n",
                    getErrorMessage(result.error_code),
                    inputs.weight_count, inputs.batch_size);
            return result;
        }
    } else if (inputs.weight_count != 0) {
        result.error_code = UnifiedLossTelemetry::ERR_INVALID_WEIGHTS;
        fprintf(stderr, "[UnifiedLoss] %s (sequence_weights null, weight_count=%d)\n",
                getErrorMessage(result.error_code),
                inputs.weight_count);
        return result;
    }

    const int total_tokens = inputs.batch_size * inputs.seq_len;

    const float focal_alpha = (config.focal_enabled && config.focal_alpha > 0.0f)
        ? config.focal_alpha : 1.0f;
    const float focal_gamma = config.focal_enabled ? config.focal_gamma : 0.0f;
    const float smoothing_epsilon = config.smoothing_enabled
        ? fminf(fmaxf(config.smoothing_epsilon, 0.0f), 0.5f)
        : 0.0f;

    DeviceTelemetryAccum init_telemetry = {};
    init_telemetry.loss_max = -FLT_MAX;
    init_telemetry.loss_min = FLT_MAX;
    init_telemetry.grad_norm_max = 0.0f;
    cudaMemcpyAsync(d_telemetry_, &init_telemetry, sizeof(DeviceTelemetryAccum),
                    cudaMemcpyHostToDevice, inputs.stream);
    cudaMemsetAsync(outputs.loss_sum, 0, sizeof(float), inputs.stream);

    const int weight_count = inputs.weight_count;

    unifiedLossKernelV2<<<launchGrid(total_tokens), kBlockSize, 0, inputs.stream>>>(
        inputs.logits,
        inputs.targets,
        inputs.sequence_weights,
        outputs.token_losses,
        outputs.grad_logits,
        outputs.loss_sum,
        d_telemetry_,
        total_tokens,
        inputs.vocab_size,
        inputs.seq_len,
        weight_count,
        focal_alpha,
        focal_gamma,
        smoothing_epsilon,
        config.strict_mode
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        result.error_code = UnifiedLossTelemetry::ERR_KERNEL_LAUNCH;
        fprintf(stderr, "[UnifiedLoss] Kernel launch failed: %s\n",
                cudaGetErrorString(err));
        return result;
    }

    DeviceTelemetryAccum host_telemetry;
    cudaMemcpyAsync(&host_telemetry, d_telemetry_, sizeof(DeviceTelemetryAccum),
                    cudaMemcpyDeviceToHost, inputs.stream);

    if (config.strict_mode) {
        cudaError_t sync_err = cudaStreamSynchronize(inputs.stream);
        if (sync_err != cudaSuccess) {
            result.error_code = UnifiedLossTelemetry::ERR_CUDA_SYNC;
            fprintf(stderr, "[UnifiedLoss] Stream sync failed: %s\n",
                    cudaGetErrorString(sync_err));
            return result;
        }
    }

    static bool logged_debug = true;  // TEMPORARILY ENABLED for debugging
    if (logged_debug) {
        logged_debug = true;  // Only print once
        const float expected_loss = host_telemetry.debug_focal_alpha *
            host_telemetry.debug_focal_weight *
            host_telemetry.debug_ce_smooth *
            host_telemetry.debug_sample_weight;

        fprintf(stderr, "  COMPUTED: alpha*fw*ce*sw = %.6f * %.9f * %.6f * %.6f = %.9f\n",
                host_telemetry.debug_focal_alpha,
                host_telemetry.debug_focal_weight,
                host_telemetry.debug_ce_smooth,
                host_telemetry.debug_sample_weight,
                expected_loss);
        
        // Detailed CE breakdown: ce_smooth = -q_on * log_p_t - q_off * sum_log_p_off
        // For standard training (no label smoothing): ce_smooth = -log(p_t) = -(target_logit - log_sum_exp)
        // p_t = exp(target_logit) / sum(exp(logits)) = softmax probability of correct token
        // log_p_t = target_logit - log_sum_exp
        // Random baseline: p_t ≈ 1/vocab_size → ce ≈ ln(vocab_size) ≈ 10.83 for vocab=50376
        fprintf(stderr, "  CE BREAKDOWN: ce_smooth=%.6f comes from:\n", host_telemetry.debug_ce_smooth);
        fprintf(stderr, "    p_t=%.9f (probability model assigned to correct token)\n", host_telemetry.debug_p_t);
        fprintf(stderr, "    -log(p_t)=%.6f (cross-entropy = negative log probability)\n", -logf(fmaxf(host_telemetry.debug_p_t, 1e-10f)));
        if (host_telemetry.debug_p_277 >= 0.0f) {
            fprintf(stderr, "    p_277=%.9f (probability model assigned to token 277)\n", host_telemetry.debug_p_277);
            fprintf(stderr, "    -log(p_277)=%.6f (cross-entropy if target=277)\n", -logf(fmaxf(host_telemetry.debug_p_277, 1e-10f)));
        } else {
            fprintf(stderr, "    p_277=N/A (token 277 out of range for vocab)\n");
        }
        fprintf(stderr, "    max_logit=%.6f sum_exp=%.6f (softmax normalization)\n",
                host_telemetry.debug_max_logit, host_telemetry.debug_sum_exp);
        fprintf(stderr, "    target_token=%d (1-indexed, 0=not set)\n", host_telemetry.debug_target);
        
        fprintf(stderr, "  STORED loss=%.9f (should match computed)\n",
                host_telemetry.debug_loss);
        const uint32_t debug_count = host_telemetry.debug_count;
        const uint32_t debug_to_print = (debug_count < static_cast<uint32_t>(kDebugSamples))
            ? debug_count
            : static_cast<uint32_t>(kDebugSamples);
        fprintf(stderr, "[UnifiedLoss] FIRST %u p_t = [", debug_to_print);
        for (uint32_t i = 0; i < debug_to_print; ++i) {
            fprintf(stderr, "%.9f%s",
                    host_telemetry.debug_p_t_10[i],
                    (i + 1 < debug_to_print) ? ", " : "");
        }
        fprintf(stderr, "]\n");
        fprintf(stderr, "[UnifiedLoss] FIRST %u ce_smooth = [", debug_to_print);
        for (uint32_t i = 0; i < debug_to_print; ++i) {
            fprintf(stderr, "%.6f%s",
                    host_telemetry.debug_ce_smooth_10[i],
                    (i + 1 < debug_to_print) ? ", " : "");
        }
        fprintf(stderr, "]\n");
        if (outputs.token_losses) {
            float first_losses[10] = {0};
            cudaMemcpy(first_losses, outputs.token_losses, sizeof(float) * 10, cudaMemcpyDeviceToHost);
            fprintf(stderr, "[UnifiedLoss] First 10 token_losses = [%.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f]\n",
                    first_losses[0], first_losses[1], first_losses[2], first_losses[3], first_losses[4],
                    first_losses[5], first_losses[6], first_losses[7], first_losses[8], first_losses[9]);
        }
        logged_debug = true;
    }
    
    // Log grad_logit[277] breakdown - KEY DIAGNOSTIC for mode collapse
    fprintf(stderr, "[Grad277Trace] grad_277_sum=%.6f target_sum=%.6f nontarget_sum=%.6f target_count=%u\n",
            host_telemetry.grad_277_sum,
            host_telemetry.grad_277_sum_target,      // Should be negative (model penalized for overpredicting)
            host_telemetry.grad_277_sum_nontarget,   // Will be positive (p_277 for non-277 targets)
            host_telemetry.target_277_count);
    
    const uint32_t valid = host_telemetry.valid_count;
    const uint32_t masked = host_telemetry.masked_count;

    result.valid_tokens = valid;
    result.masked_tokens = masked;
    result.nan_count = host_telemetry.nan_count;
    result.inf_count = host_telemetry.inf_count;
    result.invalid_target_count = host_telemetry.invalid_target_count;

    if (host_telemetry.invalid_target_count > 0) {
        result.error_code = UnifiedLossTelemetry::ERR_INVALID_TARGET;
        fprintf(stderr, "[UnifiedLoss] %s: %u invalid targets detected\n",
                getErrorMessage(result.error_code),
                host_telemetry.invalid_target_count);
        return result;
    }

    if (valid > 0) {
        const float inv_valid = 1.0f / static_cast<float>(valid);
        result.loss_mean = host_telemetry.loss_sum * inv_valid;
        result.loss_variance = (host_telemetry.loss_sq_sum * inv_valid)
            - (result.loss_mean * result.loss_mean);
        result.loss_max = host_telemetry.loss_max;
        result.loss_min = host_telemetry.loss_min;
        result.grad_norm_mean = host_telemetry.grad_norm_sum * inv_valid;
        result.grad_norm_max = host_telemetry.grad_norm_max;
        result.focal_weight_mean = host_telemetry.focal_weight_sum * inv_valid;
        result.hard_example_ratio = static_cast<float>(host_telemetry.hard_example_count) * inv_valid;
    }
     bool  strict_mode = false;

    if (strict_mode) {
        if (host_telemetry.nan_count > 0) {
            result.error_code = UnifiedLossTelemetry::ERR_NAN_IN_LOSS;
            fprintf(stderr, "[UnifiedLoss] %s: %u NaN tokens detected\n",
                    getErrorMessage(result.error_code), host_telemetry.nan_count);
            return result;
        }
        if (host_telemetry.inf_count > 0) {
            result.error_code = UnifiedLossTelemetry::ERR_INF_IN_LOSS;
            fprintf(stderr, "[UnifiedLoss] %s: %u Inf tokens detected\n",
                    getErrorMessage(result.error_code), host_telemetry.inf_count);
            return result;
        }
        if (!std::isfinite(result.loss_mean)) {
            result.error_code = UnifiedLossTelemetry::ERR_NAN_IN_LOSS;
            fprintf(stderr, "[UnifiedLoss] %s: mean loss is %f\n",
                    getErrorMessage(result.error_code), result.loss_mean);
            return result;
        }
    }

    result.error_code = UnifiedLossTelemetry::OK;

    static bool logged_init = true;
    if (logged_init) {
        fprintf(stdout, "[UnifiedLoss] Initialized: focal(α=%.2f, γ=%.2f) "
                        "smoothing(ε=%.3f) strict=%s\n",
                focal_alpha, focal_gamma, smoothing_epsilon,
                config.strict_mode ? "ON" : "off");
        fprintf(stdout, "[UnifiedLoss] First call: valid=%u masked=%u loss_sum=%.6f\n",
                valid, masked, host_telemetry.loss_sum);
        logged_init = false;
    }

    return result;
}

} // namespace GRIM::Loss
