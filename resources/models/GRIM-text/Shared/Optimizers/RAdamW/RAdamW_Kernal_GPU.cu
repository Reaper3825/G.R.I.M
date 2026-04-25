#ifndef USE_CUDA
#define USE_CUDA
#endif
//======================================================//
//  RAdamW_Kernal_GPU.cu
//  CUDA implementation for Rectified AdamW optimizer step
//  (Liu et al. 2019 + Loshchilov & Hutter 2019 decoupled WD).
//
//  Math (per-element — always rectified, that IS RAdamW):
//      m_t   = β₁·m_{t-1} + (1-β₁)·g
//      v_t   = β₂·v_{t-1} + (1-β₂)·g²
//      m̂    = m_t / (1 − β₁^t)
//      ρ_∞   = 2/(1-β₂) − 1
//      ρ_t   = ρ_∞ − 2·t·β₂^t / (1 − β₂^t)
//      if ρ_t > 4:                       // rectified, adaptive
//          v̂  = sqrt(v_t / (1 − β₂^t))
//          r_t = sqrt( ((ρ_t-4)(ρ_t-2)ρ_∞) / ((ρ_∞-4)(ρ_∞-2)ρ_t) )
//          update = r_t · m̂ / (v̂ + ε)
//      else:                             // SGD-with-momentum warmup
//          update = m̂
//      θ ← θ − lr·(update + wd·θ)        // decoupled WD always
//
//  Hyperparameters (β₁, β₂, ε) are passed through the launch
//  signature — kernel reads no globals. Defaults live in
//  HyperParameters_GPU.hpp (single source of truth).
//
//  No "plain" / non-rectified branch exists. Rectification IS RAdamW
//  — callers wanting bias-corrected AdamW must use the AdamW kernel.
//======================================================//

#include "RAdamW_Kernal_GPU.hpp"
#include "../../HyperParameters/HyperParameters_GPU.hpp"
#include "../../TensorContract/TensorContract_GPU.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <stdexcept>
#include <vector>

namespace GRIM {

namespace {

//------------------------------------------------------//
//  Per-element kernel
//------------------------------------------------------//

// Full RAdamW. use_rectified == (ρ_t > 4): true → rectified adaptive
// update (with r_t); false → SGD-with-momentum warmup step.
__global__ void RAdamWRectifiedKernel(float* __restrict__ params,
                                      const float* __restrict__ grads,
                                      float* __restrict__ moments1,
                                      float* __restrict__ moments2,
                                      std::size_t size,
                                      float learning_rate,
                                      float weight_decay,
                                      float beta1,
                                      float beta2,
                                      float epsilon,
                                      float inv_bias_correction1,
                                      float sqrt_inv_bias_correction2,
                                      float r_t,
                                      int   use_rectified) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = blockDim.x * gridDim.x;

    for (std::size_t i = idx; i < size; i += stride) {
        const float g = grads[i];
        const float p = params[i];

        const float m_old = moments1[i];
        const float v_old = moments2[i];

        const float m_new = beta1 * m_old + (1.0f - beta1) * g;
        const float v_new = beta2 * v_old + (1.0f - beta2) * g * g;

        const float m_hat = m_new * inv_bias_correction1;

        float update;
        if (use_rectified) {
            // v̂ = sqrt(v / (1 − β₂^t))  ≡  sqrt(v) * sqrt_inv_bias_correction2
            const float v_hat_sqrt = sqrtf(v_new) * sqrt_inv_bias_correction2;
            update = r_t * m_hat / (v_hat_sqrt + epsilon);
        } else {
            // Warmup: SGD-with-momentum (un-adapted)
            update = m_hat;
        }

        params[i]   = p - learning_rate * (update + weight_decay * p);
        moments1[i] = m_new;
        moments2[i] = v_new;
    }
}

inline int computeGridSize(std::size_t elements, int block_size) {
    if (block_size <= 0) {
        throw std::runtime_error("[launchRAdamWKernel] block_size must be > 0");
    }
    const std::size_t grid = (elements + static_cast<std::size_t>(block_size) - 1) /
        static_cast<std::size_t>(block_size);
    if (grid == 0 || grid > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "[launchRAdamWKernel] invalid grid size for elements=" + std::to_string(elements) +
            " block_size=" + std::to_string(block_size));
    }
    return static_cast<int>(grid);
}

} // namespace

//======================================================//
//  launchRAdamWKernel — single ParameterGroup
//======================================================//

void launchRAdamWKernel(ParameterGroup& group,
                        float learning_rate,
                        float weight_decay,
                        int   step,
                        float beta1,
                        float beta2,
                        float epsilon,
                        cudaStream_t stream) {
    float* params      = group.weights();
    const float* grads = group.grads();
    float* moments1    = group.m_state();
    float* moments2    = group.v_state();
    const std::size_t size = group.size();

    if (!std::isfinite(learning_rate) || learning_rate < 0.0f) {
        throw std::runtime_error(
            "[launchRAdamWKernel] invalid learning_rate for group '" + group.name +
            "': " + std::to_string(learning_rate));
    }
    if (!std::isfinite(weight_decay) || weight_decay < 0.0f) {
        throw std::runtime_error(
            "[launchRAdamWKernel] invalid weight_decay for group '" + group.name +
            "': " + std::to_string(weight_decay));
    }
    if (step < 0) {
        throw std::runtime_error(
            "[launchRAdamWKernel] step must be >= 0 for group '" + group.name +
            "', got " + std::to_string(step));
    }
    if (!(beta1 > 0.0f && beta1 < 1.0f)) {
        throw std::runtime_error(
            "[launchRAdamWKernel] beta1 must be in (0,1) for group '" + group.name +
            "', got " + std::to_string(beta1));
    }
    if (!(beta2 > 0.0f && beta2 < 1.0f)) {
        throw std::runtime_error(
            "[launchRAdamWKernel] beta2 must be in (0,1) for group '" + group.name +
            "', got " + std::to_string(beta2));
    }
    if (!(epsilon > 0.0f) || !std::isfinite(epsilon)) {
        throw std::runtime_error(
            "[launchRAdamWKernel] epsilon must be > 0 and finite for group '" + group.name +
            "', got " + std::to_string(epsilon));
    }
    if (stream == nullptr) {
        throw std::runtime_error(
            "[launchRAdamWKernel] stream is NULL for group '" + group.name + "'");
    }
    if (!params || !grads || !moments1 || !moments2 || size == 0) {
        throw std::runtime_error(
            "[launchRAdamWKernel] NULL buffer in group '" + group.name +
            "' params=" + std::to_string(reinterpret_cast<uintptr_t>(params)) +
            " grads=" + std::to_string(reinterpret_cast<uintptr_t>(grads)) +
            " m=" + std::to_string(reinterpret_cast<uintptr_t>(moments1)) +
            " v=" + std::to_string(reinterpret_cast<uintptr_t>(moments2)) +
            " size=" + std::to_string(size));
    }

    // Bias correction uses iteration = step + 1 to avoid 1 - β^0 = 0.
    const int iteration = step + 1;
    const float bias_correction1 = 1.0f - powf(beta1, static_cast<float>(iteration));
    const float bias_correction2 = 1.0f - powf(beta2, static_cast<float>(iteration));
    if (!std::isfinite(bias_correction1) || !std::isfinite(bias_correction2) ||
        bias_correction1 <= 0.0f || bias_correction2 <= 0.0f) {
        throw std::runtime_error(
            "[launchRAdamWKernel] invalid bias correction for group '" + group.name +
            "' step=" + std::to_string(step) +
            " bc1=" + std::to_string(bias_correction1) +
            " bc2=" + std::to_string(bias_correction2));
    }
    const float inv_bias_correction1 = 1.0f / bias_correction1;
    const float inv_bias_correction2 = 1.0f / bias_correction2;

    const int grid = computeGridSize(size, HyperParameters::CUDA_BLOCK_SIZE_STANDARD);
    const int block = HyperParameters::CUDA_BLOCK_SIZE_STANDARD;

    // RAdamW: always run ρ_∞ / ρ_t variance-rectification math.
    const float rho_inf = 2.0f / (1.0f - beta2) - 1.0f;
    const float beta2_t = powf(beta2, static_cast<float>(iteration));
    // ρ_t = ρ_∞ − 2·t·β₂^t / (1 − β₂^t)
    const float rho_t = rho_inf -
        2.0f * static_cast<float>(iteration) * beta2_t / (1.0f - beta2_t);

    const bool use_rectified = (rho_t > 4.0f);
    float r_t = 1.0f;
    if (use_rectified) {
        // r_t = sqrt( ((ρ_t-4)(ρ_t-2)ρ_∞) / ((ρ_∞-4)(ρ_∞-2)ρ_t) )
        const float num = (rho_t - 4.0f) * (rho_t - 2.0f) * rho_inf;
        const float den = (rho_inf - 4.0f) * (rho_inf - 2.0f) * rho_t;
        if (!(den > 0.0f) || !(num > 0.0f)) {
            throw std::runtime_error(
                "[launchRAdamWKernel] invalid rectification term for group '" + group.name +
                "' rho_t=" + std::to_string(rho_t) +
                " rho_inf=" + std::to_string(rho_inf) +
                " num=" + std::to_string(num) +
                " den=" + std::to_string(den));
        }
        r_t = sqrtf(num / den);
        if (!std::isfinite(r_t)) {
            throw std::runtime_error(
                "[launchRAdamWKernel] non-finite r_t for group '" + group.name +
                "' r_t=" + std::to_string(r_t));
        }
    }
    const float sqrt_inv_bias_correction2 = sqrtf(inv_bias_correction2);

    RAdamWRectifiedKernel<<<grid, block, 0, stream>>>(params, grads, moments1, moments2,
                                                      size,
                                                      learning_rate, weight_decay,
                                                      beta1, beta2, epsilon,
                                                      inv_bias_correction1,
                                                      sqrt_inv_bias_correction2,
                                                      r_t,
                                                      use_rectified ? 1 : 0);

    const cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        throw std::runtime_error(
            "[launchRAdamWKernel] kernel launch failed for group '" + group.name +
            "': " + std::string(cudaGetErrorString(launch_error)));
    }
}

//======================================================//
//  launchRAdamWStep — all ParameterGroups
//======================================================//

void launchRAdamWStep(std::vector<ParameterGroup>& groups,
                      float learning_rate,
                      float weight_decay,
                      int   step,
                      float beta1,
                      float beta2,
                      float epsilon,
                      cudaStream_t stream,
                      int   embedding_freeze_after_step) {
    if (groups.empty()) {
        throw std::runtime_error(
            "[launchRAdamWStep] parameter groups are empty - "
            "caller MUST call buildParameterGroups() first");
    }
    if (!stream) {
        throw std::runtime_error(
            "[launchRAdamWStep] stream is NULL - caller MUST provide valid CUDA stream");
    }

    const bool embedding_frozen = (embedding_freeze_after_step >= 0) &&
                                  (step >= embedding_freeze_after_step);

    for (size_t i = 0; i < groups.size(); ++i) {
        auto& group = groups[i];
        if (!group.weights() || !group.grads() || group.size() == 0) continue;

        // Match AdamW embedding-freeze convention exactly (tied / untied).
        if (embedding_frozen && group.type == ParamGroupType::EMBEDDING) {
            continue;
        }
        if (embedding_frozen && group.type == ParamGroupType::LM_HEAD &&
            group.name == "embedding_lm_head_tied") {
            continue;
        }

        if (!group.m_state() || !group.v_state()) {
            throw std::runtime_error(
                "[launchRAdamWStep] FATAL: Missing optimizer state for group '" + group.name +
                "' idx=" + std::to_string(i) +
                " size=" + std::to_string(group.size()) +
                " weights=" + std::to_string(reinterpret_cast<uintptr_t>(group.weights())) +
                " grads=" + std::to_string(reinterpret_cast<uintptr_t>(group.grads())) +
                " m_state=" + std::to_string(reinterpret_cast<uintptr_t>(group.m_state())) +
                " v_state=" + std::to_string(reinterpret_cast<uintptr_t>(group.v_state())) +
                " step=" + std::to_string(step));
        }

        // Per-group depth-aware weight decay & lr scaling — same as AdamW path.
        const float effective_weight_decay = weight_decay * group.upsilon * group.weight_decay_multiplier;
        const float effective_lr = learning_rate * group.lr_multiplier;

        launchRAdamWKernel(group, effective_lr, effective_weight_decay, step,
                           beta1, beta2, epsilon, stream);
    }
}

} // namespace GRIM
