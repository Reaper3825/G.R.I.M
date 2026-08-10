#ifndef USE_CUDA
#define USE_CUDA
#endif
//======================================================//
//  AdamW_Kernal_GPU.cu
//  CUDA implementation for AdamW optimizer update
//======================================================//

#include "AdamW_Kernal_GPU.hpp"
#include "../../HyperParameters/HyperParameters_GPU.hpp"
#include "../../TensorContract/TensorContract_GPU.hpp"
#include "../../../Common/grim_scale_buffer.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <stdexcept>
#include <vector>

namespace GRIM {

// Use centralized hyperparameters - NO LOCAL DEFINITIONS
// See HyperParameters_GPU.hpp for ADAMW_BETA1, ADAMW_BETA2, ADAMW_EPSILON

namespace {

__global__ void AdamWKernel(float* __restrict__ params,
							const float* __restrict__ grads,
							float* __restrict__ moments1,
							float* __restrict__ moments2,
							std::size_t size,
							float learning_rate,
							float weight_decay,
							float inv_bias_correction1,
							float inv_bias_correction2) {
	const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const std::size_t stride = blockDim.x * gridDim.x;

	for (std::size_t i = idx; i < size; i += stride) {
		const float grad = grads[i];
		const float param = params[i];

		const float m_old = moments1[i];
		const float v_old = moments2[i];

		// AdamW moment updates (using HyperParameters constants)
		const float m_new = HyperParameters::ADAMW_BETA1 * m_old + (1.0f - HyperParameters::ADAMW_BETA1) * grad;
		const float v_new = HyperParameters::ADAMW_BETA2 * v_old + (1.0f - HyperParameters::ADAMW_BETA2) * grad * grad;

		// Bias-corrected estimates
		const float m_hat = m_new * inv_bias_correction1;
		const float v_hat = v_new * inv_bias_correction2;

		// AdamW update: decoupled weight decay + Adam step
		const float adam_update = m_hat * rsqrtf(v_hat + HyperParameters::ADAMW_EPSILON);

		params[i] = param - learning_rate * (adam_update + weight_decay * param);
		moments1[i] = m_new;
		moments2[i] = v_new;
	}
}

inline int computeGridSize(std::size_t elements, int block_size) {
	if (block_size <= 0) {
		throw std::runtime_error("[launchAdamWKernel] block_size must be > 0");
	}
	const std::size_t grid = (elements + static_cast<std::size_t>(block_size) - 1) /
		static_cast<std::size_t>(block_size);
	if (grid == 0 || grid > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
		throw std::runtime_error(
			"[launchAdamWKernel] invalid grid size for elements=" + std::to_string(elements) +
			" block_size=" + std::to_string(block_size));
	}
	return static_cast<int>(grid);
}

} // namespace

void launchAdamWKernel(ParameterGroup& group,
					   float learning_rate,
					   float weight_decay,
					   int step,
					   cudaStream_t stream) {
	float* params   = group.weights();
	const float* grads = group.grads();
	float* moments1 = group.m_state();
	float* moments2 = group.v_state();
	const std::size_t size = group.size();

	if (!std::isfinite(learning_rate) || learning_rate < 0.0f) {
		throw std::runtime_error(
			"[launchAdamWKernel] invalid learning_rate for group '" + group.name +
			"': " + std::to_string(learning_rate));
	}
	if (!std::isfinite(weight_decay) || weight_decay < 0.0f) {
		throw std::runtime_error(
			"[launchAdamWKernel] invalid weight_decay for group '" + group.name +
			"': " + std::to_string(weight_decay));
	}
	if (step < 0) {
		throw std::runtime_error(
			"[launchAdamWKernel] step must be >= 0 for group '" + group.name +
			"', got " + std::to_string(step));
	}
	if (stream == nullptr) {
		throw std::runtime_error(
			"[launchAdamWKernel] stream is NULL for group '" + group.name + "'");
	}

	if (!params || !grads || !moments1 || !moments2 || size == 0) {
		throw std::runtime_error(
			"[launchAdamWKernel] NULL buffer in group '" + group.name +
			"' params=" + std::to_string(reinterpret_cast<uintptr_t>(params)) +
			" grads=" + std::to_string(reinterpret_cast<uintptr_t>(grads)) +
			" m=" + std::to_string(reinterpret_cast<uintptr_t>(moments1)) +
			" v=" + std::to_string(reinterpret_cast<uintptr_t>(moments2)) +
			" size=" + std::to_string(size));
	}

	// Compute bias corrections from step count
	// Use (step+1) because step=0 means "0 completed steps, doing iteration 1"
	// Avoids division by zero: 1.0 - beta^0 = 0.0
	const int iteration = step + 1;
	const float bias_correction1 = 1.0f - powf(HyperParameters::ADAMW_BETA1, static_cast<float>(iteration));
	const float bias_correction2 = 1.0f - powf(HyperParameters::ADAMW_BETA2, static_cast<float>(iteration));
	if (!std::isfinite(bias_correction1) || !std::isfinite(bias_correction2) ||
		bias_correction1 <= 0.0f || bias_correction2 <= 0.0f) {
		throw std::runtime_error(
			"[launchAdamWKernel] invalid bias correction for group '" + group.name +
			"' step=" + std::to_string(step) +
			" bc1=" + std::to_string(bias_correction1) +
			" bc2=" + std::to_string(bias_correction2));
	}
	const float inv_bias_correction1 = 1.0f / bias_correction1;
	const float inv_bias_correction2 = 1.0f / bias_correction2;

	const int grid = computeGridSize(size, HyperParameters::CUDA_BLOCK_SIZE_STANDARD);
	AdamWKernel<<<grid, HyperParameters::CUDA_BLOCK_SIZE_STANDARD, 0, stream>>>(params,
												 grads,
												 moments1,
												 moments2,
												 size,
												 learning_rate,
												 weight_decay,
												 inv_bias_correction1,
												 inv_bias_correction2);
	const cudaError_t launch_error = cudaGetLastError();
	if (launch_error != cudaSuccess) {
		throw std::runtime_error(
			"[launchAdamWKernel] kernel launch failed for group '" + group.name +
			"': " + std::string(cudaGetErrorString(launch_error)));
	}
}

//======================================================//
//  launchAdamWStep - AdamW Optimizer Step (all groups)
//======================================================//
//
//  This optimizer-owned update boundary keeps optimizer orchestration
//  independent from model topology.
//  StartupParameterRegistry owns the parameter groups (via buildParameterGroups()),
//  but stepping the optimizer is training infrastructure, not model logic.
//

void launchAdamWStep(std::vector<ParameterGroup>& groups,
                     float learning_rate,
                     float weight_decay,
                     int step,
                     cudaStream_t stream,
                     int embedding_freeze_after_step) {
    if (groups.empty()) {
        throw std::runtime_error(
            "[launchAdamWStep] parameter groups are empty - "
            "caller MUST call buildParameterGroups() first");
    }
    if (!stream) {
        throw std::runtime_error(
            "[launchAdamWStep] stream is NULL - caller MUST provide valid CUDA stream");
    }

    const bool embedding_frozen = (embedding_freeze_after_step >= 0) && (step >= embedding_freeze_after_step);

    for (size_t i = 0; i < groups.size(); ++i) {
        auto& group = groups[i];
		if (!group.weights() || !group.grads() || group.size() == 0) continue;

		if (embedding_frozen && group.stats_bucket == ParamStatsBucket::EMBEDDING) {
            continue;
        }

        // Validate optimizer state exists before update
        if (!group.m_state() || !group.v_state()) {
            throw std::runtime_error(
                "[launchAdamWStep] FATAL: Missing optimizer state for group '" + group.name +
                "' idx=" + std::to_string(i) +
                " size=" + std::to_string(group.size()) +
                " weights=" + std::to_string(reinterpret_cast<uintptr_t>(group.weights())) +
                " grads=" + std::to_string(reinterpret_cast<uintptr_t>(group.grads())) +
                " m_state=" + std::to_string(reinterpret_cast<uintptr_t>(group.m_state())) +
                " v_state=" + std::to_string(reinterpret_cast<uintptr_t>(group.v_state())) +
                " step=" + std::to_string(step));
        }

		// Apply registration-stamped depth-aware regularization.
		// Formula: Υ_l = 0.1 * sqrt(L_ref / L) where L is 1-indexed layer.
		// Deeper layers get LESS regularization (smaller effective weight_decay).
        const float effective_weight_decay = weight_decay * group.upsilon * group.weight_decay_multiplier;
        const float effective_lr = learning_rate * group.lr_multiplier;

        launchAdamWKernel(group, effective_lr, effective_weight_decay, step, stream);
    }
}

//======================================================//
//  resetAdamWMoments - Zero all moment buffers
//======================================================//

void resetAdamWMoments(std::vector<ParameterGroup>& groups, cudaStream_t stream) {
    if (groups.empty()) {
        throw std::runtime_error(
            "[resetAdamWMoments] parameter groups are empty - "
            "caller MUST call buildParameterGroups() first");
    }

    for (auto& group : groups) {
        if (group.m_state() && group.size() > 0) {
            cudaMemsetAsync(group.m_state(), 0, group.size() * sizeof(float), stream);
        }
        if (group.v_state() && group.size() > 0) {
            cudaMemsetAsync(group.v_state(), 0, group.size() * sizeof(float), stream);
        }
    }
}

//======================================================//
//  scaleAdamWMoments - Scale all moment buffers
//======================================================//

void scaleAdamWMoments(std::vector<ParameterGroup>& groups,
                       float scale,
                       cudaStream_t stream) {
    if (groups.empty()) {
        throw std::runtime_error(
            "[scaleAdamWMoments] parameter groups are empty - "
            "caller MUST call buildParameterGroups() first");
    }
    if (scale <= 0.0f) {
        throw std::runtime_error(
            "[scaleAdamWMoments] scale MUST be > 0.0f, got " + std::to_string(scale));
    }

    for (auto& group : groups) {
        if (group.m_state() && group.size() > 0) {
            scaleDeviceBuffer(group.m_state(), group.size(), scale, stream);
        }
        if (group.v_state() && group.size() > 0) {
            scaleDeviceBuffer(group.v_state(), group.size(), scale, stream);
        }
    }
}

} // namespace GRIM
