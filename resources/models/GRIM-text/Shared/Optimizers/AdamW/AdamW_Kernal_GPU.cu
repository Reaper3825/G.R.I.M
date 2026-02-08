//======================================================//
//  AdamW_Kernal_GPU.cu
//  CUDA implementation for AdamW optimizer update
//======================================================//

#include "AdamW_Kernal_GPU.hpp"
#include "../../HyperParameters/HyperParameters_GPU.hpp"
#include "../../TensorContract/TensorContract_GPU.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>

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
	return static_cast<int>((elements + block_size - 1) / block_size);
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
}

} // namespace GRIM