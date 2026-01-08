//======================================================//
//  AdamW_gpu.hpp
//  High-level AdamW optimizer interface for GPU layers
//  Uses hardcoded hyperparameters: β₁=0.9, β₂=0.999, ε=1e-8
//======================================================//

#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

namespace GRIM {

struct AdamWUpdateArgs {
	float* params = nullptr;
	const float* grads = nullptr;
	float* moments1 = nullptr;
	float* moments2 = nullptr;
	std::size_t size = 0;

	float learning_rate = 1e-3f;
	float weight_decay = 0.0f;
	int step = 1;  // 1-indexed for bias correction

	cudaStream_t stream = nullptr;
};

void launchAdamWUpdate(const AdamWUpdateArgs& args);

} // namespace GRIM

