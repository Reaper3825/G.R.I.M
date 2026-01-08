//======================================================//
//  AdamW_Kernal_GPU.hpp
//  Production AdamW optimizer kernel
//  Hardcoded hyperparameters: β₁=0.9, β₂=0.999, ε=1e-8
//======================================================//

#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

namespace GRIM {

// Production AdamW update kernel
// - learning_rate: Step size (typically 1e-4 to 1e-3)
// - weight_decay: Decoupled L2 regularization (typically 0.01)
// - step: Optimizer step count for bias correction (1-indexed)
void launchAdamWKernel(float* params,
					   const float* grads,
					   float* moments1,
					   float* moments2,
					   std::size_t size,
					   float learning_rate,
					   float weight_decay,
					   int step,
					   cudaStream_t stream);

} // namespace GRIM

