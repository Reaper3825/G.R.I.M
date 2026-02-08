//======================================================//
//  AdamW_Kernal_GPU.hpp
//  Production AdamW optimizer kernel
//  Hardcoded hyperparameters: β₁=0.9, β₂=0.999, ε=1e-8
//======================================================//

#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

namespace GRIM {

struct ParameterGroup;  // Forward declaration

/// AdamW update operating directly on a ParameterGroup.
/// Reads weights/grads/m/v from the group's Tensors — no cached raw pointers.
void launchAdamWKernel(ParameterGroup& group,
					   float learning_rate,
					   float weight_decay,
					   int step,
					   cudaStream_t stream);

} // namespace GRIM

