//======================================================//
//  OptimizerUpdate_GPU.hpp
//  Optimizer Window configured update dispatch boundary
//======================================================//

#pragma once

#include <vector>
#include <cuda_runtime_api.h>

#include "../HyperParameters/HyperparameterGroupings.hpp"

namespace GRIM {

struct ParameterGroup;

/// Optimizer Window: run one configured optimizer update across all parameter groups.
/// Dispatches to AdamW or RAdamW from the grouped optimizer update HP.
/// Rule 20: Throws on invalid HP, missing optimizer state, NULL stream, or unknown kind.
void launchOptimizerUpdate(std::vector<ParameterGroup>& groups,
                           const HyperParameters::OptimizerUpdateHP& hp,
                           float learning_rate,
                           int step,
                           cudaStream_t stream);

} // namespace GRIM
