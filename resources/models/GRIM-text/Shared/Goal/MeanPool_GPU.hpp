#pragma once

#include "../TensorContract/TensorContract_GPU.hpp"

#include <cuda_runtime.h>

namespace GRIM {

Tensor meanPoolHiddenStates(
    const Tensor& hidden_states,
    cudaStream_t stream);

} // namespace GRIM
