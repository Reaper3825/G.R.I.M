//======================================================//
//  OptimizerState_GPU.hpp
//  GPU-owned optimizer moment tensors.
//
//  Owns durable Adam/RAdam moment buffers (m, v) per ParameterGroup.
//  Step bookkeeping intentionally lives in OptimizerStep.hpp.
//======================================================//

#pragma once

#include <cstddef>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime_api.h>

#include "../TensorContract/TensorContract_GPU.hpp"

namespace GRIM {

struct OptimizerState {
    std::vector<Tensor> m_states;  // First moment per param group
    std::vector<Tensor> v_states;  // Second moment per param group
    bool allocated = false;

    void allocate(const std::vector<std::size_t>& sizes, cudaStream_t stream);
    void clear();
};

} // namespace GRIM

#endif  // USE_CUDA