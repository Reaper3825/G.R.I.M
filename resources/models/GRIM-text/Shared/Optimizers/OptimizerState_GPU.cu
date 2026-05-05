//======================================================//
//  OptimizerState_GPU.cu
//  GPU-owned optimizer moment tensors.
//======================================================//

#ifndef USE_CUDA
#define USE_CUDA
#endif

#include "OptimizerState_GPU.hpp"

#ifdef USE_CUDA

#include <stdexcept>

namespace GRIM {

void OptimizerState::allocate(const std::vector<std::size_t>& sizes, cudaStream_t stream) {
    if (stream == nullptr) {
        throw std::runtime_error("[OptimizerState::allocate] stream is NULL - caller MUST provide valid CUDA stream");
    }

    clear();

    m_states.reserve(sizes.size());
    v_states.reserve(sizes.size());

    for (std::size_t i = 0; i < sizes.size(); ++i) {
        if (sizes[i] > 0) {
            m_states.push_back(Tensor::zeros({static_cast<int>(sizes[i])}, stream, "optimizer_m"));
            v_states.push_back(Tensor::zeros({static_cast<int>(sizes[i])}, stream, "optimizer_v"));
        } else {
            m_states.emplace_back();
            v_states.emplace_back();
        }
    }

    allocated = true;
}

void OptimizerState::clear() {
    m_states.clear();
    v_states.clear();
    allocated = false;
}

} // namespace GRIM

#endif  // USE_CUDA