#pragma once

#include "TensorContract_GPU.hpp"

#include <cstdint>

namespace GRIM {

struct LoRAProjectionView {
    Tensor* A = nullptr;
    Tensor* B = nullptr;
    std::uint32_t rank = 0;
    float scale = 0.0f;
};

enum class MatmulOrientation : std::uint8_t {
    TRANSPOSED_WEIGHT,
    DIRECT_WEIGHT
};

namespace autograd {

Tensor lora_linear(
    const Tensor& x,
    const Tensor& W_base,
    const LoRAProjectionView* lora,
    MatmulOrientation orientation,
    cudaStream_t stream);

} // namespace autograd
} // namespace GRIM