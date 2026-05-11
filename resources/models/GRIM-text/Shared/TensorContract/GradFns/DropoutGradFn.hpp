#pragma once
//======================================================//
//  DropoutGradFn.hpp
//  Backward node for seeded dropout.
//======================================================//

#include "../TensorContract_GPU.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <cuda_runtime.h>

namespace GRIM {
namespace autograd {

struct DropoutGradFn : public GradFn {
    bool input_requires_grad = false;
    float* input_grad = nullptr;
    bool owns_input_grad = false;
    TensorContract::TensorShape input_shape;
    std::shared_ptr<GradFn> input_grad_fn;
    std::uint8_t* saved_mask = nullptr;
    float scale = 1.0f;
    std::size_t count = 0;

    DropoutGradFn();
    ~DropoutGradFn() override;

    void capture_input(Tensor& x, cudaStream_t stream);
    void save(const std::uint8_t* mask, float dropout_prob, std::size_t n, cudaStream_t stream);
    void apply(const Tensor& grad_output, cudaStream_t stream) override;
    void release_saved() override;
};

}  // namespace autograd
}  // namespace GRIM
