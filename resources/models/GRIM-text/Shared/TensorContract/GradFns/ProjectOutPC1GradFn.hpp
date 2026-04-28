#pragma once
//======================================================//
//  ProjectOutPC1GradFn.hpp
//  Backward node for projecting out the dominant PC1 direction.
//======================================================//

#include "../TensorContract_GPU.hpp"

#include <cstddef>
#include <memory>
#include <cuda_runtime.h>

namespace GRIM {
namespace autograd {

struct ProjectOutPC1GradFn : public GradFn {
    bool input_requires_grad = false;
    TensorContract::TensorShape input_shape;
    std::size_t element_count = 0;
    int num_rows = 0;
    int num_cols = 0;

    float* input_grad = nullptr;
    std::shared_ptr<GradFn> input_grad_fn;
    std::shared_ptr<float> owned_input_grad;

    float* g_saved = nullptr;
    std::shared_ptr<float> owned_g;

    ProjectOutPC1GradFn();

    void capture_input(Tensor& input, int rows, int cols, float* g_device, cudaStream_t stream);
    void apply(const Tensor& grad_output, cudaStream_t stream) override;
    void release_saved() override;
};

}  // namespace autograd
}  // namespace GRIM
