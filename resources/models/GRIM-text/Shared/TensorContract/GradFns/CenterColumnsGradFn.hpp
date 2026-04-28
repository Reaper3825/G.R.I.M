#pragma once
//======================================================//
//  CenterColumnsGradFn.hpp
//  Backward node for column-wise centering.
//======================================================//

#include "../TensorContract_GPU.hpp"

#include <cstddef>
#include <memory>
#include <cuda_runtime.h>

namespace GRIM {
namespace autograd {

struct CenterColumnsGradFn : public GradFn {
    bool input_requires_grad = false;
    TensorContract::TensorShape input_shape;
    std::size_t element_count = 0;
    int num_cols = 0;
    int num_rows = 0;

    float* input_grad = nullptr;
    std::shared_ptr<GradFn> input_grad_fn;
    std::shared_ptr<float> owned_input_grad;

    CenterColumnsGradFn();

    void capture_input(Tensor& input, int cols, int rows, cudaStream_t stream);
    void apply(const Tensor& grad_output, cudaStream_t stream) override;
    void release_saved() override;
};

}  // namespace autograd
}  // namespace GRIM
