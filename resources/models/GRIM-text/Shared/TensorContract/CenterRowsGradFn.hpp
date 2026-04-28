#pragma once
//======================================================//
//  CenterRowsGradFn.hpp
//  Backward node for row-wise centering.
//======================================================//

#include "TensorContract_GPU.hpp"

#include <cstddef>
#include <memory>
#include <cuda_runtime.h>

namespace GRIM {
namespace autograd {

struct CenterRowsGradFn : public GradFn {
    float* input_grad = nullptr;
    std::shared_ptr<GradFn> input_grad_fn;
    TensorContract::TensorShape input_shape;
    std::size_t element_count = 0;
    int row_dim = 0;
    int num_rows = 0;
    bool input_requires_grad = false;
    bool input_is_leaf = false;
    float* leaf_grad_buf = nullptr;

    std::shared_ptr<float> owned_input_grad;

    CenterRowsGradFn();

    void capture_input(Tensor& input, int dim, int rows, cudaStream_t stream);
    void apply(const Tensor& grad_output, cudaStream_t stream) override;
    void release_saved() override;
};

}  // namespace autograd
}  // namespace GRIM
