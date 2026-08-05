#pragma once
//======================================================//
//  SliceColumnsGradFn.hpp
//  Backward node for contiguous column slicing.
//
//  Forward: out[i, j] = x[i, col_offset + j]   (j < out_cols)
//  Backward: grad_x[i, col_offset + j] += grad_out[i, j]
//            (columns outside the slice receive zero)
//======================================================//

#include "../TensorContract_GPU.hpp"

#include <memory>
#include <cuda_runtime.h>

namespace GRIM {
namespace autograd {

struct SliceColumnsGradFn : public GradFn {
    bool x_requires_grad = false;
    std::shared_ptr<Tensor> x_gradient;
    int rows = 0;
    int in_cols = 0;
    int col_offset = 0;
    int out_cols = 0;

    SliceColumnsGradFn();

    void capture_inputs(Tensor& x, cudaStream_t stream);
    void apply_impl(const Tensor& grad_output,
                    cudaStream_t stream,
                    const Batching::BatchPayload* backward_payload,
                    const Batching::BatchDeviceBindings* backward_bindings) override;
    void release_saved() override;
};

}  // namespace autograd
}  // namespace GRIM
