#pragma once
//======================================================//
//  ZeroPadGradFn.hpp
//  Backward node for row-offset zero-padding.
//
//  Forward:  result = zeros(total_rows, cols);
//            result[row_offset : row_offset+rows, :] = x
//  Backward: grad_x += grad_result[row_offset : row_offset+rows, :]
//
//  Owns:
//    - struct ZeroPadGradFn        (declared here)
//    - autograd::zero_pad(...)     (forward op; defined in ZeroPadGradFn.cu)
//
//  No custom forward kernel — forward uses cudaMemset + cudaMemcpy directly.
//  Backward uses TensorContract/GradientAccumulation.hpp; do not add
//  per-TU kernel_accumulate_grad copies here.
//======================================================//

#include "../TensorContract_GPU.hpp"

#include <memory>
#include <cstddef>
#include <cuda_runtime.h>

namespace GRIM {
namespace autograd {

struct ZeroPadGradFn : public GradFn {
    bool input_requires_grad = false;
    std::shared_ptr<Tensor> input_gradient;
    std::size_t input_count = 0;
    std::size_t offset_elements = 0;        ///< row_offset * cols

    ZeroPadGradFn();
    ~ZeroPadGradFn() override;

    void capture_input(Tensor& x, cudaStream_t stream, std::size_t offset_elems);
    void apply_impl(const Tensor& grad_output,
                    cudaStream_t stream,
                    const Batching::BatchPayload* backward_payload,
                    const Batching::BatchDeviceBindings* backward_bindings) override;
    void release_saved() override;
};

}  // namespace autograd
}  // namespace GRIM
