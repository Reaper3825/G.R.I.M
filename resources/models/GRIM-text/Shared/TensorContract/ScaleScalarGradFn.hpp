#pragma once
//======================================================//
//  ScaleScalarGradFn.hpp
//  Backward node for scaling a 1-element tensor by a constant.
//  Forward:  result[0] = scale * t[0]   (t is scalar [1])
//  Backward: grad_t[0] = scale * grad_result[0]
//
//  Used for loss weighting (e.g. MTP alpha/K). No CUDA kernel — the math
//  is one float multiply, done host-side via cudaMemcpyAsync round-trip.
//
//  Owns:
//    - struct ScaleScalarGradFn        (declared here)
//    - autograd::scale_scalar(...)     (forward op; defined in ScaleScalarGradFn.cu)
//======================================================//

#include "TensorContract_GPU.hpp"

#include <memory>
#include <cuda_runtime.h>

namespace GRIM {
namespace autograd {

struct ScaleScalarGradFn : public GradFn {
    std::shared_ptr<GradFn> input_grad_fn;
    TensorContract::TensorShape input_shape;
    float scale = 0.0f;

    ScaleScalarGradFn();
    ~ScaleScalarGradFn() override = default;

    void apply(const Tensor& grad_output, cudaStream_t stream) override;
};

}  // namespace autograd
}  // namespace GRIM
