#pragma once
//======================================================//
//  BroadcastRowMulGradFn.hpp
//  Backward node for broadcast per-row scalar multiply.
//  Forward:  out[i, j] = scale[i, 0] * x[i, j]
//  Backward:
//    grad_scale[i] += sum_j(grad_out[i, j] * x[i, j])
//    grad_x[i, j]  += grad_out[i, j] * scale[i, 0]
//
//  Backward holds non-owning cache references to scale/x — both must
//  outlive backward (guaranteed by ModelForwardOutputs-owned per-layer
//  tensors under Issue #56).
//
//  Owns:
//    - struct BroadcastRowMulGradFn   (declared here)
//    - kernel_broadcast_row_mul_forward / _backward_x / _backward_scale (defined in .cu)
//    - autograd::broadcast_row_mul(...)                                  (defined in .cu)
//======================================================//

#include "../TensorContract_GPU.hpp"

#include <memory>
#include <cstddef>
#include <cuda_runtime.h>

namespace GRIM {
namespace autograd {

struct BroadcastRowMulGradFn : public GradFn {
    bool scale_requires_grad = false;
    bool x_requires_grad = false;
    std::shared_ptr<Tensor> scale_gradient;
    std::shared_ptr<Tensor> x_gradient;
    const float* cached_scale = nullptr;
    const float* cached_x = nullptr;
    int rows = 0;
    int cols = 0;

    BroadcastRowMulGradFn();

    void capture_inputs(Tensor& s, Tensor& x, cudaStream_t stream);
    void set_cache_refs(const float* scale_data, const float* x_data, int r, int c);

    void apply_impl(const Tensor& grad_output,
                    cudaStream_t stream,
                    const Batching::BatchPayload* backward_payload,
                    const Batching::BatchDeviceBindings* backward_bindings) override;
    void release_saved() override;
};

}  // namespace autograd
}  // namespace GRIM
