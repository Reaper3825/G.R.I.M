#pragma once
// Fused head gate + BHSD flatten. ModelForwardOutputs owns cached scale/raw
// attention storage through backward; this node only owns producer edges.
#include "../TensorContract_GPU.hpp"

#include <memory>
#include <cstddef>
#include <cuda_runtime.h>

namespace GRIM {
namespace autograd {

struct HeadGateGradFn : public GradFn {
    bool scale_requires_grad = false;
    bool x_requires_grad = false;
    // Borrow leaf destinations; non-leaf gradients live in producer accumulators.
    Tensor* leaf_scale_gradient = nullptr;
    Tensor* leaf_x_gradient = nullptr;
    std::shared_ptr<GradFn> scale_grad_fn;
    std::shared_ptr<GradFn> x_grad_fn;
    TensorContract::TensorShape scale_shape;
    TensorContract::TensorShape x_shape;
    const float* cached_scale = nullptr;
    const float* cached_x = nullptr;
    int batch = 0, heads = 0, seq = 0, head_dim = 0;

    HeadGateGradFn();
    ~HeadGateGradFn() override { release_saved(); }

    void capture_inputs(Tensor& s, Tensor& x, cudaStream_t stream);
    void set_cache_refs(const float* scale_data, const float* x_data, int b, int h, int s, int d);

    void apply_impl(const Tensor& grad_output,
                    cudaStream_t stream,
                    const Batching::BatchPayload* backward_payload,
                    const Batching::BatchDeviceBindings* backward_bindings) override;
    void release_saved() override;
};

}  // namespace autograd
}  // namespace GRIM
