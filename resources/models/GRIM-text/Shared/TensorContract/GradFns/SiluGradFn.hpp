#pragma once
//======================================================//
//  SiluGradFn.hpp
//  Backward node for SiLU (Swish) activation: y = x * sigmoid(x).
//
//  Owns:
//    - struct SiluGradFn          (declared here)
//    - kernel_silu_forward        (defined in SiluGradFn.cu)
//    - kernel_silu_backward       (defined in SiluGradFn.cu)
//    - autograd::silu(...)        (forward op; defined in SiluGradFn.cu)
//
//  Memory: uses a non-owning cache reference into ModelForwardOutputs-owned
//  per-layer or execution-step tensors (Issue #56 guarantees the source tensor
//  outlives backward).
//======================================================//

#include "../TensorContract_GPU.hpp"

#include <memory>
#include <cstddef>
#include <cuda_runtime.h>

namespace GRIM {
namespace autograd {

struct SiluGradFn : public GradFn {
    bool input_requires_grad = false;
    std::shared_ptr<Tensor> input_gradient;
    const float* cached_input = nullptr;          ///< Non-owning ref into source tensor
    std::size_t cached_size = 0;

    SiluGradFn();
    ~SiluGradFn() override;

    void capture_input(Tensor& x, cudaStream_t stream);
    void set_cache_ref(const float* data, std::size_t size);
    void apply_impl(const Tensor& grad_output,
                    cudaStream_t stream,
                    const Batching::BatchPayload* backward_payload,
                    const Batching::BatchDeviceBindings* backward_bindings) override;
    void release_saved() override;
};

}  // namespace autograd
}  // namespace GRIM
