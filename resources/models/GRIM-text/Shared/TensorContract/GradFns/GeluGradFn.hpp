#pragma once
//======================================================//
//  GeluGradFn.hpp
//  Backward node for GELU activation (TAPE-BASED, ISSUE #48 pattern).
//
//  Owns:
//    - struct GeluGradFn          (declared here)
//    - kernel_gelu_forward        (defined in GeluGradFn.cu, anon namespace)
//    - kernel_gelu_backward       (defined in GeluGradFn.cu, anon namespace)
//    - autograd::gelu(...)        (forward op; defined in GeluGradFn.cu)
//
//  GeluGradFn copies the cached input into an owned device buffer so the
//  backward pass remains valid after forward returns.
//======================================================//

#include "../TensorContract_GPU.hpp"  // GradFn, Tensor, TensorShape

#include <memory>
#include <cstddef>
#include <cuda_runtime.h>

namespace GRIM {
namespace autograd {

struct GeluGradFn : public GradFn {
    bool input_requires_grad = false;
    std::shared_ptr<Tensor> input_gradient;
    std::shared_ptr<float> owned_cache;            ///< ISSUE #51: owned copy of cached input
    const float* cached_input = nullptr;           ///< Points to owned_cache.get()
    std::size_t cached_size = 0;

    GeluGradFn();
    ~GeluGradFn() override;

    void capture_input(Tensor& x, cudaStream_t stream);

    /// Copy external_cache (size floats) into an owned device buffer for backward.
    void set_cache_copy(const float* external_cache, std::size_t size, cudaStream_t stream);

    void apply_impl(const Tensor& grad_output,
                    cudaStream_t stream,
                    const Batching::BatchPayload* backward_payload,
                    const Batching::BatchDeviceBindings* backward_bindings) override;
    void release_saved() override;
};

}  // namespace autograd
}  // namespace GRIM
