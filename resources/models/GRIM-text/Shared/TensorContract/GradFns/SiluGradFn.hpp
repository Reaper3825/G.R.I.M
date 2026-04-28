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
//  Memory: uses non-owning cache reference (ForwardIntermediates / ISSUE #56
//  guarantees the source tensor outlives backward).
//======================================================//

#include "../TensorContract_GPU.hpp"

#include <memory>
#include <cstddef>
#include <cuda_runtime.h>

namespace GRIM {
namespace autograd {

struct SiluGradFn : public GradFn {
    bool input_requires_grad = false;
    float* input_grad = nullptr;
    std::shared_ptr<float> owned_input_grad;
    TensorContract::TensorShape input_shape;
    std::shared_ptr<GradFn> input_grad_fn;
    const float* cached_input = nullptr;          ///< Non-owning ref into source tensor
    std::size_t cached_size = 0;

    SiluGradFn();
    ~SiluGradFn() override;

    void capture_input(Tensor& x, cudaStream_t stream);
    void set_cache_ref(const float* data, std::size_t size);
    void apply(const Tensor& grad_output, cudaStream_t stream) override;
    void release_saved() override;
};

}  // namespace autograd
}  // namespace GRIM
