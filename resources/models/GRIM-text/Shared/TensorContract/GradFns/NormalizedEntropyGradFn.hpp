#pragma once
//======================================================//
//  NormalizedEntropyGradFn.hpp
//  Forward/backward operation for entropy normalized by log(C).
//======================================================//

#include "../TensorContract_GPU.hpp"

#include <memory>

namespace GRIM::autograd {

struct NormalizedEntropyGradFn : public GradFn {
    int num_classes_ = 0;
    float* saved_probs_ = nullptr;

    float* grad_probs_ = nullptr;
    std::shared_ptr<float> owned_grad_probs_;
    std::shared_ptr<GradFn> probs_grad_fn_;
    TensorContract::TensorShape probs_shape_;
    bool probs_requires_grad_ = false;

    NormalizedEntropyGradFn();
    ~NormalizedEntropyGradFn() override;

    void capture(Tensor& probs_tensor, cudaStream_t stream);

    void apply_impl(const Tensor& grad_output,
                    cudaStream_t stream,
                    const Batching::BatchPayload* backward_payload,
                    const Batching::BatchDeviceBindings* backward_bindings) override;
    void release_saved() override;
};

// Returns the scalar normalized entropy of a [1, C] probability tensor:
//   -sum_i(p_i * log(p_i)) / log(C)
Tensor normalized_entropy(Tensor& probs_tensor, cudaStream_t stream);

}  // namespace GRIM::autograd
