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
    Tensor saved_probs_view_;
    Tensor grad_probs_view_;
    std::shared_ptr<GradFn> probs_grad_fn_;
    TensorContract::TensorShape probs_shape_;
    bool probs_requires_grad_ = false;
    bool probs_is_leaf_ = false;
    float* probs_leaf_grad_ = nullptr;

    NormalizedEntropyGradFn();

    void capture(Tensor& probs_tensor,
                 Tensor& saved_probs_staging,
                 Tensor& grad_probs_staging,
                 cudaStream_t stream);

    void apply_impl(const Tensor& grad_output,
                    cudaStream_t stream,
                    const Batching::BatchPayload* backward_payload,
                    const Batching::BatchDeviceBindings* backward_bindings) override;
    void release_saved() override;
};

// Returns the scalar normalized entropy of a [1, C] probability tensor.
// saved_probs_staging and grad_probs_staging are ModelForwardOutputs-owned
// buffers with the same shape as probs_tensor; this operation only borrows them.
//   -sum_i(p_i * log(p_i)) / log(C)
Tensor normalized_entropy(Tensor& probs_tensor,
                          Tensor& saved_probs_staging,
                          Tensor& grad_probs_staging,
                          cudaStream_t stream);

}  // namespace GRIM::autograd
