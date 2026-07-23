#pragma once
//======================================================//
//  ArgSelectorCeGradFn.hpp
//  Backward node for the arg/option selector cross-entropy loss.
//
//  Backward: grad_logits[t,e] = (softmax[t,e] - onehot(target[t])) * grad/N
//            (0 for unsupervised rows). The exact softmax-CE gradient.
//
//  The forward loss operation lives in Shared/Loss/ComputeLoss/ArgSelectorLoss.
//======================================================//

#include "../TensorContract_GPU.hpp"

#include <cuda_runtime.h>
#include <memory>

namespace GRIM {
namespace autograd {

struct ArgSelectorCeGradFn : public GradFn {
    std::shared_ptr<GradFn> input_grad_fn;       // selection_logits' upstream grad_fn
    TensorContract::TensorShape input_shape;     // [total_tokens, num_classes]
    float* saved_probs = nullptr;                // [total_tokens * num_classes] device, owned
    const int* targets = nullptr;                // device [total_tokens], borrowed (valid in-step)
    int total_tokens = 0;
    int num_classes = 0;
    int num_valid = 0;

    ArgSelectorCeGradFn();
    ~ArgSelectorCeGradFn() override;

    void apply_impl(const Tensor& grad_output,
                    cudaStream_t stream,
                    const Batching::BatchPayload* backward_payload,
                    const Batching::BatchDeviceBindings* backward_bindings) override;
    void release_saved() override;
};

}  // namespace autograd
}  // namespace GRIM
