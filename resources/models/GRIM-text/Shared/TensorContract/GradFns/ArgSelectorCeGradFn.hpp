#pragma once
//======================================================//
//  ArgSelectorCeGradFn.hpp
//  Selection cross-entropy for the arg/option selector.
//
//  Forward:  loss = -(1/N) Σ_{t: target[t]>=0} log softmax(selection_logits[t,:])[target[t]]
//            over num_classes = num_pool_atoms candidates; out-of-row-window
//            candidates are already masked to -inf in selection_logits, so they
//            contribute ~0 probability. N = num_valid supervised rows.
//  Backward: grad_logits[t,e] = (softmax[t,e] - onehot(target[t])) * grad/N
//            (0 for unsupervised rows). The exact softmax-CE gradient.
//
//  Self-contained (no log_softmax/NLL coupling): the forward saves the softmax
//  probabilities and the backward reconstructs the gradient directly, propagating
//  to selection_logits' upstream grad_fn — the same chaining pattern as
//  ScaleScalarGradFn.
//
//  Owns:
//    - struct ArgSelectorCeGradFn   (declared here)
//    - autograd::argSelectorLoss(...) (forward op; defined in .cu)
//======================================================//

#include "../TensorContract_GPU.hpp"

#include <memory>
#include <cuda_runtime.h>

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

// Scalar selection cross-entropy. d_targets is the per-row batch-global candidate
// index (or -1 to ignore). num_valid is the count of supervised rows (>= 0).
Tensor argSelectorLoss(const Tensor& selection_logits,
                       const int* d_targets,
                       int total_tokens,
                       int num_classes,
                       int num_valid,
                       cudaStream_t stream);

}  // namespace autograd
}  // namespace GRIM
