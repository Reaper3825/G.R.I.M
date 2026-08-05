#pragma once
//======================================================//
//  CrossEntropyLogitsGradFn.hpp
//  Backward node for single-row cross entropy from logits.
//======================================================//

#include "../TensorContract_GPU.hpp"

#include <memory>
#include <cuda_runtime.h>

namespace GRIM {
namespace autograd {

struct CrossEntropyLogitsGradFn : public GradFn {
    std::shared_ptr<Tensor> logits_gradient;
    float* saved_probs = nullptr;
    bool owns_saved = true;
    int C = 0;
    int target_idx = 0;

    CrossEntropyLogitsGradFn();
    ~CrossEntropyLogitsGradFn() override;

    void capture_input(Tensor& logits, cudaStream_t stream);
    void apply_impl(const Tensor& grad_output,
                    cudaStream_t stream,
                    const Batching::BatchPayload* backward_payload,
                    const Batching::BatchDeviceBindings* backward_bindings) override;
    void release_saved() override;
};

}  // namespace autograd
}  // namespace GRIM
