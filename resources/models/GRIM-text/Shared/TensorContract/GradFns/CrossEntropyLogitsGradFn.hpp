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
    std::shared_ptr<GradFn> input_grad_fn;
    TensorContract::TensorShape input_shape;
    float* saved_probs = nullptr;
    bool owns_saved = true;
    int C = 0;
    int target_idx = 0;

    std::shared_ptr<float> owned_grad_logits;
    float* grad_logits = nullptr;
    bool input_is_leaf = false;

    CrossEntropyLogitsGradFn();
    ~CrossEntropyLogitsGradFn() override;

    void capture_input(Tensor& logits, cudaStream_t stream);
    void apply(const Tensor& grad_output, cudaStream_t stream) override;
    void release_saved() override;
};

}  // namespace autograd
}  // namespace GRIM
