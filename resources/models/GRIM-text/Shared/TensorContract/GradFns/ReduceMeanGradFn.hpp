#pragma once
//======================================================//
//  ReduceMeanGradFn.hpp
//  Backward node for the execution block's row-local masked mean.
//======================================================//

#include "../TensorContract_GPU.hpp"

#include <cstdint>
#include <memory>

namespace GRIM::autograd {

struct ReduceMeanGradFn : public GradFn {
    int total_tokens_ = 0;
    int d_model_ = 0;
    int token_offset_ = 0;
    int row_tokens_ = 0;

    std::shared_ptr<GradFn> H_grad_fn;
    TensorContract::TensorShape H_shape;
    bool H_requires_grad = false;
    bool H_is_leaf_ = false;
    float* grad_H_buf = nullptr;
    const uint8_t* atom_mask_row_ = nullptr;

    ReduceMeanGradFn();
    ~ReduceMeanGradFn() override;

    void capture(Tensor& H,
                 int total_tokens,
                 int d_model,
                 cudaStream_t stream,
                 int token_offset = 0,
                 int row_tokens = -1,
                 const uint8_t* atom_mask_row = nullptr);

    void apply_impl(const Tensor& grad_output,
                    cudaStream_t stream,
                    const Batching::BatchPayload* backward_payload,
                    const Batching::BatchDeviceBindings* backward_bindings) override;
    void release_saved() override;
};

}  // namespace GRIM::autograd
