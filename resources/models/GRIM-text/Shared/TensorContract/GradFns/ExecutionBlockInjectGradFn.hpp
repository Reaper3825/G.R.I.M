#pragma once
//======================================================//
//  ExecutionBlockInjectGradFn.hpp
//  Backward node for row-final execution-result injection.
//======================================================//

#include "../TensorContract_GPU.hpp"

#include <memory>

namespace GRIM::autograd {

struct ExecutionBlockInjectGradFn : public GradFn {
    float* saved_result_emb = nullptr;
    float* saved_H_slot = nullptr;
    Tensor saved_gate;
    float* mod_grad_buf = nullptr;
    float inv_sqrt_d = 0.0f;
    float gate_temp = 0.5f;
    int result_slot = 0;
    int total_tokens = 0;
    int d_model = 0;

    float* grad_result_emb = nullptr;
    float* w_gate_data = nullptr;
    float* w_gate_grad = nullptr;
    std::shared_ptr<float> owned_grad_result;

    std::shared_ptr<GradFn> H_grad_fn;
    std::shared_ptr<GradFn> result_grad_fn;
    TensorContract::TensorShape H_shape;
    TensorContract::TensorShape result_shape;
    bool H_requires_grad = false;
    bool result_requires_grad = false;

    ExecutionBlockInjectGradFn();
    ~ExecutionBlockInjectGradFn() override;

    void capture(Tensor& H_t,
                 Tensor& result_t,
                 Tensor& w_gate_t,
                 const Tensor& gate_tensor,
                 float* H_slot_device,
                 float inv_sqrt_d_,
                 float gate_temp_,
                 int result_slot_,
                 int total_tokens_,
                 int d_model_,
                 cudaStream_t stream);

    void apply_impl(const Tensor& grad_output,
                    cudaStream_t stream,
                    const Batching::BatchPayload* backward_payload,
                    const Batching::BatchDeviceBindings* backward_bindings) override;
    void release_saved() override;
};

}  // namespace GRIM::autograd
