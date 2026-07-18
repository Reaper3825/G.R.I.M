#pragma once
//======================================================//
//  GatedTraceUpdateGradFn.hpp
//  Backward node for the execution trace's gated update.
//======================================================//

#include "../TensorContract_GPU.hpp"

#include <memory>

namespace GRIM::autograd {

struct GatedTraceUpdateGradFn : public GradFn {
    float* saved_old_trace = nullptr;
    float* saved_candidate = nullptr;
    float* saved_gate_vals = nullptr;
    int dm_ = 0;

    float* grad_old_trace = nullptr;
    float* grad_candidate = nullptr;
    float* grad_gate_logits = nullptr;
    std::shared_ptr<float> owned_grad_old_trace;
    std::shared_ptr<float> owned_grad_candidate;
    std::shared_ptr<float> owned_grad_gate_logits;
    std::shared_ptr<GradFn> old_trace_grad_fn;
    std::shared_ptr<GradFn> candidate_grad_fn;
    std::shared_ptr<GradFn> gate_logits_grad_fn;
    TensorContract::TensorShape old_trace_shape;
    TensorContract::TensorShape candidate_shape;
    TensorContract::TensorShape gate_logits_shape;
    bool old_trace_requires_grad = false;
    bool candidate_requires_grad = false;
    bool gate_logits_requires_grad = false;

    GatedTraceUpdateGradFn();
    ~GatedTraceUpdateGradFn() override;

    void capture(Tensor& old_trace_t,
                 Tensor& candidate_t,
                 Tensor& gate_logits_t,
                 float* gate_vals_buf,
                 int dm,
                 cudaStream_t stream);

    void apply_impl(const Tensor& grad_output,
                    cudaStream_t stream,
                    const Batching::BatchPayload* backward_payload,
                    const Batching::BatchDeviceBindings* backward_bindings) override;
    void release_saved() override;
};

}  // namespace GRIM::autograd
