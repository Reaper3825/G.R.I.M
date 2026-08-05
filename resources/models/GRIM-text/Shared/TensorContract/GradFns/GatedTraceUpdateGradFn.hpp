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

    std::shared_ptr<Tensor> old_trace_gradient;
    std::shared_ptr<Tensor> candidate_gradient;
    std::shared_ptr<Tensor> gate_logits_gradient;
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
