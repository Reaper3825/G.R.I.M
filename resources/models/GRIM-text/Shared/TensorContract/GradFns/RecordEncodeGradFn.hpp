#pragma once
//======================================================//
//  RecordEncodeGradFn.hpp
//  Backward node for execution-record encoding.
//======================================================//

#include "../TensorContract_GPU.hpp"

namespace GRIM::autograd {

struct RecordEncodeGradFn : public GradFn {
    int N_ = 0;
    int num_slots_ = 0;
    int num_ops_ = 0;
    int d_model_ = 0;

    int* saved_slot1_ = nullptr;
    int* saved_slot2_ = nullptr;
    int* saved_ops_ = nullptr;
    float* saved_scalars_ = nullptr;

    float* E_slot_grad_ = nullptr;
    float* E_op_grad_ = nullptr;
    float* W_scal_grad_ = nullptr;
    float* b_scal_grad_ = nullptr;

    RecordEncodeGradFn();
    ~RecordEncodeGradFn() override;

    void capture(int N,
                 int num_slots,
                 int num_ops,
                 int d_model,
                 const int* d_slot1,
                 const int* d_slot2,
                 const int* d_ops,
                 const float* d_scalars,
                 Tensor& E_slot,
                 Tensor& E_op,
                 Tensor& W_scal,
                 Tensor& b_scal,
                 cudaStream_t stream);

    void apply_impl(const Tensor& grad_output,
                    cudaStream_t stream,
                    const Batching::BatchPayload* backward_payload,
                    const Batching::BatchDeviceBindings* backward_bindings) override;
    void release_saved() override;
};

}  // namespace GRIM::autograd
