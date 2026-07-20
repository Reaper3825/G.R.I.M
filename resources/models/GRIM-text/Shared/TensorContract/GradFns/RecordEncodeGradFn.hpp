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

    const int* saved_slot1_view_ = nullptr;
    const int* saved_slot2_view_ = nullptr;
    const int* saved_ops_view_ = nullptr;
    const float* saved_scalars_view_ = nullptr;

    float* E_slot_grad_ = nullptr;
    float* E_op_grad_ = nullptr;
    float* W_scal_grad_ = nullptr;
    float* b_scal_grad_ = nullptr;

    RecordEncodeGradFn();

    void capture(int N,
                 int num_slots,
                 int num_ops,
                 int d_model,
                 const int* d_slot1,
                 const int* d_slot2,
                 const int* d_ops,
                 const float* d_scalars,
                 int* saved_ids_staging,
                 float* saved_scalars_staging,
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
