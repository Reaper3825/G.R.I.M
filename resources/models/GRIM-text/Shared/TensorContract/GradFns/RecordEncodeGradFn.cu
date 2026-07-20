//======================================================//
//  RecordEncodeGradFn.cu
//  Backward implementation for execution-record encoding.
//======================================================//

#include "RecordEncodeGradFn.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace GRIM::autograd {

namespace {

constexpr int kBlockSize = 256;

__global__ void kernelEncodeRecordsBackward(
    const float* __restrict__ grad_out,
    float* __restrict__ E_slot_grad,
    float* __restrict__ E_op_grad,
    float* __restrict__ W_scal_grad,
    float* __restrict__ b_scal_grad,
    const int* __restrict__ slot1_ids,
    const int* __restrict__ slot2_ids,
    const int* __restrict__ op_ids,
    const float* __restrict__ scalars,
    int N,
    int num_slots,
    int num_ops,
    int d_model)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * d_model;
    if (idx >= total) return;
    const int i = idx / d_model;
    const int j = idx % d_model;
    const float g = grad_out[idx];

    const int s1 = slot1_ids[i];
    if (s1 >= 0 && s1 < num_slots) {
        atomicAdd(&E_slot_grad[s1 * d_model + j], g);
    }
    const int s2 = slot2_ids[i];
    if (s2 >= 0 && s2 < num_slots) {
        atomicAdd(&E_slot_grad[s2 * d_model + j], g);
    }
    const int op = op_ids[i];
    if (op >= 0 && op < num_ops) {
        atomicAdd(&E_op_grad[op * d_model + j], g);
    }
    for (int k = 0; k < 3; ++k) {
        atomicAdd(&W_scal_grad[k * d_model + j], scalars[i * 3 + k] * g);
    }
    if (b_scal_grad) atomicAdd(&b_scal_grad[j], g);
}

void checkCuda(cudaError_t err, const char* caller) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(caller) + ": " + cudaGetErrorString(err));
    }
}

}  // namespace

RecordEncodeGradFn::RecordEncodeGradFn() {
    op_name = "record_encode";
}

void RecordEncodeGradFn::capture(
    int N,
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
    cudaStream_t stream)
{
    N_ = N;
    num_slots_ = num_slots;
    num_ops_ = num_ops;
    d_model_ = d_model;

    if (!saved_ids_staging || !saved_scalars_staging) {
        throw std::runtime_error(
            "RecordEncodeGradFn::capture: ModelForwardOutputs staging is NULL");
    }
    checkCuda(cudaMemcpyAsync(saved_ids_staging, d_slot1, N * sizeof(int), cudaMemcpyDeviceToDevice, stream),
              "RecordEncodeGradFn::capture slot1");
    checkCuda(cudaMemcpyAsync(saved_ids_staging + N, d_slot2, N * sizeof(int), cudaMemcpyDeviceToDevice, stream),
              "RecordEncodeGradFn::capture slot2");
    checkCuda(cudaMemcpyAsync(saved_ids_staging + 2 * N, d_ops, N * sizeof(int), cudaMemcpyDeviceToDevice, stream),
              "RecordEncodeGradFn::capture ops");
    checkCuda(cudaMemcpyAsync(saved_scalars_staging, d_scalars, N * 3 * sizeof(float), cudaMemcpyDeviceToDevice, stream),
              "RecordEncodeGradFn::capture scalars");

    saved_slot1_view_ = saved_ids_staging;
    saved_slot2_view_ = saved_ids_staging + N;
    saved_ops_view_ = saved_ids_staging + 2 * N;
    saved_scalars_view_ = saved_scalars_staging;

    E_slot_grad_ = E_slot.grad_data();
    E_op_grad_ = E_op.grad_data();
    W_scal_grad_ = W_scal.grad_data();
    b_scal_grad_ = b_scal.data ? b_scal.grad_data() : nullptr;
}

void RecordEncodeGradFn::apply_impl(
    const Tensor& grad_output,
    cudaStream_t stream,
    const Batching::BatchPayload* backward_payload,
    const Batching::BatchDeviceBindings* backward_bindings)
{
    (void)backward_payload;
    (void)backward_bindings;
    if (applied) return;
    applied = true;

    const int total = N_ * d_model_;
    const int blocks = (total + kBlockSize - 1) / kBlockSize;
    kernelEncodeRecordsBackward<<<blocks, kBlockSize, 0, stream>>>(
        grad_output.data,
        E_slot_grad_,
        E_op_grad_,
        W_scal_grad_,
        b_scal_grad_,
        saved_slot1_view_,
        saved_slot2_view_,
        saved_ops_view_,
        saved_scalars_view_,
        N_,
        num_slots_,
        num_ops_,
        d_model_);
    checkCuda(cudaGetLastError(), "RecordEncodeGradFn::apply kernelEncodeRecordsBackward");
}

void RecordEncodeGradFn::release_saved() {
    GradFn::release_saved();
    saved_slot1_view_ = nullptr;
    saved_slot2_view_ = nullptr;
    saved_ops_view_ = nullptr;
    saved_scalars_view_ = nullptr;
}

}  // namespace GRIM::autograd
