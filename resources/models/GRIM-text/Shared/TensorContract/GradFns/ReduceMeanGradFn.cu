//======================================================//
//  ReduceMeanGradFn.cu
//  Backward implementation for the row-local masked mean.
//======================================================//

#include "ReduceMeanGradFn.hpp"
#include "../../CudaAllocUtils.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace GRIM::autograd {

using CudaAlloc::cudaMallocOrThrow;

namespace {

constexpr int kBlockSize = 256;

__global__ void kernelReduceMeanBackward(
    float* __restrict__ grad_H,
    const float* __restrict__ grad_out,
    int total_tokens,
    int d_model,
    const uint8_t* __restrict__ atom_mask)
{
    const int i = blockIdx.x;
    if (i >= total_tokens) return;
    if (atom_mask && atom_mask[i] != 0) return;

    int count = 0;
    if (atom_mask) {
        for (int t = 0; t < total_tokens; ++t) {
            if (atom_mask[t] == 0) ++count;
        }
    } else {
        count = total_tokens;
    }
    if (count == 0) count = total_tokens;

    const float scale = 1.0f / static_cast<float>(count);
    float* dst = grad_H + static_cast<size_t>(i) * d_model;
    for (int j = threadIdx.x; j < d_model; j += blockDim.x) {
        dst[j] += grad_out[j] * scale;
    }
}

void checkCuda(cudaError_t err, const char* caller) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(caller) + ": " + cudaGetErrorString(err));
    }
}

}  // namespace

ReduceMeanGradFn::ReduceMeanGradFn() {
    op_name = "reduce_mean";
}

ReduceMeanGradFn::~ReduceMeanGradFn() {
    if (!H_is_leaf_ && grad_H_buf) cudaFree(grad_H_buf);
}

void ReduceMeanGradFn::capture(
    Tensor& H,
    int total_tokens,
    int d_model,
    cudaStream_t stream,
    int token_offset,
    int row_tokens,
    const uint8_t* atom_mask_row)
{
    total_tokens_ = total_tokens;
    d_model_ = d_model;
    token_offset_ = token_offset;
    row_tokens_ = (row_tokens == -1) ? total_tokens : row_tokens;
    atom_mask_row_ = atom_mask_row;
    H_requires_grad = H.requires_grad;
    H_shape = H.shape;
    H_grad_fn = H.grad_fn;
    register_input(H.grad_fn);
    H_is_leaf_ = H.is_leaf;

    if (H_requires_grad) {
        if (H.is_leaf) {
            H.ensure_grad();
            grad_H_buf = H.grad_data();
        } else {
            const size_t total = static_cast<size_t>(total_tokens) * d_model;
            cudaMallocOrThrow(
                reinterpret_cast<void**>(&grad_H_buf),
                total * sizeof(float),
                "datastream_grad_H_buf");
        }
    }
}

void ReduceMeanGradFn::apply_impl(
    const Tensor& grad_output,
    cudaStream_t stream,
    const Batching::BatchPayload* backward_payload,
    const Batching::BatchDeviceBindings* backward_bindings)
{
    if (applied) return;
    applied = true;
    if (!H_requires_grad) return;

    if (!H_is_leaf_) {
        const size_t total = static_cast<size_t>(total_tokens_) * d_model_;
        checkCuda(
            cudaMemsetAsync(grad_H_buf, 0, total * sizeof(float), stream),
            "ReduceMeanGradFn::apply cudaMemsetAsync");
    }
    kernelReduceMeanBackward<<<row_tokens_, kBlockSize, 0, stream>>>(
        grad_H_buf + static_cast<size_t>(token_offset_) * d_model_,
        grad_output.data,
        row_tokens_,
        d_model_,
        atom_mask_row_);
    checkCuda(cudaGetLastError(), "ReduceMeanGradFn::apply kernelReduceMeanBackward");

    if (H_grad_fn) {
        Tensor view;
        view.data = grad_H_buf;
        view.shape = H_shape;
        view.owns_data = false;
        view.stream = stream;
        H_grad_fn->apply(view, stream, backward_payload, backward_bindings);
    }
}

void ReduceMeanGradFn::release_saved() {
    GradFn::release_saved();
    if (!H_is_leaf_ && grad_H_buf) cudaFree(grad_H_buf);
    grad_H_buf = nullptr;
    H_grad_fn.reset();
}

}  // namespace GRIM::autograd
