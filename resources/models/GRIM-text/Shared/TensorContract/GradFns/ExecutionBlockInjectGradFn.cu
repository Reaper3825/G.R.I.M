//======================================================//
//  ExecutionBlockInjectGradFn.cu
//  Backward implementation for execution-result injection.
//======================================================//

#include "ExecutionBlockInjectGradFn.hpp"
#include "../../CudaAllocUtils.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace GRIM::autograd {

using CudaAlloc::cudaMallocOrThrow;

namespace {

constexpr int kBlockSize = 256;

__global__ void kernelInjectSlotBackward(
    float* __restrict__ grad_result,
    float* __restrict__ grad_w_gate,
    float* __restrict__ mod_grad_slot,
    const float* __restrict__ saved_result,
    const float* __restrict__ saved_H_slot,
    const float* __restrict__ w_gate,
    const float* __restrict__ saved_gate,
    float inv_sqrt_d,
    float gate_temp,
    int d_model)
{
    const float gate_val = saved_gate[0];

    __shared__ float s_d_logit;
    if (threadIdx.x == 0) {
        float dot = 0.0f;
        for (int j = 0; j < d_model; ++j) {
            dot += mod_grad_slot[j] * saved_result[j];
        }
        s_d_logit = dot * inv_sqrt_d * gate_val * (1.0f - gate_val) * gate_temp;
    }
    __syncthreads();

    for (int j = threadIdx.x; j < d_model; j += blockDim.x) {
        const float go = mod_grad_slot[j];
        grad_result[j] += inv_sqrt_d * gate_val * go;
        grad_w_gate[j] += saved_H_slot[j] * s_d_logit;
        mod_grad_slot[j] = go + w_gate[j] * s_d_logit;
    }
}

void checkKernel(const char* caller) {
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(caller) + ": " + cudaGetErrorString(err));
    }
}

}  // namespace

ExecutionBlockInjectGradFn::ExecutionBlockInjectGradFn() {
    op_name = "exec_inject_slot";
}

ExecutionBlockInjectGradFn::~ExecutionBlockInjectGradFn() {
    if (saved_result_emb) cudaFree(saved_result_emb);
    if (saved_H_slot) cudaFree(saved_H_slot);
    if (mod_grad_buf) cudaFree(mod_grad_buf);
}

void ExecutionBlockInjectGradFn::capture(
    Tensor& H_t,
    Tensor& result_t,
    Tensor& w_gate_t,
    const Tensor& gate_tensor,
    float* H_slot_device,
    float inv_sqrt_d_,
    float gate_temp_,
    int result_slot_,
    int total_tokens_,
    int d_model_,
    cudaStream_t stream)
{
    inv_sqrt_d = inv_sqrt_d_;
    gate_temp = gate_temp_;
    result_slot = result_slot_;
    total_tokens = total_tokens_;
    d_model = d_model_;

    gate_tensor.require("ExecutionBlockInjectGradFn::capture gate_tensor");
    if (!gate_tensor.shape.is_2d_layout() ||
        gate_tensor.shape.as_2d() != TensorContract::Shape2D{1, 1}) {
        throw std::runtime_error(
            "ExecutionBlockInjectGradFn::capture: gate_tensor must be [1, 1]");
    }
    saved_gate = Tensor::from_ptr(
        gate_tensor.data,
        gate_tensor.shape,
        false,
        false,
        "exec_inject_gate_saved_view");
    saved_gate.stream = gate_tensor.stream;
    saved_gate.compute_precision = gate_tensor.compute_precision;
    saved_H_slot = H_slot_device;

    cudaMallocOrThrow(
        reinterpret_cast<void**>(&saved_result_emb),
        d_model_ * sizeof(float),
        "datastream_saved_result_emb");
    cudaMemcpyAsync(
        saved_result_emb,
        result_t.data,
        d_model_ * sizeof(float),
        cudaMemcpyDeviceToDevice,
        stream);

    const size_t total_size = static_cast<size_t>(total_tokens_) * d_model_ * sizeof(float);
    cudaMallocOrThrow(
        reinterpret_cast<void**>(&mod_grad_buf),
        total_size,
        "datastream_mod_grad_buf");

    w_gate_data = w_gate_t.data;
    w_gate_t.ensure_grad();
    w_gate_grad = w_gate_t.grad_data();

    H_requires_grad = H_t.requires_grad;
    result_requires_grad = result_t.requires_grad;
    H_shape = H_t.shape;
    result_shape = result_t.shape;
    H_grad_fn = H_t.grad_fn;
    result_grad_fn = result_t.grad_fn;
    register_input(H_t.grad_fn);
    register_input(result_t.grad_fn);

    if (result_requires_grad) {
        if (result_t.is_leaf) {
            result_t.ensure_grad();
            grad_result_emb = result_t.grad_data();
        } else {
            float* buf = nullptr;
            cudaMallocOrThrow(
                reinterpret_cast<void**>(&buf),
                d_model_ * sizeof(float),
                "datastream_grad_result_emb");
            cudaMemsetAsync(buf, 0, d_model_ * sizeof(float), stream);
            owned_grad_result = std::shared_ptr<float>(buf, [](float* p) { cudaFree(p); });
            grad_result_emb = owned_grad_result.get();
        }
    }
}

void ExecutionBlockInjectGradFn::apply_impl(
    const Tensor& grad_output,
    cudaStream_t stream,
    const Batching::BatchPayload* backward_payload,
    const Batching::BatchDeviceBindings* backward_bindings)
{
    if (applied) return;
    applied = true;

    const size_t total_size = static_cast<size_t>(total_tokens) * d_model * sizeof(float);
    cudaMemcpyAsync(mod_grad_buf, grad_output.data, total_size, cudaMemcpyDeviceToDevice, stream);
    float* slot_grad = mod_grad_buf + static_cast<size_t>(result_slot) * d_model;

    if (!saved_gate.data || !saved_result_emb || !saved_H_slot) {
        throw std::runtime_error(
            "ExecutionBlockInjectGradFn::apply: saved state is NULL at " +
            std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    saved_gate.require("ExecutionBlockInjectGradFn::apply saved_gate");

    float* kernel_grad_result = grad_result_emb;
    std::shared_ptr<float> discard_buf;
    if (!kernel_grad_result) {
        float* tmp = nullptr;
        cudaMallocOrThrow(
            reinterpret_cast<void**>(&tmp),
            d_model * sizeof(float),
            "inject_discard_grad_result");
        cudaMemsetAsync(tmp, 0, d_model * sizeof(float), stream);
        discard_buf = std::shared_ptr<float>(tmp, [](float* p) { cudaFree(p); });
        kernel_grad_result = discard_buf.get();
    }

    kernelInjectSlotBackward<<<1, kBlockSize, 0, stream>>>(
        kernel_grad_result,
        w_gate_grad,
        slot_grad,
        saved_result_emb,
        saved_H_slot,
        w_gate_data,
        saved_gate.data,
        inv_sqrt_d,
        gate_temp,
        d_model);
    checkKernel("ExecutionBlockInjectGradFn::apply kernelInjectSlotBackward");

    if (result_requires_grad && result_grad_fn && grad_result_emb) {
        Tensor view;
        view.data = grad_result_emb;
        view.shape = result_shape;
        view.owns_data = false;
        view.stream = stream;
        result_grad_fn->apply(view, stream, backward_payload, backward_bindings);
    }

    if (H_requires_grad && H_grad_fn && mod_grad_buf) {
        Tensor view;
        view.data = mod_grad_buf;
        view.shape = H_shape;
        view.owns_data = false;
        view.stream = stream;
        H_grad_fn->apply(view, stream, backward_payload, backward_bindings);
    }
}

void ExecutionBlockInjectGradFn::release_saved() {
    GradFn::release_saved();
    if (saved_result_emb) { cudaFree(saved_result_emb); saved_result_emb = nullptr; }
    if (saved_H_slot) { cudaFree(saved_H_slot); saved_H_slot = nullptr; }
    saved_gate = Tensor{};
    if (mod_grad_buf) { cudaFree(mod_grad_buf); mod_grad_buf = nullptr; }
    grad_result_emb = nullptr;
    w_gate_data = nullptr;
    w_gate_grad = nullptr;
    H_grad_fn.reset();
    result_grad_fn.reset();
}

}  // namespace GRIM::autograd
