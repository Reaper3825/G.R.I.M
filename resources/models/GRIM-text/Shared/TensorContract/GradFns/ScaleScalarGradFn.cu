//======================================================//
//  ScaleScalarGradFn.cu
//  Scale a 1-element tensor by a constant. Host-side math; no CUDA kernel.
//======================================================//

#include "ScaleScalarGradFn.hpp"
#include "../TensorContract_GPU.hpp"
#include "../../CudaAllocUtils.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <string>

#define AG_TRACE(...) do { if constexpr (GRIM::VerboseLogging::ENABLE_AUTOGRAD_TRACE_LOGS) { fprintf(stderr, __VA_ARGS__); fflush(stderr); } } while(0)

namespace GRIM {

using CudaAlloc::cudaMallocOrThrow;

namespace autograd {

ScaleScalarGradFn::ScaleScalarGradFn() {
    op_name = "scale_scalar";
}

void ScaleScalarGradFn::apply_impl(const Tensor& grad_output,
                                   cudaStream_t stream,
                                   const Batching::BatchPayload* backward_payload,
                                   const Batching::BatchDeviceBindings* backward_bindings) {
    if (applied) return;
    applied = true;
    if (!input_grad_fn || !input_grad_fn->op_name) return;
    if (!grad_output.data || grad_output.numel() < 1) return;

    float h_grad = 0.0f;
    cudaMemcpyAsync(&h_grad, grad_output.data, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    const float scaled = scale * h_grad;
    float* d_scaled_raw = nullptr;
    cudaMallocOrThrow(reinterpret_cast<void**>(&d_scaled_raw), sizeof(float), "ScaleScalarGradFn_d_scaled");
    std::shared_ptr<float> d_scaled_guard(d_scaled_raw, [](float* p) { queueForDeferredCleanup(p); });
    cudaMemcpyAsync(d_scaled_raw, &scaled, sizeof(float), cudaMemcpyHostToDevice, stream);
    Tensor view;
    view.data = d_scaled_raw;
    view.shape = input_shape;
    view.owns_data = false;
    view.stream = stream;
    input_grad_fn->apply(view, stream, backward_payload, backward_bindings);
}

// ═══════════════════════════════════════════════════════════════════════════
// autograd::scale_scalar — forward op
// ═══════════════════════════════════════════════════════════════════════════

Tensor scale_scalar(const Tensor& t, float scale, cudaStream_t stream) {
    if (stream == nullptr || stream == 0) {
        throw std::runtime_error("autograd::scale_scalar: stream is NULL");
    }
    if (t.numel() != 1) {
        throw std::invalid_argument("autograd::scale_scalar: input must be scalar (1 element), got " + std::to_string(t.numel()));
    }
    float h_val = 0.0f;
    cudaMemcpyAsync(&h_val, t.data, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    const float scaled_val = scale * h_val;
    float* d_out = nullptr;
    cudaMallocOrThrow(reinterpret_cast<void**>(&d_out), sizeof(float), "scale_scalar_d_out");
    cudaMemcpyAsync(d_out, &scaled_val, sizeof(float), cudaMemcpyHostToDevice, stream);
    Tensor result;
    result.data = d_out;
    result.owns_data = true;
    result.shape = TensorContract::TensorShape::make_BSM(1, 1);
    result.is_leaf = false;
    result.requires_grad = t.requires_grad;
    result.stream = stream;
    if (t.requires_grad && t.grad_fn) {
        auto grad_fn = std::make_shared<ScaleScalarGradFn>();
        grad_fn->input_grad_fn = t.grad_fn;
        grad_fn->register_input(t.grad_fn);
        grad_fn->input_shape = t.shape;
        grad_fn->scale = scale;
        result.grad_fn = grad_fn;
    }
    return result;
}

}  // namespace autograd
}  // namespace GRIM
