//======================================================//
//  GatedTraceUpdateGradFn.cu
//  Backward implementation for the execution trace's gated update.
//======================================================//

#include "GatedTraceUpdateGradFn.hpp"
#include "../../CudaAllocUtils.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace GRIM::autograd {

using CudaAlloc::cudaMallocOrThrow;

namespace {

constexpr int kBlockSize = 256;

__global__ void kernelGatedTraceUpdateBackward(
    float* __restrict__ d_old_trace,
    float* __restrict__ d_candidate,
    float* __restrict__ d_gate_logits,
    const float* __restrict__ upstream,
    const float* __restrict__ old_trace,
    const float* __restrict__ candidate,
    const float* __restrict__ gate_vals,
    int N)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const float g = gate_vals[i];
    const float u = upstream[i];
    d_old_trace[i] += u * g;
    d_candidate[i] += u * (1.0f - g);
    d_gate_logits[i] += u * (old_trace[i] - candidate[i]) * g * (1.0f - g);
}

void checkKernel(const char* caller) {
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(caller) + ": " + cudaGetErrorString(err));
    }
}

}  // namespace

GatedTraceUpdateGradFn::GatedTraceUpdateGradFn() {
    op_name = "gated_trace_update";
}

GatedTraceUpdateGradFn::~GatedTraceUpdateGradFn() {
    if (saved_old_trace) cudaFree(saved_old_trace);
    if (saved_candidate) cudaFree(saved_candidate);
    if (saved_gate_vals) cudaFree(saved_gate_vals);
}

void GatedTraceUpdateGradFn::capture(
    Tensor& old_trace_t,
    Tensor& candidate_t,
    Tensor& gate_logits_t,
    float* gate_vals_buf,
    int dm,
    cudaStream_t stream)
{
    dm_ = dm;
    saved_gate_vals = gate_vals_buf;

    cudaMallocOrThrow(
        reinterpret_cast<void**>(&saved_old_trace),
        dm * sizeof(float),
        "gated_trace_saved_old");
    cudaMallocOrThrow(
        reinterpret_cast<void**>(&saved_candidate),
        dm * sizeof(float),
        "gated_trace_saved_cand");
    cudaMemcpyAsync(saved_old_trace, old_trace_t.data, dm * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(saved_candidate, candidate_t.data, dm * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    old_trace_requires_grad = old_trace_t.requires_grad;
    candidate_requires_grad = candidate_t.requires_grad;
    gate_logits_requires_grad = gate_logits_t.requires_grad;
    old_trace_shape = old_trace_t.shape;
    candidate_shape = candidate_t.shape;
    gate_logits_shape = gate_logits_t.shape;
    old_trace_grad_fn = old_trace_t.grad_fn;
    candidate_grad_fn = candidate_t.grad_fn;
    gate_logits_grad_fn = gate_logits_t.grad_fn;
    register_input(old_trace_t.grad_fn);
    register_input(candidate_t.grad_fn);
    register_input(gate_logits_t.grad_fn);

    auto setup_grad_buf = [&](Tensor& tensor,
                              float*& grad,
                              std::shared_ptr<float>& owned,
                              size_t count) {
        if (!tensor.requires_grad) return;
        if (tensor.is_leaf) {
            tensor.ensure_grad();
            grad = tensor.grad_data();
        } else {
            float* buf = nullptr;
            cudaMallocOrThrow(
                reinterpret_cast<void**>(&buf),
                count * sizeof(float),
                "gated_trace_grad_buf");
            cudaMemsetAsync(buf, 0, count * sizeof(float), stream);
            owned = std::shared_ptr<float>(buf, [](float* p) { cudaFree(p); });
            grad = owned.get();
        }
    };

    setup_grad_buf(old_trace_t, grad_old_trace, owned_grad_old_trace, dm);
    setup_grad_buf(candidate_t, grad_candidate, owned_grad_candidate, dm);
    setup_grad_buf(gate_logits_t, grad_gate_logits, owned_grad_gate_logits, dm);
}

void GatedTraceUpdateGradFn::apply_impl(
    const Tensor& grad_output,
    cudaStream_t stream,
    const Batching::BatchPayload* backward_payload,
    const Batching::BatchDeviceBindings* backward_bindings)
{
    if (applied) return;
    applied = true;

    const int blocks = (dm_ + kBlockSize - 1) / kBlockSize;
    kernelGatedTraceUpdateBackward<<<blocks, kBlockSize, 0, stream>>>(
        grad_old_trace,
        grad_candidate,
        grad_gate_logits,
        grad_output.data,
        saved_old_trace,
        saved_candidate,
        saved_gate_vals,
        dm_);
    checkKernel("GatedTraceUpdateGradFn::apply kernelGatedTraceUpdateBackward");

    if (old_trace_requires_grad && old_trace_grad_fn) {
        Tensor view;
        view.data = grad_old_trace;
        view.shape = old_trace_shape;
        view.owns_data = false;
        view.stream = stream;
        old_trace_grad_fn->apply(view, stream, backward_payload, backward_bindings);
    }
    if (candidate_requires_grad && candidate_grad_fn) {
        Tensor view;
        view.data = grad_candidate;
        view.shape = candidate_shape;
        view.owns_data = false;
        view.stream = stream;
        candidate_grad_fn->apply(view, stream, backward_payload, backward_bindings);
    }
    if (gate_logits_requires_grad && gate_logits_grad_fn) {
        Tensor view;
        view.data = grad_gate_logits;
        view.shape = gate_logits_shape;
        view.owns_data = false;
        view.stream = stream;
        gate_logits_grad_fn->apply(view, stream, backward_payload, backward_bindings);
    }
}

void GatedTraceUpdateGradFn::release_saved() {
    GradFn::release_saved();
    if (saved_old_trace) { cudaFree(saved_old_trace); saved_old_trace = nullptr; }
    if (saved_candidate) { cudaFree(saved_candidate); saved_candidate = nullptr; }
    if (saved_gate_vals) { cudaFree(saved_gate_vals); saved_gate_vals = nullptr; }
    grad_old_trace = nullptr;
    grad_candidate = nullptr;
    grad_gate_logits = nullptr;
    old_trace_grad_fn.reset();
    candidate_grad_fn.reset();
    gate_logits_grad_fn.reset();
}

}  // namespace GRIM::autograd
