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
    int N,
    bool old_trace_requires_grad,
    bool candidate_requires_grad,
    bool gate_logits_requires_grad)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const float g = gate_vals[i];
    const float u = upstream[i];
    if (old_trace_requires_grad) d_old_trace[i] += u * g;
    if (candidate_requires_grad) d_candidate[i] += u * (1.0f - g);
    if (gate_logits_requires_grad) {
        d_gate_logits[i] += u * (old_trace[i] - candidate[i]) * g * (1.0f - g);
    }
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
    if (old_trace_requires_grad) {
        old_trace_gradient = capture_input_gradient(
            old_trace_t, stream, "GatedTraceUpdateGradFn::capture old_trace");
    }
    if (candidate_requires_grad) {
        candidate_gradient = capture_input_gradient(
            candidate_t, stream, "GatedTraceUpdateGradFn::capture candidate");
    }
    if (gate_logits_requires_grad) {
        gate_logits_gradient = capture_input_gradient(
            gate_logits_t, stream, "GatedTraceUpdateGradFn::capture gate_logits");
    }
}

void GatedTraceUpdateGradFn::apply_impl(
    const Tensor& grad_output,
    cudaStream_t stream,
    const Batching::BatchPayload* backward_payload,
    const Batching::BatchDeviceBindings* backward_bindings)
{
    if (applied) return;
    applied = true;

    float* grad_old_trace = nullptr;
    float* grad_candidate = nullptr;
    float* grad_gate_logits = nullptr;
    if (old_trace_requires_grad) {
        if (!old_trace_gradient) {
            throw std::runtime_error("GatedTraceUpdateGradFn::apply: old_trace gradient Tensor is NULL");
        }
        grad_old_trace = old_trace_gradient->data;
    }
    if (candidate_requires_grad) {
        if (!candidate_gradient) {
            throw std::runtime_error("GatedTraceUpdateGradFn::apply: candidate gradient Tensor is NULL");
        }
        grad_candidate = candidate_gradient->data;
    }
    if (gate_logits_requires_grad) {
        if (!gate_logits_gradient) {
            throw std::runtime_error("GatedTraceUpdateGradFn::apply: gate_logits gradient Tensor is NULL");
        }
        grad_gate_logits = gate_logits_gradient->data;
    }

    const int blocks = (dm_ + kBlockSize - 1) / kBlockSize;
    kernelGatedTraceUpdateBackward<<<blocks, kBlockSize, 0, stream>>>(
        grad_old_trace,
        grad_candidate,
        grad_gate_logits,
        grad_output.data,
        saved_old_trace,
        saved_candidate,
        saved_gate_vals,
        dm_,
        old_trace_requires_grad,
        candidate_requires_grad,
        gate_logits_requires_grad);
    checkKernel("GatedTraceUpdateGradFn::apply kernelGatedTraceUpdateBackward");

    if (old_trace_requires_grad) {
        propagate_input_gradient(
            old_trace_gradient, stream, backward_payload, backward_bindings,
            "GatedTraceUpdateGradFn::apply old_trace");
    }
    if (candidate_requires_grad) {
        propagate_input_gradient(
            candidate_gradient, stream, backward_payload, backward_bindings,
            "GatedTraceUpdateGradFn::apply candidate");
    }
    if (gate_logits_requires_grad) {
        propagate_input_gradient(
            gate_logits_gradient, stream, backward_payload, backward_bindings,
            "GatedTraceUpdateGradFn::apply gate_logits");
    }
}

void GatedTraceUpdateGradFn::release_saved() {
    GradFn::release_saved();
    if (saved_old_trace) { cudaFree(saved_old_trace); saved_old_trace = nullptr; }
    if (saved_candidate) { cudaFree(saved_candidate); saved_candidate = nullptr; }
    if (saved_gate_vals) { cudaFree(saved_gate_vals); saved_gate_vals = nullptr; }
    old_trace_gradient.reset();
    candidate_gradient.reset();
    gate_logits_gradient.reset();
}

}  // namespace GRIM::autograd
