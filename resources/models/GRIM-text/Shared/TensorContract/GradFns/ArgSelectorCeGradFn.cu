//======================================================//
//  ArgSelectorCeGradFn.cu
//  Selection cross-entropy forward + backward for the arg/option selector.
//======================================================//

#include "ArgSelectorCeGradFn.hpp"
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

namespace {

constexpr int kRowBlock = 256;

// One thread per row: stable softmax over `num_classes`, store probs, and (for
// supervised rows) accumulate -log p[target] into loss_sum. num_classes is the
// candidate-pool size (small), so a serial per-row loop is fine.
__global__ void kernel_selector_ce_forward(
    const float* __restrict__ logits,
    const int* __restrict__ targets,
    float* __restrict__ probs,
    float* __restrict__ loss_sum,
    int total_tokens,
    int num_classes)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total_tokens) return;

    const float* row = logits + static_cast<size_t>(t) * num_classes;
    float* prow = probs + static_cast<size_t>(t) * num_classes;

    float m = -1e30f;
    for (int e = 0; e < num_classes; ++e) {
        m = fmaxf(m, row[e]);
    }
    float sum = 0.0f;
    for (int e = 0; e < num_classes; ++e) {
        const float ex = __expf(row[e] - m);
        prow[e] = ex;
        sum += ex;
    }
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int e = 0; e < num_classes; ++e) {
        prow[e] *= inv;
    }

    const int tgt = targets[t];
    if (tgt >= 0 && tgt < num_classes) {
        atomicAdd(loss_sum, -logf(fmaxf(prow[tgt], 1e-20f)));
    }
}

// grad_logits[t,e] = (softmax[t,e] - onehot(target[t])) * scale  (0 for target<0).
__global__ void kernel_selector_ce_backward(
    const float* __restrict__ probs,
    const int* __restrict__ targets,
    float* __restrict__ grad_logits,
    int total_tokens,
    int num_classes,
    float scale)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total_tokens) return;

    const float* prow = probs + static_cast<size_t>(t) * num_classes;
    float* grow = grad_logits + static_cast<size_t>(t) * num_classes;
    const int tgt = targets[t];
    if (tgt < 0) {
        for (int e = 0; e < num_classes; ++e) grow[e] = 0.0f;
        return;
    }
    for (int e = 0; e < num_classes; ++e) {
        const float onehot = (e == tgt) ? 1.0f : 0.0f;
        grow[e] = (prow[e] - onehot) * scale;
    }
}

}  // namespace

ArgSelectorCeGradFn::ArgSelectorCeGradFn() {
    op_name = "arg_selector_ce";
}

ArgSelectorCeGradFn::~ArgSelectorCeGradFn() {
    release_saved();
}

void ArgSelectorCeGradFn::release_saved() {
    if (saved_probs) {
        cudaFree(saved_probs);
        saved_probs = nullptr;
    }
    GradFn::release_saved();
}

void ArgSelectorCeGradFn::apply_impl(const Tensor& grad_output,
                                     cudaStream_t stream,
                                     const Batching::BatchPayload* backward_payload,
                                     const Batching::BatchDeviceBindings* backward_bindings) {
    (void)backward_payload;
    (void)backward_bindings;
    if (applied) return;
    applied = true;
    if (!input_grad_fn || !input_grad_fn->op_name) return;
    if (!grad_output.data || grad_output.numel() < 1) return;
    if (!saved_probs) {
        throw std::runtime_error("ArgSelectorCeGradFn::apply: saved softmax probs were released before backward");
    }
    if (!targets) {
        throw std::runtime_error("ArgSelectorCeGradFn::apply: targets pointer is NULL");
    }

    float h_grad = 0.0f;
    cudaMemcpyAsync(&h_grad, grad_output.data, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    const float scale = h_grad / static_cast<float>(num_valid > 0 ? num_valid : 1);

    const size_t total_elems = static_cast<size_t>(total_tokens) * num_classes;
    float* grad_logits = nullptr;
    cudaMallocOrThrow(reinterpret_cast<void**>(&grad_logits), total_elems * sizeof(float),
                      "ArgSelectorCeGradFn_grad_logits");
    std::shared_ptr<float> grad_guard(grad_logits, [](float* p) { queueForDeferredCleanup(p); });

    const int blocks = (total_tokens + kRowBlock - 1) / kRowBlock;
    kernel_selector_ce_backward<<<blocks, kRowBlock, 0, stream>>>(
        saved_probs, targets, grad_logits, total_tokens, num_classes, scale);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("ArgSelectorCeGradFn::apply: backward kernel launch failed: ") +
                                 cudaGetErrorString(err));
    }

    Tensor view;
    view.data = grad_logits;
    view.shape = input_shape;
    view.owns_data = false;
    view.stream = stream;
    input_grad_fn->apply(view, stream, backward_payload, backward_bindings);
}

Tensor argSelectorLoss(const Tensor& selection_logits,
                       const int* d_targets,
                       int total_tokens,
                       int num_classes,
                       int num_valid,
                       cudaStream_t stream) {
    if (!stream) {
        throw std::runtime_error("argSelectorLoss: stream is NULL");
    }
    if (!d_targets) {
        throw std::runtime_error("argSelectorLoss: d_targets is NULL");
    }
    selection_logits.shape.require("argSelectorLoss selection_logits");
    if (!selection_logits.shape.is_2d_layout()) {
        throw std::runtime_error("argSelectorLoss: selection_logits must be 2D [total_tokens, num_classes]");
    }
    const auto shape = selection_logits.shape.as_2d();
    if (shape.rows != total_tokens || shape.cols != num_classes) {
        throw std::runtime_error("argSelectorLoss: selection_logits shape mismatch");
    }
    if (!selection_logits.data) {
        throw std::runtime_error("argSelectorLoss: selection_logits.data is NULL");
    }

    const size_t total_elems = static_cast<size_t>(total_tokens) * num_classes;
    float* probs = nullptr;
    cudaMallocOrThrow(reinterpret_cast<void**>(&probs), total_elems * sizeof(float), "argSelectorLoss_probs");
    float* d_loss_sum = nullptr;
    cudaMallocOrThrow(reinterpret_cast<void**>(&d_loss_sum), sizeof(float), "argSelectorLoss_loss_sum");
    cudaMemsetAsync(d_loss_sum, 0, sizeof(float), stream);

    const int blocks = (total_tokens + kRowBlock - 1) / kRowBlock;
    kernel_selector_ce_forward<<<blocks, kRowBlock, 0, stream>>>(
        selection_logits.data, d_targets, probs, d_loss_sum, total_tokens, num_classes);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(probs);
        cudaFree(d_loss_sum);
        throw std::runtime_error(std::string("argSelectorLoss: forward kernel launch failed: ") +
                                 cudaGetErrorString(err));
    }

    float h_loss_sum = 0.0f;
    cudaMemcpyAsync(&h_loss_sum, d_loss_sum, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_loss_sum);

    const int n = num_valid > 0 ? num_valid : 1;
    const float loss_value = h_loss_sum / static_cast<float>(n);

    float* d_out = nullptr;
    cudaMallocOrThrow(reinterpret_cast<void**>(&d_out), sizeof(float), "argSelectorLoss_out");
    cudaMemcpyAsync(d_out, &loss_value, sizeof(float), cudaMemcpyHostToDevice, stream);

    Tensor result;
    result.data = d_out;
    result.owns_data = true;
    result.shape = TensorContract::TensorShape::make_BSM(1, 1);
    result.is_leaf = false;
    result.requires_grad = selection_logits.requires_grad;
    result.stream = stream;

    if (selection_logits.requires_grad && selection_logits.grad_fn) {
        auto grad_fn = std::make_shared<ArgSelectorCeGradFn>();
        grad_fn->input_grad_fn = selection_logits.grad_fn;
        grad_fn->register_input(selection_logits.grad_fn);
        grad_fn->input_shape = selection_logits.shape;
        grad_fn->saved_probs = probs;  // ownership transferred to the GradFn
        grad_fn->targets = d_targets;
        grad_fn->total_tokens = total_tokens;
        grad_fn->num_classes = num_classes;
        grad_fn->num_valid = n;
        result.grad_fn = grad_fn;
    } else {
        cudaFree(probs);  // no backward — probs not needed
    }
    return result;
}

}  // namespace autograd
}  // namespace GRIM
