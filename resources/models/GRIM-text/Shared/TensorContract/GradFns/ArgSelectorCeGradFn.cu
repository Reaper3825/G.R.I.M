//======================================================//
//  ArgSelectorCeGradFn.cu
//  Backward node for the arg/option selector cross-entropy loss.
//======================================================//

#include "ArgSelectorCeGradFn.hpp"
#include "../TensorContract_GPU.hpp"
#include "../../CudaAllocUtils.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace GRIM {

using CudaAlloc::cudaMallocOrThrow;

namespace autograd {

namespace {

constexpr int kRowBlock = 256;

// grad_logits[t,e] = (softmax[t,e] - onehot(target[t])) * scale (0 for target<0).
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

}  // namespace autograd
}  // namespace GRIM
