//======================================================//
//  ArgSelectorLoss.cu
//  Forward loss operation for arg/option selector supervision.
//======================================================//

#include "ArgSelectorLoss.hpp"
#include "../../CudaAllocUtils.hpp"
#include "../../TensorContract/GradFns/ArgSelectorCeGradFn.hpp"

#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <string>

namespace GRIM {

using CudaAlloc::cudaMallocOrThrow;

namespace autograd {

namespace {

constexpr int kRowBlock = 256;

// One thread per row: stable softmax over `num_classes`, store probabilities,
// and reduce supervised -log p[target] terms into the scalar loss sum.
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

}  // namespace

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
        cudaFree(probs);  // no backward; probabilities are not needed
    }
    return result;
}

}  // namespace autograd
}  // namespace GRIM
