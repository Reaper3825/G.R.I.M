//======================================================//
//  NormalizedEntropyGradFn.cu
//  Normalized entropy forward operation and backward node.
//======================================================//

#include "NormalizedEntropyGradFn.hpp"
#include "../GradientAccumulation.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace GRIM::autograd {

namespace {

constexpr int kEntropyBlockSize = 256;

__global__ void kernelNormalizedEntropyForward(
    float* __restrict__ out_entropy,
    const float* __restrict__ probs,
    int num_classes)
{
    if (threadIdx.x != 0) return;
    if (num_classes <= 1) {
        out_entropy[0] = 0.0f;
        return;
    }

    constexpr float kMinProb = 1e-10f;
    float entropy = 0.0f;
    for (int i = 0; i < num_classes; ++i) {
        const float p = fmaxf(probs[i], kMinProb);
        entropy -= p * logf(p);
    }

    const float max_entropy = logf(static_cast<float>(num_classes));
    out_entropy[0] = (max_entropy > kMinProb) ? (entropy / max_entropy) : 0.0f;
}

__global__ void kernelNormalizedEntropyBackward(
    float* __restrict__ grad_probs,
    const float* __restrict__ grad_output,
    const float* __restrict__ saved_probs,
    int num_classes)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_classes || num_classes <= 1) return;

    constexpr float kMinProb = 1e-10f;
    const float max_entropy = logf(static_cast<float>(num_classes));
    if (max_entropy <= kMinProb) return;

    const float p = fmaxf(saved_probs[idx], kMinProb);
    grad_probs[idx] += grad_output[0] * (-(logf(p) + 1.0f) / max_entropy);
}

void checkCuda(cudaError_t err, const char* caller) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(caller) + ": " + cudaGetErrorString(err));
    }
}

}  // namespace

NormalizedEntropyGradFn::NormalizedEntropyGradFn() {
    op_name = "autograd_exec_normalized_entropy";
}

void NormalizedEntropyGradFn::capture(
    Tensor& probs_tensor,
    Tensor& saved_probs_staging,
    Tensor& grad_probs_staging,
    cudaStream_t stream)
{
    if (!probs_tensor.data) {
        throw std::runtime_error("normalized_entropy: probs tensor data is NULL");
    }
    if (!probs_tensor.shape.is_2d_layout()) {
        throw std::runtime_error("normalized_entropy: probs tensor must be 2D");
    }
    if (probs_tensor.shape.flat.rows != 1) {
        throw std::runtime_error(
            "normalized_entropy: probs tensor must be [1, C], got rows=" +
            std::to_string(probs_tensor.shape.flat.rows));
    }

    num_classes_ = probs_tensor.shape.flat.cols;
    probs_requires_grad_ = probs_tensor.requires_grad;
    probs_is_leaf_ = probs_tensor.is_leaf;
    probs_shape_ = probs_tensor.shape;
    probs_grad_fn_ = probs_tensor.grad_fn;
    register_input(probs_tensor.grad_fn);

    saved_probs_staging.require("NormalizedEntropyGradFn::capture saved_probs_staging");
    grad_probs_staging.require("NormalizedEntropyGradFn::capture grad_probs_staging");
    if (!saved_probs_staging.shape.is_2d_layout() ||
        !grad_probs_staging.shape.is_2d_layout() ||
        saved_probs_staging.shape.as_2d() != probs_tensor.shape.as_2d() ||
        grad_probs_staging.shape.as_2d() != probs_tensor.shape.as_2d()) {
        throw std::runtime_error(
            "NormalizedEntropyGradFn::capture: staging tensors must match probs shape");
    }

    saved_probs_view_ = Tensor::from_ptr(
        saved_probs_staging.data,
        saved_probs_staging.shape,
        false,
        false,
        "normalized_entropy_saved_probs_view");
    saved_probs_view_.stream = stream;
    grad_probs_view_ = Tensor::from_ptr(
        grad_probs_staging.data,
        grad_probs_staging.shape,
        false,
        false,
        "normalized_entropy_grad_probs_view");
    grad_probs_view_.stream = stream;

    checkCuda(
        cudaMemcpyAsync(
            saved_probs_view_.data,
            probs_tensor.data,
            static_cast<size_t>(num_classes_) * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream),
        "NormalizedEntropyGradFn::capture cudaMemcpyAsync");

    if (!probs_requires_grad_) return;

    if (probs_is_leaf_) {
        probs_tensor.ensure_grad();
        probs_leaf_grad_ = probs_tensor.grad_data();
    }
}

void NormalizedEntropyGradFn::apply_impl(
    const Tensor& grad_output,
    cudaStream_t stream,
    const Batching::BatchPayload* backward_payload,
    const Batching::BatchDeviceBindings* backward_bindings)
{
    if (applied) return;
    applied = true;
    if (!probs_requires_grad_ || !grad_probs_view_.data) return;

    const int blocks = (num_classes_ + kEntropyBlockSize - 1) / kEntropyBlockSize;
    kernelNormalizedEntropyBackward<<<blocks, kEntropyBlockSize, 0, stream>>>(
        grad_probs_view_.data,
        grad_output.data,
        saved_probs_view_.data,
        num_classes_);
    checkCuda(cudaGetLastError(), "NormalizedEntropyGradFn::apply kernelNormalizedEntropyBackward");

    if (probs_is_leaf_ && probs_leaf_grad_) {
        accumulate_grad(
            probs_leaf_grad_,
            grad_probs_view_.data,
            static_cast<size_t>(num_classes_),
            1.0f,
            stream,
            "NormalizedEntropyGradFn::apply probs_leaf_grad");
    } else if (probs_grad_fn_) {
        Tensor view;
        view.data = grad_probs_view_.data;
        view.shape = probs_shape_;
        view.owns_data = false;
        view.stream = stream;
        probs_grad_fn_->apply(view, stream, backward_payload, backward_bindings);
    }
}

void NormalizedEntropyGradFn::release_saved() {
    GradFn::release_saved();
    saved_probs_view_ = Tensor();
    grad_probs_view_ = Tensor();
    probs_leaf_grad_ = nullptr;
    probs_grad_fn_.reset();
}

Tensor normalized_entropy(
    Tensor& probs_tensor,
    Tensor& saved_probs_staging,
    Tensor& grad_probs_staging,
    cudaStream_t stream)
{
    if (!probs_tensor.data) {
        throw std::runtime_error("normalized_entropy: probs tensor data is NULL");
    }
    if (!probs_tensor.shape.is_2d_layout()) {
        throw std::runtime_error("normalized_entropy: probs tensor must be 2D");
    }
    if (probs_tensor.shape.flat.rows != 1) {
        throw std::runtime_error(
            "normalized_entropy: probs tensor must be [1, C], got rows=" +
            std::to_string(probs_tensor.shape.flat.rows));
    }

    Tensor entropy = Tensor::zeros({1, 1}, stream, "autograd_exec_normalized_entropy");
    kernelNormalizedEntropyForward<<<1, 1, 0, stream>>>(
        entropy.data,
        probs_tensor.data,
        probs_tensor.shape.flat.cols);
    checkCuda(cudaGetLastError(), "normalized_entropy kernelNormalizedEntropyForward");

    auto grad_fn = std::make_shared<NormalizedEntropyGradFn>();
    grad_fn->capture(
        probs_tensor,
        saved_probs_staging,
        grad_probs_staging,
        stream);
    entropy.grad_fn = grad_fn;
    entropy.requires_grad = true;
    entropy.is_leaf = false;
    return entropy;
}

}  // namespace GRIM::autograd
