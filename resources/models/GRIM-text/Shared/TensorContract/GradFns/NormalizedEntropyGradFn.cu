//======================================================//
//  NormalizedEntropyGradFn.cu
//  Normalized entropy forward operation and backward node.
//======================================================//

#include "NormalizedEntropyGradFn.hpp"
#include "../../CudaAllocUtils.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace GRIM::autograd {

using CudaAlloc::cudaMallocOrThrow;

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

NormalizedEntropyGradFn::~NormalizedEntropyGradFn() {
    if (saved_probs_) cudaFree(saved_probs_);
}

void NormalizedEntropyGradFn::capture(Tensor& probs_tensor, cudaStream_t stream) {
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
    probs_shape_ = probs_tensor.shape;
    probs_grad_fn_ = probs_tensor.grad_fn;
    register_input(probs_tensor.grad_fn);

    cudaMallocOrThrow(
        reinterpret_cast<void**>(&saved_probs_),
        static_cast<size_t>(num_classes_) * sizeof(float),
        "autograd_exec_entropy_saved_probs");
    checkCuda(
        cudaMemcpyAsync(
            saved_probs_,
            probs_tensor.data,
            static_cast<size_t>(num_classes_) * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream),
        "NormalizedEntropyGradFn::capture cudaMemcpyAsync");

    if (!probs_requires_grad_) return;

    if (probs_tensor.is_leaf) {
        probs_tensor.ensure_grad();
        grad_probs_ = probs_tensor.grad_data();
    } else {
        float* buf = nullptr;
        cudaMallocOrThrow(
            reinterpret_cast<void**>(&buf),
            static_cast<size_t>(num_classes_) * sizeof(float),
            "autograd_exec_entropy_grad_probs");
        checkCuda(
            cudaMemsetAsync(
                buf,
                0,
                static_cast<size_t>(num_classes_) * sizeof(float),
                stream),
            "NormalizedEntropyGradFn::capture cudaMemsetAsync");
        owned_grad_probs_ = std::shared_ptr<float>(buf, [](float* p) { cudaFree(p); });
        grad_probs_ = owned_grad_probs_.get();
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
    if (!probs_requires_grad_ || !grad_probs_) return;

    const int blocks = (num_classes_ + kEntropyBlockSize - 1) / kEntropyBlockSize;
    kernelNormalizedEntropyBackward<<<blocks, kEntropyBlockSize, 0, stream>>>(
        grad_probs_,
        grad_output.data,
        saved_probs_,
        num_classes_);
    checkCuda(cudaGetLastError(), "NormalizedEntropyGradFn::apply kernelNormalizedEntropyBackward");

    if (probs_grad_fn_) {
        Tensor view;
        view.data = grad_probs_;
        view.shape = probs_shape_;
        view.owns_data = false;
        view.stream = stream;
        probs_grad_fn_->apply(view, stream, backward_payload, backward_bindings);
    }
}

void NormalizedEntropyGradFn::release_saved() {
    GradFn::release_saved();
    if (saved_probs_) { cudaFree(saved_probs_); saved_probs_ = nullptr; }
    grad_probs_ = nullptr;
    owned_grad_probs_.reset();
    probs_grad_fn_.reset();
}

Tensor normalized_entropy(Tensor& probs_tensor, cudaStream_t stream) {
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
    grad_fn->capture(probs_tensor, stream);
    entropy.grad_fn = grad_fn;
    entropy.requires_grad = true;
    entropy.is_leaf = false;
    return entropy;
}

}  // namespace GRIM::autograd
