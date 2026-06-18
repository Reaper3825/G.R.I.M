//======================================================//
//  CrossEntropyLogitsGradFn.cu
//  Single-row cross entropy from logits + autograd backward.
//======================================================//

#include "CrossEntropyLogitsGradFn.hpp"
#include "../TensorContract_GPU.hpp"
#include "../../CudaAllocUtils.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

__global__ void kernel_ce_logits_forward(
    const float* __restrict__ logits,
    float* __restrict__ ce_out,
    float* __restrict__ saved_probs,
    int C,
    int target_idx
) {
    if (threadIdx.x != 0) return;

    float z_max = logits[0];
    for (int i = 1; i < C; ++i) z_max = fmaxf(z_max, logits[i]);

    float sum_exp = 0.0f;
    for (int i = 0; i < C; ++i) sum_exp += expf(logits[i] - z_max);
    sum_exp = fmaxf(sum_exp, 1e-10f);

    ce_out[0] = logf(sum_exp) + z_max - logits[target_idx];

    float inv_sum = 1.0f / sum_exp;
    for (int i = 0; i < C; ++i) {
        saved_probs[i] = expf(logits[i] - z_max) * inv_sum;
    }
}

__global__ void kernel_ce_logits_backward(
    const float* __restrict__ grad_output,
    const float* __restrict__ saved_probs,
    float* __restrict__ grad_logits,
    int C,
    int target_idx
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= C) return;
    float target_indicator = (i == target_idx) ? 1.0f : 0.0f;
    atomicAdd(&grad_logits[i], grad_output[0] * (saved_probs[i] - target_indicator));
}

}  // anonymous namespace

namespace GRIM {

using CudaAlloc::cudaMallocOrThrow;

namespace autograd {

CrossEntropyLogitsGradFn::CrossEntropyLogitsGradFn() {
    op_name = "cross_entropy_logits";
}

CrossEntropyLogitsGradFn::~CrossEntropyLogitsGradFn() {
    release_saved();
}

void CrossEntropyLogitsGradFn::capture_input(Tensor& logits, cudaStream_t stream) {
    input_grad_fn = logits.grad_fn;
    register_input(logits.grad_fn);
    input_shape = logits.shape;
    C = logits.shape.as_2d().cols;
    input_is_leaf = logits.is_leaf;

    if (logits.requires_grad) {
        if (logits.is_leaf) {
            logits.ensure_grad();
            grad_logits = logits.grad_data();
        } else {
            float* buf = nullptr;
            cudaMallocOrThrow(reinterpret_cast<void**>(&buf), static_cast<size_t>(C) * sizeof(float), "CELogitsGradFn_grad_logits");
            cudaMemsetAsync(buf, 0, static_cast<size_t>(C) * sizeof(float), stream);
            owned_grad_logits = std::shared_ptr<float>(buf, [](float* p) {
                queueForDeferredCleanup(p);
            });
            grad_logits = owned_grad_logits.get();
        }
    }
}

void CrossEntropyLogitsGradFn::apply_impl(const Tensor& grad_output,
                                          cudaStream_t stream,
                                          const Batching::BatchPayload* backward_payload,
                                          const Batching::BatchDeviceBindings* backward_bindings) {
    if (applied) return;
    applied = true;

    if (!saved_probs) {
        throw std::runtime_error("CrossEntropyLogitsGradFn::apply: saved_probs is NULL");
    }
    if (!grad_logits) {
        throw std::runtime_error("CrossEntropyLogitsGradFn::apply: grad_logits is NULL");
    }

    const int threads = (C < 256) ? C : 256;
    const int blocks = (C + threads - 1) / threads;
    kernel_ce_logits_backward<<<blocks, threads, 0, stream>>>(
        grad_output.data, saved_probs, grad_logits, C, target_idx);

    if (input_grad_fn && input_grad_fn->op_name) {
        Tensor view;
        view.data = grad_logits;
        view.shape = input_shape;
        view.owns_data = false;
        view.stream = stream;
        input_grad_fn->apply(view, stream, backward_payload, backward_bindings);
    }
}

void CrossEntropyLogitsGradFn::release_saved() {
    GradFn::release_saved();
    if (owns_saved && saved_probs) {
        queueForDeferredCleanup(saved_probs);
    }
    saved_probs = nullptr;
    grad_logits = nullptr;
    input_grad_fn.reset();
}

Tensor cross_entropy_logits(const Tensor& logits, int target_idx, cudaStream_t stream) {
    if (!logits.shape.is_2d_layout()) {
        throw std::invalid_argument("autograd::cross_entropy_logits: logits must be 2D");
    }
    const auto dims = logits.shape.as_2d();
    if (dims.rows != 1) {
        throw std::invalid_argument("autograd::cross_entropy_logits: logits must be [1, C], got rows=" +
                                    std::to_string(dims.rows));
    }
    const int C = dims.cols;
    if (target_idx < 0 || target_idx >= C) {
        throw std::invalid_argument("autograd::cross_entropy_logits: target_idx=" +
                                    std::to_string(target_idx) + " out of range [0, " +
                                    std::to_string(C) + ")");
    }
    if (!logits.data) {
        throw std::invalid_argument("autograd::cross_entropy_logits: logits.data is NULL");
    }

    auto out_shape = TensorContract::TensorShape::make_BSM(1, 1);
    Tensor result = Tensor::zeros(out_shape, logits.requires_grad, stream, "ce_logits_result");

    float* saved_probs = nullptr;
    if (logits.requires_grad) {
        cudaMallocOrThrow(reinterpret_cast<void**>(&saved_probs), static_cast<size_t>(C) * sizeof(float), "ce_logits_saved_probs");
    }

    kernel_ce_logits_forward<<<1, 32, 0, stream>>>(
        logits.data, result.data, saved_probs ? saved_probs : result.data,
        C, target_idx);

    if (logits.requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<CrossEntropyLogitsGradFn>();
        grad_fn->capture_input(const_cast<Tensor&>(logits), stream);
        grad_fn->saved_probs = saved_probs;
        grad_fn->target_idx = target_idx;
        result.grad_fn = grad_fn;
    }

    return result;
}

}  // namespace autograd
}  // namespace GRIM
