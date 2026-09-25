//======================================================//
//  RMSNormGradFn.cu
//  RMSNorm forward + autograd backward.
//======================================================//

#include "RMSNormGradFn.hpp"
#include "../AutogradEngine.hpp"
#include "../TensorContract_GPU.hpp"
#include "../../Diagnostics/MemoryAllocationTracker.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef TENSOR_VERBOSE_DEBUG
#define TENSOR_VERBOSE_DEBUG 0
#endif

#define AG_TRACE(...) do { if constexpr (GRIM::VerboseLogging::ENABLE_AUTOGRAD_TRACE_LOGS) { fprintf(stderr, __VA_ARGS__); fflush(stderr); } } while(0)

void trackKernelLaunch(const char* kernel_name, cudaStream_t stream);

namespace {

constexpr int AUTOGRAD_BLOCK_SIZE = 256;

__global__ void kernel_rmsnorm_forward(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    float* __restrict__ output,
    int tokens,
    int d_model,
    float eps
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= tokens) return;

    extern __shared__ float shared[];

    const float* x = input + static_cast<size_t>(token_idx) * d_model;
    float* y = output + static_cast<size_t>(token_idx) * d_model;

    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        local_sum_sq += x[i] * x[i];
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
    }

    if (threadIdx.x % 32 == 0) {
        shared[threadIdx.x / 32] = local_sum_sq;
    }
    __syncthreads();

    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        float total = 0.0f;
        const int num_warps = blockDim.x / 32;
        for (int i = 0; i < num_warps; i++) {
            total += shared[i];
        }
        float rms_sq = total / d_model + eps;
        s_inv_rms = rsqrtf(rms_sq);
    }
    __syncthreads();

    const float inv_rms = s_inv_rms;

    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        y[i] = x[i] * inv_rms * gamma[i];
    }
}

__global__ void kernel_rmsnorm_backward(
    const float* grad_output,
    const float* input,
    const float* gamma,
    float* grad_input,   // optional: nullptr when only gamma trains (e.g. frozen input / embeddings)
    float* grad_gamma,
    int tokens,
    int d_model,
    float eps
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= tokens) return;

    extern __shared__ float shared[];
    float* s_warp_vals = shared;

    const float* x = input + static_cast<size_t>(token_idx) * d_model;
    const float* dy = grad_output + static_cast<size_t>(token_idx) * d_model;
    float* const dx =
        grad_input ? grad_input + static_cast<size_t>(token_idx) * d_model : nullptr;

    const int num_warps = blockDim.x / 32;

    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        local_sum_sq += x[i] * x[i];
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
    }

    if (threadIdx.x % 32 == 0) {
        s_warp_vals[threadIdx.x / 32] = local_sum_sq;
    }
    __syncthreads();

    __shared__ float s_rms_sq, s_inv_rms;
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int i = 0; i < num_warps; i++) {
            total += s_warp_vals[i];
        }
        s_rms_sq = total / d_model + eps;
        s_inv_rms = rsqrtf(s_rms_sq);
    }
    __syncthreads();

    const float inv_rms = s_inv_rms;
    const float rms_sq = s_rms_sq;

    float local_dgamma_x = 0.0f;
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        local_dgamma_x += dy[i] * gamma[i] * x[i];
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        local_dgamma_x += __shfl_down_sync(0xffffffff, local_dgamma_x, offset);
    }

    if (threadIdx.x % 32 == 0) {
        s_warp_vals[threadIdx.x / 32] = local_dgamma_x;
    }
    __syncthreads();

    __shared__ float s_dgamma_x;
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int i = 0; i < num_warps; i++) {
            total += s_warp_vals[i];
        }
        s_dgamma_x = total;
    }
    __syncthreads();

    const float dgamma_x_sum = s_dgamma_x;
    const float scale = dgamma_x_sum / (d_model * rms_sq);
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        if (grad_input) {
            dx[i] += (dy[i] * gamma[i] - x[i] * scale) * inv_rms;
        }

        if (grad_gamma) {
            atomicAdd(&grad_gamma[i], dy[i] * x[i] * inv_rms);
        }
    }
}

}  // anonymous namespace

namespace GRIM {

using MemoryAccounting::cudaMallocOrThrow;

namespace autograd {

RMSNormGradFn::RMSNormGradFn() {
    op_name = "rms_norm";
}

RMSNormGradFn::~RMSNormGradFn() {
    release_saved();
}

void RMSNormGradFn::capture_inputs(Tensor& x, Tensor& gamma_tensor, cudaStream_t stream) {
    if (!stream) throw std::runtime_error("RMSNormGradFn::capture: stream is NULL");
    input_requires_grad = x.requires_grad;
    gamma_requires_grad = gamma_tensor.requires_grad;
    input_shape = x.shape;
    gamma_shape = gamma_tensor.shape;
    gamma_data = gamma_tensor.data;
    auto capture = [&](Tensor& x, std::shared_ptr<GradFn>& producer, Tensor*& leaf) {
        x.require("RMSNormGradFn::capture");
        producer.reset();
        leaf = nullptr;
        if (!x.requires_grad) return;
        if (x.is_leaf) {
            if (x.grad_fn) throw std::runtime_error("RMSNormGradFn::capture: leaf has a producer");
            x.ensure_grad();
            leaf = x.grad_.get();
            if (!leaf) throw std::runtime_error("RMSNormGradFn::capture: missing leaf gradient");
        } else {
            if (!x.grad_fn) throw std::runtime_error("RMSNormGradFn::capture: non-leaf has no producer");
            producer = x.grad_fn;
            register_input(producer);
        }
    };
    capture(x, input_grad_fn, leaf_input_gradient);
    capture(gamma_tensor, gamma_grad_fn, leaf_gamma_gradient);
}

void RMSNormGradFn::set_cache_copy(const float* external_cache, size_t size, int d, float e, cudaStream_t stream) {
    if (!external_cache) {
        throw std::runtime_error("RMSNormGradFn::set_cache_copy: external_cache is NULL - caller MUST provide cache");
    }
    cached_size = size;
    d_model = d;
    eps = e;

    float* buffer = nullptr;
    cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), size * sizeof(float), "RMSNormGradFn_cache", GRIM::MemoryAccounting::Kind::Saved);
    cudaMemcpyAsync(buffer, external_cache, size * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    owned_cache = std::shared_ptr<float>(buffer, [](float* p) {
        queueForDeferredCleanup(p);
    });
    cached_input = owned_cache.get();
    AG_TRACE("[RMSNormGradFn] Copied cache: %zu floats to %p\n", size, (void*)cached_input);
}

void RMSNormGradFn::apply_impl(const Tensor& grad_output,
                               cudaStream_t stream,
                               const Batching::BatchPayload* backward_payload,
                               const Batching::BatchDeviceBindings* backward_bindings) {
    setCurrentGradFnOp("rms_norm", this);

    if (applied) {
        return;
    }
    if (!stream) throw std::runtime_error("RMSNormGradFn::apply: stream is NULL");
    grad_output.require("RMSNormGradFn::apply grad_output");

#if TENSOR_VERBOSE_DEBUG
    fprintf(stderr, "[RMSNormGradFn::apply] ENTRY - this=%p grad_output.data=%p stream=%p\n",
            (void*)this, (void*)grad_output.data, (void*)stream);
    fflush(stderr);
#endif

    if (!cached_input) {
        throw std::runtime_error("RMSNormGradFn::apply: cached_input is NULL - set_cache() must be called first");
    }
    if (d_model <= 0) {
        throw std::runtime_error("RMSNormGradFn::apply: d_model is " + std::to_string(d_model) + " - must be > 0");
    }

    if (!gamma_data || cached_size != input_shape.total_elements() ||
        cached_size % static_cast<size_t>(d_model) != 0 ||
        gamma_shape.total_elements() != static_cast<size_t>(d_model) ||
        grad_output.numel() != cached_size) {
        throw std::runtime_error("RMSNormGradFn::apply: invalid cache or gradient dimensions");
    }
    auto destination = [&](bool required, const std::shared_ptr<GradFn>& producer,
                           Tensor* leaf, const TensorContract::TensorShape& shape) -> Tensor* {
        if (!required) return nullptr;
        if (producer) return &producer->gradient_destination(shape, stream);
        if (!leaf) throw std::runtime_error("RMSNormGradFn::apply: missing leaf destination");
        leaf->require("RMSNormGradFn::apply leaf");
        return leaf;
    };
    Tensor* input_gradient = destination(input_requires_grad, input_grad_fn, leaf_input_gradient, input_shape);
    Tensor* gamma_gradient = destination(gamma_requires_grad, gamma_grad_fn, leaf_gamma_gradient, gamma_shape);
    applied = true;
    if (!input_gradient && !gamma_gradient) return;
    const int tokens = static_cast<int>(cached_size / d_model);
    const int shared_mem = (AUTOGRAD_BLOCK_SIZE / 32) * sizeof(float);
    kernel_rmsnorm_backward<<<tokens, AUTOGRAD_BLOCK_SIZE, shared_mem, stream>>>(
        grad_output.data, cached_input, gamma_data,
        input_gradient ? input_gradient->data : nullptr,
        gamma_gradient ? gamma_gradient->data : nullptr,
        tokens, d_model, eps);
    trackKernelLaunch("kernel_rmsnorm_backward", stream);

    if (input_requires_grad && !input_grad_fn) leaf_input_gradient->record_leaf_gradient_delivery();
    if (gamma_requires_grad && !gamma_grad_fn) leaf_gamma_gradient->record_leaf_gradient_delivery();
    // Complete both writes before notifying each deduplicated producer edge.
    auto notify = [&](const std::shared_ptr<GradFn>& producer) {
        if (!producer) return;
        if (auto* engine = AutogradEngine::active()) {
            engine->contribute(producer.get());
        } else {
            producer->apply(producer->pending_gradient("RMSNormGradFn::apply producer"),
                            stream, backward_payload, backward_bindings);
        }
    };
    notify(input_grad_fn);
    if (gamma_grad_fn != input_grad_fn) notify(gamma_grad_fn);
}

void RMSNormGradFn::release_saved() {
    GradFn::release_saved();
    cached_input = nullptr;
    cached_size = 0;
    leaf_input_gradient = nullptr;
    leaf_gamma_gradient = nullptr;
    gamma_data = nullptr;
    gamma_grad_fn.reset();
    input_grad_fn.reset();
}

Tensor rms_norm(const Tensor& x, const Tensor& gamma, float eps, cudaStream_t stream) {
    if (stream == nullptr || stream == 0) {
        throw std::runtime_error("autograd::rms_norm: stream is NULL - caller MUST provide valid stream");
    }

    if (!x.shape.is_2d_layout()) {
        throw std::invalid_argument("autograd::rms_norm: input must be 2D (BSM)");
    }

    const auto& dims = x.shape.as_2d();
    const int tokens = dims.rows;
    const int d_model = dims.cols;

    Tensor result = Tensor::empty(x.shape, x.requires_grad || gamma.requires_grad, stream, "rms_norm_result");

    const int shared_mem = (AUTOGRAD_BLOCK_SIZE / 32) * sizeof(float);

    kernel_rmsnorm_forward<<<tokens, AUTOGRAD_BLOCK_SIZE, shared_mem, stream>>>(
        x.data, gamma.data, result.data, tokens, d_model, eps);

    if (result.requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<RMSNormGradFn>();
        grad_fn->capture_inputs(const_cast<Tensor&>(x), const_cast<Tensor&>(gamma), stream);
        grad_fn->set_cache_copy(x.data, static_cast<size_t>(tokens) * d_model, d_model, eps, stream);
        result.grad_fn = grad_fn;
    }

    return result;
}

}  // namespace autograd
}  // namespace GRIM
