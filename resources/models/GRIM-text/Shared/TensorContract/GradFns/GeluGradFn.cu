//======================================================//
//  GeluGradFn.cu
//  GELU activation forward + autograd backward.
//======================================================//

#include "GeluGradFn.hpp"
#include "../TensorContract_GPU.hpp"
#include "../../CudaAllocUtils.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cmath>
#include <vector>

#ifndef TENSOR_VERBOSE_DEBUG
#define TENSOR_VERBOSE_DEBUG 0
#endif

#define AG_TRACE(...) do { if constexpr (GRIM::VerboseLogging::ENABLE_AUTOGRAD_TRACE_LOGS) { fprintf(stderr, __VA_ARGS__); fflush(stderr); } } while(0)

// ─── Forward declaration: defined in TensorContract_GPU.cu at global scope ───
void trackKernelLaunch(const char* kernel_name, cudaStream_t stream);

// ═══════════════════════════════════════════════════════════════════════════
// Anonymous-namespace helpers — internal linkage (no ODR conflict with the
// same names in TensorContract_GPU.cu / AutogradAttention.cu).
// ═══════════════════════════════════════════════════════════════════════════
namespace {

constexpr int AUTOGRAD_BLOCK_SIZE = 256;
constexpr int kMaxGridBlocks1DFallback = 65534;

inline int getMaxGridBlocks1D() {
    static int cached = -1;
    if (cached < 0) {
        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess) {
            cached = kMaxGridBlocks1DFallback;
        } else {
            int max_x = 0;
            if (cudaDeviceGetAttribute(&max_x, cudaDevAttrMaxGridDimX, device) != cudaSuccess) {
                cached = kMaxGridBlocks1DFallback;
            } else {
                cached = (max_x > 65534) ? 65534 : max_x;
            }
        }
    }
    return cached;
}

constexpr int kMaxGridDimY = 65535;

inline dim3 gridForCount(size_t count) {
    if (count == 0) return dim3(1, 1, 1);
    const int blocks = static_cast<int>((count + AUTOGRAD_BLOCK_SIZE - 1) / AUTOGRAD_BLOCK_SIZE);
    const int max1d = getMaxGridBlocks1D();
    if (blocks <= max1d) return dim3(blocks, 1, 1);
    int gy = (blocks + max1d - 1) / max1d;
    if (gy <= kMaxGridDimY) return dim3(max1d, gy, 1);
    int gx = (blocks + kMaxGridDimY - 1) / kMaxGridDimY;
    return dim3(gx, kMaxGridDimY, 1);
}

// GELU forward: y = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__global__ void kernel_gelu_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    size_t count
) {
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
    if (idx < count) {
        const float x = input[idx];
        const float c = 0.7978845608f;  // sqrt(2/pi)
        const float k = 0.044715f;

        const float x3 = x * x * x;
        const float inner = c * (x + k * x3);
        const float tanh_inner = tanhf(inner);

        output[idx] = 0.5f * x * (1.0f + tanh_inner);
    }
}

// GELU backward: grad_x += grad_y * gelu'(x)
// gelu'(x) = 0.5 * (1 + tanh) + 0.5 * x * sech^2 * c * (1 + 3*k*x^2)
__global__ void kernel_gelu_backward(
    const float* grad_output,
    const float* input,
    float* grad_input,
    size_t count
) {
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
    if (idx < count) {
        const float x = input[idx];
        const float c = 0.7978845608f;  // sqrt(2/pi)
        const float k = 0.044715f;

        const float x3 = x * x * x;
        const float inner = c * (x + k * x3);
        const float tanh_inner = tanhf(inner);
        const float sech2 = 1.0f - tanh_inner * tanh_inner;

        const float dgelu = 0.5f * (1.0f + tanh_inner) +
                           0.5f * x * sech2 * c * (1.0f + 3.0f * k * x * x);

        grad_input[idx] += grad_output[idx] * dgelu;
    }
}

}  // anonymous namespace

namespace GRIM {

using CudaAlloc::cudaMallocOrThrow;

namespace autograd {

GeluGradFn::GeluGradFn() {
    op_name = "gelu";
}

GeluGradFn::~GeluGradFn() {
    release_saved();
}

void GeluGradFn::capture_input(Tensor& x, cudaStream_t stream) {
    input_requires_grad = x.requires_grad;
    input_shape = x.shape;
    input_grad_fn = x.grad_fn;
    register_input(x.grad_fn);

    if (input_requires_grad) {
        if (x.is_leaf) {
            x.ensure_grad();
            input_grad = x.grad_data();
        } else {
            const size_t x_numel = x.numel();
            float* buffer = nullptr;
            cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), x_numel * sizeof(float), "GeluGradFn_input_grad");
            cudaMemsetAsync(buffer, 0, x_numel * sizeof(float), stream);
            owned_input_grad = std::shared_ptr<float>(buffer, [](float* p) { queueForDeferredCleanup(p); });
            input_grad = owned_input_grad.get();
        }
    }
}

// ISSUE #51: Copy cache data to owned buffer instead of storing dangling pointer
void GeluGradFn::set_cache_copy(const float* external_cache, size_t size, cudaStream_t stream) {
    if (!external_cache) {
        throw std::runtime_error("GeluGradFn::set_cache_copy: external_cache is NULL - caller MUST provide cache");
    }
    cached_size = size;

    float* buffer = nullptr;
    cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), size * sizeof(float), "GeluGradFn_cache");
    cudaMemcpyAsync(buffer, external_cache, size * sizeof(float),
                   cudaMemcpyDeviceToDevice, stream);

    // ISSUE #53: Deferred cleanup deleter (cudaFree blocks while GPU busy)
    owned_cache = std::shared_ptr<float>(buffer, [](float* p) {
        queueForDeferredCleanup(p);
    });
    cached_input = owned_cache.get();
    AG_TRACE("[GeluGradFn] Copied cache: %zu floats to %p\n", size, (void*)cached_input);
}

void GeluGradFn::apply_impl(const Tensor& grad_output,
                            cudaStream_t stream,
                            const Batching::BatchPayload* backward_payload,
                            const Batching::BatchDeviceBindings* backward_bindings) {
    setCurrentGradFnOp("gelu", this);

    if (applied) return;
    applied = true;

#if TENSOR_VERBOSE_DEBUG
    static int s_gelu_bwd_call = 0;
    const int gelu_call_idx = ++s_gelu_bwd_call;
    {
        cudaStreamSynchronize(stream);
        const size_t grad_elems = grad_output.numel();
        std::vector<float> samp(std::min(grad_elems, static_cast<size_t>(10000)));
        cudaMemcpy(samp.data(), grad_output.data, samp.size() * sizeof(float), cudaMemcpyDeviceToHost);
        float mx = 0.0f; double sq = 0.0;
        for (auto& v : samp) { if (!std::isnan(v) && !std::isinf(v)) { mx = std::max(mx, std::abs(v)); sq += v*v; } }
        float rms = std::sqrt(static_cast<float>(sq / samp.size()));
        fprintf(stderr, "[GELU-BWD-IN] call=%d | grad: numel=%zu max=%.6f rms=%.6f\n",
                gelu_call_idx, grad_elems, mx, rms);
    }
#endif

    if (!input_requires_grad) return;
    if (!input_grad) {
        throw std::runtime_error("GeluGradFn::apply: input_grad is NULL - capture_input() must be called first");
    }
    if (!cached_input) {
        throw std::runtime_error("GeluGradFn::apply: cached_input is NULL - set_cache() must be called first");
    }

    const size_t count = grad_output.numel();
    if (count != cached_size) {
        throw std::runtime_error("GeluGradFn::apply: size mismatch - grad_output.numel()=" +
                                 std::to_string(count) + " cached_size=" + std::to_string(cached_size));
    }

    kernel_gelu_backward<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        grad_output.data, cached_input, input_grad, count);
    trackKernelLaunch("kernel_gelu_backward", stream);

    if (input_grad_fn) {
        Tensor view;
        view.data = input_grad; view.shape = input_shape;
        view.owns_data = false; view.stream = stream;
        input_grad_fn->apply(view, stream, backward_payload, backward_bindings);
    }
}

void GeluGradFn::release_saved() {
    GradFn::release_saved();
    cached_input = nullptr;
    cached_size = 0;
    input_grad = nullptr;
    input_grad_fn.reset();
}

// ═══════════════════════════════════════════════════════════════════════════
// autograd::gelu — forward op
// ═══════════════════════════════════════════════════════════════════════════

Tensor gelu(const Tensor& x, cudaStream_t stream, const float* input_cache) {
    if (stream == nullptr || stream == 0) {
        throw std::runtime_error("autograd::gelu: stream is NULL - caller MUST provide valid stream");
    }

    Tensor result = Tensor::empty(x.shape, x.requires_grad, stream, "gelu_result");

    const size_t count = x.numel();
    kernel_gelu_forward<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(x.data, result.data, count);

    if (x.requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<GeluGradFn>();
        grad_fn->capture_input(const_cast<Tensor&>(x), stream);

        const float* effective_cache = input_cache ? input_cache : x.data;
        grad_fn->set_cache_copy(effective_cache, count, stream);
        result.grad_fn = grad_fn;
    }

    return result;
}

}  // namespace autograd
}  // namespace GRIM
