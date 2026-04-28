//======================================================//
//  ElementwiseMulGradFn.cu
//  Element-wise (Hadamard) multiply forward + autograd backward.
//
//  Forward:  y[i] = a[i] * b[i]
//  Backward:
//    grad_a[i] = grad_y[i] * b[i]
//    grad_b[i] = grad_y[i] * a[i]
//
//  Backward uses non-owning cached references to a/b — see header.
//======================================================//

#include "ElementwiseMulGradFn.hpp"
#include "../TensorContract_GPU.hpp"
#include "../../CudaAllocUtils.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>

#define AG_TRACE(...) do { if (g_autograd_verbose) { fprintf(stderr, __VA_ARGS__); fflush(stderr); } } while(0)

void trackKernelLaunch(const char* kernel_name, cudaStream_t stream);

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

// Element-wise multiply forward: output = a ⊙ b
__global__ void kernel_elementwise_mul_forward(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ output,
    size_t count
) {
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
    if (idx < count) {
        output[idx] = a[idx] * b[idx];
    }
}

// Element-wise multiply backward (used for both grad_a and grad_b):
//   grad_self[i] = grad_output[i] * other[i]
__global__ void kernel_elementwise_mul_backward(
    const float* grad_output,
    const float* other,
    float* grad_self,
    size_t count
) {
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
    if (idx < count) {
        grad_self[idx] = grad_output[idx] * other[idx];
    }
}

}  // anonymous namespace

namespace GRIM {

using CudaAlloc::cudaMallocOrThrow;

namespace autograd {

ElementwiseMulGradFn::ElementwiseMulGradFn() {
    op_name = "elementwise_mul";
}

void ElementwiseMulGradFn::capture_inputs(Tensor& a, Tensor& b, cudaStream_t stream) {
    a_requires_grad = a.requires_grad;
    b_requires_grad = b.requires_grad;
    a_shape = a.shape;
    b_shape = b.shape;
    a_grad_fn = a.grad_fn;
    b_grad_fn = b.grad_fn;

    if (a_requires_grad) {
        if (a.is_leaf) {
            a.ensure_grad();
            a_grad = a.grad_data();
        } else {
            const size_t n = a.numel();
            float* buffer = nullptr;
            cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), n * sizeof(float), "ElementwiseMulGradFn_grad_a");
            cudaMemsetAsync(buffer, 0, n * sizeof(float), stream);
            owned_a_grad = std::shared_ptr<float>(buffer, [](float* p) { queueForDeferredCleanup(p); });
            a_grad = owned_a_grad.get();
        }
    }
    if (b_requires_grad) {
        if (b.is_leaf) {
            b.ensure_grad();
            b_grad = b.grad_data();
        } else {
            const size_t n = b.numel();
            float* buffer = nullptr;
            cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), n * sizeof(float), "ElementwiseMulGradFn_grad_b");
            cudaMemsetAsync(buffer, 0, n * sizeof(float), stream);
            owned_b_grad = std::shared_ptr<float>(buffer, [](float* p) { queueForDeferredCleanup(p); });
            b_grad = owned_b_grad.get();
        }
    }
}

void ElementwiseMulGradFn::set_cache_refs(const float* a_data, const float* b_data, size_t size) {
    cached_size = size;
    if (a_requires_grad && b_data) cached_b = b_data;
    if (b_requires_grad && a_data) cached_a = a_data;
}

void ElementwiseMulGradFn::apply(const Tensor& grad_output, cudaStream_t stream) {
    setCurrentGradFnOp("elementwise_mul", this);
    if (applied) return;
    applied = true;

    const size_t count = grad_output.numel();

    if (a_requires_grad) {
        if (!a_grad || !cached_b) {
            throw std::runtime_error("ElementwiseMulGradFn::apply: a_grad or cached_b is NULL");
        }
        kernel_elementwise_mul_backward<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
            grad_output.data, cached_b, a_grad, count);
        trackKernelLaunch("kernel_elementwise_mul_backward_a", stream);

        if (a_grad_fn) {
            Tensor view;
            view.data = a_grad; view.shape = a_shape;
            view.owns_data = false; view.stream = stream;
            a_grad_fn->apply(view, stream);
        }
    }

    if (b_requires_grad) {
        if (!b_grad || !cached_a) {
            throw std::runtime_error("ElementwiseMulGradFn::apply: b_grad or cached_a is NULL");
        }
        kernel_elementwise_mul_backward<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
            grad_output.data, cached_a, b_grad, count);
        trackKernelLaunch("kernel_elementwise_mul_backward_b", stream);

        if (b_grad_fn) {
            Tensor view;
            view.data = b_grad; view.shape = b_shape;
            view.owns_data = false; view.stream = stream;
            b_grad_fn->apply(view, stream);
        }
    }
}

void ElementwiseMulGradFn::release_saved() {
    GradFn::release_saved();
    cached_a = nullptr;
    cached_b = nullptr;
    cached_size = 0;
    a_grad = nullptr;
    b_grad = nullptr;
    a_grad_fn.reset();
    b_grad_fn.reset();
}

Tensor elementwise_mul(const Tensor& a, const Tensor& b, cudaStream_t stream) {
    if (stream == nullptr || stream == 0) {
        throw std::runtime_error("autograd::elementwise_mul: stream is NULL");
    }
    if (a.numel() != b.numel()) {
        throw std::runtime_error("autograd::elementwise_mul: size mismatch a.numel()=" +
                                 std::to_string(a.numel()) + " b.numel()=" + std::to_string(b.numel()));
    }

    const bool needs_grad = a.requires_grad || b.requires_grad;
    Tensor result = Tensor::empty(a.shape, needs_grad, stream, "emul_result");

    const size_t count = a.numel();
    kernel_elementwise_mul_forward<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        a.data, b.data, result.data, count);

    if (needs_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<ElementwiseMulGradFn>();
        grad_fn->capture_inputs(const_cast<Tensor&>(a), const_cast<Tensor&>(b), stream);
        grad_fn->set_cache_refs(a.data, b.data, count);
        result.grad_fn = grad_fn;
    }

    return result;
}

}  // namespace autograd
}  // namespace GRIM
