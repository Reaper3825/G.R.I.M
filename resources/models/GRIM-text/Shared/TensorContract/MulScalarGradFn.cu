//======================================================//
//  MulScalarGradFn.cu
//  Element-wise mul-scalar forward + autograd backward.
//======================================================//

#include "MulScalarGradFn.hpp"
#include "TensorContract_GPU.hpp"
#include "../CudaAllocUtils.hpp"

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

// Mul-scalar forward: y = x * scalar
__global__ void kernel_mul_scalar_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    float scalar,
    size_t count
) {
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
    if (idx < count) {
        output[idx] = input[idx] * scalar;
    }
}

// Mul-scalar backward: grad_x += grad_y * scalar
__global__ void kernel_mul_scalar_backward(
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    float scalar,
    size_t count
) {
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
    if (idx < count) {
        grad_input[idx] += grad_output[idx] * scalar;
    }
}

}  // anonymous namespace

namespace GRIM {

using CudaAlloc::cudaMallocOrThrow;

namespace autograd {

MulScalarGradFn::MulScalarGradFn() {
    op_name = "mul_scalar";
}

MulScalarGradFn::~MulScalarGradFn() {
    release_saved();
}

void MulScalarGradFn::capture_input(Tensor& x, cudaStream_t stream) {
    input_requires_grad = x.requires_grad;
    input_shape = x.shape;
    input_grad_fn = x.grad_fn;

    if (input_requires_grad) {
        if (x.is_leaf) {
            x.ensure_grad();
            input_grad = x.grad_data();
        } else {
            const size_t x_numel = x.numel();
            float* buffer = nullptr;
            cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), x_numel * sizeof(float), "MulScalarGradFn_input_grad");
            cudaMemsetAsync(buffer, 0, x_numel * sizeof(float), stream);
            owned_input_grad = std::shared_ptr<float>(buffer, [](float* p) { queueForDeferredCleanup(p); });
            input_grad = owned_input_grad.get();
        }
    }
}

void MulScalarGradFn::apply(const Tensor& grad_output, cudaStream_t stream) {
    setCurrentGradFnOp("mul_scalar", this);
    if (applied) return;
    applied = true;

    if (!input_requires_grad) return;
    if (!input_grad) {
        throw std::runtime_error("MulScalarGradFn::apply: input_grad is NULL");
    }

    kernel_mul_scalar_backward<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        grad_output.data, input_grad, scalar, count);
    trackKernelLaunch("kernel_mul_scalar_backward", stream);

    if (input_grad_fn) {
        Tensor view;
        view.data = input_grad; view.shape = input_shape;
        view.owns_data = false; view.stream = stream;
        input_grad_fn->apply(view, stream);
    }
}

void MulScalarGradFn::release_saved() {
    GradFn::release_saved();
    input_grad = nullptr;
    input_grad_fn.reset();
}

// ═══════════════════════════════════════════════════════════════════════════
// autograd::mul_scalar — forward op
// ═══════════════════════════════════════════════════════════════════════════

Tensor mul_scalar(const Tensor& x, float scalar, cudaStream_t stream) {
    if (stream == nullptr || stream == 0) {
        throw std::runtime_error("autograd::mul_scalar: stream is NULL - caller MUST provide valid stream");
    }
    if (!x.data) {
        throw std::runtime_error("autograd::mul_scalar: input data is NULL");
    }

    Tensor result = Tensor::empty(x.shape, x.requires_grad, stream, "mul_scalar_result");
    const size_t count = x.numel();
    kernel_mul_scalar_forward<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        x.data, result.data, scalar, count);

    if (x.requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<MulScalarGradFn>();
        grad_fn->capture_input(const_cast<Tensor&>(x), stream);
        grad_fn->scalar = scalar;
        grad_fn->count = count;
        result.grad_fn = grad_fn;
    }

    return result;
}

}  // namespace autograd
}  // namespace GRIM
