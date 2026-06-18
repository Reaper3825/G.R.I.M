//======================================================//
//  AddScalarGradFn.cu
//  Element-wise add-scalar forward + autograd backward (pass-through).
//======================================================//

#include "AddScalarGradFn.hpp"
#include "../GradientAccumulation.hpp"
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

// Add scalar forward: y = x + scalar
__global__ void kernel_add_scalar_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    float scalar,
    size_t count
) {
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
    if (idx < count) {
        output[idx] = input[idx] + scalar;
    }
}

}  // anonymous namespace

namespace GRIM {

using CudaAlloc::cudaMallocOrThrow;

namespace autograd {

AddScalarGradFn::AddScalarGradFn() {
    op_name = "add_scalar";
}

AddScalarGradFn::~AddScalarGradFn() {
    release_saved();
}

void AddScalarGradFn::capture_input(Tensor& x, cudaStream_t stream) {
    input_requires_grad = x.requires_grad;
    input_shape = x.shape;
    input_grad_fn = x.grad_fn;
    register_input(x.grad_fn);
    count = x.numel();

    if (input_requires_grad) {
        if (x.is_leaf) {
            x.ensure_grad();
            input_grad = x.grad_data();
        } else {
            float* buffer = nullptr;
            cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), count * sizeof(float), "AddScalarGradFn_input_grad");
            cudaMemsetAsync(buffer, 0, count * sizeof(float), stream);
            owned_input_grad = std::shared_ptr<float>(buffer, [](float* p) { queueForDeferredCleanup(p); });
            input_grad = owned_input_grad.get();
        }
    }
}

void AddScalarGradFn::apply_impl(const Tensor& grad_output,
                                 cudaStream_t stream,
                                 const Batching::BatchPayload* backward_payload,
                                 const Batching::BatchDeviceBindings* backward_bindings) {
    setCurrentGradFnOp("add_scalar", this);
    if (applied) return;
    applied = true;

    if (!input_requires_grad) return;
    if (!input_grad) {
        throw std::runtime_error("AddScalarGradFn::apply: input_grad is NULL");
    }

    // Pure pass-through: grad_x += grad_y
    const size_t n = grad_output.numel();
    if (n != count) {
        throw std::runtime_error("AddScalarGradFn::apply: size mismatch");
    }
    accumulate_grad(input_grad, grad_output.data, n, 1.0f, stream, "AddScalarGradFn::apply input_grad");
    trackKernelLaunch("add_scalar_backward", stream);

    if (input_grad_fn) {
        Tensor view;
        view.data = input_grad; view.shape = input_shape;
        view.owns_data = false; view.stream = stream;
        input_grad_fn->apply(view, stream, backward_payload, backward_bindings);
    }
}

void AddScalarGradFn::release_saved() {
    GradFn::release_saved();
    input_grad = nullptr;
    input_grad_fn.reset();
}

// ═══════════════════════════════════════════════════════════════════════════
// autograd::add_scalar — forward op
// ═══════════════════════════════════════════════════════════════════════════

Tensor add_scalar(const Tensor& x, float scalar, cudaStream_t stream) {
    if (stream == nullptr || stream == 0) {
        throw std::runtime_error("autograd::add_scalar: stream is NULL - caller MUST provide valid stream");
    }
    if (!x.data) {
        throw std::runtime_error("autograd::add_scalar: input data is NULL");
    }

    Tensor result = Tensor::empty(x.shape, x.requires_grad, stream, "add_scalar_result");
    const size_t count = x.numel();
    kernel_add_scalar_forward<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        x.data, result.data, scalar, count);

    if (x.requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<AddScalarGradFn>();
        grad_fn->capture_input(const_cast<Tensor&>(x), stream);
        result.grad_fn = grad_fn;
    }

    return result;
}

}  // namespace autograd
}  // namespace GRIM
