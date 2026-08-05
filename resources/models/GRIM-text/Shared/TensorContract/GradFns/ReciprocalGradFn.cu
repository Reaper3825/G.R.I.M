//======================================================//
//  ReciprocalGradFn.cu
//  Element-wise reciprocal forward + autograd backward (saves output).
//======================================================//

#include "ReciprocalGradFn.hpp"
#include "../TensorContract_GPU.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>

#define AG_TRACE(...) do { if constexpr (GRIM::VerboseLogging::ENABLE_AUTOGRAD_TRACE_LOGS) { fprintf(stderr, __VA_ARGS__); fflush(stderr); } } while(0)

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

// Reciprocal forward: y = 1/x
__global__ void kernel_reciprocal_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    size_t count
) {
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
    if (idx < count) {
        output[idx] = 1.0f / input[idx];
    }
}

// Reciprocal backward: grad_x += grad_y * (-y^2)  (uses saved output)
__global__ void kernel_reciprocal_backward(
    const float* __restrict__ grad_output,
    const float* __restrict__ saved_output,
    float* __restrict__ grad_input,
    size_t count
) {
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
    if (idx < count) {
        const float y = saved_output[idx];
        grad_input[idx] += grad_output[idx] * (-y * y);
    }
}

}  // anonymous namespace

namespace GRIM {

namespace autograd {

ReciprocalGradFn::ReciprocalGradFn() {
    op_name = "reciprocal";
}

ReciprocalGradFn::~ReciprocalGradFn() {
    release_saved();
}

void ReciprocalGradFn::capture_input(Tensor& x, cudaStream_t stream) {
    input_requires_grad = x.requires_grad;

    if (input_requires_grad) {
        input_gradient = capture_input_gradient(
            x, stream, "ReciprocalGradFn::capture_input");
    }
}

void ReciprocalGradFn::save_output(const float* output_data, size_t size) {
    if (!output_data) {
        throw std::runtime_error("ReciprocalGradFn::save_output: output_data is NULL");
    }
    cached_output = output_data;
    cached_size = size;
}

void ReciprocalGradFn::apply_impl(const Tensor& grad_output,
                                  cudaStream_t stream,
                                  const Batching::BatchPayload* backward_payload,
                                  const Batching::BatchDeviceBindings* backward_bindings) {
    setCurrentGradFnOp("reciprocal", this);
    if (applied) return;
    applied = true;

    if (!input_requires_grad) return;
    if (!input_gradient) {
        throw std::runtime_error("ReciprocalGradFn::apply: input gradient Tensor is NULL");
    }
    if (!cached_output) {
        throw std::runtime_error("ReciprocalGradFn::apply: cached_output is NULL");
    }

    const size_t count = grad_output.numel();
    if (count != cached_size) {
        throw std::runtime_error("ReciprocalGradFn::apply: size mismatch");
    }

    kernel_reciprocal_backward<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        grad_output.data, cached_output, input_gradient->data, count);
    trackKernelLaunch("kernel_reciprocal_backward", stream);

    propagate_input_gradient(
        input_gradient, stream, backward_payload, backward_bindings,
        "ReciprocalGradFn::apply");
}

void ReciprocalGradFn::release_saved() {
    GradFn::release_saved();
    cached_output = nullptr;
    cached_size = 0;
    input_gradient.reset();
}

// ═══════════════════════════════════════════════════════════════════════════
// autograd::reciprocal — forward op
// ═══════════════════════════════════════════════════════════════════════════

Tensor reciprocal(const Tensor& x, cudaStream_t stream) {
    if (stream == nullptr || stream == 0) {
        throw std::runtime_error("autograd::reciprocal: stream is NULL - caller MUST provide valid stream");
    }
    if (!x.data) {
        throw std::runtime_error("autograd::reciprocal: input data is NULL");
    }

    Tensor result = Tensor::empty(x.shape, x.requires_grad, stream, "reciprocal_result");
    const size_t count = x.numel();
    kernel_reciprocal_forward<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        x.data, result.data, count);

    if (x.requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<ReciprocalGradFn>();
        grad_fn->capture_input(const_cast<Tensor&>(x), stream);
        grad_fn->save_output(result.data, count);
        result.grad_fn = grad_fn;
    }

    return result;
}

}  // namespace autograd
}  // namespace GRIM
