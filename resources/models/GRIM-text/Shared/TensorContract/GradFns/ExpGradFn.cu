//======================================================//
//  ExpGradFn.cu
//  Element-wise exp forward + autograd backward (saves output).
//======================================================//

#include "ExpGradFn.hpp"
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

// Exp forward: y = exp(x)
__global__ void kernel_exp_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    size_t count
) {
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
    if (idx < count) {
        output[idx] = expf(input[idx]);
    }
}

// Exp backward: grad_x += grad_y * y  (uses saved output)
__global__ void kernel_exp_backward(
    const float* __restrict__ grad_output,
    const float* __restrict__ saved_output,
    float* __restrict__ grad_input,
    size_t count
) {
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
    if (idx < count) {
        grad_input[idx] += grad_output[idx] * saved_output[idx];
    }
}

}  // anonymous namespace

namespace GRIM {

using CudaAlloc::cudaMallocOrThrow;

namespace autograd {

ExpGradFn::ExpGradFn() {
    op_name = "exp";
}

ExpGradFn::~ExpGradFn() {
    release_saved();
}

void ExpGradFn::capture_input(Tensor& x, cudaStream_t stream) {
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
            cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), x_numel * sizeof(float), "ExpGradFn_input_grad");
            cudaMemsetAsync(buffer, 0, x_numel * sizeof(float), stream);
            owned_input_grad = std::shared_ptr<float>(buffer, [](float* p) { queueForDeferredCleanup(p); });
            input_grad = owned_input_grad.get();
        }
    }
}

void ExpGradFn::save_output(const float* output_data, size_t size) {
    if (!output_data) {
        throw std::runtime_error("ExpGradFn::save_output: output_data is NULL");
    }
    cached_output = output_data;
    cached_size = size;
}

void ExpGradFn::apply_impl(const Tensor& grad_output,
                           cudaStream_t stream,
                           const Batching::BatchPayload* backward_payload,
                           const Batching::BatchDeviceBindings* backward_bindings) {
    setCurrentGradFnOp("exp", this);
    if (applied) return;
    applied = true;

    if (!input_requires_grad) return;
    if (!input_grad) {
        throw std::runtime_error("ExpGradFn::apply: input_grad is NULL");
    }
    if (!cached_output) {
        throw std::runtime_error("ExpGradFn::apply: cached_output is NULL");
    }

    const size_t count = grad_output.numel();
    if (count != cached_size) {
        throw std::runtime_error("ExpGradFn::apply: size mismatch");
    }

    kernel_exp_backward<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        grad_output.data, cached_output, input_grad, count);
    trackKernelLaunch("kernel_exp_backward", stream);

    if (input_grad_fn) {
        Tensor view;
        view.data = input_grad; view.shape = input_shape;
        view.owns_data = false; view.stream = stream;
        input_grad_fn->apply(view, stream, backward_payload, backward_bindings);
    }
}

void ExpGradFn::release_saved() {
    GradFn::release_saved();
    cached_output = nullptr;
    cached_size = 0;
    input_grad = nullptr;
    input_grad_fn.reset();
}

// ═══════════════════════════════════════════════════════════════════════════
// autograd::exp — forward op
// ═══════════════════════════════════════════════════════════════════════════

Tensor exp(const Tensor& x, cudaStream_t stream) {
    if (stream == nullptr || stream == 0) {
        throw std::runtime_error("autograd::exp: stream is NULL - caller MUST provide valid stream");
    }
    if (!x.data) {
        throw std::runtime_error("autograd::exp: input data is NULL");
    }

    Tensor result = Tensor::empty(x.shape, x.requires_grad, stream, "exp_result");
    const size_t count = x.numel();
    kernel_exp_forward<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(x.data, result.data, count);

    if (x.requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<ExpGradFn>();
        grad_fn->capture_input(const_cast<Tensor&>(x), stream);
        grad_fn->save_output(result.data, count);
        result.grad_fn = grad_fn;
    }

    return result;
}

}  // namespace autograd
}  // namespace GRIM
