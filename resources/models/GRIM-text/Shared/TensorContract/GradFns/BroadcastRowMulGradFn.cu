//======================================================//
//  BroadcastRowMulGradFn.cu
//  Broadcast per-row scalar multiply forward + autograd backward.
//
//  Forward:  out[i, j] = scale[i, 0] * x[i, j]
//  Backward:
//    grad_scale[i] += sum_j(grad_out[i, j] * x[i, j])  (block-reduction per row)
//    grad_x[i, j]  += grad_out[i, j] * scale[i, 0]
//======================================================//

#include "BroadcastRowMulGradFn.hpp"
#include "../TensorContract_GPU.hpp"
#include "../../CudaAllocUtils.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>

#define AG_TRACE(...) do { if constexpr (GRIM::VerboseLogging::ENABLE_AUTOGRAD_TRACE_LOGS) { fprintf(stderr, __VA_ARGS__); fflush(stderr); } } while(0)

void trackKernelLaunch(const char* kernel_name, cudaStream_t stream);

namespace {

constexpr int AUTOGRAD_BLOCK_SIZE = 256;

// Forward: out[i, j] = scale[i, 0] * x[i, j]
__global__ void kernel_broadcast_row_mul_forward(
    const float* __restrict__ scale,   // [rows, 1]
    const float* __restrict__ x,       // [rows, cols]
    float* __restrict__ output,        // [rows, cols]
    int rows, int cols
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    const int row = idx / cols;
    output[idx] = scale[row] * x[idx];
}

// Backward for x: grad_x[i, j] += grad_out[i, j] * scale[i, 0]
__global__ void kernel_broadcast_row_mul_backward_x(
    const float* __restrict__ grad_output,  // [rows, cols]
    const float* __restrict__ scale,        // [rows, 1]
    float* __restrict__ grad_x,             // [rows, cols]
    int rows, int cols
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    const int row = idx / cols;
    grad_x[idx] += grad_output[idx] * scale[row];
}

// Backward for scale: grad_scale[i] += sum_j(grad_out[i, j] * x[i, j])
//   one block per row, reduce across cols using warp-shuffle, atomicAdd
//   the per-warp partial back into grad_scale[row].
__global__ void kernel_broadcast_row_mul_backward_scale(
    const float* __restrict__ grad_output,  // [rows, cols]
    const float* __restrict__ x,            // [rows, cols]
    float* __restrict__ grad_scale,         // [rows, 1]
    int rows, int cols
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    float sum = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        sum += grad_output[row * cols + j] * x[row * cols + j];
    for (int mask = warpSize / 2; mask > 0; mask >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, mask);
    if ((threadIdx.x & (warpSize - 1)) == 0)
        atomicAdd(&grad_scale[row], sum);
}

}  // anonymous namespace

namespace GRIM {

using CudaAlloc::cudaMallocOrThrow;

namespace autograd {

BroadcastRowMulGradFn::BroadcastRowMulGradFn() {
    op_name = "broadcast_row_mul";
}

void BroadcastRowMulGradFn::capture_inputs(Tensor& s, Tensor& x, cudaStream_t stream) {
    scale_requires_grad = s.requires_grad;
    x_requires_grad = x.requires_grad;
    scale_shape = s.shape;
    x_shape = x.shape;
    scale_grad_fn = s.grad_fn;
    x_grad_fn = x.grad_fn;
    register_input(s.grad_fn);
    register_input(x.grad_fn);

    if (scale_requires_grad) {
        if (s.is_leaf) {
            s.ensure_grad();
            scale_grad = s.grad_data();
        } else {
            const size_t s_numel = s.numel();
            float* buffer = nullptr;
            cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), s_numel * sizeof(float), "BroadcastRowMulGradFn_scale_grad");
            cudaMemsetAsync(buffer, 0, s_numel * sizeof(float), stream);
            owned_scale_grad = std::shared_ptr<float>(buffer, [](float* p) { queueForDeferredCleanup(p); });
            scale_grad = owned_scale_grad.get();
        }
    }
    if (x_requires_grad) {
        if (x.is_leaf) {
            x.ensure_grad();
            x_grad = x.grad_data();
        } else {
            const size_t x_numel = x.numel();
            float* buffer = nullptr;
            cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), x_numel * sizeof(float), "BroadcastRowMulGradFn_x_grad");
            cudaMemsetAsync(buffer, 0, x_numel * sizeof(float), stream);
            owned_x_grad = std::shared_ptr<float>(buffer, [](float* p) { queueForDeferredCleanup(p); });
            x_grad = owned_x_grad.get();
        }
    }
}

void BroadcastRowMulGradFn::set_cache_refs(const float* scale_data, const float* x_data, int r, int c) {
    if (!scale_data) throw std::runtime_error("BroadcastRowMulGradFn::set_cache_refs: scale_data is NULL");
    if (!x_data) throw std::runtime_error("BroadcastRowMulGradFn::set_cache_refs: x_data is NULL");
    cached_scale = scale_data;
    cached_x = x_data;
    rows = r;
    cols = c;
}

void BroadcastRowMulGradFn::apply_impl(const Tensor& grad_output,
                                       cudaStream_t stream,
                                       const Batching::BatchPayload* backward_payload,
                                       const Batching::BatchDeviceBindings* backward_bindings) {
    setCurrentGradFnOp("broadcast_row_mul", this);
    if (applied) return;
    applied = true;

    const int total = rows * cols;

    if (x_requires_grad) {
        if (!x_grad) throw std::runtime_error("BroadcastRowMulGradFn::apply: x_grad is NULL");
        kernel_broadcast_row_mul_backward_x<<<(total + AUTOGRAD_BLOCK_SIZE - 1) / AUTOGRAD_BLOCK_SIZE, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
            grad_output.data, cached_scale, x_grad, rows, cols);
        trackKernelLaunch("kernel_broadcast_row_mul_backward_x", stream);
    }

    if (scale_requires_grad) {
        if (!scale_grad) throw std::runtime_error("BroadcastRowMulGradFn::apply: scale_grad is NULL");
        kernel_broadcast_row_mul_backward_scale<<<rows, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
            grad_output.data, cached_x, scale_grad, rows, cols);
        trackKernelLaunch("kernel_broadcast_row_mul_backward_scale", stream);
    }

    if (x_requires_grad && x_grad_fn) {
        Tensor view;
        view.data = x_grad; view.shape = x_shape;
        view.owns_data = false; view.stream = stream;
        x_grad_fn->apply(view, stream, backward_payload, backward_bindings);
    }
    if (scale_requires_grad && scale_grad_fn) {
        Tensor view;
        view.data = scale_grad; view.shape = scale_shape;
        view.owns_data = false; view.stream = stream;
        scale_grad_fn->apply(view, stream, backward_payload, backward_bindings);
    }
}

void BroadcastRowMulGradFn::release_saved() {
    GradFn::release_saved();
    cached_scale = nullptr;
    cached_x = nullptr;
    scale_grad = nullptr;
    x_grad = nullptr;
    scale_grad_fn.reset();
    x_grad_fn.reset();
}

Tensor broadcast_row_mul(const Tensor& scale, const Tensor& x, cudaStream_t stream) {
    if (stream == nullptr || stream == 0) {
        throw std::runtime_error("autograd::broadcast_row_mul: stream is NULL");
    }
    if (!scale.data || !x.data) {
        throw std::runtime_error("autograd::broadcast_row_mul: input data is NULL");
    }
    const auto s_dims = scale.shape.as_2d();
    const auto x_dims = x.shape.as_2d();
    if (s_dims.cols != 1) {
        throw std::runtime_error("autograd::broadcast_row_mul: scale must be [rows,1], got cols=" + std::to_string(s_dims.cols));
    }
    if (s_dims.rows != x_dims.rows) {
        throw std::runtime_error("autograd::broadcast_row_mul: row mismatch scale.rows=" + std::to_string(s_dims.rows) + " x.rows=" + std::to_string(x_dims.rows));
    }
    const int rows = x_dims.rows;
    const int cols = x_dims.cols;
    const int total = rows * cols;

    const bool needs_grad = scale.requires_grad || x.requires_grad;
    Tensor result = Tensor::empty(x.shape, needs_grad, stream, "brow_mul_result");

    kernel_broadcast_row_mul_forward<<<(total + AUTOGRAD_BLOCK_SIZE - 1) / AUTOGRAD_BLOCK_SIZE, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        scale.data, x.data, result.data, rows, cols);

    if (needs_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<BroadcastRowMulGradFn>();
        grad_fn->capture_inputs(const_cast<Tensor&>(scale), const_cast<Tensor&>(x), stream);
        grad_fn->set_cache_refs(scale.data, x.data, rows, cols);
        result.grad_fn = grad_fn;
    }

    return result;
}

}  // namespace autograd
}  // namespace GRIM
