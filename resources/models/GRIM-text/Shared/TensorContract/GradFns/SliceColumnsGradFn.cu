//======================================================//
//  SliceColumnsGradFn.cu
//  CUDA kernels and autograd node for contiguous column slicing.
//
//  Introduced for the latent-trajectory MTP path: the shared
//  hidden-trajectory projection emits [T, K*d_model] and each MTP
//  horizon k consumes the [T, d_model] block at column offset k*d_model.
//======================================================//

#include "SliceColumnsGradFn.hpp"
#include "../../CudaAllocUtils.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace {

constexpr int AUTOGRAD_BLOCK_SIZE = 256;

//========================================================================
// Slice Forward: out[i, j] = x[i, col_offset + j]
//========================================================================
__global__ void kernel_slice_columns_forward(
    float* __restrict__ output,
    const float* __restrict__ x,
    int N, int in_cols, int col_offset, int out_cols
) {
    const int row = blockIdx.x;
    if (row >= N) return;
    for (int j = threadIdx.x; j < out_cols; j += blockDim.x) {
        output[static_cast<size_t>(row) * out_cols + j] =
            x[static_cast<size_t>(row) * in_cols + col_offset + j];
    }
}

//========================================================================
// Slice Backward: grad_x[i, col_offset + j] += grad_out[i, j]
// Columns outside the slice stay untouched (buffer is zero-initialized
// for non-leaf inputs; leaf grads accumulate into the registry buffer).
//========================================================================
__global__ void kernel_slice_columns_backward(
    float* __restrict__ grad_x,
    const float* __restrict__ grad_out,
    int N, int in_cols, int col_offset, int out_cols
) {
    const int row = blockIdx.x;
    if (row >= N) return;
    for (int j = threadIdx.x; j < out_cols; j += blockDim.x) {
        grad_x[static_cast<size_t>(row) * in_cols + col_offset + j] +=
            grad_out[static_cast<size_t>(row) * out_cols + j];
    }
}

}  // anonymous namespace

namespace GRIM {
using CudaAlloc::cudaMallocOrThrow;
namespace autograd {

SliceColumnsGradFn::SliceColumnsGradFn() {
    op_name = "slice_columns";
}

void SliceColumnsGradFn::capture_inputs(Tensor& x, cudaStream_t stream) {
    x_requires_grad = x.requires_grad;
    x_shape = x.shape;
    x_grad_fn = x.grad_fn;
    register_input(x.grad_fn);

    if (x_requires_grad) {
        if (x.is_leaf) {
            x.ensure_grad();
            grad_x = x.grad_data();
        } else {
            const size_t n = x.numel();
            float* buffer = nullptr;
            cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), n * sizeof(float), "SliceColumnsGradFn_grad_x");
            cudaMemsetAsync(buffer, 0, n * sizeof(float), stream);
            owned_grad_x = std::shared_ptr<float>(buffer, [](float* p) { queueForDeferredCleanup(p); });
            grad_x = owned_grad_x.get();
        }
    }
}

void SliceColumnsGradFn::apply_impl(const Tensor& grad_output,
                                    cudaStream_t stream,
                                    const Batching::BatchPayload* backward_payload,
                                    const Batching::BatchDeviceBindings* backward_bindings) {
    setCurrentGradFnOp("slice_columns", this);
    if (applied) return;
    applied = true;

    if (x_requires_grad && grad_x) {
        kernel_slice_columns_backward<<<rows, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
            grad_x, grad_output.data, rows, in_cols, col_offset, out_cols);
        if (x_grad_fn) {
            Tensor view;
            view.data = grad_x; view.shape = x_shape;
            view.owns_data = false; view.stream = stream;
            x_grad_fn->apply(view, stream, backward_payload, backward_bindings);
        }
    }
}

void SliceColumnsGradFn::release_saved() {
    GradFn::release_saved();
    grad_x = nullptr;
    x_grad_fn.reset();
}

Tensor slice_columns(const Tensor& x, int col_offset, int out_cols, cudaStream_t stream) {
    if (!x.shape.is_2d_layout()) {
        throw std::invalid_argument("autograd::slice_columns: input must be 2D");
    }
    if (!x.data) {
        throw std::invalid_argument("autograd::slice_columns: null data pointer");
    }
    const auto x_dims = x.shape.as_2d();
    if (col_offset < 0 || out_cols <= 0 || col_offset + out_cols > x_dims.cols) {
        throw std::invalid_argument("autograd::slice_columns: slice [" +
            std::to_string(col_offset) + ", " + std::to_string(col_offset + out_cols) +
            ") out of range for input cols=" + std::to_string(x_dims.cols));
    }

    const int N = x_dims.rows;

    const bool needs_grad = x.requires_grad;
    auto shape = TensorContract::TensorShape::make_BSM(N, out_cols);
    Tensor result = Tensor::empty(shape, needs_grad, stream, "slice_columns_result");

    kernel_slice_columns_forward<<<N, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        result.data, x.data, N, x_dims.cols, col_offset, out_cols);

    if (needs_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<SliceColumnsGradFn>();
        grad_fn->capture_inputs(const_cast<Tensor&>(x), stream);
        grad_fn->rows = N;
        grad_fn->in_cols = x_dims.cols;
        grad_fn->col_offset = col_offset;
        grad_fn->out_cols = out_cols;
        result.grad_fn = grad_fn;
    }

    return result;
}

}  // namespace autograd
}  // namespace GRIM
