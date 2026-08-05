//======================================================//
//  SliceColumnsGradFn.cu
//  CUDA kernels and autograd node for contiguous column slicing.
//
//  Supports contiguous feature partitioning without copying.
//======================================================//

#include "SliceColumnsGradFn.hpp"

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
namespace autograd {

SliceColumnsGradFn::SliceColumnsGradFn() {
    op_name = "slice_columns";
}

void SliceColumnsGradFn::capture_inputs(Tensor& x, cudaStream_t stream) {
    x_requires_grad = x.requires_grad;

    if (x_requires_grad) {
        x_gradient = capture_input_gradient(
            x, stream, "SliceColumnsGradFn::capture_inputs");
    }
}

void SliceColumnsGradFn::apply_impl(const Tensor& grad_output,
                                    cudaStream_t stream,
                                    const Batching::BatchPayload* backward_payload,
                                    const Batching::BatchDeviceBindings* backward_bindings) {
    setCurrentGradFnOp("slice_columns", this);
    if (applied) return;
    applied = true;

    if (!x_requires_grad) return;
    if (!x_gradient) {
        throw std::runtime_error("SliceColumnsGradFn::apply: x gradient Tensor is NULL");
    }

    kernel_slice_columns_backward<<<rows, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        x_gradient->data, grad_output.data, rows, in_cols, col_offset, out_cols);
    propagate_input_gradient(
        x_gradient, stream, backward_payload, backward_bindings,
        "SliceColumnsGradFn::apply");
}

void SliceColumnsGradFn::release_saved() {
    GradFn::release_saved();
    x_gradient.reset();
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
