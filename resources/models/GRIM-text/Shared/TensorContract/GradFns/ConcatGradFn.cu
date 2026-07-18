//======================================================//
//  ConcatGradFn.cu
//  CUDA kernels and autograd node for row-wise concat.
//======================================================//

#include "ConcatGradFn.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace {

constexpr int AUTOGRAD_BLOCK_SIZE = 256;

//========================================================================
// Concat Forward: out[i, 0:D1] = a[i,:], out[i, D1:D1+D2] = b[i,:]
//========================================================================
__global__ void kernel_concat_forward(
    float* __restrict__ output,
    const float* __restrict__ a,
    const float* __restrict__ b,
    int N, int D1, int D2
) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int D = D1 + D2;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        if (j < D1)
            output[static_cast<size_t>(row) * D + j] = a[static_cast<size_t>(row) * D1 + j];
        else
            output[static_cast<size_t>(row) * D + j] = b[static_cast<size_t>(row) * D2 + (j - D1)];
    }
}

__global__ void kernel_concat_backward_a(
    float* __restrict__ grad_a,
    const float* __restrict__ grad_out,
    int N, int D1, int D_total
) {
    const int row = blockIdx.x;
    if (row >= N) return;
    for (int j = threadIdx.x; j < D1; j += blockDim.x)
        grad_a[static_cast<size_t>(row) * D1 + j] += grad_out[static_cast<size_t>(row) * D_total + j];
}

__global__ void kernel_concat_backward_b(
    float* __restrict__ grad_b,
    const float* __restrict__ grad_out,
    int N, int D1, int D2, int D_total
) {
    const int row = blockIdx.x;
    if (row >= N) return;
    for (int j = threadIdx.x; j < D2; j += blockDim.x)
        grad_b[static_cast<size_t>(row) * D2 + j] += grad_out[static_cast<size_t>(row) * D_total + (D1 + j)];
}

}  // anonymous namespace

namespace GRIM {
namespace autograd {

ConcatGradFn::ConcatGradFn() {
    op_name = "concat";
}

void ConcatGradFn::capture_inputs(Tensor& a, Tensor& b, cudaStream_t stream) {
    (void)stream;
    a_requires_grad = a.requires_grad;
    b_requires_grad = b.requires_grad;
    a_is_leaf = a.is_leaf;
    b_is_leaf = b.is_leaf;
    a_shape = a.shape;
    b_shape = b.shape;
    a_grad_fn = a.grad_fn;
    b_grad_fn = b.grad_fn;
    register_input(a.grad_fn);
    register_input(b.grad_fn);

    if (a_requires_grad) {
        if (a_is_leaf) {
            a.ensure_grad();
            grad_a = a.grad_;
        }
    }
    if (b_requires_grad) {
        if (b_is_leaf) {
            b.ensure_grad();
            grad_b = b.grad_;
        }
    }
}

void ConcatGradFn::apply_impl(const Tensor& grad_output,
                              cudaStream_t stream,
                              const Batching::BatchPayload* backward_payload,
                              const Batching::BatchDeviceBindings* backward_bindings) {
    setCurrentGradFnOp("concat", this);
    if (applied) return;
    applied = true;

    const int D_total = D1 + D2;
    const auto expected_output = TensorContract::TensorShape::make_BSM(rows, D_total);
    if (grad_output.shape.layout != expected_output.layout ||
        grad_output.shape.as_2d() != expected_output.as_2d()) {
        throw std::runtime_error(
            "ConcatGradFn::apply: grad_output shape does not match captured concat output");
    }

    // Non-leaf input gradients are real Tensor-owned contributions. Allocate
    // them lazily at backward time instead of keeping anonymous float buffers
    // alive from forward capture through the whole graph lifetime.
    if (a_requires_grad && !grad_a) {
        grad_a = std::make_shared<Tensor>(Tensor::zeros(
            a_shape, false, stream, "ConcatGradFn_grad_a"));
    }
    if (b_requires_grad && !grad_b) {
        if (a_grad_fn && b_grad_fn == a_grad_fn) {
            if (a_shape.layout != b_shape.layout || a_shape.as_2d() != b_shape.as_2d()) {
                throw std::runtime_error(
                    "ConcatGradFn::apply: inputs sharing one producer must have identical shapes");
            }
            grad_b = grad_a;
        } else {
            grad_b = std::make_shared<Tensor>(Tensor::zeros(
                b_shape, false, stream, "ConcatGradFn_grad_b"));
        }
    }

    if (a_requires_grad) {
        if (!grad_a) {
            throw std::runtime_error("ConcatGradFn::apply: grad_a Tensor is missing");
        }
        grad_a->require("ConcatGradFn::apply grad_a");
        kernel_concat_backward_a<<<rows, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
            grad_a->data, grad_output.data, rows, D1, D_total);
        const cudaError_t launch_status = cudaGetLastError();
        if (launch_status != cudaSuccess) {
            throw std::runtime_error(
                std::string("ConcatGradFn::apply: grad_a kernel launch failed: ") +
                cudaGetErrorString(launch_status));
        }
    }
    if (b_requires_grad) {
        if (!grad_b) {
            throw std::runtime_error("ConcatGradFn::apply: grad_b Tensor is missing");
        }
        grad_b->require("ConcatGradFn::apply grad_b");
        kernel_concat_backward_b<<<rows, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
            grad_b->data, grad_output.data, rows, D1, D2, D_total);
        const cudaError_t launch_status = cudaGetLastError();
        if (launch_status != cudaSuccess) {
            throw std::runtime_error(
                std::string("ConcatGradFn::apply: grad_b kernel launch failed: ") +
                cudaGetErrorString(launch_status));
        }
    }

    // Both split kernels must be enqueued before propagation. When both inputs
    // share one producer, grad_a and grad_b alias the same Tensor and the two
    // slices are accumulated before sending exactly one scheduler contribution.
    if (a_grad_fn) {
        a_grad_fn->apply(*grad_a, stream, backward_payload, backward_bindings);
    }
    if (b_grad_fn && b_grad_fn != a_grad_fn) {
        b_grad_fn->apply(*grad_b, stream, backward_payload, backward_bindings);
    }
}

void ConcatGradFn::release_saved() {
    GradFn::release_saved();
    grad_a.reset();
    grad_b.reset();
    a_grad_fn.reset();
    b_grad_fn.reset();
}

Tensor concat(const Tensor& a, const Tensor& b, cudaStream_t stream) {
    if (!a.shape.is_2d_layout() || !b.shape.is_2d_layout()) {
        throw std::invalid_argument("autograd::concat: both inputs must be 2D");
    }
    const auto a_dims = a.shape.as_2d();
    const auto b_dims = b.shape.as_2d();
    if (a_dims.rows != b_dims.rows) {
        throw std::invalid_argument("autograd::concat: row count mismatch (a=" +
            std::to_string(a_dims.rows) + " b=" + std::to_string(b_dims.rows) + ")");
    }
    if (!a.data || !b.data) {
        throw std::invalid_argument("autograd::concat: null data pointer");
    }

    const int N = a_dims.rows;
    const int D1 = a_dims.cols;
    const int D2 = b_dims.cols;

    const bool needs_grad = a.requires_grad || b.requires_grad;
    auto shape = TensorContract::TensorShape::make_BSM(N, D1 + D2);
    Tensor result = Tensor::empty(shape, needs_grad, stream, "concat_result");

    kernel_concat_forward<<<N, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        result.data, a.data, b.data, N, D1, D2);

    if (needs_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<ConcatGradFn>();
        grad_fn->capture_inputs(const_cast<Tensor&>(a), const_cast<Tensor&>(b), stream);
        grad_fn->rows = N;
        grad_fn->D1 = D1;
        grad_fn->D2 = D2;
        result.grad_fn = grad_fn;
    }

    return result;
}

}  // namespace autograd
}  // namespace GRIM
