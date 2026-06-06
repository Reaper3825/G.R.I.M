//======================================================//
//  ConcatGradFn.cu
//  CUDA kernels and autograd node for row-wise concat.
//======================================================//

#include "ConcatGradFn.hpp"
#include "../../CudaAllocUtils.hpp"

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
using CudaAlloc::cudaMallocOrThrow;
namespace autograd {

ConcatGradFn::ConcatGradFn() {
    op_name = "concat";
}

void ConcatGradFn::capture_inputs(Tensor& a, Tensor& b, cudaStream_t stream) {
    a_requires_grad = a.requires_grad;
    b_requires_grad = b.requires_grad;
    a_shape = a.shape;
    b_shape = b.shape;
    a_grad_fn = a.grad_fn;
    b_grad_fn = b.grad_fn;

    if (a_requires_grad) {
        a.ensure_grad();
        if (a.is_leaf) {
            grad_a = a.grad_data();
        } else {
            const size_t n = a.numel();
            float* buffer = nullptr;
            cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), n * sizeof(float), "ConcatGradFn_grad_a");
            cudaMemsetAsync(buffer, 0, n * sizeof(float), stream);
            owned_grad_a = std::shared_ptr<float>(buffer, [](float* p) { queueForDeferredCleanup(p); });
            grad_a = owned_grad_a.get();
        }
    }
    if (b_requires_grad) {
        b.ensure_grad();
        if (b.is_leaf) {
            grad_b = b.grad_data();
        } else {
            const size_t n = b.numel();
            float* buffer = nullptr;
            cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), n * sizeof(float), "ConcatGradFn_grad_b");
            cudaMemsetAsync(buffer, 0, n * sizeof(float), stream);
            owned_grad_b = std::shared_ptr<float>(buffer, [](float* p) { queueForDeferredCleanup(p); });
            grad_b = owned_grad_b.get();
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

    if (a_requires_grad && grad_a) {
        kernel_concat_backward_a<<<rows, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
            grad_a, grad_output.data, rows, D1, D_total);
        if (a_grad_fn) {
            Tensor view;
            view.data = grad_a; view.shape = a_shape;
            view.owns_data = false; view.stream = stream;
            a_grad_fn->apply(view, stream, backward_payload, backward_bindings);
        }
    }
    if (b_requires_grad && grad_b) {
        kernel_concat_backward_b<<<rows, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
            grad_b, grad_output.data, rows, D1, D2, D_total);
        if (b_grad_fn && b_grad_fn != a_grad_fn) {
            Tensor view;
            view.data = grad_b; view.shape = b_shape;
            view.owns_data = false; view.stream = stream;
            b_grad_fn->apply(view, stream, backward_payload, backward_bindings);
        }
    }
}

void ConcatGradFn::release_saved() {
    GradFn::release_saved();
    grad_a = nullptr;
    grad_b = nullptr;
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
