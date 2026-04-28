//======================================================//
//  CenterColumnsGradFn.cu
//  Column-wise centering forward + autograd backward.
//======================================================//

#include "CenterColumnsGradFn.hpp"
#include "../TensorContract_GPU.hpp"
#include "../../CudaAllocUtils.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>

#ifndef TENSOR_VERBOSE_DEBUG
#define TENSOR_VERBOSE_DEBUG 0
#endif

#define AG_TRACE(...) do { if (g_autograd_verbose) { fprintf(stderr, __VA_ARGS__); fflush(stderr); } } while(0)

namespace {

constexpr int AUTOGRAD_BLOCK_SIZE = 256;

__global__ void kernel_center_columns(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_cols,
    int num_rows
) {
    const int col_idx = blockIdx.x;
    if (col_idx >= num_cols) return;

    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();

    float local_sum = 0.0f;
    for (int row = threadIdx.x; row < num_rows; row += blockDim.x) {
        local_sum += input[static_cast<size_t>(row) * num_cols + col_idx];
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd(&s_sum, local_sum);
    }
    __syncthreads();

    const float mean = s_sum / static_cast<float>(num_rows);

    for (int row = threadIdx.x; row < num_rows; row += blockDim.x) {
        const size_t idx = static_cast<size_t>(row) * num_cols + col_idx;
        output[idx] = input[idx] - mean;
    }
}

}  // anonymous namespace

namespace GRIM {

using CudaAlloc::cudaMallocOrThrow;

namespace autograd {

CenterColumnsGradFn::CenterColumnsGradFn() {
    op_name = "center_columns";
}

void CenterColumnsGradFn::capture_input(Tensor& input, int cols, int rows, cudaStream_t stream) {
    input_requires_grad = input.requires_grad;
    if (!input.requires_grad) return;

    input_shape = input.shape;
    element_count = input.numel();
    num_cols = cols;
    num_rows = rows;
    input_grad_fn = input.grad_fn;

    input.ensure_grad();

    float* buf = nullptr;
    cudaMallocOrThrow(reinterpret_cast<void**>(&buf), element_count * sizeof(float), "CenterColumnsGradFn_input_grad");
    cudaMemsetAsync(buf, 0, element_count * sizeof(float), stream);
    owned_input_grad.reset(buf, [](float* p) { queueForDeferredCleanup(p); });
    input_grad = owned_input_grad.get();
    AG_TRACE("[CenterColumnsGradFn] Allocated owned input_grad buffer (Issue #136 FIX): %zu floats at %p\n",
             element_count, (void*)input_grad);
}

void CenterColumnsGradFn::apply(const Tensor& grad_output, cudaStream_t stream) {
    if (applied) return;
    if (!input_requires_grad) return;
    if (!input_grad) throw std::runtime_error("CenterColumnsGradFn::apply: input_grad is NULL - capture_input() must be called first");
    if (!grad_output.data) throw std::runtime_error("CenterColumnsGradFn::apply: grad_output.data is NULL - backward called with null gradient");
    applied = true;

    kernel_center_columns<<<num_cols, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        grad_output.data, input_grad, num_cols, num_rows);

    if (input_grad_fn) {
        Tensor input_grad_tensor;
        input_grad_tensor.data = input_grad;
        input_grad_tensor.shape = input_shape;
        input_grad_tensor.owns_data = false;
        input_grad_tensor.stream = stream;
        input_grad_fn->apply(input_grad_tensor, stream);
    }
}

void CenterColumnsGradFn::release_saved() {
    owned_input_grad.reset();
}

Tensor center_columns(const Tensor& x, cudaStream_t stream) {
    if (!x.data) {
        throw std::runtime_error("center_columns: input tensor data is NULL");
    }
    if (!x.shape.is_2d_layout()) {
        throw std::runtime_error("center_columns: expected 2D (flat) tensor, got 4D");
    }

    const int num_rows = x.shape.as_2d().rows;
    const int num_cols = x.shape.as_2d().cols;

    const bool track_grad = x.requires_grad;
    Tensor result = Tensor::empty(x.shape, track_grad, stream, "center_columns_result");

    kernel_center_columns<<<num_cols, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        x.data, result.data, num_cols, num_rows);

    if (track_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<CenterColumnsGradFn>();
        grad_fn->capture_input(const_cast<Tensor&>(x), num_cols, num_rows, stream);
        result.grad_fn = grad_fn;
    }

    return result;
}

}  // namespace autograd
}  // namespace GRIM
