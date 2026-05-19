//======================================================//
//  CenterRowsGradFn.cu
//  Row-wise centering forward + autograd backward.
//======================================================//

#include "CenterRowsGradFn.hpp"
#include "../GradientAccumulation.hpp"
#include "../TensorContract_GPU.hpp"
#include "../TokenTypeGate.hpp"
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

__global__ void kernel_center_rows(
    const float* __restrict__ input,
    float* __restrict__ output,
    int row_dim,
    int num_rows
) {
    const int row_idx = blockIdx.x;
    if (row_idx >= num_rows) return;

    const float* in_row = input + static_cast<size_t>(row_idx) * row_dim;
    float* out_row = output + static_cast<size_t>(row_idx) * row_dim;

    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < row_dim; i += blockDim.x) {
        local_sum += in_row[i];
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd(&s_sum, local_sum);
    }
    __syncthreads();

    const float mean = s_sum / static_cast<float>(row_dim);

    for (int i = threadIdx.x; i < row_dim; i += blockDim.x) {
        out_row[i] = in_row[i] - mean;
    }
}

__global__ void kernel_type_gate_rows(
    const float* __restrict__ input,
    float* __restrict__ output,
    int row_dim,
    int num_rows
) {
    const int row_idx = blockIdx.x;
    if (row_idx >= num_rows) return;

    const auto gate = GRIM::TensorContract::tokenTypeGateRangeForTokenId(row_idx, row_dim, num_rows);
    if (gate.width <= 0) {
        printf("FATAL: invalid token type gate for row_idx=%d row_dim=%d num_rows=%d in kernel_type_gate_rows\n",
               row_idx, row_dim, num_rows);
        __trap();
    }

    const float* in_row = input + static_cast<size_t>(row_idx) * row_dim;
    float* out_row = output + static_cast<size_t>(row_idx) * row_dim;
    for (int i = threadIdx.x; i < row_dim; i += blockDim.x) {
        out_row[i] = (i >= gate.start && i < gate.end) ? in_row[i] : 0.0f;
    }
}

__global__ void kernel_center_rows_by_token_type_gate(
    const float* __restrict__ input,
    float* __restrict__ output,
    int row_dim,
    int num_rows
) {
    const int row_idx = blockIdx.x;
    if (row_idx >= num_rows) return;

    const auto gate = GRIM::TensorContract::tokenTypeGateRangeForTokenId(row_idx, row_dim, num_rows);
    if (gate.width <= 0) {
        printf("FATAL: invalid token type gate for row_idx=%d row_dim=%d num_rows=%d in kernel_center_rows_by_token_type_gate\n",
               row_idx, row_dim, num_rows);
        __trap();
    }

    const float* in_row = input + static_cast<size_t>(row_idx) * row_dim;
    float* out_row = output + static_cast<size_t>(row_idx) * row_dim;

    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = gate.start + threadIdx.x; i < gate.end; i += blockDim.x) {
        local_sum += in_row[i];
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd(&s_sum, local_sum);
    }
    __syncthreads();

    const float mean = s_sum / static_cast<float>(gate.width);

    for (int i = threadIdx.x; i < row_dim; i += blockDim.x) {
        out_row[i] = (i >= gate.start && i < gate.end) ? (in_row[i] - mean) : 0.0f;
    }
}

}  // anonymous namespace

namespace GRIM {

using CudaAlloc::cudaMallocOrThrow;

namespace autograd {

CenterRowsGradFn::CenterRowsGradFn() {
    op_name = "center_rows";
}

void CenterRowsGradFn::capture_input(Tensor& input, int dim, int rows, cudaStream_t stream,
                                     bool token_type_gate, bool center_active) {
    input_requires_grad = input.requires_grad;
    if (!input.requires_grad) return;

    input_shape = input.shape;
    element_count = input.numel();
    row_dim = dim;
    num_rows = rows;
    use_token_type_gate = token_type_gate;
    center_active_subspace = center_active;
    input_grad_fn = input.grad_fn;

    input.ensure_grad();

    input_is_leaf = input.is_leaf;
    if (input_is_leaf) {
        leaf_grad_buf = input.grad_data();
        if (!leaf_grad_buf) {
            throw std::runtime_error(
                "CenterRowsGradFn::capture_input: leaf input has requires_grad but "
                "grad_data() is NULL after ensure_grad() at " __FILE__);
        }
    }

    float* buf = nullptr;
    cudaMallocOrThrow(reinterpret_cast<void**>(&buf), element_count * sizeof(float), "CenterRowsGradFn_input_grad");
    cudaMemsetAsync(buf, 0, element_count * sizeof(float), stream);
    owned_input_grad.reset(buf, [](float* p) { queueForDeferredCleanup(p); });
    input_grad = owned_input_grad.get();
    AG_TRACE("[CenterRowsGradFn] Allocated owned input_grad buffer (Issue #136 FIX): %zu floats at %p (leaf=%d token_gate=%d center_active=%d)\n",
             element_count, (void*)input_grad, (int)input_is_leaf, (int)use_token_type_gate, (int)center_active_subspace);
}

void CenterRowsGradFn::apply_impl(const Tensor& grad_output, cudaStream_t stream) {
    if (applied) return;
    if (!input_requires_grad) return;
    if (!input_grad) throw std::runtime_error("CenterRowsGradFn::apply: input_grad is NULL - capture_input() must be called first");
    if (!grad_output.data) throw std::runtime_error("CenterRowsGradFn::apply: grad_output.data is NULL - backward called with null gradient");
    applied = true;

    if (use_token_type_gate && center_active_subspace) {
        kernel_center_rows_by_token_type_gate<<<num_rows, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
            grad_output.data, input_grad, row_dim, num_rows);
    } else if (use_token_type_gate) {
        kernel_type_gate_rows<<<num_rows, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
            grad_output.data, input_grad, row_dim, num_rows);
    } else {
        kernel_center_rows<<<num_rows, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
            grad_output.data, input_grad, row_dim, num_rows);
    }

    if (input_is_leaf) {
        if (!leaf_grad_buf) {
            throw std::runtime_error("CenterRowsGradFn::apply: leaf_grad_buf is NULL for leaf input at " __FILE__);
        }
        accumulate_grad(leaf_grad_buf, input_grad, element_count, 1.0f, stream, "CenterRowsGradFn::apply leaf_grad_buf");
    }

    if (input_grad_fn) {
        Tensor input_grad_tensor;
        input_grad_tensor.data = input_grad;
        input_grad_tensor.shape = input_shape;
        input_grad_tensor.owns_data = false;
        input_grad_tensor.stream = stream;
        input_grad_fn->apply(input_grad_tensor, stream);
    }
}

void CenterRowsGradFn::release_saved() {
    owned_input_grad.reset();
}

Tensor center_rows(const Tensor& x, cudaStream_t stream) {
    if (!x.data) {
        throw std::runtime_error("center_rows: input tensor data is NULL");
    }
    if (!x.shape.is_2d_layout()) {
        throw std::runtime_error("center_rows: expected 2D (flat) tensor, got 4D");
    }

    const int num_rows = x.shape.as_2d().rows;
    const int row_dim = x.shape.as_2d().cols;

    const bool track_grad = x.requires_grad;
    Tensor result = Tensor::empty(x.shape, track_grad, stream, "center_rows_result");

    kernel_center_rows<<<num_rows, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        x.data, result.data, row_dim, num_rows);

    if (track_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<CenterRowsGradFn>();
        grad_fn->capture_input(const_cast<Tensor&>(x), row_dim, num_rows, stream);
        result.grad_fn = grad_fn;
    }

    return result;
}

Tensor type_gate_rows_by_token_type(const Tensor& x, cudaStream_t stream) {
    if (!x.data) {
        throw std::runtime_error("type_gate_rows_by_token_type: input tensor data is NULL");
    }
    if (!x.shape.is_2d_layout()) {
        throw std::runtime_error("type_gate_rows_by_token_type: expected 2D (flat) tensor, got 4D");
    }

    const int num_rows = x.shape.as_2d().rows;
    const int row_dim = x.shape.as_2d().cols;
    if (row_dim < 4) {
        throw std::runtime_error("type_gate_rows_by_token_type: row_dim must be >= 4, got " +
                                 std::to_string(row_dim));
    }

    const bool track_grad = x.requires_grad;
    Tensor result = Tensor::empty(x.shape, track_grad, stream, "type_gate_rows_by_token_type_result");

    kernel_type_gate_rows<<<num_rows, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        x.data, result.data, row_dim, num_rows);

    if (track_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<CenterRowsGradFn>();
        grad_fn->op_name = "type_gate_rows_by_token_type";
        grad_fn->capture_input(const_cast<Tensor&>(x), row_dim, num_rows, stream, true, false);
        result.grad_fn = grad_fn;
    }

    return result;
}

Tensor center_rows_by_token_type_gate(const Tensor& x, cudaStream_t stream) {
    if (!x.data) {
        throw std::runtime_error("center_rows_by_token_type_gate: input tensor data is NULL");
    }
    if (!x.shape.is_2d_layout()) {
        throw std::runtime_error("center_rows_by_token_type_gate: expected 2D (flat) tensor, got 4D");
    }

    const int num_rows = x.shape.as_2d().rows;
    const int row_dim = x.shape.as_2d().cols;
    if (row_dim < 4) {
        throw std::runtime_error("center_rows_by_token_type_gate: row_dim must be >= 4, got " +
                                 std::to_string(row_dim));
    }

    const bool track_grad = x.requires_grad;
    Tensor result = Tensor::empty(x.shape, track_grad, stream, "center_rows_by_token_type_gate_result");

    kernel_center_rows_by_token_type_gate<<<num_rows, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        x.data, result.data, row_dim, num_rows);

    if (track_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<CenterRowsGradFn>();
        grad_fn->op_name = "center_rows_by_token_type_gate";
        grad_fn->capture_input(const_cast<Tensor&>(x), row_dim, num_rows, stream, true, true);
        result.grad_fn = grad_fn;
    }

    return result;
}

}  // namespace autograd
}  // namespace GRIM
