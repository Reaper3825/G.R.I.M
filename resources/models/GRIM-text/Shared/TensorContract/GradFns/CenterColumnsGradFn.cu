//======================================================//
//  CenterColumnsGradFn.cu
//  Column-wise centering forward + autograd backward.
//======================================================//

#include "CenterColumnsGradFn.hpp"
#include "../GradientAccumulation.hpp"
#include "../TensorContract_GPU.hpp"
#include "../../CudaAllocUtils.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef TENSOR_VERBOSE_DEBUG
#define TENSOR_VERBOSE_DEBUG 0
#endif

#define AG_TRACE(...) do { if (g_autograd_verbose) { fprintf(stderr, __VA_ARGS__); fflush(stderr); } } while(0)

namespace {

constexpr int AUTOGRAD_BLOCK_SIZE = 256;

__global__ void kernel_center_columns_grouped(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_cols,
    int num_rows,
    int rows_per_group
) {
    const int col_idx = blockIdx.x;
    if (col_idx >= num_cols) return;

    const int group_idx = blockIdx.y;
    const int group_count = num_rows / rows_per_group;
    if (group_idx >= group_count) return;

    const int row_start = group_idx * rows_per_group;

    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();

    float local_sum = 0.0f;
    for (int local_row = threadIdx.x; local_row < rows_per_group; local_row += blockDim.x) {
        const int row = row_start + local_row;
        local_sum += input[static_cast<size_t>(row) * num_cols + col_idx];
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd(&s_sum, local_sum);
    }
    __syncthreads();

    const float mean = s_sum / static_cast<float>(rows_per_group);

    for (int local_row = threadIdx.x; local_row < rows_per_group; local_row += blockDim.x) {
        const int row = row_start + local_row;
        const size_t idx = static_cast<size_t>(row) * num_cols + col_idx;
        output[idx] = input[idx] - mean;
    }
}

__global__ void kernel_center_columns_group_scalar_length(
    const float* __restrict__ input,
    float* __restrict__ output,
    int valid_rows,
    int num_cols,
    int num_rows,
    int rows_per_group,
    int group_idx
) {
    const int col_idx = blockIdx.x;
    if (col_idx >= num_cols) return;

    if (valid_rows <= 1 || valid_rows > rows_per_group) {
        printf("[center_columns_by_sequence_lengths] invalid seq_lengths[%d]=%d rows_per_group=%d\n",
               group_idx, valid_rows, rows_per_group);
        asm("trap;");
    }

    const int row_start = group_idx * rows_per_group;
    if (row_start + rows_per_group > num_rows) {
        printf("[center_columns_by_sequence_lengths] invalid group span group=%d row_start=%d rows_per_group=%d num_rows=%d\n",
               group_idx, row_start, rows_per_group, num_rows);
        asm("trap;");
    }

    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();

    float local_sum = 0.0f;
    for (int local_row = threadIdx.x; local_row < valid_rows; local_row += blockDim.x) {
        const int row = row_start + local_row;
        local_sum += input[static_cast<size_t>(row) * num_cols + col_idx];
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd(&s_sum, local_sum);
    }
    __syncthreads();

    const float mean = s_sum / static_cast<float>(valid_rows);

    for (int local_row = threadIdx.x; local_row < rows_per_group; local_row += blockDim.x) {
        const int row = row_start + local_row;
        const size_t idx = static_cast<size_t>(row) * num_cols + col_idx;
        output[idx] = (local_row < valid_rows) ? (input[idx] - mean) : 0.0f;
    }
}

__global__ void kernel_center_columns_causal_prefix_scalar_length(
    const float* __restrict__ input,
    float* __restrict__ output,
    int valid_rows,
    int num_cols,
    int num_rows,
    int rows_per_group,
    int group_idx
) {
    const int col_idx = blockIdx.x;
    if (col_idx >= num_cols) return;
    if (threadIdx.x != 0) return;

    if (valid_rows <= 1 || valid_rows > rows_per_group) {
        printf("[center_columns_by_causal_prefix_lengths] invalid seq_lengths[%d]=%d rows_per_group=%d\n",
               group_idx, valid_rows, rows_per_group);
        asm("trap;");
    }

    const int row_start = group_idx * rows_per_group;
    if (row_start + rows_per_group > num_rows) {
        printf("[center_columns_by_causal_prefix_lengths] invalid group span group=%d row_start=%d rows_per_group=%d num_rows=%d\n",
               group_idx, row_start, rows_per_group, num_rows);
        asm("trap;");
    }

    float prefix_sum = 0.0f;
    for (int local_row = 0; local_row < valid_rows; ++local_row) {
        const int row = row_start + local_row;
        const size_t idx = static_cast<size_t>(row) * num_cols + col_idx;
        if (local_row == 0) {
            output[idx] = input[idx];
        } else {
            const float prefix_mean = prefix_sum / static_cast<float>(local_row);
            output[idx] = input[idx] - prefix_mean;
        }
        prefix_sum += input[idx];
    }

    for (int local_row = valid_rows; local_row < rows_per_group; ++local_row) {
        const int row = row_start + local_row;
        const size_t idx = static_cast<size_t>(row) * num_cols + col_idx;
        output[idx] = 0.0f;
    }
}

__global__ void kernel_center_columns_causal_prefix_backward_scalar_length(
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int valid_rows,
    int num_cols,
    int num_rows,
    int rows_per_group,
    int group_idx
) {
    const int col_idx = blockIdx.x;
    if (col_idx >= num_cols) return;
    if (threadIdx.x != 0) return;

    if (valid_rows <= 1 || valid_rows > rows_per_group) {
        printf("[CenterColumnsGradFn::apply causal_prefix] invalid seq_lengths[%d]=%d rows_per_group=%d\n",
               group_idx, valid_rows, rows_per_group);
        asm("trap;");
    }

    const int row_start = group_idx * rows_per_group;
    if (row_start + rows_per_group > num_rows) {
        printf("[CenterColumnsGradFn::apply causal_prefix] invalid group span group=%d row_start=%d rows_per_group=%d num_rows=%d\n",
               group_idx, row_start, rows_per_group, num_rows);
        asm("trap;");
    }

    float suffix_weighted_sum = 0.0f;
    for (int local_row = valid_rows - 1; local_row >= 0; --local_row) {
        const int row = row_start + local_row;
        const size_t idx = static_cast<size_t>(row) * num_cols + col_idx;
        grad_input[idx] = grad_output[idx] - suffix_weighted_sum;
        if (local_row > 0) {
            suffix_weighted_sum += grad_output[idx] / static_cast<float>(local_row);
        }
    }

    for (int local_row = valid_rows; local_row < rows_per_group; ++local_row) {
        const int row = row_start + local_row;
        const size_t idx = static_cast<size_t>(row) * num_cols + col_idx;
        grad_input[idx] = 0.0f;
    }
}

void require_center_columns_group_shape(
    const GRIM::Tensor& x,
    int group_rows,
    const char* caller,
    int& num_rows,
    int& num_cols
) {
    if (!x.data) {
        throw std::runtime_error(std::string(caller) + ": input tensor data is NULL");
    }
    if (!x.shape.is_2d_layout()) {
        throw std::runtime_error(std::string(caller) + ": expected 2D (flat) tensor, got 4D");
    }

    num_rows = x.shape.as_2d().rows;
    num_cols = x.shape.as_2d().cols;
    if (num_rows <= 0 || num_cols <= 0) {
        throw std::runtime_error(std::string(caller) + ": invalid flat shape rows=" +
                                 std::to_string(num_rows) + " cols=" + std::to_string(num_cols));
    }
    if (group_rows <= 0) {
        throw std::runtime_error(std::string(caller) + ": rows_per_group must be > 0, got " +
                                 std::to_string(group_rows));
    }
    if (group_rows > num_rows) {
        throw std::runtime_error(std::string(caller) + ": rows_per_group=" +
                                 std::to_string(group_rows) + " exceeds tensor rows=" +
                                 std::to_string(num_rows));
    }
    if (num_rows % group_rows != 0) {
        throw std::runtime_error(std::string(caller) + ": tensor rows=" +
                                 std::to_string(num_rows) + " is not divisible by rows_per_group=" +
                                 std::to_string(group_rows));
    }
}

void require_center_columns_length_shape(
    const GRIM::Tensor& x,
    const std::vector<int>& sequence_lengths,
    int batch_size,
    int group_rows,
    const char* caller,
    int& num_rows,
    int& num_cols
) {
    require_center_columns_group_shape(x, group_rows, caller, num_rows, num_cols);
    if (batch_size <= 0) {
        throw std::runtime_error(std::string(caller) + ": batch_size must be > 0, got " +
                                 std::to_string(batch_size));
    }
    if (static_cast<int>(sequence_lengths.size()) != batch_size) {
        throw std::runtime_error(std::string(caller) + ": sequence_lengths size (" +
                                 std::to_string(sequence_lengths.size()) + ") != batch_size (" +
                                 std::to_string(batch_size) + ")");
    }
    for (int b = 0; b < batch_size; ++b) {
        const int valid_rows = sequence_lengths[static_cast<size_t>(b)];
        if (valid_rows <= 1 || valid_rows > group_rows) {
            throw std::runtime_error(std::string(caller) + ": invalid sequence_lengths[" +
                                     std::to_string(b) + "]=" + std::to_string(valid_rows) +
                                     " for rows_per_group=" + std::to_string(group_rows));
        }
    }
    const int expected_rows = batch_size * group_rows;
    if (num_rows != expected_rows) {
        throw std::runtime_error(std::string(caller) + ": tensor rows=" +
                                 std::to_string(num_rows) + " does not match batch_size * rows_per_group=" +
                                 std::to_string(batch_size) + " * " + std::to_string(group_rows) +
                                 " = " + std::to_string(expected_rows));
    }
}

void check_center_columns_kernel_launch(const char* caller, cudaStream_t stream) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(caller) + " launch failed: " +
                                 std::string(cudaGetErrorString(err)));
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(caller) + " stream synchronization failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

}  // anonymous namespace

namespace GRIM {

using CudaAlloc::cudaMallocOrThrow;

namespace autograd {

CenterColumnsGradFn::CenterColumnsGradFn() {
    op_name = "center_columns";
}

void CenterColumnsGradFn::capture_input(Tensor& input, int cols, int rows, int group_rows,
                                        const std::vector<int>* sequence_lengths_, int groups,
                                        cudaStream_t stream,
                                        bool causal_prefix) {
    input_requires_grad = input.requires_grad;
    if (!input.requires_grad) return;
    if (group_rows <= 0 || rows <= 0 || cols <= 0 || rows % group_rows != 0) {
        throw std::runtime_error("CenterColumnsGradFn::capture_input: invalid grouped shape rows=" +
                                 std::to_string(rows) + " cols=" + std::to_string(cols) +
                                 " rows_per_group=" + std::to_string(group_rows));
    }

    input_shape = input.shape;
    element_count = input.numel();
    num_cols = cols;
    num_rows = rows;
    rows_per_group = group_rows;
    group_count = rows / group_rows;
    use_sequence_lengths = (sequence_lengths_ != nullptr);
    use_causal_prefix = causal_prefix;
    if (use_sequence_lengths) {
        if (groups <= 0 || groups != group_count) {
            throw std::runtime_error("CenterColumnsGradFn::capture_input: invalid sequence length group count groups=" +
                                     std::to_string(groups) + " expected=" + std::to_string(group_count));
        }
        if (static_cast<int>(sequence_lengths_->size()) != group_count) {
            throw std::runtime_error("CenterColumnsGradFn::capture_input: sequence_lengths size (" +
                                     std::to_string(sequence_lengths_->size()) + ") != expected group_count=" +
                                     std::to_string(group_count));
        }
        sequence_lengths = *sequence_lengths_;
    }
    input_grad_fn = input.grad_fn;
    register_input(input.grad_fn);

    input.ensure_grad();
    input_is_leaf = input.is_leaf;
    if (input_is_leaf) {
        leaf_grad_buf = input.grad_data();
        if (!leaf_grad_buf) {
            throw std::runtime_error(
                "CenterColumnsGradFn::capture_input: leaf input has requires_grad but "
                "grad_data() is NULL after ensure_grad() at " __FILE__);
        }
    }

    float* buf = nullptr;
    cudaMallocOrThrow(reinterpret_cast<void**>(&buf), element_count * sizeof(float), "CenterColumnsGradFn_input_grad");
    cudaMemsetAsync(buf, 0, element_count * sizeof(float), stream);
    owned_input_grad.reset(buf, [](float* p) { queueForDeferredCleanup(p); });
    input_grad = owned_input_grad.get();
    AG_TRACE("[CenterColumnsGradFn] Allocated owned input_grad buffer (Issue #136 FIX): %zu floats at %p\n",
             element_count, (void*)input_grad);
}

void CenterColumnsGradFn::apply_impl(const Tensor& grad_output,
                                     cudaStream_t stream,
                                     const Batching::BatchPayload* backward_payload,
                                     const Batching::BatchDeviceBindings* backward_bindings) {
    if (applied) return;
    if (!input_requires_grad) return;
    if (!input_grad) throw std::runtime_error("CenterColumnsGradFn::apply: input_grad is NULL - capture_input() must be called first");
    if (!grad_output.data) throw std::runtime_error("CenterColumnsGradFn::apply: grad_output.data is NULL - backward called with null gradient");
    applied = true;

    if (use_sequence_lengths) {
        if (static_cast<int>(sequence_lengths.size()) != group_count) {
            throw std::runtime_error("CenterColumnsGradFn::apply: saved sequence_lengths size does not match group_count");
        }
        if (use_causal_prefix) {
            for (int group_idx = 0; group_idx < group_count; ++group_idx) {
                kernel_center_columns_causal_prefix_backward_scalar_length<<<num_cols, 1, 0, stream>>>(
                    grad_output.data, input_grad, sequence_lengths[static_cast<size_t>(group_idx)],
                    num_cols, num_rows, rows_per_group, group_idx);
            }
            check_center_columns_kernel_launch("CenterColumnsGradFn::apply(causal-prefix length-aware)", stream);
        } else {
            for (int group_idx = 0; group_idx < group_count; ++group_idx) {
                kernel_center_columns_group_scalar_length<<<num_cols, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
                    grad_output.data, input_grad, sequence_lengths[static_cast<size_t>(group_idx)],
                    num_cols, num_rows, rows_per_group, group_idx);
            }
            check_center_columns_kernel_launch("CenterColumnsGradFn::apply(length-aware)", stream);
        }
    } else {
        kernel_center_columns_grouped<<<dim3(num_cols, group_count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
            grad_output.data, input_grad, num_cols, num_rows, rows_per_group);
        check_center_columns_kernel_launch("CenterColumnsGradFn::apply(grouped)", stream);
    }

    if (input_is_leaf) {
        if (!leaf_grad_buf) {
            throw std::runtime_error("CenterColumnsGradFn::apply: leaf_grad_buf is NULL for leaf input at " __FILE__);
        }
        accumulate_grad(leaf_grad_buf, input_grad, element_count, 1.0f, stream, "CenterColumnsGradFn::apply leaf_grad_buf");
    }

    if (input_grad_fn) {
        Tensor input_grad_tensor;
        input_grad_tensor.data = input_grad;
        input_grad_tensor.shape = input_shape;
        input_grad_tensor.owns_data = false;
        input_grad_tensor.stream = stream;
        input_grad_fn->apply(input_grad_tensor, stream, backward_payload, backward_bindings);
    }
}

void CenterColumnsGradFn::release_saved() {
    owned_input_grad.reset();
    sequence_lengths.clear();
    input_grad = nullptr;
    leaf_grad_buf = nullptr;
    input_is_leaf = false;
    input_grad_fn.reset();
}

Tensor center_columns(const Tensor& x, cudaStream_t stream) {
    int num_rows = 0;
    int num_cols = 0;
    require_center_columns_group_shape(x, x.shape.is_2d_layout() ? x.shape.as_2d().rows : 0,
                                       "center_columns", num_rows, num_cols);

    const bool track_grad = x.requires_grad;
    Tensor result = Tensor::empty(x.shape, track_grad, stream, "center_columns_result");

    kernel_center_columns_grouped<<<dim3(num_cols, 1), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        x.data, result.data, num_cols, num_rows, num_rows);
    check_center_columns_kernel_launch("center_columns", stream);

    if (track_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<CenterColumnsGradFn>();
        grad_fn->capture_input(const_cast<Tensor&>(x), num_cols, num_rows, num_rows, nullptr, 1, stream);
        result.grad_fn = grad_fn;
    }

    return result;
}

Tensor center_columns_by_sequence(const Tensor& x, int rows_per_sequence, cudaStream_t stream) {
    int num_rows = 0;
    int num_cols = 0;
    require_center_columns_group_shape(x, rows_per_sequence,
                                       "center_columns_by_sequence", num_rows, num_cols);

    const bool track_grad = x.requires_grad;
    Tensor result = Tensor::empty(x.shape, track_grad, stream, "center_columns_by_sequence_result");

    const int sequence_count = num_rows / rows_per_sequence;
    kernel_center_columns_grouped<<<dim3(num_cols, sequence_count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        x.data, result.data, num_cols, num_rows, rows_per_sequence);
    check_center_columns_kernel_launch("center_columns_by_sequence", stream);

    if (track_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<CenterColumnsGradFn>();
        grad_fn->capture_input(const_cast<Tensor&>(x), num_cols, num_rows, rows_per_sequence, nullptr, sequence_count, stream);
        result.grad_fn = grad_fn;
    }

    return result;
}

Tensor center_columns_by_sequence_lengths(const Tensor& x,
                                          const std::vector<int>& sequence_lengths,
                                          int batch_size,
                                          int rows_per_sequence,
                                          cudaStream_t stream) {
    int num_rows = 0;
    int num_cols = 0;
    require_center_columns_length_shape(x, sequence_lengths, batch_size, rows_per_sequence,
                                        "center_columns_by_sequence_lengths", num_rows, num_cols);

    const bool track_grad = x.requires_grad;
    Tensor result = Tensor::empty(x.shape, track_grad, stream, "center_columns_by_sequence_lengths_result");

    for (int group_idx = 0; group_idx < batch_size; ++group_idx) {
        kernel_center_columns_group_scalar_length<<<num_cols, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
            x.data, result.data, sequence_lengths[static_cast<size_t>(group_idx)],
            num_cols, num_rows, rows_per_sequence, group_idx);
    }
    check_center_columns_kernel_launch("center_columns_by_sequence_lengths", stream);

    if (track_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<CenterColumnsGradFn>();
        grad_fn->capture_input(const_cast<Tensor&>(x), num_cols, num_rows, rows_per_sequence,
                               &sequence_lengths, batch_size, stream);
        result.grad_fn = grad_fn;
    }

    return result;
}

Tensor center_columns_by_causal_prefix_lengths(const Tensor& x,
                                               const std::vector<int>& sequence_lengths,
                                               int batch_size,
                                               int rows_per_sequence,
                                               cudaStream_t stream) {
    int num_rows = 0;
    int num_cols = 0;
    require_center_columns_length_shape(x, sequence_lengths, batch_size, rows_per_sequence,
                                        "center_columns_by_causal_prefix_lengths", num_rows, num_cols);

    const bool track_grad = x.requires_grad;
    Tensor result = Tensor::empty(x.shape, track_grad, stream, "center_columns_by_causal_prefix_lengths_result");

    for (int group_idx = 0; group_idx < batch_size; ++group_idx) {
        kernel_center_columns_causal_prefix_scalar_length<<<num_cols, 1, 0, stream>>>(
            x.data, result.data, sequence_lengths[static_cast<size_t>(group_idx)],
            num_cols, num_rows, rows_per_sequence, group_idx);
    }
    check_center_columns_kernel_launch("center_columns_by_causal_prefix_lengths", stream);

    if (track_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<CenterColumnsGradFn>();
        grad_fn->capture_input(const_cast<Tensor&>(x), num_cols, num_rows, rows_per_sequence,
                               &sequence_lengths, batch_size, stream, true);
        result.grad_fn = grad_fn;
    }

    return result;
}

}  // namespace autograd
}  // namespace GRIM
