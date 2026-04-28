//======================================================//
//  CenterRowsGradFn.cu
//  Row-wise centering forward + autograd backward.
//======================================================//

#include "CenterRowsGradFn.hpp"
#include "TensorContract_GPU.hpp"
#include "../CudaAllocUtils.hpp"

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
constexpr int kMaxGridBlocks1DFallback = 65534;
constexpr int kMaxGridDimY = 65535;

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

__global__ void kernel_accumulate_grad(float* dst, const float* src, size_t count, float scale) {
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] += src[idx] * scale;
    }
}

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

}  // anonymous namespace

namespace GRIM {

using CudaAlloc::cudaMallocOrThrow;

namespace autograd {

CenterRowsGradFn::CenterRowsGradFn() {
    op_name = "center_rows";
}

void CenterRowsGradFn::capture_input(Tensor& input, int dim, int rows, cudaStream_t stream) {
    input_requires_grad = input.requires_grad;
    if (!input.requires_grad) return;

    input_shape = input.shape;
    element_count = input.numel();
    row_dim = dim;
    num_rows = rows;
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
    AG_TRACE("[CenterRowsGradFn] Allocated owned input_grad buffer (Issue #136 FIX): %zu floats at %p (leaf=%d)\n",
             element_count, (void*)input_grad, (int)input_is_leaf);
}

void CenterRowsGradFn::apply(const Tensor& grad_output, cudaStream_t stream) {
    if (applied) return;
    if (!input_requires_grad) return;
    if (!input_grad) throw std::runtime_error("CenterRowsGradFn::apply: input_grad is NULL - capture_input() must be called first");
    if (!grad_output.data) throw std::runtime_error("CenterRowsGradFn::apply: grad_output.data is NULL - backward called with null gradient");
    applied = true;

    kernel_center_rows<<<num_rows, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        grad_output.data, input_grad, row_dim, num_rows);

    if (input_is_leaf) {
        if (!leaf_grad_buf) {
            throw std::runtime_error("CenterRowsGradFn::apply: leaf_grad_buf is NULL for leaf input at " __FILE__);
        }
        kernel_accumulate_grad<<<gridForCount(element_count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
            leaf_grad_buf, input_grad, element_count, 1.0f);
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

}  // namespace autograd
}  // namespace GRIM
