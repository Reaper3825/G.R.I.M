//======================================================//
//  ProjectOutPC1GradFn.cu
//  PC1 projection forward + autograd backward.
//======================================================//

#include "ProjectOutPC1GradFn.hpp"
#include "../TensorContract_GPU.hpp"
#include "../../CudaAllocUtils.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>

#ifndef TENSOR_VERBOSE_DEBUG
#define TENSOR_VERBOSE_DEBUG 0
#endif

#define AG_TRACE(...) do { if (g_autograd_verbose) { fprintf(stderr, __VA_ARGS__); fflush(stderr); } } while(0)

namespace {

static __global__ void kernel_pc1_col_mean(
    const float* __restrict__ H, float* __restrict__ g, int T, int D)
{
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float s = 0.f;
        for (int t = 0; t < T; t++) s += H[(size_t)t * D + d];
        g[d] = s / (float)T;
    }
}

static __global__ void kernel_pc1_normalize(float* __restrict__ g, int D)
{
    assert(blockDim.x == 256);
    __shared__ float sdata[256];
    __shared__ float s_inv;
    __shared__ int s_degenerate;
    float local = 0.f;
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        local += g[d] * g[d];
    sdata[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float sum_sq = sdata[0];
        const float kFloor = 1e-20f;
        s_degenerate = (sum_sq < kFloor) ? 1 : 0;
        s_inv = (sum_sq < kFloor) ? 0.f : (1.f / sqrtf(sum_sq / (float)D));
    }
    __syncthreads();
    if (s_degenerate) {
        for (int d = threadIdx.x; d < D; d += blockDim.x)
            g[d] = (d == 0) ? 1.f : 0.f;
    } else {
        const float inv = s_inv;
        for (int d = threadIdx.x; d < D; d += blockDim.x)
            g[d] *= inv;
    }
}

static __global__ void kernel_pc1_gemv_Hg(
    const float* __restrict__ H, const float* __restrict__ g,
    float* __restrict__ v, int T, int D)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    float dot = 0.f;
    for (int d = 0; d < D; d++) dot += H[(size_t)t * D + d] * g[d];
    v[t] = dot;
}

static __global__ void kernel_pc1_gemv_HtV(
    const float* __restrict__ H, const float* __restrict__ v,
    float* __restrict__ g_out, int T, int D)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float dot = 0.f;
    for (int t = 0; t < T; t++) dot += H[(size_t)t * D + d] * v[t];
    g_out[d] = dot;
}

static __global__ void kernel_pc1_project(
    const float* __restrict__ H, const float* __restrict__ g,
    float* __restrict__ H_out, int T, int D)
{
    assert(blockDim.x == 256);
    int t = blockIdx.x;
    if (t >= T) return;
    __shared__ float sdata[256];
    float local = 0.f;
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        local += H[(size_t)t * D + d] * g[d];
    sdata[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float coeff = sdata[0] / (float)D;
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        H_out[(size_t)t * D + d] = H[(size_t)t * D + d] - coeff * g[d];
}

static __global__ void kernel_pc1_project_accum(
    const float* __restrict__ H, const float* __restrict__ g,
    float* __restrict__ H_out, int T, int D)
{
    assert(blockDim.x == 256);
    int t = blockIdx.x;
    if (t >= T) return;
    __shared__ float sdata[256];
    float local = 0.f;
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        local += H[(size_t)t * D + d] * g[d];
    sdata[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float coeff = sdata[0] / (float)D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const size_t idx = (size_t)t * D + d;
        H_out[idx] += H[idx] - coeff * g[d];
    }
}

}  // anonymous namespace

namespace GRIM {

using CudaAlloc::cudaMallocOrThrow;

namespace autograd {

ProjectOutPC1GradFn::ProjectOutPC1GradFn() {
    op_name = "project_out_pc1";
}

void ProjectOutPC1GradFn::capture_input(Tensor& input, int rows, int cols,
                                        float* g_device, cudaStream_t stream) {
    if (g_device == nullptr)
        throw std::runtime_error("ProjectOutPC1GradFn::capture_input: g_device is NULL");
    if (rows <= 0 || cols <= 0)
        throw std::runtime_error("ProjectOutPC1GradFn::capture_input: invalid dims rows=" +
                                 std::to_string(rows) + " cols=" + std::to_string(cols));
    if (input.numel() != (std::size_t)rows * (std::size_t)cols)
        throw std::runtime_error("ProjectOutPC1GradFn::capture_input: input.numel()=" +
                                 std::to_string(input.numel()) + " != rows*cols=" +
                                 std::to_string((std::size_t)rows * (std::size_t)cols));

    input_requires_grad = input.requires_grad;
    input_shape = input.shape;
    element_count = input.numel();
    num_rows = rows;
    num_cols = cols;

    owned_g.reset(g_device, [](float* p) { queueForDeferredCleanup(p); });
    g_saved = owned_g.get();

    if (!input.requires_grad) return;

    input_grad_fn = input.grad_fn;

    if (input.is_leaf) {
        input.ensure_grad();
        input_grad = input.grad_data();
        AG_TRACE("[ProjectOutPC1GradFn] Using persistent input_grad buffer (leaf): %p\n", (void*)input_grad);
    } else {
        float* buf = nullptr;
        cudaMallocOrThrow(reinterpret_cast<void**>(&buf), element_count * sizeof(float), "ProjectOutPC1GradFn_input_grad");
        cudaMemsetAsync(buf, 0, element_count * sizeof(float), stream);
        owned_input_grad.reset(buf, [](float* p) { queueForDeferredCleanup(p); });
        input_grad = owned_input_grad.get();
        AG_TRACE("[ProjectOutPC1GradFn] Allocated input_grad buffer (non-leaf): %zu floats at %p\n",
                 element_count, (void*)input_grad);
    }
}

void ProjectOutPC1GradFn::apply(const Tensor& grad_output, cudaStream_t stream) {
    if (applied) return;
    if (!input_requires_grad) return;
    if (!g_saved) throw std::runtime_error("ProjectOutPC1GradFn::apply: g_saved is NULL - g direction was freed before backward");
    if (grad_output.data == nullptr)
        throw std::runtime_error("ProjectOutPC1GradFn::apply: grad_output.data is NULL");
    if (grad_output.numel() != element_count)
        throw std::runtime_error("ProjectOutPC1GradFn::apply: grad_output.numel()=" +
                                 std::to_string(grad_output.numel()) +
                                 " != captured element_count=" + std::to_string(element_count));
    if (!input_grad)
        throw std::runtime_error("ProjectOutPC1GradFn::apply: input_grad is NULL - capture_input did not run or wiring is broken");
    applied = true;

    kernel_pc1_project_accum<<<num_rows, 256, 0, stream>>>(
        grad_output.data, g_saved, input_grad, num_rows, num_cols);

    if (input_grad_fn) {
        Tensor input_grad_tensor;
        input_grad_tensor.data = input_grad;
        input_grad_tensor.shape = input_shape;
        input_grad_tensor.owns_data = false;
        input_grad_tensor.stream = stream;
        input_grad_fn->apply(input_grad_tensor, stream);
    }
}

void ProjectOutPC1GradFn::release_saved() {
    owned_input_grad.reset();
    owned_g.reset();
    g_saved = nullptr;
}

Tensor project_out_pc1(const Tensor& x, int n_power_iters, cudaStream_t stream) {
    if (x.numel() == 0)
        throw std::runtime_error("project_out_pc1: input tensor is empty");
    if (x.data == nullptr)
        throw std::runtime_error("project_out_pc1: input tensor data is NULL");
    if (!x.shape.is_2d_layout())
        throw std::runtime_error("project_out_pc1: expected 2D (flat) tensor [T, D]");
    if (n_power_iters < 0)
        throw std::runtime_error("project_out_pc1: n_power_iters must be >= 0, got " + std::to_string(n_power_iters));

    const int D = (int)x.shape.flat.cols;
    const int T = (int)(x.numel() / (std::size_t)D);
    if (D <= 0 || T <= 0)
        throw std::runtime_error("project_out_pc1: invalid dimensions T=" + std::to_string(T) + " D=" + std::to_string(D));
    if (T < 2)
        throw std::runtime_error("project_out_pc1: requires T >= 2 (got T=" + std::to_string(T) + "); projection of a single row collapses to zero");
    if (D < 2)
        throw std::runtime_error("project_out_pc1: requires D >= 2 (got D=" + std::to_string(D) + "); projection in 1D collapses to zero");

    const bool track_grad = x.requires_grad;

    float* g_buf = nullptr;
    float* g_tmp = nullptr;
    float* v_buf = nullptr;
    cudaMallocOrThrow(reinterpret_cast<void**>(&g_buf), D * sizeof(float), "pc1_g_buf");
    cudaMallocOrThrow(reinterpret_cast<void**>(&g_tmp), D * sizeof(float), "pc1_g_tmp");
    cudaMallocOrThrow(reinterpret_cast<void**>(&v_buf), T * sizeof(float), "pc1_v_buf");

    kernel_pc1_col_mean<<<1, 256, 0, stream>>>(x.data, g_buf, T, D);
    kernel_pc1_normalize<<<1, 256, 0, stream>>>(g_buf, D);

    const int blk = 256;
    for (int iter = 0; iter < n_power_iters; ++iter) {
        kernel_pc1_gemv_Hg<<<(T + blk - 1) / blk, blk, 0, stream>>>(x.data, g_buf, v_buf, T, D);
        kernel_pc1_gemv_HtV<<<(D + blk - 1) / blk, blk, 0, stream>>>(x.data, v_buf, g_tmp, T, D);
        kernel_pc1_normalize<<<1, 256, 0, stream>>>(g_tmp, D);
        std::swap(g_buf, g_tmp);
    }

    cudaError_t pc1_err = cudaStreamSynchronize(stream);
    if (pc1_err != cudaSuccess)
        throw std::runtime_error("project_out_pc1: PC1 kernels failed: " + std::string(cudaGetErrorString(pc1_err)));

    float* out_data = nullptr;
    cudaMallocOrThrow(reinterpret_cast<void**>(&out_data), (std::size_t)T * D * sizeof(float), "pc1_output");

    Tensor result;
    result.data = out_data;
    result.shape = x.shape;
    result.owns_data = true;
    result.requires_grad = track_grad;
    result.is_leaf = false;
    result.stream = stream;

    kernel_pc1_project<<<T, 256, 0, stream>>>(x.data, g_buf, out_data, T, D);

    cudaFreeAsync(v_buf, stream);
    cudaFreeAsync(g_tmp, stream);

    if (track_grad) {
        auto grad_fn = std::make_shared<ProjectOutPC1GradFn>();
        grad_fn->capture_input(const_cast<Tensor&>(x), T, D, g_buf, stream);
        result.grad_fn = grad_fn;
    } else {
        cudaFreeAsync(g_buf, stream);
    }

    return result;
}

}  // namespace autograd
}  // namespace GRIM
