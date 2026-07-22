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

#define AG_TRACE(...) do { if constexpr (GRIM::VerboseLogging::ENABLE_AUTOGRAD_TRACE_LOGS) { fprintf(stderr, __VA_ARGS__); fflush(stderr); } } while(0)

namespace {

inline void throwIfCudaFailed(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(err));
    }
}

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        reset();
    }

    T* get() const {
        return ptr_;
    }

    T* release() {
        T* out = ptr_;
        ptr_ = nullptr;
        return out;
    }

    void reset(T* next = nullptr) {
        if (ptr_) cudaFree(ptr_);
        ptr_ = next;
    }

    void allocate(std::size_t count, const char* label) {
        if (count == 0)
            throw std::runtime_error(std::string(label) + ": allocation count is 0");
        reset();
        T* raw = nullptr;
        GRIM::CudaAlloc::cudaMallocOrThrow(reinterpret_cast<void**>(&raw), count * sizeof(T), label);
        ptr_ = raw;
    }

    void swap(DeviceBuffer& other) {
        std::swap(ptr_, other.ptr_);
    }

private:
    T* ptr_ = nullptr;
};

static __global__ void kernel_pc1_col_mean(
    const float* __restrict__ H, float* __restrict__ g, int T, int D)
{
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float s = 0.f;
        for (int t = 0; t < T; t++) s += H[(size_t)t * D + d];
        g[d] = s / (float)T;
    }
}

static __global__ void kernel_pc1_normalize(
    float* __restrict__ g, int D, int* __restrict__ degenerate_flag,
    float* __restrict__ inv_norms, int inv_index)
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
        if (inv_norms) {
            if (inv_index < 0) {
                printf("[kernel_pc1_normalize] invalid inv_index=%d\n", inv_index);
                asm("trap;");
            }
            inv_norms[inv_index] = s_inv;
        }
    }
    __syncthreads();
    if (s_degenerate) {
        if (threadIdx.x == 0 && degenerate_flag)
            *degenerate_flag = 1;
        for (int d = threadIdx.x; d < D; d += blockDim.x)
            g[d] = 0.f;
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

static __global__ void kernel_pc1_projection_g_adjoint(
    const float* __restrict__ x_t_beta,
    const float* __restrict__ grad_t_a,
    float* __restrict__ adj_g,
    int D)
{
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    adj_g[d] = -(x_t_beta[d] + grad_t_a[d]) / (float)D;
}

static __global__ void kernel_pc1_normalize_backward(
    const float* __restrict__ adj_normalized,
    const float* __restrict__ normalized,
    const float* __restrict__ inv_norms,
    int inv_index,
    float* __restrict__ adj_pre_norm,
    int D)
{
    assert(blockDim.x == 256);
    __shared__ float sdata[256];
    __shared__ float s_dot_over_d;
    __shared__ float s_inv;

    float local = 0.f;
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        local += normalized[d] * adj_normalized[d];
    sdata[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (!inv_norms || inv_index < 0) {
            printf("[kernel_pc1_normalize_backward] missing inv_norms or invalid inv_index=%d\n", inv_index);
            asm("trap;");
        }
        s_inv = inv_norms[inv_index];
        if (!(s_inv > 0.f)) {
            printf("[kernel_pc1_normalize_backward] non-positive inv_norm[%d]=%.9e\n", inv_index, s_inv);
            asm("trap;");
        }
        s_dot_over_d = sdata[0] / (float)D;
    }
    __syncthreads();

    const float dot_over_d = s_dot_over_d;
    const float inv = s_inv;
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        adj_pre_norm[d] = inv * (adj_normalized[d] - normalized[d] * dot_over_d);
}

static __global__ void kernel_pc1_outer_accum(
    const float* __restrict__ row_vec,
    const float* __restrict__ col_vec,
    float* __restrict__ out,
    int T,
    int D,
    float scale)
{
    const size_t n = (size_t)T * (size_t)D;
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int t = (int)(idx / (size_t)D);
    const int d = (int)(idx - (size_t)t * (size_t)D);
    out[idx] += scale * row_vec[t] * col_vec[d];
}

static __global__ void kernel_pc1_mean_backward_accum(
    const float* __restrict__ adj_mean,
    float* __restrict__ out,
    int T,
    int D)
{
    const size_t n = (size_t)T * (size_t)D;
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int d = (int)(idx - (idx / (size_t)D) * (size_t)D);
    out[idx] += adj_mean[d] / (float)T;
}

}  // anonymous namespace

namespace GRIM {

using CudaAlloc::cudaMallocOrThrow;

namespace autograd {

ProjectOutPC1GradFn::ProjectOutPC1GradFn() {
    op_name = "project_out_pc1";
}

void ProjectOutPC1GradFn::capture_input(Tensor& input, int rows, int cols, int n_power_iters,
                                        float* g_device, float* g_history_device, float* inv_norm_device,
                                        cudaStream_t stream) {
    if (g_device == nullptr)
        throw std::runtime_error("ProjectOutPC1GradFn::capture_input: g_device is NULL");
    if (g_history_device == nullptr)
        throw std::runtime_error("ProjectOutPC1GradFn::capture_input: g_history_device is NULL");
    if (inv_norm_device == nullptr)
        throw std::runtime_error("ProjectOutPC1GradFn::capture_input: inv_norm_device is NULL");
    if (n_power_iters < 1)
        throw std::runtime_error("ProjectOutPC1GradFn::capture_input: n_power_iters must be >= 1, got " +
                                 std::to_string(n_power_iters));
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
    power_iters = n_power_iters;

    if (!input.requires_grad) {
        owned_g.reset(g_device, [](float* p) { queueForDeferredCleanup(p); });
        g_saved = owned_g.get();
        owned_g_history.reset(g_history_device, [](float* p) { queueForDeferredCleanup(p); });
        g_history_saved = owned_g_history.get();
        owned_inv_norms.reset(inv_norm_device, [](float* p) { queueForDeferredCleanup(p); });
        inv_norm_saved = owned_inv_norms.get();
        return;
    }

    input_grad_fn = input.grad_fn;
    register_input(input.grad_fn);

    float* input_copy = nullptr;
    cudaMallocOrThrow(reinterpret_cast<void**>(&input_copy), element_count * sizeof(float), "ProjectOutPC1GradFn_input_data");
    throwIfCudaFailed(cudaMemcpyAsync(input_copy, input.data, element_count * sizeof(float), cudaMemcpyDeviceToDevice, stream),
                      "ProjectOutPC1GradFn::capture_input: cudaMemcpyAsync(input_data) failed");
    owned_input_data.reset(input_copy, [](float* p) { queueForDeferredCleanup(p); });
    input_data_saved = owned_input_data.get();

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

    owned_g.reset(g_device, [](float* p) { queueForDeferredCleanup(p); });
    g_saved = owned_g.get();
    owned_g_history.reset(g_history_device, [](float* p) { queueForDeferredCleanup(p); });
    g_history_saved = owned_g_history.get();
    owned_inv_norms.reset(inv_norm_device, [](float* p) { queueForDeferredCleanup(p); });
    inv_norm_saved = owned_inv_norms.get();
}

void ProjectOutPC1GradFn::apply_impl(const Tensor& grad_output,
                                     cudaStream_t stream,
                                     const Batching::BatchPayload* backward_payload,
                                     const Batching::BatchDeviceBindings* backward_bindings) {
    if (applied) return;
    if (!input_requires_grad) {
        applied = true;
        return;
    }
    if (!g_saved) throw std::runtime_error("ProjectOutPC1GradFn::apply: g_saved is NULL - g direction was freed before backward");
    if (!g_history_saved)
        throw std::runtime_error("ProjectOutPC1GradFn::apply: g_history_saved is NULL - forward PC1 tape was not captured");
    if (!inv_norm_saved)
        throw std::runtime_error("ProjectOutPC1GradFn::apply: inv_norm_saved is NULL - normalization tape was not captured");
    if (!input_data_saved)
        throw std::runtime_error("ProjectOutPC1GradFn::apply: input_data_saved is NULL - input forward data was not captured");
    if (grad_output.data == nullptr)
        throw std::runtime_error("ProjectOutPC1GradFn::apply: grad_output.data is NULL");
    if (grad_output.numel() != element_count)
        throw std::runtime_error("ProjectOutPC1GradFn::apply: grad_output.numel()=" +
                                 std::to_string(grad_output.numel()) +
                                 " != captured element_count=" + std::to_string(element_count));
    if (!input_grad)
        throw std::runtime_error("ProjectOutPC1GradFn::apply: input_grad is NULL - capture_input did not run or wiring is broken");
    if (power_iters < 1)
        throw std::runtime_error("ProjectOutPC1GradFn::apply: power_iters must be >= 1, got " +
                                 std::to_string(power_iters));

    applied = true;

    const int blk = 256;
    const int elem_grid = static_cast<int>((element_count + (std::size_t)blk - 1) / (std::size_t)blk);

    DeviceBuffer<float> a_buf;
    DeviceBuffer<float> beta_buf;
    DeviceBuffer<float> tmp1_buf;
    DeviceBuffer<float> tmp2_buf;
    DeviceBuffer<float> adj_a_buf;
    DeviceBuffer<float> adj_b_buf;
    DeviceBuffer<float> adj_pre_norm_buf;
    DeviceBuffer<float> v_buf;
    DeviceBuffer<float> dv_buf;
    a_buf.allocate((std::size_t)num_rows, "ProjectOutPC1GradFn_a");
    beta_buf.allocate((std::size_t)num_rows, "ProjectOutPC1GradFn_beta");
    tmp1_buf.allocate((std::size_t)num_cols, "ProjectOutPC1GradFn_tmp1");
    tmp2_buf.allocate((std::size_t)num_cols, "ProjectOutPC1GradFn_tmp2");
    adj_a_buf.allocate((std::size_t)num_cols, "ProjectOutPC1GradFn_adj_a");
    adj_b_buf.allocate((std::size_t)num_cols, "ProjectOutPC1GradFn_adj_b");
    adj_pre_norm_buf.allocate((std::size_t)num_cols, "ProjectOutPC1GradFn_adj_pre_norm");
    v_buf.allocate((std::size_t)num_rows, "ProjectOutPC1GradFn_v");
    dv_buf.allocate((std::size_t)num_rows, "ProjectOutPC1GradFn_dv");

    kernel_pc1_gemv_Hg<<<(num_rows + blk - 1) / blk, blk, 0, stream>>>(
        input_data_saved, g_saved, a_buf.get(), num_rows, num_cols);
    kernel_pc1_gemv_Hg<<<(num_rows + blk - 1) / blk, blk, 0, stream>>>(
        grad_output.data, g_saved, beta_buf.get(), num_rows, num_cols);
    kernel_pc1_project_accum<<<num_rows, 256, 0, stream>>>(
        grad_output.data, g_saved, input_grad, num_rows, num_cols);
    kernel_pc1_gemv_HtV<<<(num_cols + blk - 1) / blk, blk, 0, stream>>>(
        input_data_saved, beta_buf.get(), tmp1_buf.get(), num_rows, num_cols);
    kernel_pc1_gemv_HtV<<<(num_cols + blk - 1) / blk, blk, 0, stream>>>(
        grad_output.data, a_buf.get(), tmp2_buf.get(), num_rows, num_cols);
    kernel_pc1_projection_g_adjoint<<<(num_cols + blk - 1) / blk, blk, 0, stream>>>(
        tmp1_buf.get(), tmp2_buf.get(), adj_a_buf.get(), num_cols);
    throwIfCudaFailed(cudaGetLastError(), "ProjectOutPC1GradFn::apply: projection reverse launch failed");
    throwIfCudaFailed(cudaStreamSynchronize(stream), "ProjectOutPC1GradFn::apply: projection reverse failed");

    float* adj_next = adj_a_buf.get();
    float* adj_current = adj_b_buf.get();
    for (int iter = power_iters - 1; iter >= 0; --iter) {
        const float* g_i = g_history_saved + (std::size_t)iter * (std::size_t)num_cols;
        const float* g_next = g_history_saved + (std::size_t)(iter + 1) * (std::size_t)num_cols;

        kernel_pc1_normalize_backward<<<1, 256, 0, stream>>>(
            adj_next, g_next, inv_norm_saved, iter + 1, adj_pre_norm_buf.get(), num_cols);
        kernel_pc1_gemv_Hg<<<(num_rows + blk - 1) / blk, blk, 0, stream>>>(
            input_data_saved, g_i, v_buf.get(), num_rows, num_cols);
        kernel_pc1_outer_accum<<<elem_grid, blk, 0, stream>>>(
            v_buf.get(), adj_pre_norm_buf.get(), input_grad, num_rows, num_cols, 1.0f);
        kernel_pc1_gemv_Hg<<<(num_rows + blk - 1) / blk, blk, 0, stream>>>(
            input_data_saved, adj_pre_norm_buf.get(), dv_buf.get(), num_rows, num_cols);
        kernel_pc1_outer_accum<<<elem_grid, blk, 0, stream>>>(
            dv_buf.get(), g_i, input_grad, num_rows, num_cols, 1.0f);
        kernel_pc1_gemv_HtV<<<(num_cols + blk - 1) / blk, blk, 0, stream>>>(
            input_data_saved, dv_buf.get(), adj_current, num_rows, num_cols);
        throwIfCudaFailed(cudaGetLastError(), "ProjectOutPC1GradFn::apply: power-iteration reverse launch failed");
        throwIfCudaFailed(cudaStreamSynchronize(stream), "ProjectOutPC1GradFn::apply: power-iteration reverse failed");
        std::swap(adj_next, adj_current);
    }

    kernel_pc1_normalize_backward<<<1, 256, 0, stream>>>(
        adj_next, g_history_saved, inv_norm_saved, 0, adj_pre_norm_buf.get(), num_cols);
    kernel_pc1_mean_backward_accum<<<elem_grid, blk, 0, stream>>>(
        adj_pre_norm_buf.get(), input_grad, num_rows, num_cols);
    throwIfCudaFailed(cudaGetLastError(), "ProjectOutPC1GradFn::apply: seed reverse launch failed");
    throwIfCudaFailed(cudaStreamSynchronize(stream), "ProjectOutPC1GradFn::apply: seed reverse failed");

    if (input_grad_fn) {
        Tensor input_grad_tensor;
        input_grad_tensor.data = input_grad;
        input_grad_tensor.shape = input_shape;
        input_grad_tensor.owns_data = false;
        input_grad_tensor.stream = stream;
        input_grad_fn->apply(input_grad_tensor, stream, backward_payload, backward_bindings);
    }
}

void ProjectOutPC1GradFn::release_saved() {
    owned_input_grad.reset();
    owned_input_data.reset();
    owned_g.reset();
    owned_g_history.reset();
    owned_inv_norms.reset();
    input_grad = nullptr;
    input_data_saved = nullptr;
    g_saved = nullptr;
    g_history_saved = nullptr;
    inv_norm_saved = nullptr;
    input_grad_fn.reset();
    power_iters = 0;
}

Tensor project_out_pc1(const Tensor& x, int n_power_iters, cudaStream_t stream) {
    if (x.numel() == 0)
        throw std::runtime_error("project_out_pc1: input tensor is empty");
    if (x.data == nullptr)
        throw std::runtime_error("project_out_pc1: input tensor data is NULL");
    if (!x.shape.is_2d_layout())
        throw std::runtime_error("project_out_pc1: expected 2D (flat) tensor [T, D]");
    if (n_power_iters < 1)
        throw std::runtime_error(
            "project_out_pc1: n_power_iters must be >= 1 because 0 projects the mean direction instead of PC1; got " +
            std::to_string(n_power_iters));

    if (x.shape.total_elements() != x.numel())
        throw std::runtime_error("project_out_pc1: shape total_elements=" +
                                 std::to_string(x.shape.total_elements()) +
                                 " != tensor numel=" + std::to_string(x.numel()));

    const int T = x.shape.flat.rows;
    const int D = x.shape.flat.cols;
    if (D <= 0 || T <= 0)
        throw std::runtime_error("project_out_pc1: invalid dimensions T=" + std::to_string(T) + " D=" + std::to_string(D));
    if (T < 2)
        throw std::runtime_error("project_out_pc1: requires T >= 2 (got T=" + std::to_string(T) + "); projection of a single row collapses to zero");
    if (D < 2)
        throw std::runtime_error("project_out_pc1: requires D >= 2 (got D=" + std::to_string(D) + "); projection in 1D collapses to zero");

    const bool track_grad = x.requires_grad;

    DeviceBuffer<float> g_buf;
    DeviceBuffer<float> g_tmp;
    DeviceBuffer<float> v_buf;
    DeviceBuffer<float> g_history_buf;
    DeviceBuffer<float> inv_norm_buf;
    DeviceBuffer<int> degenerate_flag;
    g_buf.allocate((std::size_t)D, "pc1_g_buf");
    g_tmp.allocate((std::size_t)D, "pc1_g_tmp");
    v_buf.allocate((std::size_t)T, "pc1_v_buf");
    if (track_grad) {
        g_history_buf.allocate((std::size_t)(n_power_iters + 1) * (std::size_t)D, "pc1_g_history");
        inv_norm_buf.allocate((std::size_t)(n_power_iters + 1), "pc1_inv_norm_history");
    }
    degenerate_flag.allocate(1, "pc1_degenerate_flag");
    throwIfCudaFailed(cudaMemsetAsync(degenerate_flag.get(), 0, sizeof(int), stream),
                      "project_out_pc1: cudaMemsetAsync(degenerate_flag) failed");

    kernel_pc1_col_mean<<<1, 256, 0, stream>>>(x.data, g_buf.get(), T, D);
    kernel_pc1_normalize<<<1, 256, 0, stream>>>(
        g_buf.get(), D, degenerate_flag.get(), track_grad ? inv_norm_buf.get() : nullptr, track_grad ? 0 : -1);
    if (track_grad) {
        throwIfCudaFailed(cudaMemcpyAsync(g_history_buf.get(), g_buf.get(), (std::size_t)D * sizeof(float),
                                          cudaMemcpyDeviceToDevice, stream),
                          "project_out_pc1: failed to save initial PC1 direction");
    }

    const int blk = 256;
    for (int iter = 0; iter < n_power_iters; ++iter) {
        kernel_pc1_gemv_Hg<<<(T + blk - 1) / blk, blk, 0, stream>>>(x.data, g_buf.get(), v_buf.get(), T, D);
        kernel_pc1_gemv_HtV<<<(D + blk - 1) / blk, blk, 0, stream>>>(x.data, v_buf.get(), g_tmp.get(), T, D);
        kernel_pc1_normalize<<<1, 256, 0, stream>>>(
            g_tmp.get(), D, degenerate_flag.get(), track_grad ? inv_norm_buf.get() : nullptr, track_grad ? iter + 1 : -1);
        if (track_grad) {
            throwIfCudaFailed(cudaMemcpyAsync(g_history_buf.get() + (std::size_t)(iter + 1) * (std::size_t)D,
                                              g_tmp.get(), (std::size_t)D * sizeof(float),
                                              cudaMemcpyDeviceToDevice, stream),
                              "project_out_pc1: failed to save power-iteration PC1 direction");
        }
        g_buf.swap(g_tmp);
    }

    throwIfCudaFailed(cudaGetLastError(), "project_out_pc1: PC1 estimation kernel launch failed");
    throwIfCudaFailed(cudaStreamSynchronize(stream), "project_out_pc1: PC1 estimation kernels failed");

    int host_degenerate_flag = 0;
    throwIfCudaFailed(cudaMemcpy(&host_degenerate_flag, degenerate_flag.get(), sizeof(int), cudaMemcpyDeviceToHost),
                      "project_out_pc1: failed to read degeneracy flag");
    if (host_degenerate_flag != 0) {
        throw std::runtime_error(
            "project_out_pc1: degenerate PC1 direction; initial mean/power-iteration vector has near-zero norm and cannot be normalized");
    }

    DeviceBuffer<float> out_data;
    out_data.allocate((std::size_t)T * (std::size_t)D, "pc1_output");

    Tensor result;
    result.data = nullptr;
    result.shape = x.shape;
    result.owns_data = true;
    result.requires_grad = track_grad;
    result.is_leaf = false;
    result.stream = stream;
    result.name = "project_out_pc1_result";

    kernel_pc1_project<<<T, 256, 0, stream>>>(x.data, g_buf.get(), out_data.get(), T, D);
    throwIfCudaFailed(cudaGetLastError(), "project_out_pc1: kernel_pc1_project launch failed");
    throwIfCudaFailed(cudaStreamSynchronize(stream), "project_out_pc1: kernel_pc1_project failed");

    if (track_grad) {
        auto grad_fn = std::make_shared<ProjectOutPC1GradFn>();
        grad_fn->capture_input(const_cast<Tensor&>(x), T, D, n_power_iters,
                               g_buf.get(), g_history_buf.get(), inv_norm_buf.get(), stream);
        g_buf.release();
        g_history_buf.release();
        inv_norm_buf.release();
        result.grad_fn = grad_fn;
    }

    result.data = out_data.release();

    return result;
}

}  // namespace autograd
}  // namespace GRIM
