//======================================================//
//  LogSoftmaxGradFn.cu
//  Log-softmax forward + autograd backward.
//======================================================//

#include "LogSoftmaxGradFn.hpp"
#include "../TensorContract_GPU.hpp"
#include "../../CudaAllocUtils.hpp"

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>

void trackKernelLaunch(const char* kernel_name, cudaStream_t stream);

namespace {

constexpr int AUTOGRAD_BLOCK_SIZE = 256;

inline void throwIfCudaFailed(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(err));
    }
}

__global__ void kernel_log_softmax_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    int tokens,
    int dim
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= tokens) return;

    const float* row = input + static_cast<size_t>(token_idx) * dim;
    float* out_row = output + static_cast<size_t>(token_idx) * dim;

    constexpr int kMaxWarps = 8;
    __shared__ float s_warp[kMaxWarps];
    __shared__ float s_val;

    float local_max = -FLT_MAX;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        local_max = fmaxf(local_max, row[i]);
    }
    for (int off = warpSize / 2; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, off));

    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;
    const int num_warps = (blockDim.x + warpSize - 1) / warpSize;

    if (lane_id == 0 && warp_id < kMaxWarps) s_warp[warp_id] = local_max;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = s_warp[0];
        for (int w = 1; w < num_warps && w < kMaxWarps; w++) m = fmaxf(m, s_warp[w]);
        s_val = m;
    }
    __syncthreads();
    const float max_val = s_val;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        local_sum += expf(row[i] - max_val);

    for (int off = warpSize / 2; off > 0; off >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, off);

    if (lane_id == 0 && warp_id < kMaxWarps) s_warp[warp_id] = local_sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int w = 0; w < num_warps && w < kMaxWarps; w++) s += s_warp[w];
        s_val = s;
    }
    __syncthreads();
    const float sum_exp = s_val;

    if (threadIdx.x == 0 && (sum_exp < 1e-6f || isnan(sum_exp) || isinf(sum_exp))) {
        printf("[LOG_SOFTMAX_EQUATION] FATAL: sum_exp=%e at token %d - logits corrupted! "
               "max_val=%e, dim=%d\n", sum_exp, token_idx, max_val, dim);
    }

    const float log_sum_exp = logf(sum_exp) + max_val;

    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        out_row[i] = row[i] - log_sum_exp;
}

__global__ void kernel_log_softmax_backward(
    const float* __restrict__ grad_output,
    const float* __restrict__ log_softmax,
    float* __restrict__ grad_input,
    int tokens,
    int dim
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= tokens) return;

    const float* dy = grad_output + static_cast<size_t>(token_idx) * dim;
    const float* log_p = log_softmax + static_cast<size_t>(token_idx) * dim;
    float* dx = grad_input + static_cast<size_t>(token_idx) * dim;

    constexpr int kMaxWarps = 8;
    __shared__ float s_warp[kMaxWarps];
    __shared__ float s_sum;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        local_sum += dy[i];

    for (int off = warpSize / 2; off > 0; off >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, off);

    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;
    const int num_warps = (blockDim.x + warpSize - 1) / warpSize;

    if (lane_id == 0 && warp_id < kMaxWarps) s_warp[warp_id] = local_sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int w = 0; w < num_warps && w < kMaxWarps; w++) s += s_warp[w];
        s_sum = s;
    }
    __syncthreads();

    const float sum_dy = s_sum;

    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        dx[i] += dy[i] - expf(log_p[i]) * sum_dy;
}

}  // anonymous namespace

namespace GRIM {

using CudaAlloc::cudaMallocOrThrow;

namespace autograd {

LogSoftmaxGradFn::LogSoftmaxGradFn() {
    op_name = "log_softmax";
}

LogSoftmaxGradFn::~LogSoftmaxGradFn() {
    release_saved();
}

void LogSoftmaxGradFn::capture_input(Tensor& x, cudaStream_t stream) {
    input_requires_grad = x.requires_grad;
    input_shape = x.shape;
    input_grad_fn = x.grad_fn;
    register_input(x.grad_fn);

    if (input_requires_grad) {
        x.ensure_grad();
        if (x.is_leaf) {
            input_grad = x.grad_data();
            owns_input_grad = false;
        } else {
            const size_t bytes = x.shape.total_elements() * sizeof(float);
            cudaMallocOrThrow(reinterpret_cast<void**>(&input_grad), bytes, "LogSoftmaxGradFn_input_grad");
            throwIfCudaFailed(
                cudaMemsetAsync(input_grad, 0, bytes, stream),
                "LogSoftmaxGradFn::capture_input: cudaMemsetAsync(input_grad) failed");
            owns_input_grad = true;
        }
    }
}

void LogSoftmaxGradFn::save(const float* log_softmax_output, int tokens, int d, cudaStream_t stream, bool copy) {
    num_tokens = tokens;
    dim = d;
    if (copy) {
        const size_t bytes = static_cast<size_t>(tokens) * d * sizeof(float);
        cudaMallocOrThrow(reinterpret_cast<void**>(&saved_log_softmax), bytes, "LogSoftmaxGradFn_saved");
        throwIfCudaFailed(
            cudaMemcpyAsync(saved_log_softmax, log_softmax_output, bytes, cudaMemcpyDeviceToDevice, stream),
            "LogSoftmaxGradFn::save: cudaMemcpyAsync(saved_log_softmax) failed");
        owns_saved_log_softmax = true;
    } else {
        saved_log_softmax = const_cast<float*>(log_softmax_output);
        owns_saved_log_softmax = false;
    }
}

void LogSoftmaxGradFn::apply_impl(const Tensor& grad_output,
                                  cudaStream_t stream,
                                  const Batching::BatchPayload* backward_payload,
                                  const Batching::BatchDeviceBindings* backward_bindings) {
    setCurrentGradFnOp("log_softmax", this);

    if (applied) return;
    applied = true;

    if (!input_requires_grad) return;
    if (!saved_log_softmax) throw std::runtime_error("LogSoftmaxGradFn::apply: saved_log_softmax is NULL - forward must save output for backward");
    if (!input_grad) {
        throw std::runtime_error(
            "LogSoftmaxGradFn::apply: input_grad is NULL - "
            "capture_input() must be called first");
    }

    kernel_log_softmax_backward<<<num_tokens, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        grad_output.data, saved_log_softmax, input_grad, num_tokens, dim);
    trackKernelLaunch("kernel_log_softmax_backward", stream);
    throwIfCudaFailed(cudaGetLastError(), "LogSoftmaxGradFn::apply: kernel_log_softmax_backward launch failed");

    if (input_grad_fn) {
        Tensor view;
        view.data = input_grad;
        view.shape = input_shape;
        view.owns_data = false;
        view.stream = stream;
        input_grad_fn->apply(view, stream, backward_payload, backward_bindings);
    }
}

void LogSoftmaxGradFn::release_saved() {
    GradFn::release_saved();
    if (owns_saved_log_softmax && saved_log_softmax) {
        cudaFree(saved_log_softmax);
        saved_log_softmax = nullptr;
    } else {
        saved_log_softmax = nullptr;
    }
    if (owns_input_grad && input_grad) {
        cudaFree(input_grad);
        input_grad = nullptr;
    }
    owns_input_grad = false;
    input_grad_fn.reset();
}

Tensor log_softmax(const Tensor& x, cudaStream_t stream, bool save_output_copy) {
    if (!x.shape.is_2d_layout()) {
        throw std::invalid_argument("autograd::log_softmax: input must be 2D [tokens, dim]");
    }
    if (!x.data) {
        throw std::invalid_argument("autograd::log_softmax: input data is NULL");
    }

    const auto dims = x.shape.as_2d();
    const int num_tokens = dims.rows;
    const int dim = dims.cols;

    Tensor result = Tensor::empty(x.shape, x.requires_grad, stream, "log_softmax_result");

    kernel_log_softmax_forward<<<num_tokens, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        x.data, result.data, num_tokens, dim);
    trackKernelLaunch("kernel_log_softmax_forward", stream);
    throwIfCudaFailed(cudaGetLastError(), "autograd::log_softmax: kernel_log_softmax_forward launch failed");

    if (x.requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<LogSoftmaxGradFn>();
        grad_fn->capture_input(const_cast<Tensor&>(x), stream);
        grad_fn->save(result.data, num_tokens, dim, stream, save_output_copy);
        result.grad_fn = grad_fn;
    }

    return result;
}

}  // namespace autograd
}  // namespace GRIM
