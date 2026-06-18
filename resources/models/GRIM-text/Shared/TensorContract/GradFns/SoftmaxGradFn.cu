//======================================================//
//  SoftmaxGradFn.cu
//  Softmax forward + autograd backward.
//======================================================//

#include "SoftmaxGradFn.hpp"
#include "../TensorContract_GPU.hpp"
#include "../../CudaAllocUtils.hpp"

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

constexpr int AUTOGRAD_BLOCK_SIZE = 256;

__global__ void kernel_softmax_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    int tokens,
    int dim,
    float inv_temperature
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= tokens) return;

    const float* row = input + static_cast<size_t>(token_idx) * dim;
    float* out_row = output + static_cast<size_t>(token_idx) * dim;

    constexpr int kMaxWarps = 8;
    __shared__ float s_warp[kMaxWarps];
    __shared__ float s_val;

    float local_max = -FLT_MAX;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        local_max = fmaxf(local_max, row[i] * inv_temperature);
    for (int off = warpSize / 2; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, off));

    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;
    const int num_warps = (blockDim.x + warpSize - 1) / warpSize;

    if (lane_id == 0 && warp_id < kMaxWarps) s_warp[warp_id] = local_max;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = -FLT_MAX;
        for (int w = 0; w < num_warps && w < kMaxWarps; w++) m = fmaxf(m, s_warp[w]);
        s_val = m;
    }
    __syncthreads();
    const float max_val = s_val;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float e = expf(row[i] * inv_temperature - max_val);
        out_row[i] = e;
        local_sum += e;
    }
    for (int off = warpSize / 2; off > 0; off >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, off);

    if (lane_id == 0 && warp_id < kMaxWarps) s_warp[warp_id] = local_sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int w = 0; w < num_warps && w < kMaxWarps; w++) s += s_warp[w];
        s_val = 1.0f / (s + 1e-7f);
    }
    __syncthreads();
    const float inv_sum = s_val;

    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        out_row[i] *= inv_sum;
}

__global__ void kernel_softmax_backward(
    const float* __restrict__ grad_output,
    const float* __restrict__ softmax_out,
    float* __restrict__ grad_input,
    int tokens,
    int dim,
    float inv_temperature
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= tokens) return;

    const float* dy = grad_output + static_cast<size_t>(token_idx) * dim;
    const float* p = softmax_out + static_cast<size_t>(token_idx) * dim;
    float* dx = grad_input + static_cast<size_t>(token_idx) * dim;

    constexpr int kMaxWarps = 8;
    __shared__ float s_warp[kMaxWarps];
    __shared__ float s_dot;

    float local_dot = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        local_dot += dy[i] * p[i];
    for (int off = warpSize / 2; off > 0; off >>= 1)
        local_dot += __shfl_down_sync(0xffffffff, local_dot, off);

    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;
    const int num_warps = (blockDim.x + warpSize - 1) / warpSize;

    if (lane_id == 0 && warp_id < kMaxWarps) s_warp[warp_id] = local_dot;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int w = 0; w < num_warps && w < kMaxWarps; w++) s += s_warp[w];
        s_dot = s;
    }
    __syncthreads();
    const float dot = s_dot;

    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        dx[i] += p[i] * (dy[i] - dot) * inv_temperature;
}

}  // anonymous namespace

namespace GRIM {

using CudaAlloc::cudaMallocOrThrow;

namespace autograd {

SoftmaxGradFn::SoftmaxGradFn() {
    op_name = "softmax";
}

SoftmaxGradFn::~SoftmaxGradFn() {
    release_saved();
}

void SoftmaxGradFn::capture_input(Tensor& x, cudaStream_t stream) {
    input_requires_grad = x.requires_grad;
    input_shape = x.shape;
    input_grad_fn = x.grad_fn;
    register_input(x.grad_fn);

    if (input_requires_grad) {
        x.ensure_grad();
        if (x.is_leaf) {
            input_grad = x.grad_data();
        } else {
            const size_t n = x.numel();
            float* buffer = nullptr;
            cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), n * sizeof(float), "SoftmaxGradFn_input_grad");
            cudaMemsetAsync(buffer, 0, n * sizeof(float), stream);
            owned_input_grad = std::shared_ptr<float>(buffer, [](float* p) {
                queueForDeferredCleanup(p);
            });
            input_grad = owned_input_grad.get();
        }
    }
}

void SoftmaxGradFn::save(const float* softmax_output, int tokens_, int dim_, float inv_temp, cudaStream_t stream) {
    num_tokens = tokens_;
    dim = dim_;
    inv_temperature = inv_temp;
    const size_t bytes = static_cast<size_t>(tokens_) * dim_ * sizeof(float);
    cudaMallocOrThrow(reinterpret_cast<void**>(&saved_softmax), bytes, "SoftmaxGradFn_saved");
    cudaMemcpyAsync(saved_softmax, softmax_output, bytes, cudaMemcpyDeviceToDevice, stream);
}

void SoftmaxGradFn::apply_impl(const Tensor& grad_output,
                               cudaStream_t stream,
                               const Batching::BatchPayload* backward_payload,
                               const Batching::BatchDeviceBindings* backward_bindings) {
    setCurrentGradFnOp("softmax", this);
    if (applied) return;
    applied = true;
    if (!input_requires_grad) return;
    if (!saved_softmax || !input_grad) {
        throw std::runtime_error("SoftmaxGradFn::apply: saved data or grad buffer is NULL");
    }

    kernel_softmax_backward<<<num_tokens, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        grad_output.data, saved_softmax, input_grad, num_tokens, dim, inv_temperature);

    if (input_grad_fn) {
        Tensor view;
        view.data = input_grad; view.shape = input_shape;
        view.owns_data = false; view.stream = stream;
        input_grad_fn->apply(view, stream, backward_payload, backward_bindings);
    }
}

void SoftmaxGradFn::release_saved() {
    GradFn::release_saved();
    if (saved_softmax) {
        cudaFree(saved_softmax);
        saved_softmax = nullptr;
    }
    input_grad = nullptr;
    input_grad_fn.reset();
}

Tensor softmax(const Tensor& x, float temperature, cudaStream_t stream) {
    if (!x.shape.is_2d_layout()) {
        throw std::invalid_argument("autograd::softmax: input must be 2D [tokens, dim]");
    }
    if (!x.data) {
        throw std::invalid_argument("autograd::softmax: input data is NULL");
    }
    if (temperature < 1e-6f) temperature = 1e-6f;
    const float inv_temp = 1.0f / temperature;

    const auto dims = x.shape.as_2d();
    const int num_tokens = dims.rows;
    const int dim = dims.cols;

    Tensor result = Tensor::empty(x.shape, x.requires_grad, stream, "softmax_result");

    kernel_softmax_forward<<<num_tokens, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        x.data, result.data, num_tokens, dim, inv_temp);

    if (x.requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<SoftmaxGradFn>();
        grad_fn->capture_input(const_cast<Tensor&>(x), stream);
        grad_fn->save(result.data, num_tokens, dim, inv_temp, stream);
        result.grad_fn = grad_fn;
    }

    return result;
}

}  // namespace autograd
}  // namespace GRIM
