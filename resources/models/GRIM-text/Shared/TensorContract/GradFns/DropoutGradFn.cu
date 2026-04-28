//======================================================//
//  DropoutGradFn.cu
//  Seeded dropout forward + autograd backward.
//======================================================//

#include "DropoutGradFn.hpp"
#include "AddGradFn.hpp"
#include "../TensorContract_GPU.hpp"
#include "../../CudaAllocUtils.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>

void trackKernelLaunch(const char* kernel_name, cudaStream_t stream);

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

inline void throwIfCudaFailed(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(err));
    }
}

__global__ void kernel_dropout_forward(
    const float* __restrict__ input,
    const std::uint8_t* __restrict__ mask,
    float* __restrict__ output,
    float scale,
    size_t count
) {
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
    if (idx < count) {
        output[idx] = input[idx] * (mask[idx] ? scale : 0.0f);
    }
}

__global__ void kernel_generate_dropout_mask(
    std::uint8_t* __restrict__ mask,
    size_t count,
    float dropout_prob,
    std::uint64_t seed
) {
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
    if (idx < count) {
        curandStatePhilox4_32_10_t state;
        curand_init(seed, idx, 0, &state);
        const float rnd = curand_uniform(&state);
        mask[idx] = (rnd > dropout_prob) ? 1 : 0;
    }
}

__global__ void kernel_dropout_backward(
    const float* grad_output,
    const std::uint8_t* mask,
    float* grad_input,
    float scale,
    size_t count
) {
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
    if (idx < count) {
        grad_input[idx] = grad_output[idx] * (mask[idx] ? scale : 0.0f);
    }
}

}  // anonymous namespace

namespace GRIM {

using CudaAlloc::cudaMallocOrThrow;

namespace autograd {

DropoutGradFn::DropoutGradFn() {
    op_name = "dropout";
}

DropoutGradFn::~DropoutGradFn() {
    release_saved();
}

void DropoutGradFn::capture_input(Tensor& x) {
    input_requires_grad = x.requires_grad;
    input_shape = x.shape;
    input_grad_fn = x.grad_fn;

    if (input_requires_grad) {
        const size_t grad_size = x.numel();
        cudaMallocOrThrow(reinterpret_cast<void**>(&input_grad), grad_size * sizeof(float), "DropoutGradFn_input_grad");
        cudaMemset(input_grad, 0, grad_size * sizeof(float));
        owns_input_grad = true;
    }
}

void DropoutGradFn::save(const std::uint8_t* mask, float dropout_prob, size_t n, cudaStream_t stream) {
    count = n;
    scale = (dropout_prob < 1.0f) ? 1.0f / (1.0f - dropout_prob) : 0.0f;

    cudaMallocOrThrow(reinterpret_cast<void**>(&saved_mask), n * sizeof(std::uint8_t), "DropoutGradFn_saved_mask");
    cudaMemcpyAsync(saved_mask, mask, n * sizeof(std::uint8_t), cudaMemcpyDeviceToDevice, stream);
}

void DropoutGradFn::apply(const Tensor& grad_output, cudaStream_t stream) {
    setCurrentGradFnOp("dropout", this);

    if (applied) {
        return;
    }
    applied = true;

    if (!input_requires_grad) return;
    if (!saved_mask) {
        throw std::runtime_error("DropoutGradFn::apply: saved_mask is NULL - forward must save dropout mask for backward");
    }
    if (!input_grad) {
        throw std::runtime_error("DropoutGradFn::apply: input_grad is NULL - capture_input() must be called first");
    }
    if (!grad_output.data) {
        throw std::runtime_error("DropoutGradFn::apply: grad_output.data is NULL");
    }

    kernel_dropout_backward<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        grad_output.data, saved_mask, input_grad, scale, count);
    trackKernelLaunch("kernel_dropout_backward", stream);

    if (input_grad_fn) {
        Tensor view;
        view.data = input_grad; view.shape = input_shape;
        view.owns_data = false; view.stream = stream;
        input_grad_fn->apply(view, stream);
    }
}

void DropoutGradFn::release_saved() {
    GradFn::release_saved();
    if (saved_mask) {
        cudaFree(saved_mask);
        saved_mask = nullptr;
    }
    if (owns_input_grad && input_grad) {
        cudaFree(input_grad);
        owns_input_grad = false;
    }
    input_grad = nullptr;
    input_grad_fn.reset();
}

Tensor dropout(const Tensor& x, float p, std::uint64_t seed, bool training, cudaStream_t stream) {
    Tensor result = Tensor::empty(x.shape, x.requires_grad, stream, "dropout_seeded_result");

    if (!training || p == 0.0f) {
        cudaMemcpyAsync(result.data, x.data, x.size_bytes(), cudaMemcpyDeviceToDevice, stream);

        if (x.requires_grad) {
            result.is_leaf = false;
            auto grad_fn = std::make_shared<AddGradFn>();
            grad_fn->capture_single_input(const_cast<Tensor&>(x), stream);
            result.grad_fn = grad_fn;
        }
        return result;
    }

    if (p < 0.0f || p >= 1.0f) {
        throw std::invalid_argument(
            "autograd::dropout: dropout probability p must be in [0, 1), got " +
            std::to_string(p));
    }

    const size_t count = x.numel();
    if (count == 0) {
        if (x.requires_grad) {
            result.is_leaf = false;
            auto grad_fn = std::make_shared<AddGradFn>();
            grad_fn->capture_single_input(const_cast<Tensor&>(x), stream);
            result.grad_fn = grad_fn;
        }
        return result;
    }
    const float scale = 1.0f / (1.0f - p);

    std::uint8_t* mask = nullptr;
    cudaMallocOrThrow(reinterpret_cast<void**>(&mask), count * sizeof(std::uint8_t), "dropout_mask");
    if (!x.data || !result.data || !mask) {
        throw std::runtime_error("autograd::dropout: null buffer(s) before kernel launch");
    }

    kernel_generate_dropout_mask<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        mask, count, p, seed);
    trackKernelLaunch("kernel_generate_dropout_mask", stream);
    throwIfCudaFailed(cudaGetLastError(), "autograd::dropout: kernel_generate_dropout_mask launch failed");

    kernel_dropout_forward<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        x.data, mask, result.data, scale, count);
    trackKernelLaunch("kernel_dropout_forward", stream);
    throwIfCudaFailed(cudaGetLastError(), "autograd::dropout: kernel_dropout_forward launch failed");

    if (x.requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<DropoutGradFn>();
        grad_fn->capture_input(const_cast<Tensor&>(x));
        grad_fn->save(mask, p, count, stream);
        result.grad_fn = grad_fn;
    }

    const cudaError_t free_err = cudaFreeAsync(mask, stream);
    if (free_err != cudaSuccess) {
        throw std::runtime_error(
            std::string("autograd::dropout: cudaFreeAsync(mask) failed: ") +
            cudaGetErrorString(free_err));
    }

    return result;
}

}  // namespace autograd
}  // namespace GRIM
