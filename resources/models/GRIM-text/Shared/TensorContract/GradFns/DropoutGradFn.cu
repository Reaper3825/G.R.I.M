//======================================================//
//  DropoutGradFn.cu
//  Seeded dropout forward + autograd backward.
//======================================================//

#include "DropoutGradFn.hpp"
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

inline std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

inline std::uint64_t deriveDropoutMaskSeed(std::uint64_t base_seed, std::uint64_t mask_stream_id) {
    if (mask_stream_id == 0) {
        throw std::runtime_error("autograd::dropout: mask_stream_id is 0 - caller MUST provide a non-zero call-specific dropout mask stream id");
    }
    return splitmix64(base_seed ^ splitmix64(mask_stream_id));
}

inline bool tensorShapesEqual(const TensorContract::TensorShape& a, const TensorContract::TensorShape& b) {
    if (a.layout != b.layout) return false;
    if (a.is_2d_layout() && b.is_2d_layout()) return a.as_2d() == b.as_2d();
    if (a.is_4d() && b.is_4d()) return a.as_4d() == b.as_4d();
    return false;
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
        grad_input[idx] += grad_output[idx] * (mask[idx] ? scale : 0.0f);
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

void DropoutGradFn::capture_input(Tensor& x, cudaStream_t stream) {
    input_requires_grad = x.requires_grad;
    input_shape = x.shape;
    input_grad_fn = x.grad_fn;
    register_input(x.grad_fn);

    if (input_requires_grad) {
        if (x.is_leaf) {
            x.ensure_grad();
            input_grad = x.grad_data();
            owns_input_grad = false;
        } else {
            const size_t grad_size = x.numel();
            cudaMallocOrThrow(reinterpret_cast<void**>(&input_grad), grad_size * sizeof(float), "DropoutGradFn_input_grad");
            throwIfCudaFailed(
                cudaMemsetAsync(input_grad, 0, grad_size * sizeof(float), stream),
                "DropoutGradFn::capture_input: cudaMemsetAsync(input_grad) failed");
            owns_input_grad = true;
        }
    }
}

void DropoutGradFn::save(const std::uint8_t* mask, float dropout_prob, size_t n, cudaStream_t stream) {
    count = n;
    scale = (dropout_prob < 1.0f) ? 1.0f / (1.0f - dropout_prob) : 0.0f;

    cudaMallocOrThrow(reinterpret_cast<void**>(&saved_mask), n * sizeof(std::uint8_t), "DropoutGradFn_saved_mask");
    cudaMemcpyAsync(saved_mask, mask, n * sizeof(std::uint8_t), cudaMemcpyDeviceToDevice, stream);
}

void DropoutGradFn::apply_impl(const Tensor& grad_output,
                               cudaStream_t stream,
                               const Batching::BatchPayload* backward_payload,
                               const Batching::BatchDeviceBindings* backward_bindings) {
    setCurrentGradFnOp("dropout", this);

    if (applied) {
        return;
    }

    if (!input_requires_grad) {
        applied = true;
        return;
    }
    if (!saved_mask) {
        throw std::runtime_error("DropoutGradFn::apply: saved_mask is NULL - forward must save dropout mask for backward");
    }
    if (!input_grad) {
        throw std::runtime_error("DropoutGradFn::apply: input_grad is NULL - capture_input() must be called first");
    }
    if (!grad_output.data) {
        throw std::runtime_error("DropoutGradFn::apply: grad_output.data is NULL");
    }
    if (grad_output.numel() != count) {
        throw std::runtime_error(
            "DropoutGradFn::apply: grad_output numel mismatch, expected " +
            std::to_string(count) + " got " + std::to_string(grad_output.numel()));
    }
    input_shape.require("DropoutGradFn::apply input_shape");
    grad_output.shape.require("DropoutGradFn::apply grad_output.shape");
    if (!tensorShapesEqual(grad_output.shape, input_shape)) {
        throw std::runtime_error("DropoutGradFn::apply: grad_output shape mismatch with captured input_shape");
    }

    kernel_dropout_backward<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        grad_output.data, saved_mask, input_grad, scale, count);
    trackKernelLaunch("kernel_dropout_backward", stream);
    throwIfCudaFailed(cudaGetLastError(), "DropoutGradFn::apply: kernel_dropout_backward launch failed");

    applied = true;

    if (input_grad_fn) {
        Tensor view;
        view.data = input_grad; view.shape = input_shape;
        view.owns_data = false; view.stream = stream;
        input_grad_fn->apply(view, stream, backward_payload, backward_bindings);
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

Tensor dropout(const Tensor& x, float p, std::uint64_t seed, cudaStream_t stream,
               std::uint64_t mask_stream_id) {
    if (p <= 0.0f || p >= 1.0f) {
        throw std::invalid_argument(
            "autograd::dropout: training-only dropout probability p must be in (0, 1), got " +
            std::to_string(p));
    }
    if (!stream) {
        throw std::runtime_error("autograd::dropout: stream is NULL - caller MUST provide valid stream");
    }
    if (!x.data) {
        throw std::runtime_error("autograd::dropout: x.data is NULL");
    }

    const size_t count = x.numel();
    if (count == 0) {
        throw std::runtime_error("autograd::dropout: x.numel() is 0 - dropout caller must provide a non-empty training tensor");
    }

    Tensor result = Tensor::empty(x.shape, x.requires_grad, stream, "dropout_seeded_result");
    const float scale = 1.0f / (1.0f - p);
    const std::uint64_t effective_seed = deriveDropoutMaskSeed(seed, mask_stream_id);

    std::uint8_t* mask = nullptr;
    cudaMallocOrThrow(reinterpret_cast<void**>(&mask), count * sizeof(std::uint8_t), "dropout_mask");
    if (!x.data || !result.data || !mask) {
        throw std::runtime_error("autograd::dropout: null buffer(s) before kernel launch");
    }

    kernel_generate_dropout_mask<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        mask, count, p, effective_seed);
    trackKernelLaunch("kernel_generate_dropout_mask", stream);
    throwIfCudaFailed(cudaGetLastError(), "autograd::dropout: kernel_generate_dropout_mask launch failed");

    kernel_dropout_forward<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        x.data, mask, result.data, scale, count);
    trackKernelLaunch("kernel_dropout_forward", stream);
    throwIfCudaFailed(cudaGetLastError(), "autograd::dropout: kernel_dropout_forward launch failed");

    if (x.requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<DropoutGradFn>();
        grad_fn->capture_input(const_cast<Tensor&>(x), stream);
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
