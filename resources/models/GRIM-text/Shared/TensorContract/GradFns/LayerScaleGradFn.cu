//======================================================//
//  LayerScaleGradFn.cu
//  Learnable scalar multiply forward + autograd backward.
//======================================================//

#include "LayerScaleGradFn.hpp"
#include "../TensorContract_GPU.hpp"
#include "../../CudaAllocUtils.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <stdexcept>

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

__global__ void kernel_dot_accumulate_scalar(float* dst, const float* a, const float* b, size_t count) {
    __shared__ float sdata[256];
    size_t idx = threadIdx.x;
    float sum = 0.0f;
    for (size_t i = idx; i < count; i += blockDim.x) {
        sum += a[i] * b[i];
    }
    sdata[idx] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (idx < static_cast<size_t>(s)) sdata[idx] += sdata[idx + s];
        __syncthreads();
    }
    if (idx == 0) atomicAdd(dst, sdata[0]);
}

}  // anonymous namespace

namespace GRIM {

using CudaAlloc::cudaMallocOrThrow;

namespace autograd {

LayerScaleGradFn::LayerScaleGradFn() {
    op_name = "layer_scale";
}

void LayerScaleGradFn::capture_inputs(Tensor& input, Tensor& scale_param, float cached_scale_value, cudaStream_t stream) {
    input_shape = input.shape;
    element_count = input.numel();
    scale_value = cached_scale_value;
    input_grad_fn = input.grad_fn;

    if (input.requires_grad) {
        input.ensure_grad();
        if (input.is_leaf) {
            input_grad = input.grad_data();
        } else {
            float* buffer = nullptr;
            cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), element_count * sizeof(float), "LayerScaleGradFn_input_grad");
            cudaMemsetAsync(buffer, 0, element_count * sizeof(float), stream);
            owned_input_grad = std::shared_ptr<float>(buffer, [](float* p) {
                queueForDeferredCleanup(p);
            });
            input_grad = owned_input_grad.get();
        }
    }

    if (scale_param.requires_grad) {
        scale_param.ensure_grad();
        scale_grad = scale_param.grad_data();
    }

    if (input.is_leaf) {
        input_data = input.data;
    } else {
        float* buffer = nullptr;
        cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), element_count * sizeof(float), "LayerScaleGradFn_input_data");
        cudaMemcpyAsync(buffer, input.data, element_count * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        owned_input_data = std::shared_ptr<float>(buffer, [](float* p) {
            queueForDeferredCleanup(p);
        });
        input_data = owned_input_data.get();
    }
}

void LayerScaleGradFn::apply(const Tensor& grad_output, cudaStream_t stream) {
    if (applied) {
        AG_TRACE("[LayerScaleGradFn] apply() SKIPPED (already applied)\n");
        return;
    }
    applied = true;

    const size_t n = element_count;

    if (input_grad && grad_output.data) {
        kernel_accumulate_grad<<<gridForCount(n), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
            input_grad, grad_output.data, n, scale_value);
    }

    if (scale_grad && grad_output.data && input_data) {
        kernel_dot_accumulate_scalar<<<1, 256, 0, stream>>>(
            scale_grad, grad_output.data, input_data, n);
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

void LayerScaleGradFn::release_saved() {
    owned_input_grad.reset();
    owned_input_data.reset();
}

Tensor layer_scale(const Tensor& x, Tensor& scale_param, cudaStream_t stream) {
    if (!scale_param.data) {
        throw std::runtime_error("layer_scale: scale_param is NULL");
    }

    float scale_value = 1.0f;
    cudaMemcpyAsync(&scale_value, scale_param.data, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    const bool track_grad = x.requires_grad || scale_param.requires_grad;
    Tensor result = Tensor::empty(x.shape, track_grad, stream, "layer_scale_result");

    TensorContract::TensorView x_view(const_cast<float*>(x.data), x.shape);
    TensorContract::TensorView out_view(result.data, result.shape);
    TensorContract::scale(x_view, scale_value, out_view, stream);

    if (track_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<LayerScaleGradFn>();
        grad_fn->capture_inputs(const_cast<Tensor&>(x), scale_param, scale_value, stream);
        result.grad_fn = grad_fn;
    }

    return result;
}

}  // namespace autograd
}  // namespace GRIM
