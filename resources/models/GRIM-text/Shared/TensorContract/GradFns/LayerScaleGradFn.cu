//======================================================//
//  LayerScaleGradFn.cu
//  Learnable per-channel residual scaling forward + autograd backward.
//======================================================//

#include "LayerScaleGradFn.hpp"
#include "../TensorContract_GPU.hpp"
#include "../../CudaAllocUtils.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>

#ifndef TENSOR_VERBOSE_DEBUG
#define TENSOR_VERBOSE_DEBUG 0
#endif

#define AG_TRACE(...) do { if constexpr (GRIM::VerboseLogging::ENABLE_AUTOGRAD_TRACE_LOGS) { fprintf(stderr, __VA_ARGS__); fflush(stderr); } } while(0)

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

__global__ void kernel_layer_scale_forward(const float* __restrict__ input,
                                           const float* __restrict__ scale,
                                           float* __restrict__ output,
                                           int rows,
                                           int cols) {
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    if (idx < total) {
        const int col = static_cast<int>(idx % static_cast<size_t>(cols));
        output[idx] = input[idx] * scale[col];
    }
}

__global__ void kernel_layer_scale_backward_input(float* __restrict__ input_grad,
                                                  const float* __restrict__ grad_output,
                                                  const float* __restrict__ scale,
                                                  int rows,
                                                  int cols) {
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    if (idx < total) {
        const int col = static_cast<int>(idx % static_cast<size_t>(cols));
        input_grad[idx] += grad_output[idx] * scale[col];
    }
}

__global__ void kernel_layer_scale_backward_scale(const float* __restrict__ grad_output,
                                                  const float* __restrict__ input,
                                                  float* __restrict__ scale_grad,
                                                  int rows,
                                                  int cols) {
    __shared__ float sdata[256];
    const int col = blockIdx.x;
    const int idx = threadIdx.x;
    if (col >= cols) {
        return;
    }

    float sum = 0.0f;
    for (int row = idx; row < rows; row += blockDim.x) {
        const size_t offset = static_cast<size_t>(row) * static_cast<size_t>(cols) + static_cast<size_t>(col);
        sum += grad_output[offset] * input[offset];
    }
    // Local parameter GradFns sum reductions. CrossEntropy/root backward already
    // injects mean-reduction and accumulation scaling into grad_output, so dividing
    // by rows here would double-average LayerScale and under-scale gamma updates.
    sdata[idx] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (idx < static_cast<size_t>(s)) sdata[idx] += sdata[idx + s];
        __syncthreads();
    }
    if (idx == 0) atomicAdd(&scale_grad[col], sdata[0]);
}

inline void checkCudaLaunch(const char* kernel_name) {
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(kernel_name) + " launch failed: " + cudaGetErrorString(err));
    }
}

}  // anonymous namespace

namespace GRIM {

using CudaAlloc::cudaMallocOrThrow;

namespace autograd {

LayerScaleGradFn::LayerScaleGradFn() {
    op_name = "layer_scale";
}

void LayerScaleGradFn::capture_inputs(Tensor& input, Tensor& scale_param, cudaStream_t stream) {
    element_count = input.numel();

    const auto input_dims = input.shape.as_2d();
    rows = input_dims.rows;
    cols = input_dims.cols;

    if (input.requires_grad) {
        input_gradient = capture_input_gradient(
            input, stream, "LayerScaleGradFn::capture_inputs input");
    }

    if (scale_param.requires_grad) {
        scale_gradient = capture_input_gradient(
            scale_param, stream, "LayerScaleGradFn::capture_inputs scale");
    }

    float* scale_buffer = nullptr;
    cudaMallocOrThrow(reinterpret_cast<void**>(&scale_buffer), scale_param.numel() * sizeof(float), "LayerScaleGradFn_scale_data");
    cudaMemcpyAsync(scale_buffer, scale_param.data, scale_param.numel() * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    owned_scale_data = std::shared_ptr<float>(scale_buffer, [](float* p) {
        queueForDeferredCleanup(p);
    });
    scale_data = owned_scale_data.get();

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

void LayerScaleGradFn::apply_impl(const Tensor& grad_output,
                                  cudaStream_t stream,
                                  const Batching::BatchPayload* backward_payload,
                                  const Batching::BatchDeviceBindings* backward_bindings) {
    setCurrentGradFnOp("layer_scale", this);
    if (applied) {
        AG_TRACE("[LayerScaleGradFn] apply() SKIPPED (already applied)\n");
        return;
    }
    applied = true;

    const size_t n = element_count;
    if (!grad_output.data) {
        throw std::runtime_error("LayerScaleGradFn::apply: grad_output.data is NULL");
    }
    if (grad_output.numel() != n) {
        throw std::runtime_error("LayerScaleGradFn::apply: grad_output element count mismatch. expected=" +
                                 std::to_string(n) + " got=" + std::to_string(grad_output.numel()));
    }
    if (!scale_data) {
        throw std::runtime_error("LayerScaleGradFn::apply: saved scale_data is NULL");
    }

    if (input_gradient) {
        kernel_layer_scale_backward_input<<<gridForCount(n), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
            input_gradient->data, grad_output.data, scale_data, rows, cols);
        checkCudaLaunch("kernel_layer_scale_backward_input");
    }

    if (scale_gradient) {
        if (!input_data) {
            throw std::runtime_error("LayerScaleGradFn::apply: saved input_data is NULL while scale requires grad");
        }
        if (cols > getMaxGridBlocks1D()) {
            throw std::runtime_error("LayerScaleGradFn::apply: cols exceeds 1D grid limit for scale gradient");
        }
        kernel_layer_scale_backward_scale<<<cols, 256, 0, stream>>>(
            grad_output.data, input_data, scale_gradient->data, rows, cols);
        checkCudaLaunch("kernel_layer_scale_backward_scale");
    }

    if (input_gradient) {
        propagate_input_gradient(
            input_gradient, stream, backward_payload, backward_bindings,
            "LayerScaleGradFn::apply input");
    }

    if (scale_gradient) {
        propagate_input_gradient(
            scale_gradient, stream, backward_payload, backward_bindings,
            "LayerScaleGradFn::apply scale");
    }
}

void LayerScaleGradFn::release_saved() {
    GradFn::release_saved();
    owned_input_data.reset();
    owned_scale_data.reset();
    input_data = nullptr;
    scale_data = nullptr;
    input_gradient.reset();
    scale_gradient.reset();
}

Tensor layer_scale(const Tensor& x, Tensor& scale_param, cudaStream_t stream) {
    if (stream == nullptr || stream == 0) {
        throw std::runtime_error("layer_scale: stream is NULL - caller MUST provide valid stream");
    }
    if (!x.data) {
        throw std::runtime_error("layer_scale: input x.data is NULL");
    }
    if (!scale_param.data) {
        throw std::runtime_error("layer_scale: scale_param is NULL");
    }

    x.shape.require("layer_scale input");
    scale_param.shape.require("layer_scale scale_param");
    if (!x.shape.is_2d_layout()) {
        throw std::runtime_error("layer_scale: input must be a 2D [tokens,d_model] tensor");
    }
    if (!scale_param.shape.is_2d_layout()) {
        throw std::runtime_error("layer_scale: scale_param must be a 2D [1,d_model] tensor");
    }

    const auto x_dims = x.shape.as_2d();
    const auto scale_dims = scale_param.shape.as_2d();
    if (scale_dims.rows != 1 || scale_dims.cols != x_dims.cols) {
        throw std::runtime_error("layer_scale: standard LayerScale requires per-channel scale_param shape [1,d_model]. input shape=[" +
                                 std::to_string(x_dims.rows) + "," + std::to_string(x_dims.cols) +
                                 "] scale shape=[" + std::to_string(scale_dims.rows) + "," +
                                 std::to_string(scale_dims.cols) + "]");
    }

    const bool track_grad = x.requires_grad || scale_param.requires_grad;
    Tensor result = Tensor::empty(x.shape, track_grad, stream, "layer_scale_result");

    kernel_layer_scale_forward<<<gridForCount(x.numel()), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        x.data, scale_param.data, result.data, x_dims.rows, x_dims.cols);
    checkCudaLaunch("kernel_layer_scale_forward");

    if (track_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<LayerScaleGradFn>();
        grad_fn->capture_inputs(const_cast<Tensor&>(x), scale_param, stream);
        result.grad_fn = grad_fn;
    }

    return result;
}

}  // namespace autograd
}  // namespace GRIM
