//======================================================//
//  BiasAddGradFn.cu
//  Broadcast bias-add forward + autograd backward.
//
//  Forward: out = input; out[i, j] += bias[j]
//    - implemented as an in-place memcpy (input → out) followed by
//      biasAddKernel that broadcasts bias[j] across all tokens i.
//
//  Backward:
//    grad_input = grad_output               (pass-through accumulate)
//    grad_bias[j] = sum_i(grad_output[i,j]) (block-reduction per feature)
//
//  The bias kernels and their launch wrappers used to live inside
//  TensorContract_GPU.cu; they are owned by the autograd layer (not
//  FFN-specific) and only consumed by this op, so they are pulled into
//  this TU under internal linkage to keep the .cu self-contained.
//======================================================//

#include "BiasAddGradFn.hpp"
#include "TensorContract_GPU.hpp"
#include "HyperParameters/HyperParameters_GPU.hpp"
#include "../CudaAllocUtils.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

#define AG_TRACE(...) do { if (g_autograd_verbose) { fprintf(stderr, __VA_ARGS__); fflush(stderr); } } while(0)

namespace {

constexpr int AUTOGRAD_BLOCK_SIZE = 256;
constexpr int kMaxGridBlocks1DFallback = 65534;

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

constexpr int kMaxGridDimY = 65535;

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

// Bias add forward (broadcast bias[features] across [total_elements] flat tokens)
__global__ void biasAddKernel(float* __restrict__ tensor,
                              const float* __restrict__ bias,
                              int total_elements,
                              int features) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        const int feature_idx = idx % features;
        tensor[idx] += bias[feature_idx];
    }
}

// Bias backward: grad_bias[j] += sum_i(grad_output[i, j])
//   one block per feature, reduce over tokens via shared-memory tree.
__global__ void biasBackwardKernel(const float* __restrict__ grad_output,
                                   float* __restrict__ grad_bias,
                                   int total_tokens,
                                   int features) {
    extern __shared__ float sdata[];

    const int feature_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (feature_idx >= features) return;

    float local_sum = 0.0f;
    for (int t = tid; t < total_tokens; t += blockDim.x) {
        local_sum += grad_output[t * features + feature_idx];
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        grad_bias[feature_idx] += sdata[0];
    }
}

void launchBiasAdd(float* tensor, const float* bias,
                   int total_tokens, int features,
                   cudaStream_t stream) {
    if (!tensor) throw std::runtime_error("launchBiasAdd: tensor is NULL at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    if (!bias) throw std::runtime_error("launchBiasAdd: bias is NULL at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    if (total_tokens <= 0 || features <= 0) throw std::runtime_error("launchBiasAdd: invalid dimensions (" + std::to_string(total_tokens) + ", " + std::to_string(features) + ")");

    const int total_elements = total_tokens * features;
    constexpr int kBlockSize = GRIM::HyperParameters::CUDA_BLOCK_SIZE_STANDARD;
    const int grid = (total_elements + kBlockSize - 1) / kBlockSize;
    biasAddKernel<<<grid, kBlockSize, 0, stream>>>(tensor, bias, total_elements, features);
}

void launchBiasBackward(const float* grad_output, float* grad_bias,
                        int total_tokens, int features,
                        cudaStream_t stream) {
    if (!grad_output) throw std::runtime_error("launchBiasBackward: grad_output is NULL at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    if (!grad_bias) throw std::runtime_error("launchBiasBackward: grad_bias is NULL at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    if (total_tokens <= 0 || features <= 0) throw std::runtime_error("launchBiasBackward: invalid dimensions (" + std::to_string(total_tokens) + ", " + std::to_string(features) + ")");

    constexpr int kBlockSize = GRIM::HyperParameters::CUDA_BLOCK_SIZE_STANDARD;
    const int shared_bytes = kBlockSize * sizeof(float);
    biasBackwardKernel<<<features, kBlockSize, shared_bytes, stream>>>(
        grad_output, grad_bias, total_tokens, features);
}

}  // anonymous namespace

namespace GRIM {

using CudaAlloc::cudaMallocOrThrow;

namespace autograd {

BiasAddGradFn::BiasAddGradFn() {
    op_name = "bias_add";
}

void BiasAddGradFn::capture_inputs(Tensor& input, Tensor& bias,
                                   int num_tokens, int num_features,
                                   cudaStream_t stream) {
    input_requires_grad = input.requires_grad;
    bias_requires_grad = bias.requires_grad;
    input_shape = input.shape;
    bias_shape = bias.shape;
    total_tokens = static_cast<size_t>(num_tokens);
    features = static_cast<size_t>(num_features);

    input_grad_fn = input.grad_fn;

    if (input_requires_grad) {
        input.ensure_grad();
        if (input.is_leaf) {
            grad_input = input.grad_data();
            AG_TRACE("[BiasAddGradFn] Using persistent grad_input buffer (leaf): %p\n", (void*)grad_input);
        } else {
            const size_t input_numel = input.numel();
            float* buffer_input = nullptr;
            cudaMallocOrThrow(reinterpret_cast<void**>(&buffer_input), input_numel * sizeof(float), "BiasAddGradFn_grad_input");
            cudaMemsetAsync(buffer_input, 0, input_numel * sizeof(float), stream);
            owned_grad_input = std::shared_ptr<float>(buffer_input, [](float* p) {
                queueForDeferredCleanup(p);
            });
            grad_input = owned_grad_input.get();
            AG_TRACE("[BiasAddGradFn] Allocated owned grad_input buffer (non-leaf): %zu floats at %p\n", input_numel, (void*)grad_input);
        }
    }

    if (bias_requires_grad) {
        bias.ensure_grad();
        if (bias.is_leaf) {
            grad_bias = bias.grad_data();
            AG_TRACE("[BiasAddGradFn] Using persistent grad_bias buffer (leaf): %p\n", (void*)grad_bias);
        } else {
            float* buffer_bias = nullptr;
            cudaMallocOrThrow(reinterpret_cast<void**>(&buffer_bias), features * sizeof(float), "BiasAddGradFn_grad_bias");
            cudaMemsetAsync(buffer_bias, 0, features * sizeof(float), stream);
            owned_grad_bias = std::shared_ptr<float>(buffer_bias, [](float* p) {
                queueForDeferredCleanup(p);
            });
            grad_bias = owned_grad_bias.get();
            AG_TRACE("[BiasAddGradFn] Allocated owned grad_bias buffer (non-leaf): %zu floats at %p\n", features, (void*)grad_bias);
        }
    }
}

void BiasAddGradFn::apply(const Tensor& grad_output, cudaStream_t stream) {
    setCurrentGradFnOp("bias_add", this);

    if (applied) {
        return;
    }
    applied = true;

    if (!grad_output.data) {
        throw std::runtime_error("BiasAddGradFn::apply: grad_output.data is NULL - backward called with null gradient");
    }

    const size_t count = grad_output.numel();

    // Backward for input: grad_input = grad_output (pass-through, no shape change)
    if (input_requires_grad && grad_input) {
        kernel_accumulate_grad<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
            grad_input, grad_output.data, count, 1.0f);
    }

    // Backward for bias: grad_bias[j] = sum_i(grad_output[i,j])
    if (bias_requires_grad && grad_bias) {
        launchBiasBackward(grad_output.data, grad_bias,
                           static_cast<int>(total_tokens), static_cast<int>(features), stream);
    }

    // CONTINUE AUTOGRAD CHAIN for input
    if (input_requires_grad && input_grad_fn && input_grad_fn->op_name) {
        Tensor view;
        view.data = grad_output.data;  // ISSUE #58: Pass incoming gradient
        view.shape = input_shape;
        view.owns_data = false;
        view.stream = stream;
        input_grad_fn->apply(view, stream);
    }
}

void BiasAddGradFn::release_saved() {
    GradFn::release_saved();
    grad_input = nullptr;
    grad_bias = nullptr;
    input_grad_fn.reset();
}

Tensor broadcast_add(const Tensor& input, const Tensor& bias, cudaStream_t stream) {
    if (stream == nullptr || stream == 0) {
        throw std::runtime_error("autograd::broadcast_add: stream is NULL - caller MUST provide valid stream");
    }
    if (!input.data) {
        throw std::runtime_error("autograd::broadcast_add: input.data is NULL");
    }
    if (!bias.data) {
        throw std::runtime_error("autograd::broadcast_add: bias.data is NULL");
    }

    if (!input.shape.is_2d_layout()) {
        throw std::runtime_error("autograd::broadcast_add: input must have 2D flat layout (BSM)");
    }
    const int total_tokens = input.shape.flat.rows;
    const int features = input.shape.flat.cols;
    const int bias_size = static_cast<int>(bias.numel());

    if (features != bias_size) {
        throw std::runtime_error("autograd::broadcast_add: feature dimension mismatch. input features=" +
                                 std::to_string(features) + " bias size=" + std::to_string(bias_size));
    }

    Tensor result = Tensor::empty(input.shape, input.requires_grad || bias.requires_grad, stream, "broadcast_add_result");

    // Forward: copy input to output, then add bias in-place via broadcast
    const size_t total_bytes = static_cast<size_t>(total_tokens) * features * sizeof(float);
    cudaMemcpyAsync(result.data, input.data, total_bytes, cudaMemcpyDeviceToDevice, stream);

    launchBiasAdd(result.data, bias.data, total_tokens, features, stream);

    if (result.requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<BiasAddGradFn>();
        grad_fn->capture_inputs(const_cast<Tensor&>(input), const_cast<Tensor&>(bias),
                                total_tokens, features, stream);
        result.grad_fn = grad_fn;
    }

    AG_TRACE("[autograd::broadcast_add] input[%d,%d] + bias[%d] -> output[%d,%d] requires_grad=%d\n",
             total_tokens, features, bias_size, total_tokens, features, result.requires_grad);

    return result;
}

}  // namespace autograd
}  // namespace GRIM
