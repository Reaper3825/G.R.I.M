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
//    grad_bias[j] += sum_i(grad_output[i,j]) (block-reduction per feature)
//
//  The bias kernels and their launch wrappers used to live inside
//  TensorContract_GPU.cu; they are owned by the autograd layer (not
//  FFN-specific) and only consumed by this op, so they are pulled into
//  this TU under internal linkage to keep the .cu self-contained.
//======================================================//

#include "BiasAddGradFn.hpp"
#include "../AutogradQKVDiagnostics.hpp"
#include "../GradientAccumulation.hpp"
#include "../TensorContract_GPU.hpp"
#include "../../HyperParameters/HyperParameters_GPU.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

#define AG_TRACE(...) do { if constexpr (GRIM::VerboseLogging::ENABLE_AUTOGRAD_TRACE_LOGS) { fprintf(stderr, __VA_ARGS__); fflush(stderr); } } while(0)

namespace {

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

namespace autograd {

BiasAddGradFn::BiasAddGradFn() {
    op_name = "bias_add";
}

void BiasAddGradFn::capture_inputs(Tensor& input, Tensor& bias,
                                   int num_tokens, int num_features,
                                   cudaStream_t stream) {
    input_requires_grad = input.requires_grad;
    bias_requires_grad = bias.requires_grad;
    total_tokens = static_cast<size_t>(num_tokens);
    features = static_cast<size_t>(num_features);

    if (input_requires_grad) {
        input_gradient = capture_input_gradient(
            input, stream, "BiasAddGradFn::capture_inputs input");
        AG_TRACE("[BiasAddGradFn] Captured input gradient Tensor data: %p\n",
                 static_cast<void*>(input_gradient->data));
    }

    if (bias_requires_grad) {
        bias_gradient = capture_input_gradient(
            bias, stream, "BiasAddGradFn::capture_inputs bias");
        AG_TRACE("[BiasAddGradFn] Captured bias gradient Tensor data: %p\n",
                 static_cast<void*>(bias_gradient->data));
    }
}

void BiasAddGradFn::apply_impl(const Tensor& grad_output,
                               cudaStream_t stream,
                               const Batching::BatchPayload* backward_payload,
                               const Batching::BatchDeviceBindings* backward_bindings) {
    setCurrentGradFnOp("bias_add", this);

    if (applied) {
        return;
    }
    applied = true;

    if (!grad_output.data) {
        throw std::runtime_error("BiasAddGradFn::apply: grad_output.data is NULL - backward called with null gradient");
    }

    const size_t count = grad_output.numel();
    logGradFlowTensorStats("BiasAdd.apply grad_output", grad_output.data, count, stream);

    // Backward for input: grad_input = grad_output (pass-through, no shape change)
    if (input_requires_grad) {
        if (!input_gradient) {
            throw std::runtime_error(
                "BiasAddGradFn::apply: input gradient Tensor is NULL");
        }
        accumulate_grad(
            input_gradient->data,
            grad_output.data,
            count,
            1.0f,
            stream,
            "BiasAddGradFn::apply grad_input");
        logGradFlowTensorStats(
            "BiasAdd.apply grad_input_accum", input_gradient->data, count, stream);
    }

    // Backward for bias: grad_bias[j] += sum_i(grad_output[i,j])
    if (bias_requires_grad) {
        if (!bias_gradient) {
            throw std::runtime_error(
                "BiasAddGradFn::apply: bias gradient Tensor is NULL");
        }
        launchBiasBackward(grad_output.data, bias_gradient->data,
                           static_cast<int>(total_tokens), static_cast<int>(features), stream);
        logGradFlowTensorStats(
            "BiasAdd.apply grad_bias_accum", bias_gradient->data, features, stream);
    }

    if (input_requires_grad) {
        propagate_input_gradient(
            input_gradient,
            stream,
            backward_payload,
            backward_bindings,
            "BiasAddGradFn::apply input");
    }
    if (bias_requires_grad &&
        (bias_gradient->is_leaf || !input_requires_grad ||
         bias_gradient->grad_fn != input_gradient->grad_fn)) {
        propagate_input_gradient(
            bias_gradient,
            stream,
            backward_payload,
            backward_bindings,
            "BiasAddGradFn::apply bias");
    }
}

void BiasAddGradFn::release_saved() {
    GradFn::release_saved();
    input_gradient.reset();
    bias_gradient.reset();
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
