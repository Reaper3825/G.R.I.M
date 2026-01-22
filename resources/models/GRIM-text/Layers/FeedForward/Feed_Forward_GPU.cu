//======================================================//
//  Feed_Forward_GPU.cu
//  GPU-accelerated FeedForward layer using autograd
//
//  Two-layer MLP with GELU activation:
//    hidden = GELU(input @ W1 + b1)
//    output = hidden @ W2 + b2
//
//  NOTE:
//  - Autograd covers matmul + GELU.
//  - Bias add is done by custom kernels; bias gradients must be computed
//    via launchFFNBiasBackward (outside this forward graph), unless you
//    later add an autograd broadcast_add op.
//======================================================//

#include "Feed_Forward_GPU.hpp"
#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"
#include "../../Shared/StreamController/StreamController_GPU.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <algorithm>
#include <cstdio>

namespace GRIM {

//======================================================//
//  FFN-specific CUDA Kernels (Bias operations)
//======================================================//

namespace {

__global__ void ffnBiasAddKernel(float* __restrict__ tensor,
                                 const float* __restrict__ bias,
                                 int total_elements,
                                 int features) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        const int feature_idx = idx % features;
        tensor[idx] += bias[feature_idx];
    }
}

// Bias backward: one block per feature; reduction within block.
// No atomic needed because each feature is handled by exactly one block.
__global__ void ffnBiasBackwardKernel(const float* __restrict__ grad_output,
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

    // Reduction in shared memory (assumes blockDim.x is power-of-two)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        // Accumulate into grad_bias (caller may be doing grad accumulation)
        grad_bias[feature_idx] += sdata[0];
    }
}

} // anonymous namespace

//======================================================//
//  FFN Kernel Launch Functions
//======================================================//

void launchFFNBiasAdd(float* tensor, const float* bias,
                      int total_tokens, int features,
                      cudaStream_t stream) {
    // Rule 20: Crash on invalid input
    if (!tensor) {
        throw std::runtime_error("launchFFNBiasAdd: tensor is NULL at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    if (!bias) {
        throw std::runtime_error("launchFFNBiasAdd: bias is NULL at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    if (total_tokens <= 0 || features <= 0) {
        throw std::runtime_error("launchFFNBiasAdd: invalid dimensions (" + std::to_string(total_tokens) + ", " + std::to_string(features) + ")");
    }

    const int total_elements = total_tokens * features;
    constexpr int kBlockSize = HyperParameters::CUDA_BLOCK_SIZE_STANDARD;
    const int grid = (total_elements + kBlockSize - 1) / kBlockSize;

    ffnBiasAddKernel<<<grid, kBlockSize, 0, stream>>>(tensor, bias, total_elements, features);
}

void launchFFNBiasBackward(const float* grad_output, float* grad_bias,
                           int total_tokens, int features,
                           cudaStream_t stream) {
    // Rule 20: Crash on invalid input
    if (!grad_output) {
        throw std::runtime_error("launchFFNBiasBackward: grad_output is NULL at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    if (!grad_bias) {
        throw std::runtime_error("launchFFNBiasBackward: grad_bias is NULL at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    if (total_tokens <= 0 || features <= 0) {
        throw std::runtime_error("launchFFNBiasBackward: invalid dimensions (" + std::to_string(total_tokens) + ", " + std::to_string(features) + ")");
    }

    constexpr int kBlockSize = HyperParameters::CUDA_BLOCK_SIZE_STANDARD;
    const int shared_bytes = kBlockSize * sizeof(float);

    ffnBiasBackwardKernel<<<features, kBlockSize, shared_bytes, stream>>>(
        grad_output, grad_bias, total_tokens, features);
}

//======================================================//
//  FeedForwardLayer Implementation
//======================================================//

FeedForwardLayer::FeedForwardLayer(const FeedForwardConfig& config)
    : config_(config) {
    if (!config_.cublas_handle) {
        throw std::runtime_error("FeedForwardLayer: cublas_handle is NULL - caller MUST provide handle");
    }
    // Set cuBLAS handle for autograd
    autograd::set_autograd_cublas_handle(config_.cublas_handle);
}

FeedForwardLayer::~FeedForwardLayer() {
    // Tensors clean up via RAII
}

FeedForwardLayer::FeedForwardLayer(FeedForwardLayer&& other) noexcept
    : config_(other.config_)
    , W1_(std::move(other.W1_))
    , b1_(std::move(other.b1_))
    , W2_(std::move(other.W2_))
    , b2_(std::move(other.b2_)) {
    other.config_.cublas_handle = nullptr;
}

FeedForwardLayer& FeedForwardLayer::operator=(FeedForwardLayer&& other) noexcept {
    if (this != &other) {
        config_ = other.config_;
        W1_ = std::move(other.W1_);
        b1_ = std::move(other.b1_);
        W2_ = std::move(other.W2_);
        b2_ = std::move(other.b2_);
        other.config_.cublas_handle = nullptr;
    }
    return *this;
}

void FeedForwardLayer::setConfig(const FeedForwardConfig& cfg) {
    config_ = cfg;
    if (config_.cublas_handle) {
        autograd::set_autograd_cublas_handle(config_.cublas_handle);
    }
}

void FeedForwardLayer::ensureWeightStorage() {
    if (config_.d_model <= 0 || config_.d_ff <= 0) {
        throw std::runtime_error("FeedForwardLayer: d_model and d_ff must be positive");
    }
    if (!config_.stream) {
        throw std::runtime_error("FeedForwardLayer: stream is NULL");
    }
    StreamController::fatalIfDefaultStream(config_.stream, "FeedForwardLayer::ensureWeightStorage");

    // W1: stored as [d_model, d_ff] for matmul: input [T, d_model] @ W1 [d_model, d_ff] = [T, d_ff]
    if (W1_.numel() == 0) {
        auto w1_shape = TensorContract::TensorShape::make_BSM(config_.d_model, config_.d_ff);
        W1_ = Tensor::xavier_uniform(w1_shape, true, config_.stream);
        W1_.name = "ffn.W1";
    }

    if (b1_.numel() == 0) {
        auto b1_shape = TensorContract::TensorShape::make_BSM(1, config_.d_ff);
        b1_ = Tensor::zeros(b1_shape, true, config_.stream);
        b1_.name = "ffn.b1";
    }

    // W2: stored as [d_ff, d_model] for matmul: hidden [T, d_ff] @ W2 [d_ff, d_model] = [T, d_model]
    if (W2_.numel() == 0) {
        auto w2_shape = TensorContract::TensorShape::make_BSM(config_.d_ff, config_.d_model);
        W2_ = Tensor::xavier_uniform(w2_shape, true, config_.stream);
        W2_.name = "ffn.W2";
    }

    if (b2_.numel() == 0) {
        auto b2_shape = TensorContract::TensorShape::make_BSM(1, config_.d_model);
        b2_ = Tensor::zeros(b2_shape, true, config_.stream);
        b2_.name = "ffn.b2";
    }

    // AUTOGRAD MIGRATION: Allocate gradient buffers for all trainable tensors
    W1_.ensure_grad();
    b1_.ensure_grad();
    W2_.ensure_grad();
    b2_.ensure_grad();
}

void FeedForwardLayer::useExternalWeights(Tensor& w1, Tensor& b1, Tensor& w2, Tensor& b2) {
    const int d_model = config_.d_model;
    const int d_ff = config_.d_ff;

    // Strong shape enforcement (numel alone is not enough; orientation matters)
    // Expected internal layout:
    //   W1: [d_model, d_ff]
    //   b1: [1, d_ff] (optional)
    //   W2: [d_ff, d_model]
    //   b2: [1, d_model] (optional)
    {
        const auto& s1 = w1.shape.as_2d();
        if (s1.rows != d_model || s1.cols != d_ff) {
            throw std::invalid_argument(
                "FeedForwardLayer::useExternalWeights: W1 shape mismatch. Expected [" +
                std::to_string(d_model) + "," + std::to_string(d_ff) + "], got [" +
                std::to_string(s1.rows) + "," + std::to_string(s1.cols) + "]"
            );
        }
    }
    {
        const auto& s2 = w2.shape.as_2d();
        if (s2.rows != d_ff || s2.cols != d_model) {
            throw std::invalid_argument(
                "FeedForwardLayer::useExternalWeights: W2 shape mismatch. Expected [" +
                std::to_string(d_ff) + "," + std::to_string(d_model) + "], got [" +
                std::to_string(s2.rows) + "," + std::to_string(s2.cols) + "]"
            );
        }
    }

    // Biases are optional but if provided, enforce expected shapes.
    if (b1.data) {
        const auto& sb1 = b1.shape.as_2d();
        if (sb1.rows != 1 || sb1.cols != d_ff) {
            throw std::invalid_argument(
                "FeedForwardLayer::useExternalWeights: b1 shape mismatch. Expected [1," +
                std::to_string(d_ff) + "], got [" +
                std::to_string(sb1.rows) + "," + std::to_string(sb1.cols) + "]"
            );
        }
    }

    if (b2.data) {
        const auto& sb2 = b2.shape.as_2d();
        if (sb2.rows != 1 || sb2.cols != d_model) {
            throw std::invalid_argument(
                "FeedForwardLayer::useExternalWeights: b2 shape mismatch. Expected [1," +
                std::to_string(d_model) + "], got [" +
                std::to_string(sb2.rows) + "," + std::to_string(sb2.cols) + "]"
            );
        }
    }

    // Create view Tensors that reference external buffers
    W1_ = Tensor::from_ptr(w1.data, w1.shape, false, true);
    W1_.grad = w1.grad;
    W1_.owns_data = false;
    W1_.owns_grad = false;
    W1_.name = "ffn.W1_ext";

    if (b1.data) {
        b1_ = Tensor::from_ptr(b1.data, b1.shape, false, true);
        b1_.grad = b1.grad;
        b1_.owns_data = false;
        b1_.owns_grad = false;
        b1_.name = "ffn.b1_ext";
    }

    W2_ = Tensor::from_ptr(w2.data, w2.shape, false, true);
    W2_.grad = w2.grad;
    W2_.owns_data = false;
    W2_.owns_grad = false;
    W2_.name = "ffn.W2_ext";

    if (b2.data) {
        b2_ = Tensor::from_ptr(b2.data, b2.shape, false, true);
        b2_.grad = b2.grad;
        b2_.owns_data = false;
        b2_.owns_grad = false;
        b2_.name = "ffn.b2_ext";
    }

    std::fprintf(stderr, "[FeedForwardLayer] Using external weights: W1=[%zu], W2=[%zu]\n",
                 W1_.numel(), W2_.numel());
}

//======================================================//
//  Forward Pass - Autograd
//======================================================//

Tensor FeedForwardLayer::forward(const Tensor& input, float* cache_pre_gelu) {
    // Rule 20: Crash on invalid input
    if (!input.data) {
        throw std::runtime_error("FeedForwardLayer::forward: input.data is NULL");
    }
    if (!W1_.data || !W2_.data) {
        throw std::runtime_error("FeedForwardLayer::forward: weights not initialized - call ensureWeightStorage() first");
    }
    if (!config_.stream) {
        throw std::runtime_error("FeedForwardLayer::forward: stream is NULL");
    }

    const cudaStream_t stream = config_.stream;

    // Extract dimensions from input shape
    const auto& in_shape = input.shape.as_2d();
    const int total_tokens = in_shape.rows;
    const int d_model = in_shape.cols;

    if (d_model != config_.d_model) {
        throw std::runtime_error("FeedForwardLayer::forward: input d_model mismatch (" +
                                 std::to_string(d_model) + " vs " + std::to_string(config_.d_model) + ")");
    }

    // Sanity-check weight shapes (avoid silent wrong matmuls)
    {
        const auto& s1 = W1_.shape.as_2d();
        if (s1.rows != config_.d_model || s1.cols != config_.d_ff) {
            throw std::runtime_error("FeedForwardLayer::forward: W1 shape mismatch. Expected [" +
                                     std::to_string(config_.d_model) + "," + std::to_string(config_.d_ff) +
                                     "], got [" + std::to_string(s1.rows) + "," + std::to_string(s1.cols) + "]");
        }
    }
    {
        const auto& s2 = W2_.shape.as_2d();
        if (s2.rows != config_.d_ff || s2.cols != config_.d_model) {
            throw std::runtime_error("FeedForwardLayer::forward: W2 shape mismatch. Expected [" +
                                     std::to_string(config_.d_ff) + "," + std::to_string(config_.d_model) +
                                     "], got [" + std::to_string(s2.rows) + "," + std::to_string(s2.cols) + "]");
        }
    }

    // Set cuBLAS handle for autograd ops
    autograd::set_autograd_cublas_handle(config_.cublas_handle);

    //--------------------------------------------------
    // Layer 1: hidden = GELU(input @ W1 + b1)
    //--------------------------------------------------

    // matmul: input [tokens, d_model] @ W1 [d_model, d_ff] = pre_gelu [tokens, d_ff]
    Tensor pre_gelu = autograd::matmul(input, W1_, stream,
                                       input.data,  // cache input for W1 grad
                                       nullptr);    // W1 persists, no cache needed

    // Add bias b1 (broadcasted)
    launchFFNBiasAdd(pre_gelu.data, b1_.data, total_tokens, config_.d_ff, stream);

    // HYBRID FIX: Cache pre_gelu (after bias, before GELU) for legacy backward
    if (cache_pre_gelu) {
        cudaMemcpyAsync(cache_pre_gelu, pre_gelu.data,
                        static_cast<size_t>(total_tokens) * config_.d_ff * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    }

    // GELU activation
    Tensor post_gelu = autograd::gelu(pre_gelu, stream, pre_gelu.data);

    //--------------------------------------------------
    // Layer 2: output = post_gelu @ W2 + b2
    //--------------------------------------------------

    // matmul: post_gelu [tokens, d_ff] @ W2 [d_ff, d_model] = output [tokens, d_model]
    Tensor output = autograd::matmul(post_gelu, W2_, stream,
                                     post_gelu.data,  // cache post_gelu for W2 grad
                                     nullptr);        // W2 persists

    // Add bias b2 (broadcasted)
    launchFFNBiasAdd(output.data, b2_.data, total_tokens, config_.d_model, stream);

    return output;
}

} // namespace GRIM
