//======================================================//
//  AutogradLoss.cu
//  CUDA implementation of autograd-enabled cross-entropy loss
//======================================================//

#include "AutogradLoss.hpp"
#include "../../TensorContract/TensorContract_GPU.hpp"
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <memory>  // For std::make_shared

namespace GRIM {
namespace autograd {

//========================================================================
// CUDA Kernels
//========================================================================

/**
 * Cross-entropy forward kernel
 * Each block processes one token, uses shared memory for max reduction
 * Computes: loss[t] = -log(softmax(logits[t])[target[t]])
 */
__global__ void kernelCrossEntropyForward(
    const float* __restrict__ logits,      // [tokens, vocab_size]
    const int* __restrict__ targets,       // [tokens]
    const float* __restrict__ valid_mask,  // [tokens] or nullptr
    float* __restrict__ per_token_loss,    // [tokens]
    float* __restrict__ loss_sum,          // scalar (atomicAdd)
    int* __restrict__ valid_count,         // scalar (atomicAdd)
    int num_tokens,
    int vocab_size
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;
    
    const float mask = valid_mask ? valid_mask[token_idx] : 1.0f;
    
    if (mask < 0.5f) {
        per_token_loss[token_idx] = 0.0f;
        return;
    }
    
    const float* row = logits + static_cast<size_t>(token_idx) * vocab_size;
    const int target = targets[token_idx];
    
    // Step 1: Find max for numerical stability
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = -FLT_MAX;
    __syncthreads();
    
    float local_max = -FLT_MAX;
    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        local_max = fmaxf(local_max, row[v]);
    }
    
    // Warp reduction for max
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    
    if (threadIdx.x % warpSize == 0) {
        atomicMax(reinterpret_cast<int*>(&s_max), __float_as_int(local_max));
    }
    __syncthreads();
    
    const float max_val = s_max;
    
    // Step 2: Compute sum of exp(logit - max) and exp(target_logit - max)
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();
    
    float local_sum = 0.0f;
    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        local_sum += expf(row[v] - max_val);
    }
    
    // Warp reduction for sum
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    
    if (threadIdx.x % warpSize == 0) {
        atomicAdd(&s_sum, local_sum);
    }
    __syncthreads();
    
    // Step 3: Compute loss = -log(exp(target_logit - max) / sum_exp)
    //                     = -(target_logit - max - log(sum_exp))
    //                     = log(sum_exp) - (target_logit - max)
    if (threadIdx.x == 0) {
        const float log_sum = logf(s_sum + 1e-10f);
        const float target_logit = (target >= 0 && target < vocab_size) ? row[target] : 0.0f;
        const float loss = log_sum - (target_logit - max_val);
        
        per_token_loss[token_idx] = loss;
        atomicAdd(loss_sum, loss);
        atomicAdd(valid_count, 1);
    }
}

/**
 * Cross-entropy backward kernel
 * Computes: grad_logits = (softmax - one_hot(target)) / valid_count
 */
__global__ void kernelCrossEntropyBackward(
    const float* __restrict__ logits,      // [tokens, vocab_size]
    const int* __restrict__ targets,       // [tokens]
    const float* __restrict__ valid_mask,  // [tokens] or nullptr
    float* __restrict__ grad_logits,       // [tokens, vocab_size]
    int num_tokens,
    int vocab_size,
    float inv_valid_count                  // 1.0 / valid_count for mean reduction
) {
    // Grid-stride loop over all elements
    const size_t total = static_cast<size_t>(num_tokens) * vocab_size;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridDim.x * blockDim.x) {
        const int token_idx = idx / vocab_size;
        const int vocab_idx = idx % vocab_size;
        
        const float mask = valid_mask ? valid_mask[token_idx] : 1.0f;
        
        if (mask < 0.5f) {
            grad_logits[idx] = 0.0f;
            continue;
        }
        
        const float* row = logits + static_cast<size_t>(token_idx) * vocab_size;
        const int target = targets[token_idx];
        
        // Compute softmax probability for this vocab element
        // First find max for numerical stability
        float max_val = -FLT_MAX;
        for (int v = 0; v < vocab_size; ++v) {
            max_val = fmaxf(max_val, row[v]);
        }
        
        // Compute softmax
        float sum_exp = 0.0f;
        for (int v = 0; v < vocab_size; ++v) {
            sum_exp += expf(row[v] - max_val);
        }
        
        const float prob = expf(row[vocab_idx] - max_val) / (sum_exp + 1e-10f);
        const float one_hot = (vocab_idx == target) ? 1.0f : 0.0f;
        
        // Gradient: (softmax - one_hot) / valid_count
        grad_logits[idx] = (prob - one_hot) * inv_valid_count;
    }
}

/**
 * Optimized backward kernel - one block per token
 * Computes softmax + gradient in one pass with shared memory
 */
__global__ void kernelCrossEntropyBackwardOptimized(
    const float* __restrict__ logits,
    const int* __restrict__ targets,
    const float* __restrict__ valid_mask,
    float* __restrict__ grad_logits,
    int num_tokens,
    int vocab_size,
    float inv_valid_count
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;
    
    const float mask = valid_mask ? valid_mask[token_idx] : 1.0f;
    const float* row = logits + static_cast<size_t>(token_idx) * vocab_size;
    float* grad_row = grad_logits + static_cast<size_t>(token_idx) * vocab_size;
    const int target = targets[token_idx];
    
    if (mask < 0.5f) {
        // Zero out gradients for padding tokens
        for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
            grad_row[v] = 0.0f;
        }
        return;
    }
    
    // Step 1: Find max
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = -FLT_MAX;
    __syncthreads();
    
    float local_max = -FLT_MAX;
    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        local_max = fmaxf(local_max, row[v]);
    }
    
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    if (threadIdx.x % warpSize == 0) {
        atomicMax(reinterpret_cast<int*>(&s_max), __float_as_int(local_max));
    }
    __syncthreads();
    const float max_val = s_max;
    
    // Step 2: Compute sum_exp
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();
    
    float local_sum = 0.0f;
    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        local_sum += expf(row[v] - max_val);
    }
    
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if (threadIdx.x % warpSize == 0) {
        atomicAdd(&s_sum, local_sum);
    }
    __syncthreads();
    
    const float inv_sum = 1.0f / (s_sum + 1e-10f);
    
    // Step 3: Compute gradient = softmax - one_hot
    // Issue #46 FIX: REMOVED inv_valid_count scaling!
    // Standard CE gradient is just (p - y), no per-token normalization.
    // The 1/N scaling was making gradients 4000x too small (~5e-9 instead of ~2e-5).
    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        const float prob = expf(row[v] - max_val) * inv_sum;
        const float one_hot = (v == target) ? 1.0f : 0.0f;
        grad_row[v] = prob - one_hot;  // NO scaling!
    }
}

//========================================================================
// Launch Functions
//========================================================================

void launchCrossEntropyForward(
    const float* logits,
    const int* targets,
    const float* valid_mask,
    float* per_token_loss,
    float* loss_sum,
    int* valid_count,
    int num_tokens,
    int vocab_size,
    cudaStream_t stream
) {
    // Zero outputs
    cudaMemsetAsync(loss_sum, 0, sizeof(float), stream);
    cudaMemsetAsync(valid_count, 0, sizeof(int), stream);
    
    // One block per token, 256 threads per block
    const int block_size = 256;
    kernelCrossEntropyForward<<<num_tokens, block_size, 0, stream>>>(
        logits, targets, valid_mask,
        per_token_loss, loss_sum, valid_count,
        num_tokens, vocab_size
    );
}

void launchCrossEntropyBackward(
    const float* logits,
    const int* targets,
    const float* valid_mask,
    float* grad_logits,
    int num_tokens,
    int vocab_size,
    int valid_count,
    cudaStream_t stream
) {
    // Issue #46 FIX: Pass 1.0f instead of 1/valid_count
    // Standard CE gradient is (softmax - one_hot), no per-token normalization.
    // The loss is already averaged in forward; backward just needs raw gradient.
    const float scale = 1.0f;  // Was: 1.0f / valid_count (WRONG!)
    
    // One block per token for optimized kernel
    const int block_size = 256;
    kernelCrossEntropyBackwardOptimized<<<num_tokens, block_size, 0, stream>>>(
        logits, targets, valid_mask, grad_logits,
        num_tokens, vocab_size, scale
    );
}

//========================================================================
// CrossEntropyGradFn - Autograd node
//========================================================================

/**
 * GradFn for cross-entropy loss
 * Writes gradient directly to the logits tensor's grad field
 */
struct CrossEntropyLossGradFn : public GradFn {
    // Forward tensors (kept alive by shared_ptr)
    std::shared_ptr<float> logits_data;  // Owns GPU memory copy
    size_t logits_size;
    
    // Forward pass metadata
    const int* targets;          // Device pointer (not owned - must stay valid!)
    const float* valid_mask;     // Device pointer (not owned)
    int num_tokens;
    int vocab_size;
    int valid_count;
    
    // Pointer to logits tensor so we can write to its grad
    Tensor* logits_tensor_ptr;
    
    __host__ CrossEntropyLossGradFn(
        float* logits, size_t logits_numel,
        const int* targets_, const float* valid_mask_,
        int num_tokens_, int vocab_size_, int valid_count_,
        Tensor* logits_tensor,
        cudaStream_t stream_
    ) : logits_data(nullptr), logits_size(logits_numel),
        targets(targets_), valid_mask(valid_mask_),
        num_tokens(num_tokens_), vocab_size(vocab_size_),
        valid_count(valid_count_), logits_tensor_ptr(logits_tensor)
    {
        op_name = "cross_entropy_loss";
        
        // Copy logits to owned buffer (so they persist for backward)
        float* buffer = nullptr;
        cudaMalloc(&buffer, logits_numel * sizeof(float));
        cudaMemcpyAsync(buffer, logits, logits_numel * sizeof(float), 
                        cudaMemcpyDeviceToDevice, stream_);
        logits_data = std::shared_ptr<float>(buffer, [](float* p) { cudaFree(p); });
    }
    
    __host__ void apply(const Tensor& grad_output, cudaStream_t stream) override {
        // grad_output is scalar (1 element) - the loss gradient (usually 1.0)
        
        // Ensure logits tensor has gradient buffer
        if (logits_tensor_ptr) {
            logits_tensor_ptr->ensure_grad();
            
            // Compute gradient: (softmax - one_hot) / valid_count
            // This writes to logits_tensor_ptr->grad
            launchCrossEntropyBackward(
                logits_data.get(),
                targets,
                valid_mask,
                logits_tensor_ptr->grad,  // Write directly to tensor's grad
                num_tokens,
                vocab_size,
                valid_count,
                stream
            );
        }
    }
    
    __host__ void release_saved() override {
        logits_data.reset();  // Free the copied logits
        GradFn::release_saved();
    }
};

//========================================================================
// Main API Function (Host-only)
//========================================================================

__host__ Tensor cross_entropy_loss(
    Tensor& logits,
    const int* targets,
    const float* valid_mask,
    int num_tokens,
    int vocab_size,
    cudaStream_t stream
) {
    // Allocate temporary buffers
    float* per_token_loss = nullptr;
    float* d_loss_sum = nullptr;
    int* d_valid_count = nullptr;
    
    cudaMalloc(&per_token_loss, num_tokens * sizeof(float));
    cudaMalloc(&d_loss_sum, sizeof(float));
    cudaMalloc(&d_valid_count, sizeof(int));
    
    // Compute forward pass
    launchCrossEntropyForward(
        logits.data,
        targets,
        valid_mask,
        per_token_loss,
        d_loss_sum,
        d_valid_count,
        num_tokens,
        vocab_size,
        stream
    );
    
    // Copy results to host
    float h_loss_sum = 0.0f;
    int h_valid_count = 0;
    cudaMemcpyAsync(&h_loss_sum, d_loss_sum, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_valid_count, d_valid_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // Compute mean loss
    const float mean_loss = (h_valid_count > 0) ? (h_loss_sum / h_valid_count) : 0.0f;
    
    // Create scalar loss tensor on GPU
    float* d_loss = nullptr;
    cudaMalloc(&d_loss, sizeof(float));
    cudaMemcpyAsync(d_loss, &mean_loss, sizeof(float), cudaMemcpyHostToDevice, stream);
    
    Tensor loss;
    loss.data = d_loss;
    loss.owns_data = true;  // Loss tensor owns its GPU memory
    loss.shape = TensorContract::TensorShape::make_BSM(1, 1);  // Scalar as [1, 1]
    loss.is_leaf = false;
    loss.requires_grad = logits.requires_grad;
    loss.stream = stream;
    
    // Attach grad_fn if logits requires grad
    // NOTE: We use 'new' and the Tensor destructor / release_saved() handles cleanup
    if (logits.requires_grad) {
        auto* grad_fn = new CrossEntropyLossGradFn(
            logits.data, logits.numel(),
            targets, valid_mask,
            num_tokens, vocab_size, h_valid_count,
            &logits,  // Pass pointer to logits tensor for grad writing
            stream
        );
        loss.grad_fn = grad_fn;  // Tensor takes ownership
    }
    
    // Cleanup temporary buffers
    cudaFree(per_token_loss);
    cudaFree(d_loss_sum);
    cudaFree(d_valid_count);
    
    return loss;
}

}  // namespace autograd
}  // namespace GRIM
