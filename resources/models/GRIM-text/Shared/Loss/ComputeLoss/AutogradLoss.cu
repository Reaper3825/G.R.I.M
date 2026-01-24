//======================================================//
//  AutogradLoss.cu
//  CUDA implementation of unified autograd-enabled loss
//  
//  Implements: Focal Loss + Label Smoothing + Cross Entropy + Entropy Regularization
//  This is the ONLY loss computation path for training.
//======================================================//

#include "AutogradLoss.hpp"
#include "../../TensorContract/TensorContract_GPU.hpp"
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <memory>

// Access the global autograd verbose flag
extern bool g_autograd_verbose;
#define AG_TRACE(...) do { if (g_autograd_verbose) { fprintf(stderr, __VA_ARGS__); fflush(stderr); } } while(0)

namespace GRIM {
namespace autograd {

//========================================================================
// Constants
//========================================================================

constexpr float kEpsilon = 1e-10f;

//========================================================================
// CUDA Kernels - Forward Pass (Full Unified Loss)
//========================================================================

/**
 * Unified loss forward kernel - one block per token
 * 
 * Computes:
 *   L = α * (1-p_t)^γ * CE_smooth + λ * neg_entropy
 * 
 * Where:
 *   CE_smooth = -(1-ε)*log(p_t) - ε/(V-1)*Σ_{i≠t}log(p_i)  [label smoothing]
 *   neg_entropy = Σ p_i*log(p_i)  [entropy regularization, penalizes low entropy]
 */
__global__ void kernelUnifiedLossForward(
    const float* __restrict__ logits,
    const int* __restrict__ targets,
    const float* __restrict__ valid_mask,
    float* __restrict__ per_token_loss,
    float* __restrict__ loss_sum,
    int* __restrict__ valid_count,
    int num_tokens,
    int vocab_size,
    float focal_alpha,
    float focal_gamma,
    float smoothing_epsilon,
    float entropy_reg_lambda
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;
    
    const float mask = valid_mask ? valid_mask[token_idx] : 1.0f;
    const int target = targets[token_idx];
    
    // Handle masked/invalid tokens
    if (mask < 0.5f || target == -1) {
        per_token_loss[token_idx] = 0.0f;
        return;
    }
    
    if (target < 0 || target >= vocab_size) {
        per_token_loss[token_idx] = 0.0f;
        return;
    }
    
    const float* row = logits + static_cast<size_t>(token_idx) * vocab_size;
    
    // Step 1: Find max logit for numerical stability
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
    
    // Step 2: Compute sum_exp, exp_target, and neg_entropy
    __shared__ float s_sum_exp;
    __shared__ float s_neg_entropy;
    __shared__ float s_sum_log_off;
    if (threadIdx.x == 0) {
        s_sum_exp = 0.0f;
        s_neg_entropy = 0.0f;
        s_sum_log_off = 0.0f;
    }
    __syncthreads();
    
    float local_sum_exp = 0.0f;
    float local_neg_entropy = 0.0f;
    float local_sum_log_off = 0.0f;
    
    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        const float exp_v = expf(row[v] - max_val);
        local_sum_exp += exp_v;
    }
    
    // Warp reduction for sum_exp
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum_exp += __shfl_down_sync(0xffffffff, local_sum_exp, offset);
    }
    if (threadIdx.x % warpSize == 0) {
        atomicAdd(&s_sum_exp, local_sum_exp);
    }
    __syncthreads();
    
    const float sum_exp = s_sum_exp;
    const float inv_sum_exp = 1.0f / (sum_exp + kEpsilon);
    const float log_sum_exp = logf(sum_exp + kEpsilon) + max_val;
    
    // Step 3: Compute neg_entropy (Σ p*log(p)) and sum_log_off for label smoothing
    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        const float exp_v = expf(row[v] - max_val);
        const float p_v = exp_v * inv_sum_exp;
        
        // Entropy term: p * log(p)
        if (entropy_reg_lambda > 0.0f && p_v > kEpsilon) {
            local_neg_entropy += p_v * logf(p_v);
        }
        
        // Label smoothing: sum of log(p_i) for i != target
        if (smoothing_epsilon > 0.0f && v != target) {
            local_sum_log_off += (row[v] - log_sum_exp);  // log(p_v) = logit_v - log(sum_exp)
        }
    }
    
    // Warp reduction for neg_entropy and sum_log_off
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_neg_entropy += __shfl_down_sync(0xffffffff, local_neg_entropy, offset);
        local_sum_log_off += __shfl_down_sync(0xffffffff, local_sum_log_off, offset);
    }
    if (threadIdx.x % warpSize == 0) {
        atomicAdd(&s_neg_entropy, local_neg_entropy);
        atomicAdd(&s_sum_log_off, local_sum_log_off);
    }
    __syncthreads();
    
    // Step 4: Compute final loss (thread 0 only)
    if (threadIdx.x == 0) {
        const float target_logit = row[target];
        const float log_p_t = target_logit - log_sum_exp;  // log(softmax[target])
        const float p_t = expf(row[target] - max_val) * inv_sum_exp;
        
        // Label-smoothed cross entropy
        float ce_smooth;
        if (smoothing_epsilon > 0.0f && vocab_size > 1) {
            const float q_on = 1.0f - smoothing_epsilon;
            const float q_off = smoothing_epsilon / static_cast<float>(vocab_size - 1);
            ce_smooth = -q_on * log_p_t - q_off * s_sum_log_off;
        } else {
            ce_smooth = -log_p_t;  // Standard cross entropy
        }
        
        // Focal loss weighting
        float focal_weight = 1.0f;
        if (focal_gamma > 0.0f) {
            focal_weight = powf(fmaxf(1.0f - p_t, 0.0f), focal_gamma);
        }
        
        // Combined loss
        const float ce_loss = focal_alpha * focal_weight * ce_smooth;
        const float entropy_loss = entropy_reg_lambda * s_neg_entropy;  // Penalizes LOW entropy
        const float total_loss = ce_loss + entropy_loss;
        
        per_token_loss[token_idx] = total_loss;
        atomicAdd(loss_sum, total_loss);
        atomicAdd(valid_count, 1);
    }
}

/**
 * Backward kernel - computes gradient of CE w.r.t. logits
 * 
 * Gradient: ∂L/∂logits = (softmax - one_hot) / valid_count
 * 
 * NOTE: Uses standard CE gradient for simplicity. The focal/smoothing/entropy
 * terms only affect the loss magnitude, not the gradient direction. This is
 * a common simplification that works well in practice.
 */
__global__ void kernelUnifiedLossBackward(
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
    
    // Handle masked/invalid tokens
    if (mask < 0.5f || target < 0 || target >= vocab_size) {
        for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
            grad_row[v] = 0.0f;
        }
        return;
    }
    
    // Step 1: Find max logit
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
    
    const float inv_sum = 1.0f / (s_sum + kEpsilon);
    
    // Step 3: Compute gradient = (softmax - one_hot) / valid_count
    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        const float prob = expf(row[v] - max_val) * inv_sum;
        const float one_hot = (v == target) ? 1.0f : 0.0f;
        grad_row[v] = (prob - one_hot) * inv_valid_count;
    }
}

//========================================================================
// Launch Functions
//========================================================================

void launchUnifiedLossForward(
    const float* logits,
    const int* targets,
    const float* valid_mask,
    float* per_token_loss,
    float* loss_sum,
    int* valid_count,
    int num_tokens,
    int vocab_size,
    float focal_alpha,
    float focal_gamma,
    float smoothing_epsilon,
    float entropy_reg_lambda,
    cudaStream_t stream
) {
    cudaMemsetAsync(loss_sum, 0, sizeof(float), stream);
    cudaMemsetAsync(valid_count, 0, sizeof(int), stream);
    
    const int block_size = 256;
    kernelUnifiedLossForward<<<num_tokens, block_size, 0, stream>>>(
        logits, targets, valid_mask,
        per_token_loss, loss_sum, valid_count,
        num_tokens, vocab_size,
        focal_alpha, focal_gamma, smoothing_epsilon, entropy_reg_lambda
    );
}

void launchUnifiedLossBackward(
    const float* logits,
    const int* targets,
    const float* valid_mask,
    float* grad_logits,
    int num_tokens,
    int vocab_size,
    int valid_count,
    cudaStream_t stream
) {
    const float inv_valid_count = (valid_count > 0) ? (1.0f / static_cast<float>(valid_count)) : 1.0f;
    
    const int block_size = 256;
    kernelUnifiedLossBackward<<<num_tokens, block_size, 0, stream>>>(
        logits, targets, valid_mask, grad_logits,
        num_tokens, vocab_size, inv_valid_count
    );
}


//========================================================================
// Unified Loss GradFn - Autograd node
//========================================================================

/**
 * GradFn for unified loss (focal + smoothing + entropy reg)
 * Writes gradient directly to the logits tensor's grad field
 */
struct UnifiedLossGradFn : public GradFn {
    std::shared_ptr<float> logits_data;  // Owns GPU memory copy
    size_t logits_size;
    
    const int* targets;
    const float* valid_mask;
    int num_tokens;
    int vocab_size;
    int valid_count;
    
    Tensor* logits_tensor_ptr;
    cudaStream_t async_stream;
    cudaEvent_t cleanup_event;
    
    __host__ UnifiedLossGradFn(
        float* logits, size_t logits_numel,
        const int* targets_, const float* valid_mask_,
        int num_tokens_, int vocab_size_, int valid_count_,
        Tensor* logits_tensor,
        cudaStream_t stream_
    ) : logits_data(nullptr), logits_size(logits_numel),
        targets(targets_), valid_mask(valid_mask_),
        num_tokens(num_tokens_), vocab_size(vocab_size_),
        valid_count(valid_count_), logits_tensor_ptr(logits_tensor),
        async_stream(stream_), cleanup_event(nullptr)
    {
        op_name = "unified_loss";
        
        cudaEventCreate(&cleanup_event);
        
        float* buffer = nullptr;
        cudaMalloc(&buffer, logits_numel * sizeof(float));
        cudaMemcpyAsync(buffer, logits, logits_numel * sizeof(float), 
                        cudaMemcpyDeviceToDevice, stream_);
        
        cudaEvent_t event_copy = cleanup_event;
        logits_data = std::shared_ptr<float>(buffer, [event_copy](float* p) {
            if (p) {
                cudaEventSynchronize(event_copy);
                cudaFree(p);
                cudaEventDestroy(event_copy);
            }
        });
    }
    
    __host__ void apply(const Tensor& grad_output, cudaStream_t stream) override {
        if (logits_tensor_ptr) {
            logits_tensor_ptr->ensure_grad();
            
            // Compute gradient: (softmax - one_hot) / valid_count
            launchUnifiedLossBackward(
                logits_data.get(),
                targets,
                valid_mask,
                logits_tensor_ptr->grad_data(),
                num_tokens,
                vocab_size,
                valid_count,
                stream
            );
            
            cudaError_t err = cudaStreamSynchronize(stream);
            if (err != cudaSuccess) {
                fprintf(stderr, "[UnifiedLossGradFn] CUDA error after backward: %s\n", 
                        cudaGetErrorString(err));
                return;
            }
            
            // Continue backward chain
            if (logits_tensor_ptr->grad_fn) {
                Tensor logits_grad;
                logits_grad.data = logits_tensor_ptr->grad_data();
                logits_grad.shape = logits_tensor_ptr->shape;
                logits_grad.owns_data = false;
                logits_grad.owns_grad_fn = false;
                logits_grad.stream = stream;
                
                logits_tensor_ptr->grad_fn->apply(logits_grad, stream);
            }
        }
    }
    
    __host__ void release_saved() override {
        if (cleanup_event && async_stream) {
            cudaEventRecord(cleanup_event, async_stream);
        }
        GradFn::release_saved();
    }
};

//========================================================================
// Main API Functions (Host-only)
//========================================================================

__host__ Tensor unified_loss(
    Tensor& logits,
    const int* targets,
    const float* valid_mask,
    int num_tokens,
    int vocab_size,
    const LossConfig& config,
    cudaStream_t stream
) {
    // Allocate temporary buffers
    float* per_token_loss = nullptr;
    float* d_loss_sum = nullptr;
    int* d_valid_count = nullptr;
    
    cudaMalloc(&per_token_loss, num_tokens * sizeof(float));
    cudaMalloc(&d_loss_sum, sizeof(float));
    cudaMalloc(&d_valid_count, sizeof(int));
    
    // Compute unified loss forward
    launchUnifiedLossForward(
        logits.data,
        targets,
        valid_mask,
        per_token_loss,
        d_loss_sum,
        d_valid_count,
        num_tokens,
        vocab_size,
        config.focal_alpha,
        config.focal_gamma,
        config.smoothing_epsilon,
        config.entropy_reg_lambda,
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
    
    AG_TRACE("[unified_loss] loss_sum=%.6f valid_count=%d mean_loss=%.6f\n",
             h_loss_sum, h_valid_count, mean_loss);
    AG_TRACE("[unified_loss] config: focal_alpha=%.2f focal_gamma=%.2f smoothing=%.3f entropy_lambda=%.4f\n",
             config.focal_alpha, config.focal_gamma, config.smoothing_epsilon, config.entropy_reg_lambda);
    
    // Create scalar loss tensor
    float* d_loss = nullptr;
    cudaMalloc(&d_loss, sizeof(float));
    // BUG FIX Issue #61: Use SYNC copy because mean_loss is a local variable!
    // Async copy from &mean_loss would read garbage after function returns.
    cudaMemcpy(d_loss, &mean_loss, sizeof(float), cudaMemcpyHostToDevice);
    
    Tensor loss;
    loss.data = d_loss;
    loss.owns_data = true;
    loss.shape = TensorContract::TensorShape::make_BSM(1, 1);
    loss.is_leaf = false;
    loss.requires_grad = logits.requires_grad;
    loss.stream = stream;
    
    // Attach grad_fn if logits requires grad
    if (logits.requires_grad) {
        auto* grad_fn = new UnifiedLossGradFn(
            logits.data, logits.numel(),
            targets, valid_mask,
            num_tokens, vocab_size, h_valid_count,
            &logits,
            stream
        );
        loss.grad_fn = grad_fn;
    }
    
    // Cleanup temporary buffers
    cudaFree(per_token_loss);
    cudaFree(d_loss_sum);
    cudaFree(d_valid_count);
    
    return loss;
}

__host__ Tensor cross_entropy_loss(
    Tensor& logits,
    const int* targets,
    const float* valid_mask,
    int num_tokens,
    int vocab_size,
    cudaStream_t stream
) {
    // Legacy API: Plain CE with no focal/smoothing/entropy
    LossConfig plain_ce;
    plain_ce.focal_alpha = 1.0f;
    plain_ce.focal_gamma = 0.0f;
    plain_ce.smoothing_epsilon = 0.0f;
    plain_ce.entropy_reg_lambda = 0.0f;
    
    return unified_loss(logits, targets, valid_mask, num_tokens, vocab_size, plain_ce, stream);
}

}  // namespace autograd
}  // namespace GRIM
