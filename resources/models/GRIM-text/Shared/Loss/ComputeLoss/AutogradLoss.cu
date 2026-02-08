//======================================================//
//  AutogradLoss.cu
//  CUDA implementation of unified autograd-enabled loss
//  
//  Implements: Focal Loss + Label Smoothing + Cross Entropy + Entropy Regularization
//  This is the ONLY loss computation path for training.
//======================================================//

#include "AutogradLoss.hpp"
#include "../../TensorContract/TensorContract_GPU.hpp"
#include "../../EquationLogging/EquationLogging.hpp"
#include "../../EquationLogging/PyTorchVerify.hpp"  // PyTorch verification for side-by-side comparison
#include <cuda_runtime.h>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <memory>

// ========================================================================
// Finite Difference Gradient Verification (Issue Investigation)
// ========================================================================
// Define GRIM_FD_GRAD_VERIFY to enable finite difference gradient verification.
// This adds significant overhead (2 full loss computations per sample) so should
// only be enabled for debugging gradient sign errors.
//
// Usage: Uncomment the line below to enable FD verification:
// #define GRIM_FD_GRAD_VERIFY  // DISABLED for production - significant overhead
// ========================================================================

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
// CUDA Kernels — NLL Loss on log-probabilities
// These kernels receive log_probs = log_softmax(logits), NOT raw logits.
// Softmax is computed ONCE in autograd::log_softmax() and saved for backward.
//========================================================================

/**
 * NLL Loss Forward kernel — one block per token
 *
 * log_probs already contains log(softmax(logits)), computed by log_softmax.
 * We just pick -log_probs[target] and add focal/smoothing/entropy.
 *
 * Computes:
 *   L = α * (1 - p_t)^γ * CE_smooth + λ * neg_entropy
 *
 * Where:
 *   p_t  = exp(log_probs[target])          — ONE exp, not 50K
 *   CE_smooth = -(1-ε)*log_probs[target] - ε/(V-1)*Σ_{i≠t} log_probs[i]
 *   neg_entropy = Σ exp(log_probs[i]) * log_probs[i]
 */
__global__ void kernelNLLLossForward(
    const float* __restrict__ log_probs,    // [num_tokens, vocab_size] — log-probabilities
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

    if (mask < 0.5f || target == -1) {
        per_token_loss[token_idx] = 0.0f;
        return;
    }
    if (target < 0 || target >= vocab_size) {
        per_token_loss[token_idx] = 0.0f;
        return;
    }

    const float* row = log_probs + static_cast<size_t>(token_idx) * vocab_size;

    // p_t = exp(log_probs[target]) — just ONE exp call
    const float log_p_t = row[target];
    const float p_t = expf(log_p_t);

    // ── Entropy: Σ p_i * log(p_i) = Σ exp(lp_i) * lp_i ──
    __shared__ float s_neg_entropy;
    __shared__ float s_sum_log_off;
    if (threadIdx.x == 0) {
        s_neg_entropy = 0.0f;
        s_sum_log_off = 0.0f;
    }
    __syncthreads();

    float local_neg_entropy = 0.0f;
    float local_sum_log_off = 0.0f;

    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        if (entropy_reg_lambda > 0.0f) {
            const float p_v = expf(row[v]);
            if (p_v > 0.0f) {
                local_neg_entropy += p_v * row[v];  // p * log(p)
            }
        }
        if (smoothing_epsilon > 0.0f && v != target) {
            local_sum_log_off += row[v];  // log(p_v) — already in log space!
        }
    }

    // Warp + cross-warp reduction
    for (int off = warpSize / 2; off > 0; off /= 2) {
        local_neg_entropy += __shfl_down_sync(0xffffffff, local_neg_entropy, off);
        local_sum_log_off += __shfl_down_sync(0xffffffff, local_sum_log_off, off);
    }
    if (threadIdx.x % warpSize == 0) {
        atomicAdd(&s_neg_entropy, local_neg_entropy);
        atomicAdd(&s_sum_log_off, local_sum_log_off);
    }
    __syncthreads();

    // ── Thread 0: assemble loss ──
    if (threadIdx.x == 0) {
        // Label-smoothed cross entropy
        float ce_smooth;
        if (smoothing_epsilon > 0.0f && vocab_size > 1) {
            const float q_on  = 1.0f - smoothing_epsilon;
            const float q_off = smoothing_epsilon / static_cast<float>(vocab_size - 1);
            ce_smooth = -q_on * log_p_t - q_off * s_sum_log_off;
        } else {
            ce_smooth = -log_p_t;  // Standard CE: -log(p_target)
        }

        // Focal loss weight
        float focal_weight = 1.0f;
        if (focal_gamma > 0.0f) {
            focal_weight = powf(fmaxf(1.0f - p_t, 0.0f), focal_gamma);
        }

        const float ce_loss      = focal_alpha * focal_weight * ce_smooth;
        const float entropy_loss = entropy_reg_lambda * s_neg_entropy;
        const float total_loss   = ce_loss + entropy_loss;

        per_token_loss[token_idx] = total_loss;
        atomicAdd(loss_sum, total_loss);
        atomicAdd(valid_count, 1);
    }
}

/**
 * NLL Loss Backward kernel — gradient w.r.t. LOG-PROBABILITIES
 *
 * For plain CE:  ∂L/∂(log_p_i) = (1/N) * (δ_{i≠t} ? 0 : -1)
 *   But we actually want d/d(log_p) of the full loss, so:
 *
 *   ∂(-log_p_t)/∂(log_p_i) = -δ_{i=t}
 *
 * With label smoothing:
 *   ∂CE_smooth/∂(log_p_i) = -q_i  where q_t = 1-ε, q_i = ε/(V-1) for i≠t
 *
 * With focal:
 *   ∂[α*(1-p_t)^γ * CE_smooth]/∂(log_p_i)
 *     = α * [focal_weight * (-q_i) + (1-p_t)^γ's derivative via p_t = exp(log_p_t)]
 *
 * With entropy reg:
 *   ∂[λ * Σ p_j*log(p_j)]/∂(log_p_i) = λ * p_i * (log(p_i) + 1 - neg_entropy)  [Issue #124 centering]
 *   But NOTE: p_j = exp(log_p_j), so dp_j/d(log_p_i) = p_j * (δ_{ij} - p_i)  [via softmax Jacobian]
 *   HOWEVER: d(p_j*log_p_j)/d(log_p_i) = dp_j/d(log_p_i)*(log_p_j+1)
 *   This Jacobian chain belongs in LogSoftmaxGradFn (the upstream), NOT here.
 *
 * KEY INSIGHT: Since we're differentiating w.r.t. log_probs (not logits),
 * and log_probs are INDEPENDENT variables from NLL's perspective
 * (the coupling happens in LogSoftmaxGradFn), the NLL gradient is simple:
 *
 *   Base CE: grad[i] = -q_i  (only non-zero for smoothed targets)
 *            grad[target] = -(1-ε)   or  -1 if no smoothing
 *
 * With focal:
 *   The focal weight (1-p_t)^γ depends on log_p_t = log_probs[target].
 *   p_t = exp(log_p_t), so dp_t/d(log_p_i) = p_t if i=t, else 0
 *   focal_weight = (1-p_t)^γ
 *   d(focal_weight)/d(log_p_t) = -γ*(1-p_t)^(γ-1)*p_t
 *   Only affects the target position.
 *
 * With entropy reg:
 *   H(p) = Σ p_j * log_p_j where p_j = exp(log_p_j)
 *   ∂H/∂(log_p_i) = ∂(p_i * log_p_i)/∂(log_p_i) = p_i * (log_p_i + 1)
 *   BUT the coupling terms (dp_j/d(log_p_i) for j≠i) are handled by LogSoftmaxGradFn.
 *   So NLL's local gradient for entropy is just: λ * p_i * (log_p_i + 1)
 *   With Issue #124 centering: λ * p_i * (log_p_i + 1 - (H+1)) = λ * p_i * (log_p_i - H)
 *
 * WAIT — this is wrong. log_probs are NOT independent variables. They're
 * constrained by logsumexp normalization. The LogSoftmaxGradFn accounts for
 * this constraint. But the NLL gradient w.r.t. each log_p[i] treats them as
 * if they were independent — the Jacobian correction happens upstream.
 *
 * FINAL GRADIENT (same as PyTorch F.nll_loss):
 *   For plain CE:  grad_log_p[i] = -δ_{i=t} / N
 *   For smoothed:  grad_log_p[i] = -q_i / N
 *
 * When this flows into LogSoftmaxGradFn:
 *   grad_logits[j] = grad_log_p[j] - p_j * Σ_i grad_log_p[i]
 *                  = -q_j/N - p_j * (-1/N)     [since Σ q_i = 1]
 *                  = (p_j - q_j) / N
 * Which is EXACTLY the standard CE gradient! ✓
 */
__global__ void kernelNLLLossBackward(
    const float* __restrict__ log_probs,    // [num_tokens, vocab_size]
    const int* __restrict__ targets,
    const float* __restrict__ valid_mask,
    float* __restrict__ grad_log_probs,     // [num_tokens, vocab_size] — OUTPUT
    int num_tokens,
    int vocab_size,
    float inv_valid_count,  // 1/N
    float focal_alpha,
    float focal_gamma,
    float smoothing_epsilon,
    float entropy_reg_lambda
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    const float mask = valid_mask ? valid_mask[token_idx] : 1.0f;
    const float* row = log_probs + static_cast<size_t>(token_idx) * vocab_size;
    float* grad_row = grad_log_probs + static_cast<size_t>(token_idx) * vocab_size;
    const int target = targets[token_idx];

    // Masked/invalid → zero gradient
    if (mask < 0.5f || target < 0 || target >= vocab_size) {
        for (int v = threadIdx.x; v < vocab_size; v += blockDim.x)
            grad_row[v] = 0.0f;
        return;
    }

    // Label smoothing targets
    const float q_on  = (smoothing_epsilon > 0.0f && vocab_size > 1)
                        ? (1.0f - smoothing_epsilon) : 1.0f;
    const float q_off = (smoothing_epsilon > 0.0f && vocab_size > 1)
                        ? (smoothing_epsilon / static_cast<float>(vocab_size - 1)) : 0.0f;

    // Focal precomputation
    const float p_t = expf(row[target]);
    float focal_weight = 1.0f;
    float focal_deriv_factor = 0.0f;
    if (focal_gamma > 0.0f) {
        const float one_minus_pt = fmaxf(1.0f - p_t, 0.0f);
        if (one_minus_pt > 0.0f) {
            focal_weight = powf(one_minus_pt, focal_gamma);
            focal_deriv_factor = focal_gamma * powf(one_minus_pt, focal_gamma - 1.0f);
        } else {
            focal_weight = 0.0f;
            focal_deriv_factor = 0.0f;
        }
    }

    // Entropy centering term (Issue #124)
    // Need neg_entropy = Σ p_v * log_p_v for centering
    __shared__ float s_neg_entropy;
    if (threadIdx.x == 0) s_neg_entropy = 0.0f;
    __syncthreads();

    float local_ne = 0.0f;
    if (entropy_reg_lambda > 0.0f) {
        for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
            const float p_v = expf(row[v]);
            if (p_v > 0.0f) local_ne += p_v * row[v];
        }
        for (int off = warpSize / 2; off > 0; off /= 2)
            local_ne += __shfl_down_sync(0xffffffff, local_ne, off);
        if (threadIdx.x % warpSize == 0) atomicAdd(&s_neg_entropy, local_ne);
    }
    __syncthreads();
    const float neg_entropy = s_neg_entropy;

    // ── Compute gradient ──
    // NLL gradient w.r.t. log_probs (LogSoftmaxGradFn does the Jacobian correction)
    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        const float q_v = (v == target) ? q_on : q_off;

        // Base: -q_v (plain NLL loss gradient w.r.t. log_probs)
        float grad_v = focal_alpha * focal_weight * (-q_v);

        // Focal derivative: only affects gradient through p_t = exp(log_p_t)
        if (focal_gamma > 0.0f) {
            // CE_smooth for focal deriv
            const float ce_smooth = (smoothing_epsilon > 0.0f) 
                ? -(q_on * row[target])  // Simplified: dominant term
                : -row[target];
            // d(focal_weight)/d(log_p_v) = -γ*(1-p_t)^(γ-1) * p_t  if v==target, else 0
            if (v == target) {
                grad_v += focal_alpha * (-focal_deriv_factor) * p_t * ce_smooth;
            }
        }

        // Entropy regularization gradient w.r.t. log_probs
        // NLL's local derivative: λ * p_v * (log_p_v - neg_entropy)
        // The cross-term Jacobian is handled by LogSoftmaxGradFn upstream.
        if (entropy_reg_lambda > 0.0f) {
            const float p_v = expf(row[v]);
            if (p_v > 0.0f) {
                grad_v += entropy_reg_lambda * p_v * (row[v] - neg_entropy);
            }
        }

        // Apply mean reduction
        grad_row[v] = grad_v * inv_valid_count;
    }
}

// OLD kernelUnifiedLossBackward DELETED (Rule 20) — replaced by kernelNLLLossBackward above.
// The old kernel recomputed softmax INDEPENDENTLY from the forward kernel (non-deterministic atomicAdd),
// causing forward/backward probability inconsistency. The new architecture:
//   logits → autograd::log_softmax() → log_probs → kernelNLLLossForward/Backward
// ensures forward and backward use IDENTICAL probability values.
// Gradient centering (Issue #124 GRAD_CENTER_EQUATION) is now automatic:
//   After LogSoftmaxGradFn: grad_logits[v] = grad_log_p[v] - p_v * Σ grad_log_p[v]
//                         = (p_v - q_v) / N → Σ_v = 0  ✓

// ── Kept: kernelFiniteDiffGradVerify (Rule 21 diagnostic) operates on raw logits ──
// ── It's a standalone verification tool, not part of the training gradient path. ──



//========================================================================
// Launch Functions — NLL Loss (operates on log_probs, not raw logits)
//========================================================================

void launchUnifiedLossForward(
    const float* log_probs,     // [num_tokens, vocab_size] — output of log_softmax
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
    kernelNLLLossForward<<<num_tokens, block_size, 0, stream>>>(
        log_probs, targets, valid_mask,
        per_token_loss, loss_sum, valid_count,
        num_tokens, vocab_size,
        focal_alpha, focal_gamma, smoothing_epsilon, entropy_reg_lambda
    );
}

void launchUnifiedLossBackward(
    const float* log_probs,     // [num_tokens, vocab_size] — output of log_softmax
    const int* targets,
    const float* valid_mask,
    float* grad_log_probs,      // [num_tokens, vocab_size] — OUTPUT gradient w.r.t. log_probs
    int num_tokens,
    int vocab_size,
    int valid_count,
    float focal_alpha,
    float focal_gamma,
    float smoothing_epsilon,
    float entropy_reg_lambda,
    cudaStream_t stream
) {
    // Mean reduction: divide gradient by valid_count (matching PyTorch's F.nll_loss(reduction='mean'))
    if (valid_count <= 0) {
        throw std::runtime_error("[launchUnifiedLossBackward] valid_count=" + std::to_string(valid_count) 
            + " — no valid tokens, caller MUST ensure valid_count > 0");
    }
    const float inv_valid_count = 1.0f / static_cast<float>(valid_count);
    
    const int block_size = 256;
    kernelNLLLossBackward<<<num_tokens, block_size, 0, stream>>>(
        log_probs, targets, valid_mask, grad_log_probs,
        num_tokens, vocab_size, inv_valid_count,
        focal_alpha, focal_gamma, smoothing_epsilon, entropy_reg_lambda
    );
}

//========================================================================
// [FD_GRAD_VERIFY_EQUATION] Finite Difference Gradient Verification Kernel
// PURPOSE: Compare sign(analytical gradient) vs sign(finite difference gradient)
// EQUATION: FD_grad = (L(z_v + ε) - L(z_v - ε)) / (2ε)
//           If sign(FD_grad) != sign(analytical_grad), we have a sign error!
// This is RULE 21 diagnostic logging - mathematical proof of gradient correctness
//========================================================================

__global__ void kernelFiniteDiffGradVerify(
    const float* __restrict__ logits,       // [num_tokens, vocab_size]
    const int* __restrict__ targets,        // [num_tokens]
    const float* __restrict__ valid_mask,   // [num_tokens] or nullptr
    const float* __restrict__ grad_logits,  // [num_tokens, vocab_size] - analytical gradient
    int num_tokens,
    int vocab_size,
    float inv_valid_count,                  // 1/N for mean reduction
    float focal_alpha,
    float focal_gamma,
    float smoothing_epsilon,
    float entropy_reg_lambda,
    int sample_token_idx,                   // Which token position to verify
    int sample_vocab_idx                    // Which vocab position to verify (e.g., 277 for SPACE)
) {
    // Only thread 0 of block 0 runs this diagnostic
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    
    // Bounds check
    if (sample_token_idx >= num_tokens || sample_vocab_idx >= vocab_size) {
        printf("[FD_GRAD_VERIFY_EQUATION] ERROR: sample indices out of bounds (tok=%d/%d, vocab=%d/%d)\n",
               sample_token_idx, num_tokens, sample_vocab_idx, vocab_size);
        return;
    }
    
    // Check if token is valid (not masked)
    const bool is_valid = (valid_mask == nullptr) || (valid_mask[sample_token_idx] >= 0.5f);
    if (!is_valid) {
        printf("[FD_GRAD_VERIFY_EQUATION] SKIP: token %d is masked\n", sample_token_idx);
        return;
    }
    
    const int target = targets[sample_token_idx];
    if (target < 0 || target >= vocab_size) {
        printf("[FD_GRAD_VERIFY_EQUATION] SKIP: invalid target %d for token %d\n", target, sample_token_idx);
        return;
    }
    
    const float* row = logits + static_cast<size_t>(sample_token_idx) * vocab_size;
    const float analytical_grad = grad_logits[static_cast<size_t>(sample_token_idx) * vocab_size + sample_vocab_idx];
    
    // Epsilon for finite difference (should be small but not too small to avoid FP errors)
    const float epsilon = 1e-4f;
    const float kEpsilon_local = 1e-10f;  // For numerical stability in log
    
    //========================================================================
    // Compute L(z_v + ε) - perturbed loss with logit[v] increased by epsilon
    //========================================================================
    
    // Step 1: Find max logit for numerical stability (with perturbation at sample_vocab_idx)
    float max_val_plus = -1e30f;
    for (int v = 0; v < vocab_size; v++) {
        float logit_v = row[v];
        if (v == sample_vocab_idx) logit_v += epsilon;  // Perturbation
        if (logit_v > max_val_plus) max_val_plus = logit_v;
    }
    
    // Step 2: Compute softmax denominator, neg_entropy, and sum_log_off with perturbation
    float sum_exp_plus = 0.0f;
    float neg_entropy_plus = 0.0f;
    float sum_log_off_plus = 0.0f;
    
    for (int v = 0; v < vocab_size; v++) {
        float logit_v = row[v];
        if (v == sample_vocab_idx) logit_v += epsilon;  // Perturbation
        
        float exp_v = expf(logit_v - max_val_plus);
        sum_exp_plus += exp_v;
    }
    
    const float log_sum_exp_plus = logf(sum_exp_plus) + max_val_plus;
    
    for (int v = 0; v < vocab_size; v++) {
        float logit_v = row[v];
        if (v == sample_vocab_idx) logit_v += epsilon;  // Perturbation
        
        float log_p_v = logit_v - log_sum_exp_plus;
        float p_v = expf(log_p_v);
        
        // Entropy term: p_v * log(p_v)
        if (p_v > kEpsilon_local) {
            neg_entropy_plus += p_v * log_p_v;
        }
        
        // Label smoothing sum (for non-target)
        if (v != target) {
            sum_log_off_plus += log_p_v;
        }
    }
    
    // Step 3: Compute loss L(z + ε)
    float log_p_target_plus = row[target] + (target == sample_vocab_idx ? epsilon : 0.0f) - log_sum_exp_plus;
    float p_target_plus = expf(log_p_target_plus);
    
    float ce_smooth_plus;
    if (smoothing_epsilon > 0.0f) {
        const float q_on = 1.0f - smoothing_epsilon;
        const float q_off = smoothing_epsilon / (vocab_size - 1);
        ce_smooth_plus = -q_on * log_p_target_plus - q_off * sum_log_off_plus;
    } else {
        ce_smooth_plus = -log_p_target_plus;
    }
    
    float focal_weight_plus = 1.0f;
    if (focal_gamma > 0.0f) {
        focal_weight_plus = powf(1.0f - p_target_plus + kEpsilon_local, focal_gamma);
    }
    
    float loss_plus = focal_alpha * focal_weight_plus * ce_smooth_plus + entropy_reg_lambda * neg_entropy_plus;
    
    //========================================================================
    // Compute L(z_v - ε) - perturbed loss with logit[v] decreased by epsilon
    //========================================================================
    
    // Step 1: Find max logit with minus perturbation
    float max_val_minus = -1e30f;
    for (int v = 0; v < vocab_size; v++) {
        float logit_v = row[v];
        if (v == sample_vocab_idx) logit_v -= epsilon;  // Perturbation
        if (logit_v > max_val_minus) max_val_minus = logit_v;
    }
    
    // Step 2: Compute softmax components with minus perturbation
    float sum_exp_minus = 0.0f;
    float neg_entropy_minus = 0.0f;
    float sum_log_off_minus = 0.0f;
    
    for (int v = 0; v < vocab_size; v++) {
        float logit_v = row[v];
        if (v == sample_vocab_idx) logit_v -= epsilon;  // Perturbation
        
        float exp_v = expf(logit_v - max_val_minus);
        sum_exp_minus += exp_v;
    }
    
    const float log_sum_exp_minus = logf(sum_exp_minus) + max_val_minus;
    
    for (int v = 0; v < vocab_size; v++) {
        float logit_v = row[v];
        if (v == sample_vocab_idx) logit_v -= epsilon;  // Perturbation
        
        float log_p_v = logit_v - log_sum_exp_minus;
        float p_v = expf(log_p_v);
        
        if (p_v > kEpsilon_local) {
            neg_entropy_minus += p_v * log_p_v;
        }
        
        if (v != target) {
            sum_log_off_minus += log_p_v;
        }
    }
    
    // Step 3: Compute loss L(z - ε)
    float log_p_target_minus = row[target] + (target == sample_vocab_idx ? -epsilon : 0.0f) - log_sum_exp_minus;
    float p_target_minus = expf(log_p_target_minus);
    
    float ce_smooth_minus;
    if (smoothing_epsilon > 0.0f) {
        const float q_on = 1.0f - smoothing_epsilon;
        const float q_off = smoothing_epsilon / (vocab_size - 1);
        ce_smooth_minus = -q_on * log_p_target_minus - q_off * sum_log_off_minus;
    } else {
        ce_smooth_minus = -log_p_target_minus;
    }
    
    float focal_weight_minus = 1.0f;
    if (focal_gamma > 0.0f) {
        focal_weight_minus = powf(1.0f - p_target_minus + kEpsilon_local, focal_gamma);
    }
    
    float loss_minus = focal_alpha * focal_weight_minus * ce_smooth_minus + entropy_reg_lambda * neg_entropy_minus;
    
    //========================================================================
    // Finite Difference Gradient Computation
    // FD_grad = (L(z+ε) - L(z-ε)) / (2ε) ... this is PER-TOKEN loss gradient
    // Then apply inv_valid_count to match mean-reduced gradient
    //========================================================================
    
    float fd_grad_per_token = (loss_plus - loss_minus) / (2.0f * epsilon);
    float fd_grad = fd_grad_per_token * inv_valid_count;  // Apply mean reduction scaling
    
    // Determine signs
    int sign_fd = (fd_grad > 1e-10f) ? 1 : ((fd_grad < -1e-10f) ? -1 : 0);
    int sign_analytical = (analytical_grad > 1e-10f) ? 1 : ((analytical_grad < -1e-10f) ? -1 : 0);
    bool sign_match = (sign_fd == sign_analytical);
    
    // Compute relative error
    float abs_fd = fabsf(fd_grad);
    float abs_anal = fabsf(analytical_grad);
    float rel_error = (abs_fd > 1e-10f || abs_anal > 1e-10f) 
        ? fabsf(fd_grad - analytical_grad) / fmaxf(abs_fd, abs_anal)
        : 0.0f;
    
    //========================================================================
    // [FD_GRAD_VERIFY_EQUATION] RULE 21 Logging
    //========================================================================
    const char* match_str = sign_match ? "MATCH" : "MISMATCH";
    const bool is_target_pos = (sample_vocab_idx == target);
    
    printf("[FD_GRAD_VERIFY_EQUATION] token=%d vocab=%d (is_target=%s) target=%d\n",
           sample_token_idx, sample_vocab_idx, is_target_pos ? "YES" : "NO", target);
    printf("  INPUTS: L_plus=%.10f L_minus=%.10f epsilon=%.6f inv_valid=%e\n",
           loss_plus, loss_minus, epsilon, inv_valid_count);
    printf("  FD_grad_per_token = (L+ - L-) / (2ε) = (%.10f - %.10f) / %.6f = %.10e\n",
           loss_plus, loss_minus, 2.0f * epsilon, fd_grad_per_token);
    printf("  FD_grad (mean-reduced) = FD_per_token × inv_valid = %.10e × %e = %.10e\n",
           fd_grad_per_token, inv_valid_count, fd_grad);
    printf("  ANALYTICAL_grad = %.10e\n", analytical_grad);
    printf("  SIGN CHECK: sign(FD)=%d sign(anal)=%d => %s\n", sign_fd, sign_analytical, match_str);
    printf("  RELATIVE ERROR: |FD - anal| / max(|FD|,|anal|) = %.6f%%\n", rel_error * 100.0f);
    
    // Issue #124b FIX: Distinguish TRUE sign mismatch from PRECISION-LIMITED cases
    // When FD=0 because L_plus≈L_minus (float32 precision limit), this is NOT an anomaly
    // Only flag ANOMALY when FD and analytical have genuinely OPPOSITE non-zero signs
    const bool fd_is_zero = (sign_fd == 0);
    const bool loss_identical = (fabsf(loss_plus - loss_minus) < 1e-10f);
    const bool analytical_small = (abs_anal < 1e-6f);  // Gradient too small for FD detection
    
    if (!sign_match) {
        if (fd_is_zero && loss_identical && analytical_small) {
            // FD=0 due to precision limit, NOT a gradient bug
            printf("  [PRECISION LIMITED] FD=0 because L_plus==L_minus to float32 precision\n");
            printf("  [PRECISION LIMITED] Analytical grad %.2e is below FD detection threshold (~1e-6)\n", abs_anal);
            printf("  [PRECISION LIMITED] This is EXPECTED for non-target tokens with tiny gradients - NOT A BUG\n");
        } else if (fd_is_zero && !loss_identical) {
            // FD numerically zero but loss values differ - suspicious
            printf("  [WARNING] FD≈0 but L_plus≠L_minus (diff=%.2e), numerical instability?\n", 
                   fabsf(loss_plus - loss_minus));
        } else {
            // TRUE sign mismatch: FD>0 but anal<0, or FD<0 but anal>0
            printf("  [ANOMALY] GRADIENT SIGN MISMATCH! Your gradient has OPPOSITE sign from finite difference!\n");
            printf("  [ANOMALY] FD=%+.6e, Analytical=%+.6e (genuinely opposite non-zero signs)\n", fd_grad, analytical_grad);
            printf("  [ANOMALY] This means either: (1) logging negative gradient, or (2) update uses wrong sign\n");
        }
    }
    
    if (rel_error > 0.01f && sign_match) {  // >1% error but same sign
        printf("  [WARNING] Magnitude differs >1%% (could be expected for focal/entropy terms)\n");
    }
}

//========================================================================
// Launch function for finite difference gradient verification
//========================================================================

/**
 * DIAGNOSTIC: Launches finite difference gradient verification
 * This should be called AFTER launchUnifiedLossBackward to verify the gradients
 * 
 * @param logits          Input logits [num_tokens, vocab_size]
 * @param targets         Target token IDs [num_tokens]
 * @param valid_mask      Validity mask (or nullptr)
 * @param grad_logits     The computed gradients to verify
 * @param num_tokens      Number of tokens
 * @param vocab_size      Vocabulary size
 * @param valid_count     Number of valid (non-masked) tokens
 * @param focal_alpha     Focal loss alpha
 * @param focal_gamma     Focal loss gamma
 * @param smoothing_eps   Label smoothing epsilon
 * @param entropy_lambda  Entropy regularization lambda
 * @param sample_token_idx Token index to verify (should be non-masked)
 * @param sample_vocab_idx Vocab index to verify (e.g., 277 for SPACE)
 * @param stream          CUDA stream
 */
void launchFiniteDiffGradVerify(
    const float* logits,  
    const int* targets,
    const float* valid_mask,
    const float* grad_logits,
    int num_tokens,
    int vocab_size,
    int valid_count,
    float focal_alpha,
    float focal_gamma,
    float smoothing_eps,
    float entropy_lambda,
    int sample_token_idx,
    int sample_vocab_idx,
    cudaStream_t stream
) {
    // Validate inputs
    if (!logits || !targets || !grad_logits) {
        fprintf(stderr, "[launchFiniteDiffGradVerify] ERROR: null input pointer\n");
        return;
    }
    
    if (sample_token_idx < 0 || sample_token_idx >= num_tokens) {
        fprintf(stderr, "[launchFiniteDiffGradVerify] ERROR: sample_token_idx=%d out of range [0,%d)\n",
                sample_token_idx, num_tokens);
        return;
    }
    
    if (sample_vocab_idx < 0 || sample_vocab_idx >= vocab_size) {
        fprintf(stderr, "[launchFiniteDiffGradVerify] ERROR: sample_vocab_idx=%d out of range [0,%d)\n",
                sample_vocab_idx, vocab_size);
        return;
    }
    
    const float inv_valid_count = (valid_count > 0) ? (1.0f / static_cast<float>(valid_count)) : 1.0f;
    
    // Launch single-thread diagnostic kernel
    kernelFiniteDiffGradVerify<<<1, 1, 0, stream>>>(
        logits,
        targets,
        valid_mask,
        grad_logits,
        num_tokens,
        vocab_size,
        inv_valid_count,
        focal_alpha,
        focal_gamma,
        smoothing_eps,
        entropy_lambda,
        sample_token_idx,
        sample_vocab_idx
    );
    
    // Sync to ensure printf output is flushed
    cudaStreamSynchronize(stream);
}


//========================================================================
// [TOKEN277_DIAGNOSTIC] Rule 21 Device-Side Equation Logging for Mode Collapse Detection
// Uses the centralized EquationLogging system to track Token 277 (SPACE) metrics
//========================================================================

__global__ void kernelToken277Diagnostic(
    const float* __restrict__ logits,       // [num_tokens, vocab_size]
    const int* __restrict__ targets,        // [num_tokens]
    const float* __restrict__ valid_mask,   // [num_tokens] or nullptr
    const float* __restrict__ grad_logits,  // [num_tokens, vocab_size] - computed gradients
    int num_tokens,
    int vocab_size,
    int batch_idx,
    int step_idx,
    GRIM::EquationLogEntryDevice* d_eq_buffer,
    GRIM::EquationLogBufferState* d_eq_state
) {
    // Each block processes one token position
    const int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;
    
    // Only thread 0 per block does the logging (to avoid races)
    if (threadIdx.x != 0) return;
    
    // Skip masked/invalid tokens
    const float mask = valid_mask ? valid_mask[token_idx] : 1.0f;
    const int target = targets[token_idx];
    if (mask < 0.5f || target < 0 || target >= vocab_size) return;
    
    // Only log for a sample of positions (every 100th) to avoid spam
    if (token_idx % 100 != 0) return;
    
    constexpr int TOKEN_277 = 277;  // SPACE token
    const float* row = logits + static_cast<size_t>(token_idx) * vocab_size;
    const float* grad_row = grad_logits + static_cast<size_t>(token_idx) * vocab_size;
    
    // ========================================================================
    // Step 1: Compute max logit and softmax for Token 277
    // ========================================================================
    float max_logit = -1e30f;
    int argmax_token = 0;
    for (int v = 0; v < vocab_size; v++) {
        if (row[v] > max_logit) {
            max_logit = row[v];
            argmax_token = v;
        }
    }
    
    // Compute sum_exp for softmax
    float sum_exp = 0.0f;
    for (int v = 0; v < vocab_size; v++) {
        sum_exp += expf(row[v] - max_logit);
    }
    const float log_sum_exp = logf(sum_exp);
    
    // Token 277 probability
    const float logit_277 = row[TOKEN_277];
    const float prob_277 = expf(logit_277 - max_logit) / sum_exp;
    const float expected_uniform = 1.0f / static_cast<float>(vocab_size);
    
    // ========================================================================
    // Step 2: Log Token 277 Softmax Probability
    // ========================================================================
    GRIM::logToken277SoftmaxProb(
        d_eq_buffer, d_eq_state,
        logit_277, max_logit, log_sum_exp, prob_277, expected_uniform,
        token_idx, batch_idx, step_idx
    );
    
    // ========================================================================
    // Step 3: Log Argmax Analysis - is Token 277 the argmax?
    // ========================================================================
    const bool is_277_argmax = (argmax_token == TOKEN_277);
    const bool is_277_target = (target == TOKEN_277);
    const float gap_to_argmax = max_logit - logit_277;  // Positive if 277 is NOT argmax
    
    GRIM::logToken277ArgmaxAnalysis(
        d_eq_buffer, d_eq_state,
        argmax_token, max_logit, logit_277, gap_to_argmax,
        target, is_277_argmax, is_277_target,
        token_idx, batch_idx, step_idx
    );
    
    // ========================================================================
    // Step 4: Log Token 277 Loss Contribution (if 277 is the target)
    // ========================================================================
    if (is_277_target) {
        const float log_prob_277 = logf(fmaxf(prob_277, 1e-10f));
        const float ce_loss_277 = -log_prob_277;  // Plain CE loss for this token
        const float focal_weight = 1.0f;  // Assume gamma=0 for simplicity
        
        // Gradient at target position
        const float grad_277_at_target = grad_row[TOKEN_277];
        const float expected_grad = prob_277 - 1.0f;  // For plain CE: p - 1 at target
        
        GRIM::logToken277LossContribution(
            d_eq_buffer, d_eq_state,
            ce_loss_277, focal_weight, prob_277, log_prob_277,
            grad_277_at_target, expected_grad,
            token_idx, batch_idx, step_idx
        );
    }
}

/**
 * Launch Token 277 diagnostic kernel
 * This should be called AFTER backward pass to analyze gradients
 * 
 * TEMPORARILY DISABLED: The inlined __device__ functions use ~2KB of stack space
 * each (3x 256-byte char arrays), causing GPU stack overflow. Need to either:
 *   1. Enable -rdc=true for cross-TU device linking
 *   2. Reduce EQ_LOG_STRING_LEN 
 *   3. Use static device memory instead of local arrays
 */
void launchToken277Diagnostic(
    const float* logits,
    const int* targets,
    const float* valid_mask,
    const float* grad_logits,
    int num_tokens,
    int vocab_size,
    int batch_idx,
    int step_idx,
    cudaStream_t stream
) {
    // DISABLED: Stack overflow from inlined equation logging functions
    // TODO: Re-enable after fixing device function linking with -rdc=true
    return;
    
    /*
    auto& logger = GRIM::getEquationLogger();
    if (!logger.isEnabled()) return;
    
    auto* d_eq_buffer = logger.getDeviceBuffer();
    auto* d_eq_state = logger.getDeviceState();
    if (!d_eq_buffer || !d_eq_state) return;
    
    // Launch with 1 thread per block (each block handles one token position)
    const int num_blocks = (num_tokens + 255) / 256;  // Up to 256 tokens logged
    kernelToken277Diagnostic<<<num_blocks, 1, 0, stream>>>(
        logits, targets, valid_mask, grad_logits,
        num_tokens, vocab_size,
        batch_idx, step_idx,
        d_eq_buffer, d_eq_state
    );
    */
}


//========================================================================
// Unified Loss GradFn - Autograd node
//========================================================================

/**
 * GradFn for unified loss (focal + smoothing + entropy reg)
 * Writes gradient directly to the logits tensor's grad field
 */
/**
 * NLLLossGradFn — Backward pass for NLL loss operating on log-probabilities.
 *
 * Architecture (replaces UnifiedLossGradFn):
 *   Forward:  logits → autograd::log_softmax() → log_probs → NLLLossForward → scalar loss
 *   Backward: scalar grad → NLLLossBackward(log_probs) → grad_log_probs → LogSoftmaxGradFn → grad_logits
 *
 * KEY IMPROVEMENT over old UnifiedLossGradFn:
 *   Old: recomputed softmax independently in backward (non-deterministic atomicAdd → different probs)
 *   New: uses SAME log_probs from forward pass (saved), exact forward/backward consistency
 *
 * Gradient chain:
 *   1. kernelNLLLossBackward computes ∂L/∂(log_p_v) = -q_v * inv_N  (+ focal/entropy terms)
 *   2. LogSoftmaxGradFn computes ∂(log_p)/∂(logits) = δ_{ij} - p_j  (softmax Jacobian)
 *   3. Composition:  grad_logits[j] = (p_j - q_j) / N  ← standard CE gradient ✓
 *
 * Memory management:
 *   - Takes ownership of log_probs.data (no copy — old Tensor gives up owns_data)
 *   - Takes ownership of LogSoftmaxGradFn (old Tensor gives up owns_grad_fn)
 *   - Allocates grad_log_probs_buffer for backward output
 */
struct NLLLossGradFn : public GradFn {
    // Saved forward data
    float* log_probs_data;          // OWNED GPU buffer: log-probabilities [num_tokens * vocab_size]
    float* grad_log_probs_buffer;   // OWNED GPU buffer: backward output [num_tokens * vocab_size]
    
    const int* targets;             // NOT OWNED — stable for batch lifetime
    const float* valid_mask;        // NOT OWNED — stable for batch lifetime
    int num_tokens;
    int vocab_size;
    int valid_count;
    
    // Loss configuration
    float focal_alpha;
    float focal_gamma;
    float smoothing_epsilon;
    float entropy_reg_lambda;
    
    // Upstream gradient chain
    GradFn* log_probs_grad_fn;      // OWNED — LogSoftmaxGradFn
    TensorContract::TensorShape grad_shape;
    
    cudaStream_t async_stream;
    cudaEvent_t cleanup_event;
    
    __host__ NLLLossGradFn(
        float* log_probs,           // Takes ownership of this GPU buffer
        size_t log_probs_numel,
        const int* targets_, const float* valid_mask_,
        int num_tokens_, int vocab_size_, int valid_count_,
        float focal_alpha_, float focal_gamma_,
        float smoothing_epsilon_, float entropy_reg_lambda_,
        GradFn* upstream_grad_fn,   // Takes ownership
        const TensorContract::TensorShape& shape,
        cudaStream_t stream_
    )
        : log_probs_data(log_probs)
        , grad_log_probs_buffer(nullptr)
        , targets(targets_), valid_mask(valid_mask_)
        , num_tokens(num_tokens_), vocab_size(vocab_size_)
        , valid_count(valid_count_)
        , focal_alpha(focal_alpha_), focal_gamma(focal_gamma_)
        , smoothing_epsilon(smoothing_epsilon_), entropy_reg_lambda(entropy_reg_lambda_)
        , log_probs_grad_fn(upstream_grad_fn)
        , grad_shape(shape)
        , async_stream(stream_), cleanup_event(nullptr)
    {
        op_name = "nll_loss";
        cudaEventCreate(&cleanup_event);
        
        // Allocate gradient buffer for backward output
        cudaMalloc(&grad_log_probs_buffer, log_probs_numel * sizeof(float));
    }
    
    __host__ ~NLLLossGradFn() {
        // We own the upstream LogSoftmaxGradFn
        delete log_probs_grad_fn;
        
        if (cleanup_event) {
            cudaEventSynchronize(cleanup_event);
            cudaEventDestroy(cleanup_event);
        }
        if (log_probs_data) { cudaFree(log_probs_data); log_probs_data = nullptr; }
        if (grad_log_probs_buffer) { cudaFree(grad_log_probs_buffer); grad_log_probs_buffer = nullptr; }
    }
    
    __host__ void apply(const Tensor& grad_output, cudaStream_t stream) override {
        AG_TRACE("[NLLLossGradFn::apply] ENTER: log_probs_data=%p upstream=%p\n",
                 (void*)log_probs_data, (void*)log_probs_grad_fn);
        
        // ── Step 1: Compute NLL backward → gradient w.r.t. log_probs ──
        launchUnifiedLossBackward(
            log_probs_data, targets, valid_mask, grad_log_probs_buffer,
            num_tokens, vocab_size, valid_count,
            focal_alpha, focal_gamma, smoothing_epsilon, entropy_reg_lambda,
            stream
        );
        
        cudaError_t err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "[NLLLossGradFn] CUDA error after NLL backward: %s\n",
                    cudaGetErrorString(err));
            throw std::runtime_error(std::string("[NLLLossGradFn] CUDA error: ") + cudaGetErrorString(err));
        }
        
        // ── Step 2: Diagnostic — sample grad_log_probs magnitude (Rule 21) ──
        {
            const size_t grad_elems = static_cast<size_t>(num_tokens) * vocab_size;
            // Sample from token 50+ to skip BOS/masked positions
            const size_t sample_offset = std::min(static_cast<size_t>(50) * vocab_size, grad_elems);
            const size_t sample_sz = std::min(static_cast<size_t>(50000), grad_elems - sample_offset);
            std::vector<float> samp(sample_sz);
            cudaMemcpy(samp.data(), grad_log_probs_buffer + sample_offset,
                       sample_sz * sizeof(float), cudaMemcpyDeviceToHost);
            float mx = 0.0f; double sq = 0.0;
            for (auto& v : samp) {
                if (!std::isnan(v) && !std::isinf(v)) { mx = std::max(mx, std::abs(v)); sq += v*v; }
            }
            float rms = std::sqrt(static_cast<float>(sq / sample_sz));
            
            char inputs_buf[256];
            char outputs_buf[128];
            snprintf(inputs_buf, sizeof(inputs_buf),
                "num_tokens=%d vocab=%d valid=%d", num_tokens, vocab_size, valid_count);
            snprintf(outputs_buf, sizeof(outputs_buf),
                "max=%.6f rms=%.6f", mx, rms);
            EQ_LOG_HOST("NLL-BWD-OUT", 
                "grad_log_probs = dL/d(log_p) [NLL backward, before LogSoftmaxGradFn]",
                inputs_buf, outputs_buf, "max~1/N rms~1e-5", outputs_buf,
                0, 0, 0, EquationPhase::LOSS_BACKWARD);
        }
        
        // ── Step 3: Chain to LogSoftmaxGradFn → computes grad w.r.t. raw logits ──
        // LogSoftmaxGradFn::apply() does:
        //   grad_logits[i] = grad_log_p[i] - exp(log_p[i]) * Σ_j grad_log_p[j]
        // Then chains to logits.grad_fn (MatMulGradFn from LM head)
        if (log_probs_grad_fn) {
            Tensor grad_log_probs_tensor;
            grad_log_probs_tensor.data = grad_log_probs_buffer;
            grad_log_probs_tensor.shape = grad_shape;
            grad_log_probs_tensor.owns_data = false;
            grad_log_probs_tensor.owns_grad_fn = false;
            grad_log_probs_tensor.stream = stream;
            
            {
                char inputs_buf[128];
                snprintf(inputs_buf, sizeof(inputs_buf), "grad_log_probs.data=%p", 
                         (void*)grad_log_probs_buffer);
                EQ_LOG_HOST("NLL-TO-LOGSOFTMAX", 
                    "chaining grad_log_probs to LogSoftmaxGradFn → produces grad_logits",
                    inputs_buf, "calling upstream->apply()", "non-null ptr", inputs_buf,
                    0, 0, 0, EquationPhase::LOSS_BACKWARD);
            }
            
            AG_TRACE("[NLLLossGradFn::apply] Chaining to LogSoftmaxGradFn\n");
            log_probs_grad_fn->apply(grad_log_probs_tensor, stream);
            AG_TRACE("[NLLLossGradFn::apply] LogSoftmaxGradFn returned\n");
        } else {
            // Rule 20: upstream grad_fn is REQUIRED for backpropagation
            throw std::runtime_error("[NLLLossGradFn::apply] log_probs_grad_fn is NULL "
                "— LogSoftmaxGradFn was not attached. Cannot backpropagate.");
        }
        
        AG_TRACE("[NLLLossGradFn::apply] EXIT\n");
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
    // ══════════════════════════════════════════════════════════════════════
    // Architecture: logits → log_softmax() → log_probs → NLL loss → scalar
    //
    // This is the PyTorch gold standard: F.log_softmax + F.nll_loss
    // Softmax is computed ONCE (in log_softmax), saved in LogSoftmaxGradFn.
    // NLL backward produces grad w.r.t. log_probs, which chains through
    // LogSoftmaxGradFn to produce grad w.r.t. logits: (p - q) / N
    // ══════════════════════════════════════════════════════════════════════
    
    // ── Step 1: log_softmax(logits) → log_probs ──
    // Creates LogSoftmaxGradFn if logits.requires_grad
    Tensor log_probs = autograd::log_softmax(logits, stream);
    
    // ── Step 2: NLL loss forward on log_probs ──
    float* per_token_loss = nullptr;
    float* d_loss_sum = nullptr;
    int* d_valid_count = nullptr;
    
    cudaMalloc(&per_token_loss, num_tokens * sizeof(float));
    cudaMalloc(&d_loss_sum, sizeof(float));
    cudaMalloc(&d_valid_count, sizeof(int));
    
    launchUnifiedLossForward(
        log_probs.data,
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
    
    // ── Step 3: Copy results to host ──
    float h_loss_sum = 0.0f;
    int h_valid_count = 0;
    cudaMemcpyAsync(&h_loss_sum, d_loss_sum, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_valid_count, d_valid_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    if (h_valid_count <= 0) {
        throw std::runtime_error("[unified_loss] valid_count=0 — no valid tokens in batch. "
            "Check valid_mask and targets for corruption.");
    }
    const float mean_loss = h_loss_sum / static_cast<float>(h_valid_count);
    
    AG_TRACE("[unified_loss] loss_sum=%.6f valid_count=%d mean_loss=%.6f\n",
             h_loss_sum, h_valid_count, mean_loss);
    AG_TRACE("[unified_loss] config: focal_alpha=%.2f focal_gamma=%.2f smoothing=%.3f entropy_lambda=%.4f\n",
             config.focal_alpha, config.focal_gamma, config.smoothing_epsilon, config.entropy_reg_lambda);
    
    // ── Step 4: Create scalar loss tensor ──
    float* d_loss = nullptr;
    cudaMalloc(&d_loss, sizeof(float));
    // BUG FIX Issue #61: Use SYNC copy because mean_loss is a local variable!
    cudaMemcpy(d_loss, &mean_loss, sizeof(float), cudaMemcpyHostToDevice);
    
    Tensor loss;
    loss.data = d_loss;
    loss.owns_data = true;
    loss.shape = TensorContract::TensorShape::make_BSM(1, 1);
    loss.is_leaf = false;
    loss.requires_grad = logits.requires_grad;
    loss.stream = stream;
    
    // ── Step 5: Attach NLLLossGradFn ──
    if (logits.requires_grad) {
        auto* grad_fn = new NLLLossGradFn(
            log_probs.data,       // Takes ownership of log_probs GPU buffer
            log_probs.numel(),
            targets, valid_mask,
            num_tokens, vocab_size, h_valid_count,
            config.focal_alpha, config.focal_gamma,
            config.smoothing_epsilon, config.entropy_reg_lambda,
            log_probs.grad_fn,    // Takes ownership of LogSoftmaxGradFn
            log_probs.shape,
            stream
        );
        // Transfer ownership from Tensor to NLLLossGradFn
        log_probs.owns_data = false;     // NLLLossGradFn now owns log_probs.data
        log_probs.owns_grad_fn = false;  // NLLLossGradFn now owns LogSoftmaxGradFn
        
        loss.grad_fn = grad_fn;
        loss.owns_grad_fn = true;
    }
    
    // ── Step 6: Cleanup temporary buffers ──
    cudaFree(per_token_loss);
    cudaFree(d_loss_sum);
    cudaFree(d_valid_count);
    
    return loss;
    // log_probs goes out of scope: data NOT freed (transferred), grad_fn NOT deleted (transferred)
}

__host__ Tensor cross_entropy_loss(
    Tensor& logits,
    const int* targets,
    const float* valid_mask,
    int num_tokens,
    int vocab_size,
    cudaStream_t stream
) {
    // Plain CE: focal_gamma=0 (no focal), smoothing=0 (no smoothing), entropy=0
    LossConfig plain_ce;
    plain_ce.focal_alpha = 1.0f;
    plain_ce.focal_gamma = 0.0f;
    plain_ce.smoothing_epsilon = 0.0f;
    plain_ce.entropy_reg_lambda = 0.0f;
    
    return unified_loss(logits, targets, valid_mask, num_tokens, vocab_size, plain_ce, stream);
}

}  // namespace autograd
}  // namespace GRIM
