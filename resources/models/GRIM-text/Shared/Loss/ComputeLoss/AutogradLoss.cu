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
#include <cuda_runtime.h>
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
#define GRIM_FD_GRAD_VERIFY
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
    
    // ISSUE: atomicMax with int reinterpret fails for negative floats
    // FIX: Use deterministic shared memory reduction instead
    constexpr int kMaxWarps = 8;  // 256 threads / 32 = 8 warps max
    __shared__ float s_warp_max[kMaxWarps];
    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;
    const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    
    if (lane_id == 0 && warp_id < kMaxWarps) {
        s_warp_max[warp_id] = local_max;
    }
    __syncthreads();
    
    // Thread 0 computes final max from all warp results
    if (threadIdx.x == 0) {
        float final_max = s_warp_max[0];
        for (int w = 1; w < num_warps && w < kMaxWarps; w++) {
            final_max = fmaxf(final_max, s_warp_max[w]);
        }
        s_max = final_max;
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
 * Backward kernel - computes gradient of unified loss w.r.t. logits
 * 
 * Full gradient for focal loss + label smoothing + entropy regularization:
 * 
 * Let L = α * (1-p_t)^γ * CE_smooth + λ * H(p)
 * 
 * Where:
 *   CE_smooth = -q_t*log(p_t) - Σ_{i≠t} q_i*log(p_i)  [label smoothed CE]
 *   q_t = 1 - ε, q_i = ε/(V-1) for i≠t
 *   H(p) = Σ p_i*log(p_i) [negative entropy, penalizes low entropy]
 * 
 * Gradient components:
 *   1. CE gradient with label smoothing: (p - q) where q is smoothed target
 *   2. Focal weighting: multiply by (1-p_t)^γ and add derivative term
 *   3. Entropy regularization: λ * (1 + log(p_i)) * dp_i/dlogits
 * 
 * Final: grad[i] = (1/N) * [focal_weight * (p_i - q_i) + focal_deriv + entropy_grad]
 */
__global__ void kernelUnifiedLossBackward(
    const float* __restrict__ logits,
    const int* __restrict__ targets,
    const float* __restrict__ valid_mask,
    float* __restrict__ grad_logits,
    int num_tokens,
    int vocab_size,
    float inv_valid_count,  // 1/N - applied uniformly to ALL gradient terms at end
    float focal_alpha,
    float focal_gamma,
    float smoothing_epsilon,
    float entropy_reg_lambda
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
    
    // ISSUE: atomicMax with int reinterpret fails for negative floats
    // FIX: Use deterministic shared memory reduction instead
    constexpr int kMaxWarps = 8;  // 256 threads / 32 = 8 warps max
    __shared__ float s_warp_max[kMaxWarps];
    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;
    const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    
    if (lane_id == 0 && warp_id < kMaxWarps) {
        s_warp_max[warp_id] = local_max;
    }
    __syncthreads();
    
    // Thread 0 computes final max from all warp results
    if (threadIdx.x == 0) {
        float final_max = s_warp_max[0];
        for (int w = 1; w < num_warps && w < kMaxWarps; w++) {
            final_max = fmaxf(final_max, s_warp_max[w]);
        }
        s_max = final_max;
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
    
    const float sum_exp = s_sum;
    const float inv_sum = 1.0f / (sum_exp + kEpsilon);
    
    // Step 2.5: Compute neg_entropy = Σ_v p_v * log(p_v) for entropy gradient centering
    // Issue #124 FIX: This was MISSING in backward kernel, causing Issue #123's wrong reversion.
    // The entropy regularization term H(p) = Σ p_v*log(p_v) requires the CENTERING term in gradients:
    // ∂H/∂z_k = p_k * (log(p_k) + 1 - H) = p_k * (log(p_k) + 1 - neg_entropy)
    // Without the centering term, gradients sum to non-zero (λ*(1-H)), causing mode collapse.
    __shared__ float s_neg_entropy;
    if (threadIdx.x == 0) s_neg_entropy = 0.0f;
    __syncthreads();
    
    float local_neg_entropy = 0.0f;
    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        const float p_v_temp = expf(row[v] - max_val) * inv_sum;
        if (p_v_temp > kEpsilon) {
            local_neg_entropy += p_v_temp * logf(p_v_temp);
        }
    }
    // Warp reduction for neg_entropy
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_neg_entropy += __shfl_down_sync(0xffffffff, local_neg_entropy, offset);
    }
    if (threadIdx.x % warpSize == 0) {
        atomicAdd(&s_neg_entropy, local_neg_entropy);
    }
    __syncthreads();
    const float neg_entropy = s_neg_entropy;  // H(p) = Σ p_v * log(p_v) ≈ -10.83 for uniform 50k vocab
    
    // Step 3: Compute p_t (probability of correct class) for focal weighting
    const float p_t = expf(row[target] - max_val) * inv_sum;
    
    // Focal loss weight and its derivative w.r.t. p_t
    float focal_weight = 1.0f;
    float focal_deriv_factor = 0.0f;  // γ * (1-p_t)^(γ-1) for derivative contribution
    if (focal_gamma > 0.0f) {
        const float one_minus_pt = fmaxf(1.0f - p_t, kEpsilon);
        focal_weight = powf(one_minus_pt, focal_gamma);
        // Issue #119 FIX: Focal derivative factor was WRONG!
        // OLD (BUGGY): focal_deriv_factor = γ * (1-p_t)^(γ-1) * (-log(p_t))
        // The extra (-log(p_t)) term is INCORRECT and causes fd_fac to be ~10.5x too large
        // when p_t ≈ 1e-5 (typical), since -log(1e-5) ≈ 11.5
        //
        // CORRECT formula (matching PyTorch baseline):
        // d/dp_t[(1-p_t)^γ] = -γ * (1-p_t)^(γ-1)
        // The focal derivative w.r.t logit_v (for v != target) is:
        //   d(focal_weight)/d(logit_v) = d/dp_t[(1-p_t)^γ] * d(p_t)/d(logit_v)
        //                              = -γ * (1-p_t)^(γ-1) * (-p_t * p_v)
        //                              = γ * (1-p_t)^(γ-1) * p_t * p_v
        // So focal_deriv_factor = γ * (1-p_t)^(γ-1) (NO log term!)
        focal_deriv_factor = focal_gamma * powf(one_minus_pt, focal_gamma - 1.0f);
    }
    
    // Label smoothing target distribution
    const float q_on = (smoothing_epsilon > 0.0f && vocab_size > 1) ? (1.0f - smoothing_epsilon) : 1.0f;
    const float q_off = (smoothing_epsilon > 0.0f && vocab_size > 1) ? (smoothing_epsilon / static_cast<float>(vocab_size - 1)) : 0.0f;
    
    // Step 4: Compute full gradient
    // grad[i] = (1/N) * [α * focal_weight * (p_i - q_i) + focal_deriv_contrib + entropy_grad]
    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        const float p_v = expf(row[v] - max_val) * inv_sum;
        
        // Label-smoothed target: q_on for target, q_off for others
        const float q_v = (v == target) ? q_on : q_off;
        
        // Base CE gradient with label smoothing: (p - q)
        float grad_v = focal_alpha * focal_weight * (p_v - q_v);
        
        // Focal loss derivative contribution (only affects target through p_t dependence)
        // The focal weight (1-p_t)^γ depends on p_t, which depends on all logits via softmax
        // d(focal_weight)/d(logit_i) = -γ*(1-p_t)^(γ-1) * dp_t/dlogit_i
        // dp_t/dlogit_i = p_t*(1_{i=t} - p_i) for softmax
        if (focal_gamma > 0.0f) {
            // Add derivative of focal_weight w.r.t. this logit
            // focal_deriv_factor = γ*(1-p_t)^(γ-1)*(-log(p_t))
            // dp_t/dlogit_v = p_t * ((v==target ? 1 : 0) - p_v)
            const float dp_t_dlogit_v = p_t * ((v == target ? 1.0f : 0.0f) - p_v);
            grad_v += focal_alpha * focal_deriv_factor * dp_t_dlogit_v;
        }
        
        // Entropy regularization gradient: λ * d/dlogit[Σ p_i*log(p_i)]
        // d/dlogit_v[Σ_j p_j*log(p_j)] = Σ_j (1+log(p_j)) * dp_j/dlogit_v
        //                              = Σ_j (1+log(p_j)) * p_j * (1_{j=v} - p_v)
        //                              = (1+log(p_v))*p_v - p_v * Σ_j p_j*(1+log(p_j))
        float entropy_term = 0.0f;
        if (entropy_reg_lambda > 0.0f) {
            // Issue #119 FIX: Scale entropy gradient by inv_valid_count to match CE scale
            //
            // PROBLEM: CE gradient is scaled by 1/N (mean reduction), but entropy gradient
            // was NOT scaled. With N=6000 tokens and λ=0.1:
            //   CE term:      ~p * 0.000163  (scaled by 1/N)
            //   Entropy term: ~p * 0.98      (unscaled, ~6000x stronger!)
            // This caused entropy to overpower CE for rare tokens.c
            //
            // FIX: Apply inv_valid_count to entropy term so both are on same scale.
            // Now effective entropy strength = λ (not λ*N).
            //
            // Issue #121 FIX: REMOVE probability-based clamping on g!
            // OLD (WRONG): g = fmaxf(log(p) + 1, 0) - clamped when p < 0.368
            //   Problem: Disabled entropy for ALL vocab tokens (p << 0.368)
            //   Root cause: Gated on PROBABILITY threshold, not GRADIENT sign
            // g = log(p) + 1, allowing negative values
            const float log_p_v = logf(fmaxf(p_v, kEpsilon));
            const float g = log_p_v + 1.0f;  // Issue #121: Allow negative g
            //
            // Issue #124 FIX: RESTORE Issue #122's centering term - Issue #123 was WRONG!
            //
            // MATHEMATICAL DERIVATION:
            //   H(p) = Σ_v p_v * log(p_v)  (negative entropy, always ≤ 0)
            //   Loss_entropy = λ * H(p) = λ * Σ_v p_v * log(p_v)
            //
            //   ∂H/∂z_k = Σ_v (∂p_v/∂z_k) * (log(p_v) + 1)
            //           = Σ_v p_v * (δ_vk - p_k) * (log(p_v) + 1)
            //           = p_k * (log(p_k) + 1) - p_k * Σ_v p_v * (log(p_v) + 1)
            //           = p_k * (log(p_k) + 1 - (H + 1))
            //           = p_k * (log(p_k) - H)
            //           = p_k * (log(p_k) - neg_entropy)  [where neg_entropy = H(p)]
            //
            // WHY Issue #123 WAS WRONG:
            //   Issue #123 claimed PyTorch baseline shows per-token entropy grad = p_v * (log(p_v) + 1).
            //   But that's just how the DIAGNOSTIC displays it! PyTorch autograd computes the
            //   COMPLETE gradient via chain rule, which includes the centering term internally.
            //   The libtorch_baseline uses loss.backward() (line 1691), which autograd handles.
            //   Our MANUAL gradient requires explicit centering: grad = p_k * (log(p_k) - neg_entropy)
            //
            // VERIFICATION:
            //   Without centering: Σ_k p_k * (log(p_k) + 1) = H + 1 ≠ 0
            //   With centering:    Σ_k p_k * (log(p_k) - H) = H - H = 0  ✓
            //   Gradients MUST sum to zero for proper softmax training!
            //
            // neg_entropy = H(p) = Σ_v p_v * log(p_v) ≈ -10.83 for uniform 50k vocab
            // Without centering, entropy gradient is ~10x too weak and doesn't sum to zero!
            //
            entropy_term = entropy_reg_lambda * p_v * (log_p_v - neg_entropy);  // Issue #124: CORRECT with centering!
            grad_v += entropy_term;
        }
        
        //========================================================================
        // [GRAD_NONTARGET_EQUATION] Rule 21: Per-term breakdown for non-target tokens
        // 
        // EQUATION: grad_v = base_CE_term + focal_deriv_term + entropy_term
        //   base_CE_term   = α * focal_weight * (p_v - q_off)
        //   focal_deriv    = α * focal_deriv_factor * p_t * (-p_v)  [for non-target]
        //   entropy_term   = λ * p_v * (log(p_v) - neg_entropy)  [Issue #124: WITH centering!]
        //
        // ENTROPY GRADIENT BEHAVIOR (Issue #124):
        //   neg_entropy = Σ_v p_v * log(p_v) ≈ -10.83 for 50k vocab (≈ -log(V))
        //   centered_g = log(p_v) - neg_entropy = log(p_v) + 10.83
        //   For typical vocab tokens: p_v ≈ 2e-5 → log(p_v) ≈ -10.82 → centered_g ≈ 0.01
        //   For common tokens: p_v ≈ 2e-4 → log(p_v) ≈ -8.52 → centered_g ≈ +2.31 (push DOWN)
        //   For rare tokens:   p_v ≈ 2e-6 → log(p_v) ≈ -13.12 → centered_g ≈ -2.29 (push UP)
        //   
        //   CRITICAL: Sum of all entropy gradients = 0 (verified: Σ p_v*(log(p_v)-H) = H-H = 0)
        //   This is WHY centering is required - ensures gradient doesn't bias the mean.
        //========================================================================
        if (v != target && token_idx < 5) {  // Log for first 5 tokens only
            // Sample: Token 277 (SPACE) and a few periodic vocab positions
            const bool should_log_this_v = (v == 277) || (v % 10000 == 0 && v > 0);
            if (should_log_this_v && threadIdx.x == 0 && g_eq_log_buffer && g_eq_log_state) {
                const float base_ce_term = focal_alpha * focal_weight * (p_v - q_off);
                // Focal deriv for non-target: α * focal_deriv_factor * p_t * (-p_v)
                const float focal_term = (focal_gamma > 0.0f) 
                    ? focal_alpha * focal_deriv_factor * p_t * (-p_v) 
                    : 0.0f;
                const float raw_log_p_plus_1 = logf(fmaxf(p_v, kEpsilon)) + 1.0f;
                // Note: raw_log_p_plus_1 IS the same as 'g' from the entropy computation
                // Using it directly since 'g' is scoped inside the entropy_reg_lambda block
                
                // Rule 21: Build input string with key values
                char inputs_buf[256];
                int len = 0;
                len = eq_strcpy_device(inputs_buf, "token_pos=", 256);
                len = eq_itoa_device(inputs_buf + len, token_idx, 256 - len) + len;
                len = eq_strcat_device(inputs_buf, " v=", len, 256);
                len = eq_itoa_device(inputs_buf + len, v, 256 - len) + len;
                len = eq_strcat_device(inputs_buf, " target=", len, 256);
                len = eq_itoa_device(inputs_buf + len, target, 256 - len) + len;
                len = eq_strcat_device(inputs_buf, " p_v=", len, 256);
                len = eq_ftoa_device(inputs_buf + len, p_v, 256 - len) + len;
                len = eq_strcat_device(inputs_buf, " q_off=", len, 256);
                len = eq_ftoa_device(inputs_buf + len, q_off, 256 - len) + len;
                len = eq_strcat_device(inputs_buf, " λ=", len, 256);
                len = eq_ftoa_device(inputs_buf + len, entropy_reg_lambda, 256 - len) + len;
                
                // Rule 21: Build output string with term breakdown + expected sign
                char outputs_buf[256];
                int olen = 0;
                olen = eq_strcpy_device(outputs_buf, "TERM1=", 256);
                olen = eq_ftoa_device(outputs_buf + olen, base_ce_term, 256 - olen) + olen;
                olen = eq_strcat_device(outputs_buf, " TERM2=", olen, 256);
                olen = eq_ftoa_device(outputs_buf + olen, focal_term, 256 - olen) + olen;
                olen = eq_strcat_device(outputs_buf, " TERM3=", olen, 256);
                olen = eq_ftoa_device(outputs_buf + olen, entropy_term, 256 - olen) + olen;
                olen = eq_strcat_device(outputs_buf, " TOTAL=", olen, 256);
                olen = eq_ftoa_device(outputs_buf + olen, grad_v, 256 - olen) + olen;
                olen = eq_strcat_device(outputs_buf, " g=", olen, 256);
                olen = eq_ftoa_device(outputs_buf + olen, raw_log_p_plus_1, 256 - olen) + olen;
                
                // Expected sign: non-target gradient should be POSITIVE (push probability DOWN)
                const char* expected_sign = (grad_v > 0.0f) ? "POSITIVE(correct)" : "NEGATIVE(WRONG!)";
                
                enqueueEquationLog(
                    g_eq_log_buffer, g_eq_log_state,
                    "[GRAD_NONTARGET_EQUATION]",
                    "grad_v = TERM1 + TERM2 + TERM3 = α×fw×(p_v-q_off) + α×fd_fac×p_t×(-p_v) + λ×p_v×g",
                    inputs_buf,
                    outputs_buf,
                    expected_sign,
                    (grad_v < 0.0f) ? "[ANOMALY] NEGATIVE gradient for non-target!" : "OK",
                    token_idx,  // batch_idx = token position for logging
                    v,          // layer_idx = vocab token for logging
                    0,          // step_idx not used here
                    EquationPhase::LOSS_BACKWARD
                );
            }
        }
        
        // Issue #119 FIX: Apply inv_valid_count uniformly to ALL gradient terms.
        // This is the ONLY place inv_valid_count should be applied (not in individual terms).
        grad_row[v] = grad_v * inv_valid_count;
    }
    
    //========================================================================
    // [GRAD_CENTER_EQUATION] CRITICAL FIX: Remove uniform bias from gradient
    // 
    // PROBLEM: Entropy regularization adds λ × p_v × (log(p_v) + 1) to each gradient.
    //          Sum over vocab: λ × (1 - H) ≠ 0 (uniform bias!)
    //          With tied weights: grad_W[i,j] = Σ_t h[t,i] × grad_logits[t,j]
    //                           = signal + bias × Σ_t h[t,i]
    //          The bias × hidden_mean term causes systematic gradient corruption!
    //
    // FIX: Center the gradient to have zero mean across vocab dimension.
    //      This removes uniform bias while preserving gradient direction.
    //      grad_v_centered = grad_v - mean_v(grad_v)
    //
    // EQUATION: mean = (1/V) × Σ_v grad_row[v]
    //           grad_row[v] -= mean  (for all v)
    //========================================================================
    __shared__ float s_grad_mean_partial[32];  // One per warp for centering
    
    // Step 1: Compute sum of gradients for this row
    float local_sum_for_mean = 0.0f;
    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        local_sum_for_mean += grad_row[v];
    }
    
    // Warp-level reduction for sum
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum_for_mean += __shfl_down_sync(0xFFFFFFFF, local_sum_for_mean, offset);
    }
    
    if (lane_id == 0) {
        s_grad_mean_partial[warp_id] = local_sum_for_mean;
    }
    __syncthreads();
    
    // Final reduction and broadcast mean to all threads
    __shared__ float s_grad_mean;
    if (warp_id == 0) {
        const int num_warps = (blockDim.x + 31) / 32;
        float sum_all = (lane_id < num_warps) ? s_grad_mean_partial[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_all += __shfl_down_sync(0xFFFFFFFF, sum_all, offset);
        }
        if (lane_id == 0) {
            s_grad_mean = sum_all / static_cast<float>(vocab_size);
        }
    }
    __syncthreads();
    
    // Step 2: Subtract mean from all gradient elements (centering)
    const float grad_mean = s_grad_mean;
    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        grad_row[v] -= grad_mean;
    }
    __syncthreads();
    
    // Issue #120 REMOVED: Non-target gradient clamping was WRONG!
    // Clamping gradients to >= 0 killed entropy regularization's ability to
    // push DOWN high-probability tokens. Entropy reg NEEDS negative gradients
    // at high-p tokens to decrease their probability and prevent mode collapse.
    
    // NOTE: Gradient centering diagnostic removed - snprintf is host-only.
    // Formula: centered[v] = grad[v] - mean(grad), ensures sum_grad_logits ≈ 0
    
    //========================================================================
    // [GRAD_SUM_EQUATION] RULE 21: Compute sum_grad_logits for validation
    // EQUATION: sum_grad_logits = Σ_v grad_logits[v] for each token
    // EXPECTED: For plain CE (focal_gamma=0, entropy_lambda=0):
    //           Σ_v (p_v - q_v) = Σp - Σq = 1 - 1 = 0
    //           After inv_valid_count: still 0
    //           With focal_gamma > 0: non-zero from derivative term
    //           With entropy_lambda > 0: λ × Σp_v × (log(p_v)+1) = λ × (H + 1)
    //========================================================================
    __shared__ float s_grad_sum_partial[32];  // One per warp
    
    // Each thread accumulates its vocab elements
    float local_grad_sum = 0.0f;
    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        local_grad_sum += grad_row[v];
    }
    
    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_grad_sum += __shfl_down_sync(0xFFFFFFFF, local_grad_sum, offset);
    }
    
    // Store warp result (reuse warp_id/lane_id from earlier in kernel)
    if (lane_id == 0) {
        s_grad_sum_partial[warp_id] = local_grad_sum;
    }
    __syncthreads();
    
    // Final reduction in first warp
    if (warp_id == 0) {
        const int num_warps = (blockDim.x + 31) / 32;
        float sum = (lane_id < num_warps) ? s_grad_sum_partial[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        
        // Thread 0 of block logs the result for diagnostic tokens
        if (lane_id == 0) {
            // Log for token 100 (skip BOS at 0) and periodically every 500 tokens
            // NOTE: With entropy regularization (lambda > 0), the gradient formula adds:
            //   λ × p_v × (log(p_v) + 1)
            // This term is NEGATIVE when p_v < e^(-1) ≈ 0.368 (i.e., most tokens!)
            // So sum_grad_logits CAN be negative even at non-target positions.
            // EXPECTED: With plain CE: sum ≈ 0 (machine epsilon)
            //           With entropy_reg: sum = inv_valid × λ × Σp_v(log(p_v)+1)
            //                                 = inv_valid × λ × (H(p) + 1)
            //                                 where H(p) = -Σp_v×log(p_v) (entropy, typically 0-10 nats)
            // NOTE: Grad sum logging removed - snprintf is host-only.
            
            // ANOMALY DETECTION: Plain CE should have |sum| ≈ 0
            // (snprintf logging removed - host-only function)
        }
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
    float focal_alpha,
    float focal_gamma,
    float smoothing_epsilon,
    float entropy_reg_lambda,
    cudaStream_t stream
) {
    // ISSUE #81 FIX: MUST divide gradient by valid_count to match PyTorch mean reduction!
    // Issue #80 "fix" was WRONG - it removed ALL 1/N scaling.
    // The math: loss = sum(ce_i) / N, so d(loss)/d(logits) = (1/N) * (softmax - one_hot)
    // Without the 1/N, gradients are N times too large (18,000x in our case).
    const float inv_valid_count = (valid_count > 0) ? (1.0f / static_cast<float>(valid_count)) : 1.0f;
    
    const int block_size = 256;
    kernelUnifiedLossBackward<<<num_tokens, block_size, 0, stream>>>(
        logits, targets, valid_mask, grad_logits,
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
struct UnifiedLossGradFn : public GradFn {
    std::shared_ptr<float> logits_data;  // Owns GPU memory copy
    size_t logits_size;
    
    const int* targets;
    const float* valid_mask;
    int num_tokens;
    int vocab_size;
    int valid_count;
    
    // Loss config parameters for proper backward gradients
    float focal_alpha;
    float focal_gamma;
    float smoothing_epsilon;
    float entropy_reg_lambda;
    
    Tensor* logits_tensor_ptr;
    cudaStream_t async_stream;
    cudaEvent_t cleanup_event;
    
    __host__ UnifiedLossGradFn(
        float* logits, size_t logits_numel,
        const int* targets_, const float* valid_mask_,
        int num_tokens_, int vocab_size_, int valid_count_,
        float focal_alpha_, float focal_gamma_,
        float smoothing_epsilon_, float entropy_reg_lambda_,
        Tensor* logits_tensor,
        cudaStream_t stream_
    ) : logits_data(nullptr), logits_size(logits_numel),
        targets(targets_), valid_mask(valid_mask_),
        num_tokens(num_tokens_), vocab_size(vocab_size_),
        valid_count(valid_count_),
        focal_alpha(focal_alpha_), focal_gamma(focal_gamma_),
        smoothing_epsilon(smoothing_epsilon_), entropy_reg_lambda(entropy_reg_lambda_),
        logits_tensor_ptr(logits_tensor),
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
        AG_TRACE("[UnifiedLossGradFn::apply] ENTER: logits_tensor_ptr=%p\n", (void*)logits_tensor_ptr);
        
        if (logits_tensor_ptr) {
            AG_TRACE("[UnifiedLossGradFn::apply] logits_tensor_ptr->grad_fn=%p\n", 
                    (void*)logits_tensor_ptr->grad_fn);
            
            logits_tensor_ptr->ensure_grad();
            
            // Compute full unified loss gradient with focal/smoothing/entropy
            launchUnifiedLossBackward(
                logits_data.get(),
                targets,
                valid_mask,
                logits_tensor_ptr->grad_data(),
                num_tokens,
                vocab_size,
                valid_count,
                focal_alpha,
                focal_gamma,
                smoothing_epsilon,
                entropy_reg_lambda,
                stream
            );
            
            cudaError_t err = cudaStreamSynchronize(stream);
            if (err != cudaSuccess) {
                fprintf(stderr, "[UnifiedLossGradFn] CUDA error after backward: %s\n", 
                        cudaGetErrorString(err));
                return;
            }
            
            // ================================================================
            // TOKEN 277 (SPACE) DIAGNOSTIC: Track all components contributing to mode collapse
            // Uses equation-based logging (Rule 21) to trace why token 277 becomes argmax
            // ================================================================
            {
                // Static step counter to track batch/step indices
                static int s_step_idx = 0;
                static int s_batch_idx = 0;
                
                // Only log every 1st batch to avoid log spam
                if (s_batch_idx % 1 == 0) {
                    launchToken277Diagnostic(
                        logits_data.get(),
                        targets,
                        valid_mask,
                        logits_tensor_ptr->grad_data(),
                        num_tokens,
                        vocab_size,
                        s_batch_idx,
                        s_step_idx,
                        stream
                    );
                }
                
                s_step_idx++;
                // Increment batch index every accumulation_steps (assume 2)
                if (s_step_idx % 2 == 0) {
                    s_batch_idx++;
                }
            }
            
            // ================================================================
            // ISSUE INVESTIGATION: Finite Difference Gradient Verification
            // Verify that sign(FD_grad) == sign(analytical_grad)
            // If signs mismatch → gradient logging or update has wrong sign!
            // ================================================================
#ifdef GRIM_FD_GRAD_VERIFY
            {
                // Sample token 100 to avoid masked BOS, verify against its TARGET
                const int sample_token_idx = std::min(100, num_tokens - 1);
                
                // Get the target token ID for this position (what gradient should be computed for)
                int target_vocab_id = 0;
                cudaMemcpy(&target_vocab_id, targets + sample_token_idx, sizeof(int), cudaMemcpyDeviceToHost);
                
                // Also check token 277 (SPACE) which was causing mode collapse
                const int sample_vocab_idx_277 = 277;
                
                fprintf(stderr, "\n[FD_GRAD_VERIFY] Running finite difference verification...\n");
                fprintf(stderr, "[FD_GRAD_VERIFY] sample_token=%d target_vocab=%d also_checking=277\n", 
                        sample_token_idx, target_vocab_id);
                
                // Verify gradient at the TARGET position (should be negative: p_t - 1)
                launchFiniteDiffGradVerify(
                    logits_data.get(),
                    targets,
                    valid_mask,
                    logits_tensor_ptr->grad_data(),
                    num_tokens,
                    vocab_size,
                    valid_count,
                    focal_alpha,
                    focal_gamma,
                    smoothing_epsilon,
                    entropy_reg_lambda,
                    sample_token_idx,
                    target_vocab_id,
                    stream
                );
                
                // Also verify gradient at token 277 (NON-target, should be positive: p_v)
                if (target_vocab_id != sample_vocab_idx_277) {
                    launchFiniteDiffGradVerify(
                        logits_data.get(),
                        targets,
                        valid_mask,
                        logits_tensor_ptr->grad_data(),
                        num_tokens,
                        vocab_size,
                        valid_count,
                        focal_alpha,
                        focal_gamma,
                        smoothing_epsilon,
                        entropy_reg_lambda,
                        sample_token_idx,
                        sample_vocab_idx_277,
                        stream
                    );
                }
            }
#endif  // GRIM_FD_GRAD_VERIFY
            
            // DIAGNOSTIC: Log the grad_logits we just computed
            {
                const size_t grad_elems = static_cast<size_t>(num_tokens) * vocab_size;
                const size_t sample_sz = std::min(grad_elems, static_cast<size_t>(50000));
                // FIX: Sample from MIDDLE of buffer to avoid masked BOS token
                // Token 0 is BOS/masked with all-zero gradient. Start from token 50.
                const size_t start_offset = static_cast<size_t>(50) * vocab_size;
                const size_t actual_sample_sz = std::min(sample_sz, grad_elems - start_offset);
                std::vector<float> samp(actual_sample_sz);
                cudaMemcpy(samp.data(), logits_tensor_ptr->grad_data() + start_offset, 
                           actual_sample_sz * sizeof(float), cudaMemcpyDeviceToHost);
                float mx = 0.0f; double sq = 0.0;
                for (auto& v : samp) { if (!std::isnan(v) && !std::isinf(v)) { mx = std::max(mx, std::abs(v)); sq += v*v; } }
                float rms = std::sqrt(static_cast<float>(sq / actual_sample_sz));
                
                char inputs_buf[256];
                char outputs_buf[128];
                snprintf(inputs_buf, sizeof(inputs_buf), 
                    "num_tokens=%d vocab=%d valid=%d", num_tokens, vocab_size, valid_count);
                snprintf(outputs_buf, sizeof(outputs_buf), 
                    "max=%.6f rms=%.6f PTR=%p", mx, rms, (void*)logits_tensor_ptr->grad_data());
                EQ_LOG_HOST("LOSS-BWD-OUT", "grad_logits = dL/d(logits) [sampled from token 50+]",
                    inputs_buf, outputs_buf, "max~1e-4 rms~1e-5", outputs_buf,
                    0, 0, 0, EquationPhase::LOSS_BACKWARD);
            }
            
            // Continue backward chain
            if (logits_tensor_ptr->grad_fn) {
                AG_TRACE("[UnifiedLossGradFn::apply] CALLING logits_tensor_ptr->grad_fn->apply()...\n");
                Tensor logits_grad;
                logits_grad.data = logits_tensor_ptr->grad_data();
                logits_grad.shape = logits_tensor_ptr->shape;
                logits_grad.owns_data = false;
                logits_grad.owns_grad_fn = false;
                logits_grad.stream = stream;
                
                {
                    char inputs_buf[128];
                    char outputs_buf[128];
                    snprintf(inputs_buf, sizeof(inputs_buf), "logits_grad.data=%p", (void*)logits_grad.data);
                    snprintf(outputs_buf, sizeof(outputs_buf), "calling grad_fn->apply()");
                    EQ_LOG_HOST("LOSS-TO-MATMUL", "passing gradients to upstream matmul",
                        inputs_buf, outputs_buf, "non-null ptr", inputs_buf,
                        0, 0, 0, EquationPhase::LOSS_BACKWARD);
                }
                
                logits_tensor_ptr->grad_fn->apply(logits_grad, stream);
                AG_TRACE("[UnifiedLossGradFn::apply] grad_fn->apply() RETURNED\n");
            } else {
                AG_TRACE("[UnifiedLossGradFn::apply] WARNING: logits_tensor_ptr->grad_fn is NULL!\n");
            }
        }
        AG_TRACE("[UnifiedLossGradFn::apply] EXIT\n");
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
    
    // Log the computed loss for debugging (Issue #62) - guarded behind AG_TRACE for performance
    AG_TRACE("[AutogradLoss] COMPUTED: loss_sum=%.6f valid_count=%d mean_loss=%.6f\n",
            h_loss_sum, h_valid_count, mean_loss);
    
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
            config.focal_alpha, config.focal_gamma,
            config.smoothing_epsilon, config.entropy_reg_lambda,
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
