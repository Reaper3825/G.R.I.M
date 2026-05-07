//======================================================//
//  AutogradLoss.cu
//  CUDA implementation of unified autograd-enabled loss
//  
//  Implements: Focal Loss + Label Smoothing + Cross Entropy + Entropy Regularization
//  This is the ONLY loss computation path for training.
//======================================================//

#include "AutogradLoss.hpp"
#include "CrossEntropyNLL.hpp"
#include "../../TensorContract/TensorContract_GPU.hpp"
// BatchPayload + BatchDeviceBindings are part of the public unified_loss boundary.
#include "../../LogRecorder/BatchLogTape.hpp"
#include "../../VerboseLogging.hpp"  // Compile-time diagnostic guards (Issue #151)
#include "../../CudaAllocUtils.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <sstream>
#include <cfloat>
#include <cmath>
#include <memory>

using GRIM::CudaAlloc::cudaMallocOrThrow;

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
// Cross-entropy / NLL kernels live in CrossEntropyNLL.cu.
// AutogradLoss.cu owns the autograd graph node and delegates CE math through
// computeCrossEntropyForwardFromLogProbs() / computeCrossEntropyBackwardToLogProbs().
//========================================================================

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
    const float* __restrict__ grad_logits,  // [num_tokens, vocab_size] - analytical gradient
    int num_tokens,
    int vocab_size,
    float inv_valid_count,                  // 1/N for mean reduction
    float focal_alpha,
    float focal_gamma,
    bool focal_enabled,
    float smoothing_epsilon,
    float entropy_reg_lambda,
    int sample_token_idx,                   // Which token position to verify
    int sample_vocab_idx                    // Which vocab position to verify
) {
    // Only thread 0 of block 0 runs this diagnostic
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    
    // Bounds check
    if (sample_token_idx >= num_tokens || sample_vocab_idx >= vocab_size) {
        printf("[FD_GRAD_VERIFY_EQUATION] ERROR: sample indices out of bounds (tok=%d/%d, vocab=%d/%d)\n",
               sample_token_idx, num_tokens, sample_vocab_idx, vocab_size);
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
    if (focal_enabled && focal_gamma > 0.0f) {
        focal_weight_plus = powf(1.0f - p_target_plus + kEpsilon_local, focal_gamma);
    }
    
    float loss_plus = focal_enabled
        ? (focal_alpha * focal_weight_plus * ce_smooth_plus + entropy_reg_lambda * neg_entropy_plus)
        : (ce_smooth_plus + entropy_reg_lambda * neg_entropy_plus);
    
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
    if (focal_enabled && focal_gamma > 0.0f) {
        focal_weight_minus = powf(1.0f - p_target_minus + kEpsilon_local, focal_gamma);
    }
    
    float loss_minus = focal_enabled
        ? (focal_alpha * focal_weight_minus * ce_smooth_minus + entropy_reg_lambda * neg_entropy_minus)
        : (ce_smooth_minus + entropy_reg_lambda * neg_entropy_minus);
    
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
 * This should be called after computeCrossEntropyBackwardToLogProbs() to verify the gradients
 * 
 * @param logits          Input logits [num_tokens, vocab_size]
 * @param targets         Target token IDs [num_tokens]
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
    const float* grad_logits,
    int num_tokens,
    int vocab_size,
    int valid_count,
    float focal_alpha,
    float focal_gamma,
    bool focal_enabled,
    float smoothing_eps,
    float entropy_lambda,
    int sample_token_idx,
    int sample_vocab_idx,
    cudaStream_t stream
) {
    // Validate inputs
    if (!logits || !targets || !grad_logits) {
        throw std::runtime_error("[launchFiniteDiffGradVerify] null input pointer: logits=" 
                                 + std::string(logits ? "OK" : "NULL") + " targets=" 
                                 + std::string(targets ? "OK" : "NULL") + " grad_logits=" 
                                 + std::string(grad_logits ? "OK" : "NULL"));
    }
    
    if (sample_token_idx < 0 || sample_token_idx >= num_tokens) {
        throw std::runtime_error("[launchFiniteDiffGradVerify] sample_token_idx=" 
                                 + std::to_string(sample_token_idx) + " out of range [0," 
                                 + std::to_string(num_tokens) + ")");
    }
    
    if (sample_vocab_idx < 0 || sample_vocab_idx >= vocab_size) {
        throw std::runtime_error("[launchFiniteDiffGradVerify] sample_vocab_idx=" 
                                 + std::to_string(sample_vocab_idx) + " out of range [0," 
                                 + std::to_string(vocab_size) + ")");
    }
    
    if (valid_count <= 0) {
        throw std::runtime_error("[launchFiniteDiffGradVerify] valid_count=" + std::to_string(valid_count) + " - must be > 0");
    }
    const float inv_valid_count = 1.0f / static_cast<float>(valid_count);
    
    // Launch single-thread diagnostic kernel
    kernelFiniteDiffGradVerify<<<1, 1, 0, stream>>>(
        logits,
        targets,
        grad_logits,
        num_tokens,
        vocab_size,
        inv_valid_count,
        focal_alpha,
        focal_gamma,
        focal_enabled,
        smoothing_eps,
        entropy_lambda,
        sample_token_idx,
        sample_vocab_idx
    );
    
    // Sync to ensure printf output is flushed
    cudaStreamSynchronize(stream);
}


//========================================================================
// [TOKEN277_DIAGNOSTIC_ACTUAL] Lightweight diagnostic using REAL loss computation
// Computes actual loss for Token 277 (SPACE) with focal/smoothing/entropy applied
// Logs via printf to avoid stack overflow from EquationLogging
//========================================================================

__global__ void kernelToken277DiagnosticActual(
    const float* __restrict__ log_probs,       // [num_tokens, vocab_size] from log_softmax
    const float* __restrict__ logits,          // [num_tokens, vocab_size] for reference
    const int* __restrict__ targets,           // [num_tokens]
    const float* __restrict__ grad_log_probs,  // [num_tokens, vocab_size] gradients
    int num_tokens,
    int vocab_size,
    float focal_alpha,
    float focal_gamma,
    float smoothing_epsilon,
    float entropy_reg_lambda,
    int batch_idx,
    int tracked_token  // Dynamically detected collapse token
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;
    if (token_idx % 50 != 0) return;  // Sample every 50th token to avoid spam
    if (threadIdx.x != 0) return;     // Only thread 0 logs
    
    const int target = targets[token_idx];
    if (target < 0 || target >= vocab_size) return;
    if (tracked_token < 0 || tracked_token >= vocab_size) return;
    
    const float* log_row = log_probs + static_cast<size_t>(token_idx) * vocab_size;
    const float* grad_row = grad_log_probs + static_cast<size_t>(token_idx) * vocab_size;
    
    // Compute softmax probabilities from log_probs for the tracked token
    const float log_p_tracked = log_row[tracked_token];
    const float p_tracked = expf(log_p_tracked);
    
    // Compute entropy for entropy regularization term
    float neg_entropy = 0.0f;
    if (entropy_reg_lambda > 0.0f) {
        for (int v = 0; v < vocab_size; v++) {
            const float p_v = expf(log_row[v]);
            if (p_v > 0.0f) neg_entropy += p_v * log_row[v];
        }
    }
    
    // Compute ACTUAL loss for tracked token target position
    float ce_loss = -log_p_tracked;  // Base CE
    
    // Label smoothing if enabled
    if (smoothing_epsilon > 0.0f && vocab_size > 1) {
        const float q_on = 1.0f - smoothing_epsilon;
        const float q_off = smoothing_epsilon / (vocab_size - 1.0f);
        float sum_log_off = 0.0f;
        for (int v = 0; v < vocab_size; v++) {
            if (v != tracked_token) sum_log_off += log_row[v];
        }
        ce_loss = -(q_on * log_p_tracked + q_off * sum_log_off);
    }
    
    // Focal loss if enabled
    float focal_weight = 1.0f;
    if (focal_gamma > 0.0f) {
        const float one_minus_pt = fmaxf(1.0f - p_tracked, 0.0f);
        if (one_minus_pt > 0.0f) {
            focal_weight = powf(one_minus_pt, focal_gamma);
        }
    }
    
    float total_loss = focal_alpha * focal_weight * ce_loss;
    
    // Entropy regularization (for this token)
    if (entropy_reg_lambda > 0.0f) {
        total_loss += entropy_reg_lambda * neg_entropy;
    }
    
    // Gradient info
    const float grad_tracked = grad_row[tracked_token];
    const bool is_tracked_target = (target == tracked_token);
    
    // Log in simple format (no stack overflow)
    printf("[CollapseTokenDiagnostic] batch=%d token_idx=%d target=%d "
           "tracked=%d is_tracked_target=%d log_p=%.6f p=%.6f loss=%.6f "
           "focal_w=%.4f entropy=%.4f grad=%.8f\n",
           batch_idx, token_idx, target, tracked_token,
           is_tracked_target, log_p_tracked, p_tracked, total_loss,
           focal_weight, neg_entropy, grad_tracked);
}

/**
 * Launch collapse token diagnostic with ACTUAL loss computation
 * This computes the real loss (focal + smoothing + entropy) per-token
 * to help identify mode collapse and gradient issues for the tracked token
 */
void launchToken277DiagnosticActual(
    const float* log_probs,
    const float* logits,
    const int* targets,
    const float* grad_log_probs,
    int num_tokens,
    int vocab_size,
    float focal_alpha,
    float focal_gamma,
    float smoothing_epsilon,
    float entropy_reg_lambda,
    int batch_idx,
    int tracked_token,
    cudaStream_t stream
) {
    if (tracked_token < 0) return;  // No collapse token detected yet
    const int num_blocks = (num_tokens + 255) / 256;
    kernelToken277DiagnosticActual<<<num_blocks, 256, 0, stream>>>(
        log_probs, logits, targets, grad_log_probs,
        num_tokens, vocab_size,
        focal_alpha, focal_gamma, smoothing_epsilon, entropy_reg_lambda,
        batch_idx, tracked_token
    );
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
 *   1. computeCrossEntropyBackwardToLogProbs() computes ∂L/∂(log_p_v) = -q_v * inv_N  (+ focal/entropy terms)
 *   2. LogSoftmaxGradFn computes ∂(log_p)/∂(logits) = δ_{ij} - p_j  (softmax Jacobian)
 *   3. Composition:  grad_logits[j] = (p_j - q_j) / N  ← standard CE gradient ✓
 *
 * Memory management:
 *   - Takes ownership of log_probs.data (no copy — old Tensor gives up owns_data)
 *   - Shares LogSoftmaxGradFn via shared_ptr
 *   - Allocates grad_log_probs_buffer for backward output
 */
struct NLLLossGradFn : public GradFn {
    // Saved forward data
    float* log_probs_data;          // OWNED GPU buffer: log-probabilities [num_tokens * vocab_size]
    float* grad_log_probs_buffer;   // OWNED GPU buffer: backward output [num_tokens * vocab_size]
    
    const int* targets;             // NOT OWNED — stable for batch lifetime
    int num_tokens;
    int vocab_size;
    int valid_count;
    
    // Loss configuration
    float focal_alpha;
    float focal_gamma;
    bool focal_enabled;
    float smoothing_epsilon;
    bool smoothing_enabled;
    float entropy_reg_lambda;
    bool entropy_reg_enabled;
    
    // Class-balanced loss: per-token weight = 1/freq(target)^β
    const float* class_weights;     // NOT OWNED — points to TrainingState::class_weights_tensor.data
    float weight_sum;               // Sum of per-token class weights for this batch
    
    // Upstream gradient chain
    std::shared_ptr<GradFn> log_probs_grad_fn;
    TensorContract::TensorShape grad_shape;
    
    cudaStream_t async_stream;
    cudaEvent_t cleanup_event;
    
    __host__ NLLLossGradFn(
        float* log_probs,           // Takes ownership of this GPU buffer
        const int* targets_,
        int num_tokens_, int vocab_size_, int valid_count_,
        float focal_alpha_, float focal_gamma_, bool focal_enabled_,
        float smoothing_epsilon_, bool smoothing_enabled_,
        float entropy_reg_lambda_, bool entropy_reg_enabled_,
        const float* class_weights_, float weight_sum_,
        std::shared_ptr<GradFn> upstream_grad_fn,
        const TensorContract::TensorShape& shape,
        cudaStream_t stream_
    )
        : log_probs_data(log_probs)
        , grad_log_probs_buffer(nullptr)
        , targets(targets_)
        , num_tokens(num_tokens_), vocab_size(vocab_size_)
        , valid_count(valid_count_)
        , focal_alpha(focal_alpha_), focal_gamma(focal_gamma_), focal_enabled(focal_enabled_)
        , smoothing_epsilon(smoothing_epsilon_), smoothing_enabled(smoothing_enabled_)
        , entropy_reg_lambda(entropy_reg_lambda_), entropy_reg_enabled(entropy_reg_enabled_)
        , class_weights(class_weights_), weight_sum(weight_sum_)
        , log_probs_grad_fn(std::move(upstream_grad_fn))
        , grad_shape(shape)
        , async_stream(stream_), cleanup_event(nullptr)
    {
        op_name = "nll_loss";
        cudaEventCreate(&cleanup_event);
        
        // OOM FIX: grad_log_probs_buffer allocation DEFERRED to apply().
        // This saves 1.37GB during forward pass — the buffer is only needed
        // during backward when NLL gradients are computed.
    }
    
    __host__ ~NLLLossGradFn() {
        log_probs_grad_fn.reset();  // Release shared_ptr to upstream LogSoftmaxGradFn
        
        if (cleanup_event) {
            cudaEventSynchronize(cleanup_event);
            cudaEventDestroy(cleanup_event);
        }
        if (log_probs_data) { cudaFree(log_probs_data); log_probs_data = nullptr; }
        if (grad_log_probs_buffer) { cudaFree(grad_log_probs_buffer); grad_log_probs_buffer = nullptr; }
    }
    
    __host__ void apply(const Tensor& grad_output, cudaStream_t stream) override {
        AG_TRACE("[NLLLossGradFn::apply] ENTER: log_probs_data=%p upstream=%p\n",
                 (void*)log_probs_data, (void*)log_probs_grad_fn.get());
        
        // ── Read upstream scalar gradient for chain rule ──
        // Loss is scalar, so grad_output is a single float. When this loss
        // is the terminal objective seeded with 1.0, grad_scale == 1.0.
        // When composed with other losses or rescaled, grad_scale carries
        // the upstream derivative so magnitude accounting stays correct.
        //
        // CRITICAL: Must use the SAME stream for the D2H copy!
        // Tensor::backward() writes grad_output via cudaMemcpyAsync on the
        // training stream, which uses cudaStreamNonBlocking. Plain cudaMemcpy
        // (NULL stream) does NOT synchronize with non-blocking streams,
        // causing a data race that reads uninitialized GPU memory.
        float grad_scale = 1.0f;
        if (grad_output.data) {
            cudaMemcpyAsync(&grad_scale, grad_output.data, sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            if (!std::isfinite(grad_scale)) {
                throw std::runtime_error("[NLLLossGradFn::apply] grad_output is non-finite ("
                    + std::to_string(grad_scale) + ") — upstream gradient is corrupt");
            }
        }
        AG_TRACE("[NLLLossGradFn::apply] grad_output_scale=%.6f\n", grad_scale);
        
        // OOM FIX: Lazy allocation — only allocate grad buffer when actually
        // needed for backward, not during forward pass. Saves 1.37GB peak memory.
        if (!grad_log_probs_buffer) {
            const size_t grad_bytes = static_cast<size_t>(num_tokens) * vocab_size * sizeof(float);
            cudaMallocOrThrow(reinterpret_cast<void**>(&grad_log_probs_buffer), grad_bytes, "NLLLossGradFn_grad_log_probs");
        }
        
        // ── Step 1: Compute CE/NLL backward → gradient w.r.t. log_probs ──
        // grad_scale is folded into the mean-reduction denominator in the CE module.
        LossConfig loss_config;
        loss_config.focal_alpha = focal_alpha;
        loss_config.focal_gamma = focal_gamma;
        loss_config.focal_enabled = focal_enabled;
        loss_config.smoothing_epsilon = smoothing_epsilon;
        loss_config.smoothing_enabled = smoothing_enabled;
        loss_config.entropy_reg_lambda = entropy_reg_lambda;
        loss_config.entropy_reg_enabled = entropy_reg_enabled;
        loss_config.d_class_weights = class_weights;
        loss_config.class_balanced_enabled = (class_weights != nullptr);
        computeCrossEntropyBackwardToLogProbs(
            log_probs_data, targets, grad_log_probs_buffer,
            num_tokens, vocab_size, valid_count,
            weight_sum,
            loss_config,
            grad_scale,
            stream
        );
        
        // Issue #152: Removed cudaStreamSynchronize error-check here.
        // Same-stream kernel ordering guarantees grad_log_probs_buffer is written
        // before the next kernel (LogSoftmaxGradFn::apply) reads it.
        // Errors will surface at the next natural sync point (loss D2H readback).
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("[NLLLossGradFn] CUDA error after NLL backward launch: ") + cudaGetErrorString(err));
        }
        
        // ── Step 2: Diagnostic — sample grad_log_probs at TARGET columns of valid tokens ──
        // FIX: Previous code sampled a contiguous block starting at token 50.
        // Problems: (a) token 50 might be masked (kernel writes 0.0f for masked rows),
        // (b) with plain CE (no smoothing), only grad[target] is non-zero — a contiguous
        // block mostly reads the 50,375 zero entries. Now we read the TARGET column of
        // each sampled valid token, which is where the actual gradient lives.
        if constexpr (GRIM::VerboseLogging::ENABLE_LOSS_BACKWARD_SAMPLING) {
            // Copy targets to host to find valid positions
            const int sample_max = std::min(200, num_tokens);
            std::vector<int> h_targets(sample_max);
            cudaMemcpy(h_targets.data(), targets, sample_max * sizeof(int), cudaMemcpyDeviceToHost);
            
            float mx = 0.0f; double sq = 0.0; int valid_sampled = 0;
            for (int t = 0; t < sample_max; ++t) {
                if (h_targets[t] < 0 || h_targets[t] >= vocab_size) continue;  // skip masked
                // Read grad_log_probs[t, target[t]] — the one non-zero entry per valid row
                const size_t elem_offset = static_cast<size_t>(t) * vocab_size + h_targets[t];
                float val = 0.0f;
                cudaMemcpy(&val, grad_log_probs_buffer + elem_offset, sizeof(float), cudaMemcpyDeviceToHost);
                if (!std::isnan(val) && !std::isinf(val)) {
                    mx = std::max(mx, std::abs(val));
                    sq += static_cast<double>(val) * val;
                    valid_sampled++;
                }
            }
            float rms = (valid_sampled > 0) ? std::sqrt(static_cast<float>(sq / valid_sampled)) : 0.0f;
            
            // Expected: with plain CE, grad_log_probs[t, target[t]] = -1/N
            // With smoothing: grad_log_probs[t, target[t]] = -(1-epsilon)/N
            const float expected_target_grad = smoothing_enabled
                ? (1.0f - smoothing_epsilon) / valid_count
                : 1.0f / valid_count;
            
            std::ostringstream eq;
            eq << "[NLL-BWD-OUT] grad_log_probs = dL/d(log_p) [NLL backward, before LogSoftmaxGradFn]\n"
               << "  INPUTS: num_tokens=" << num_tokens << " vocab=" << vocab_size << " valid=" << valid_count << "\n"
               << "  EXPECTED |grad[t,target[t]]|=1/N=" << expected_target_grad << "\n"
               << "  ACTUAL max=" << mx << " rms=" << rms << " (sampled " << valid_sampled << " valid target entries)";
            EQ_LOG(GRIM::Logging::getGlobalTape(), GRIM::Logging::LogGroup::Loss, GRIM::Logging::LogPhase::LOSS_BACKWARD, -1, "NLL-BWD-OUT", eq.str().c_str());
        } // if constexpr ENABLE_LOSS_BACKWARD_SAMPLING (NLL-BWD-OUT)
        
        // ── Step 3: Chain to LogSoftmaxGradFn → computes grad w.r.t. raw logits ──
        // LogSoftmaxGradFn::apply() does:
        //   grad_logits[i] = grad_log_p[i] - exp(log_p[i]) * Σ_j grad_log_p[j]
        // Then chains to logits.grad_fn (MatMulGradFn from LM head)
        if (log_probs_grad_fn) {
            Tensor grad_log_probs_tensor;
            grad_log_probs_tensor.data = grad_log_probs_buffer;
            grad_log_probs_tensor.shape = grad_shape;
            grad_log_probs_tensor.owns_data = false;
            grad_log_probs_tensor.stream = stream;
            
            {
                std::ostringstream eq;
                eq << "[NLL-TO-LOGSOFTMAX] chaining grad_log_probs to LogSoftmaxGradFn -> grad_logits\n"
                   << "  grad_log_probs.data=" << (void*)grad_log_probs_buffer;
                EQ_LOG(GRIM::Logging::getGlobalTape(), GRIM::Logging::LogGroup::Loss, GRIM::Logging::LogPhase::LOSS_BACKWARD, -1, "NLL-TO-LOGSOFTMAX", eq.str().c_str());
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
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    const LossConfig& config,
    cudaStream_t stream
) {
    payload.validate("unified_loss");
    if (!bindings.d_target_ids) {
        throw std::runtime_error("[unified_loss] BatchDeviceBindings.d_target_ids is NULL — caller MUST upload BatchPayload before loss");
    }
    if (bindings.batch_size != payload.batch_size) {
        throw std::runtime_error("[unified_loss] bindings.batch_size=" + std::to_string(bindings.batch_size) +
            " != payload.batch_size=" + std::to_string(payload.batch_size));
    }
    if (bindings.max_seq_len != payload.max_seq_len) {
        throw std::runtime_error("[unified_loss] bindings.max_seq_len=" + std::to_string(bindings.max_seq_len) +
            " != payload.max_seq_len=" + std::to_string(payload.max_seq_len));
    }

    return unified_loss_from_target_buffer(
        logits,
        bindings.d_target_ids,
        payload.total_tokens,
        payload.vocab_size,
        config,
        stream
    );
}

__host__ Tensor unified_loss_from_target_buffer(
    Tensor& logits,
    const int* targets,
    int num_tokens,
    int vocab_size,
    const LossConfig& config,
    cudaStream_t stream
) {
    // ══════════════════════════════════════════════════════════════════════
    // Rule 20: FAIL LOUD on invalid data
    //
    // Callers pass explicit (num_tokens, vocab_size) — no BatchPayload coupling.
    //
    // Validation CONTRACT (caller MUST ensure):
    //   - targets[i] ∈ {-1} ∪ [0, vocab_size)  where -1 = masked position
    //   - Padding handled by target == -1 (no separate valid_mask needed)
    //
    // RATIONALE: Full validation would require GPU->CPU memcpy for all tokens
    // (slow and unnecessary). Data validation belongs in the data loader,
    // not the loss computation. If caller passes corrupt data, GPU kernel
    // will exhibit undefined behavior (which is appropriate for a bug in
    // the upstream data pipeline).
    // ══════════════════════════════════════════════════════════════════════
    
    if (!targets) {
        throw std::runtime_error("[unified_loss] targets pointer is NULL — caller MUST provide valid target token IDs");
    }
    if (num_tokens <= 0) {
        throw std::runtime_error("[unified_loss] num_tokens=" + std::to_string(num_tokens) + " — must be > 0");
    }
    if (vocab_size <= 0) {
        throw std::runtime_error("[unified_loss] vocab_size=" + std::to_string(vocab_size) + " — must be > 0");
    }

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
    // OOM FIX: Pass save_output_copy=false so LogSoftmaxGradFn stores a
    // non-owning pointer to result.data instead of copying 1.37GB.
    // This is safe because NLLLossGradFn takes ownership of result.data
    // and keeps it alive through backward. Lifecycle:
    //   NLLLossGradFn::apply() → calls LogSoftmaxGradFn::apply() (reads shared data) → returns
    //   NLLLossGradFn::~DTOR() → deletes LogSoftmaxGradFn (doesn't free shared data) → frees log_probs_data
    Tensor log_probs = autograd::log_softmax(logits, stream, false);
    
    // ── Step 2: CE/NLL loss forward on log_probs ──
    const CrossEntropyForwardResult ce_result = computeCrossEntropyForwardFromLogProbs(
        log_probs.data,
        targets,
        num_tokens,
        vocab_size,
        config,
        stream
    );
    const float mean_loss = ce_result.mean_loss;
    const int h_valid_count = ce_result.valid_count;
    const float h_weight_sum = ce_result.weight_sum;
    
    AG_TRACE("[unified_loss] valid_count=%d mean_loss=%.6f\n",
             h_valid_count, mean_loss);
    if (config.d_class_weights) {
        AG_TRACE("[unified_loss] class_balanced: weight_sum=%.2f effective_N=%.2f (vs raw N=%d)\n",
                 h_weight_sum, h_weight_sum, h_valid_count);
    }
    AG_TRACE("[unified_loss] config: focal_alpha=%.2f focal_gamma=%.2f smoothing=%.3f entropy_lambda=%.4f\n",
             config.focal_alpha, config.focal_gamma, config.smoothing_epsilon, config.entropy_reg_lambda);
    
    // ── Step 4: Create scalar loss tensor ──
    float* d_loss = nullptr;
    cudaMallocOrThrow(reinterpret_cast<void**>(&d_loss), sizeof(float), "unified_loss_d_loss");
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
        auto grad_fn = std::make_shared<NLLLossGradFn>(
            log_probs.data,       // Takes ownership of log_probs GPU buffer
            targets,
            num_tokens, vocab_size, h_valid_count,
            config.focal_alpha, config.focal_gamma, config.focal_enabled,
            config.smoothing_epsilon, config.smoothing_enabled,
            config.entropy_reg_lambda, config.entropy_reg_enabled,
            config.d_class_weights, h_weight_sum,
            log_probs.grad_fn,    // Takes ownership of LogSoftmaxGradFn
            log_probs.shape,
            stream
        );
        // Transfer data ownership from Tensor to NLLLossGradFn
        log_probs.owns_data = false;     // NLLLossGradFn now owns log_probs.data
        
        loss.grad_fn = grad_fn;
    }
    
    return loss;
    // log_probs goes out of scope: data NOT freed (transferred), grad_fn NOT deleted (transferred)
}
}  // namespace autograd
}  // namespace GRIM
