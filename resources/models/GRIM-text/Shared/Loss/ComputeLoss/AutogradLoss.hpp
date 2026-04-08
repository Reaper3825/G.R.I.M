//======================================================//
//  AutogradLoss.hpp
//  Unified autograd-enabled loss: Focal + Label Smoothing + Cross Entropy + Entropy Reg
//
//  This is the ONLY loss computation path. 
//  UnifiedLoss_GPU.cu and ComputeLoss_GPU.cu are deprecated.
//======================================================//

#pragma once

#include "../../TensorContract/TensorContract_GPU.hpp"
#include <cuda_runtime.h>
#include <cstdint>

// Forward declaration — full definition in Batching/BatchPayload.hpp
namespace GRIM { namespace Batching { struct BatchPayload; } }

namespace GRIM {
namespace autograd {

//=============================================================================
// LOSS CONFIGURATION
//=============================================================================

struct LossConfig {
    // Focal Loss: L = α * (1-p_t)^γ * CE
    // When focal_enabled=false, focal_alpha and focal_gamma are IGNORED.
    // CE term becomes plain ce_smooth (no scaling by focal_alpha).
    float focal_alpha = 1.0f;       // Class balance weight (1.0 = none)
    float focal_gamma = 0.0f;       // Focusing parameter (0 = standard CE, 2 = strong focus)
    bool  focal_enabled = false;
    
    // Label Smoothing: CE_smooth = -(1-ε)*log(p_t) - ε/(V-1)*Σlog(p_i)
    float smoothing_epsilon = 0.0f; // Target smoothing (0 = hard targets)
    bool  smoothing_enabled = false;
    
    // Entropy Regularization: reg = -λ * H(p) = λ * Σ p*log(p)
    // Penalizes low entropy (overconfidence), encourages diversity
    float entropy_reg_lambda = 0.0f;
    bool  entropy_reg_enabled = false;
    
    // Class-balanced loss: per-token weight = 1/freq(target)^β
    // GPU array [vocab_size], nullptr = disabled (all weights 1.0)
    const float* d_class_weights = nullptr;
    bool  class_balanced_enabled = false;
    
    // Static factory for plain CE (testing/default)
    static LossConfig plain_ce() { return LossConfig{}; }
};

//=============================================================================
// MAIN API
//=============================================================================

/**
 * Compute unified loss with autograd support
 * 
 * Architecture (PyTorch gold standard):
 *   logits → autograd::log_softmax() → log_probs → NLL loss → scalar loss
 *
 * Internally composes:
 *   1. log_softmax(logits) — numerically stable log(softmax(x)) = x - logsumexp(x)
 *   2. NLL loss on log_probs — -log_probs[target] with focal/smoothing/entropy
 *
 * Backward chain:
 *   NLLLossGradFn → LogSoftmaxGradFn → upstream (MatMulGradFn)
 *   Produces: grad_logits[j] = (p_j - q_j) / N  (standard CE gradient)
 *
 * Loss formula:
 *   When focal_enabled:  L = α * (1-p_t)^γ * CE_smooth + λ * Σ p*log(p)
 *   When focal disabled: L = CE_smooth + λ * Σ p*log(p)
 *   (focal_alpha is ONLY applied when focal_enabled=true)
 * 
 * @param logits      [total_tokens, vocab_size] - raw logits from LM head
 * @param targets     [total_tokens] - target token IDs (on GPU), -1 = masked
 * @param payload     BatchPayload — single source of truth for total_tokens, vocab_size, valid_tokens
 * @param config      Loss configuration (focal, label smoothing, entropy reg)
 * @param stream      CUDA stream
 * @return Scalar loss tensor with grad_fn attached (if logits.requires_grad)
 */
Tensor unified_loss(
    Tensor& logits,
    const int* targets,
    const Batching::BatchPayload& payload,
    const LossConfig& config,
    cudaStream_t stream
);

//=============================================================================
// KERNEL LAUNCH FUNCTIONS (internal — operate on log_probs, NOT raw logits)
//=============================================================================

/**
 * NLL loss forward on log-probabilities (output of log_softmax)
 * @param log_probs [num_tokens, vocab_size] — log-probabilities from log_softmax()
 * @param focal_enabled True if focal loss should be applied (rules out checking focal_gamma > 0)
 * @param smoothing_enabled True if label smoothing should be applied (rules out checking smoothing_epsilon > 0)
 * @param entropy_reg_enabled True if entropy regularization should be applied (rules out checking entropy_reg_lambda > 0)
 */
void launchUnifiedLossForward(
    const float* log_probs,
    const int* targets,
    float* per_token_loss,
    float* loss_sum,
    int* valid_count,
    float* weight_sum,
    int num_tokens,
    int vocab_size,
    float focal_alpha,
    float focal_gamma,
    bool focal_enabled,
    float smoothing_epsilon,
    bool smoothing_enabled,
    float entropy_reg_lambda,
    bool entropy_reg_enabled,
    const float* class_weights,
    cudaStream_t stream
);

/**
 * NLL loss backward — gradient w.r.t. log-probabilities
 * @param log_probs      [num_tokens, vocab_size] — saved log-probabilities
 * @param grad_log_probs [num_tokens, vocab_size] — OUTPUT: gradient w.r.t. log_probs
 * @param grad_output_scale  Upstream scalar gradient for chain rule (1.0 when terminal loss)
 * @param focal_enabled True if focal loss should be applied (rules out checking focal_gamma > 0)
 * @param smoothing_enabled True if label smoothing should be applied (rules out checking smoothing_epsilon > 0)
 * @param entropy_reg_enabled True if entropy regularization should be applied (rules out checking entropy_reg_lambda > 0)
 */
void launchUnifiedLossBackward(
    const float* log_probs,
    const int* targets,
    float* grad_log_probs,
    int num_tokens,
    int vocab_size,
    int valid_count,
    float weight_sum,
    float focal_alpha,
    float focal_gamma,
    bool focal_enabled,
    float smoothing_epsilon,
    bool smoothing_enabled,
    float entropy_reg_lambda,
    bool entropy_reg_enabled,
    const float* class_weights,
    float grad_output_scale,
    cudaStream_t stream
);

//=============================================================================
// TOKEN 277 DIAGNOSTIC LOGGING (Rule 21 Equation-Based)
//=============================================================================

/**
 * Launch collapse token diagnostic with ACTUAL loss computation.
 * Computes real loss (focal + smoothing + entropy) per-token to identify
 * mode collapse and gradient issues for the tracked token.
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
);

/**
 * Build shifted target vector for MTP head k: out[t] = targets[t+shift] if in-bounds else -1.
 * @param targets      [total_tokens] device target IDs
 * @param shifted_out   [total_tokens] device output (caller-allocated)
 * @param total_tokens  total_tokens
 * @param seq_len       sequence length (positions per batch item)
 * @param shift         k+1 for head k (number of positions to shift)
 * @param stream        CUDA stream
 */
void launchShiftTargetsKernel(
    const int* targets,
    int* shifted_out,
    int total_tokens,
    int seq_len,
    int shift,
    cudaStream_t stream
);

/**
 * Compute MTP head accuracy: correct count and valid count (target != -1).
 * Caller copies d_correct and d_valid to host and computes acc = correct / valid.
 */
void launchMTPAccuracyKernel(
    const float* logits,
    const int* targets,
    int total_tokens,
    int vocab_size,
    int* d_correct,
    int* d_valid,
    cudaStream_t stream
);

// Issue #142: cross_entropy_loss() DELETED (Rule 26: dead code).
// Was a thin wrapper calling unified_loss() with hardcoded plain CE config.
// All callers now use unified_loss() directly with real config from model.

}  // namespace autograd
}  // namespace GRIM
