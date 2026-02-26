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

namespace GRIM {
namespace autograd {

//=============================================================================
// LOSS CONFIGURATION
//=============================================================================

struct LossConfig {
    // Focal Loss: L = α * (1-p_t)^γ * CE
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
 *   L = α * (1-p_t)^γ * CE_smooth + λ * Σ p*log(p)
 * 
 * @param logits      [total_tokens, vocab_size] - raw logits from LM head
 * @param targets     [total_tokens] - target token IDs (on GPU), -1 = masked
 * @param valid_mask  [total_tokens] - 1.0 for valid, 0.0 for padding (optional)
 * @param num_tokens  Number of tokens
 * @param vocab_size  Vocabulary size
 * @param config      Loss configuration (focal, label smoothing, entropy reg)
 * @param stream      CUDA stream
 * @return Scalar loss tensor with grad_fn attached (if logits.requires_grad)
 */
Tensor unified_loss(
    Tensor& logits,
    const int* targets,
    const float* valid_mask,
    int num_tokens,
    int vocab_size,
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
    const float* valid_mask,
    float* per_token_loss,
    float* loss_sum,
    int* valid_count,
    int num_tokens,
    int vocab_size,
    float focal_alpha,
    float focal_gamma,
    bool focal_enabled,
    float smoothing_epsilon,
    bool smoothing_enabled,
    float entropy_reg_lambda,
    bool entropy_reg_enabled,
    cudaStream_t stream
);

/**
 * NLL loss backward — gradient w.r.t. log-probabilities
 * @param log_probs      [num_tokens, vocab_size] — saved log-probabilities
 * @param grad_log_probs [num_tokens, vocab_size] — OUTPUT: gradient w.r.t. log_probs
 * @param focal_enabled True if focal loss should be applied (rules out checking focal_gamma > 0)
 * @param smoothing_enabled True if label smoothing should be applied (rules out checking smoothing_epsilon > 0)
 * @param entropy_reg_enabled True if entropy regularization should be applied (rules out checking entropy_reg_lambda > 0)
 */
void launchUnifiedLossBackward(
    const float* log_probs,
    const int* targets,
    const float* valid_mask,
    float* grad_log_probs,
    int num_tokens,
    int vocab_size,
    int valid_count,
    float focal_alpha,
    float focal_gamma,
    bool focal_enabled,
    float smoothing_epsilon,
    bool smoothing_enabled,
    float entropy_reg_lambda,
    bool entropy_reg_enabled,
    cudaStream_t stream
);

//=============================================================================
// TOKEN 277 DIAGNOSTIC LOGGING (Rule 21 Equation-Based)
//=============================================================================

/**
 * Launch Token 277 diagnostic logging kernel
 * 
 * Tracks all components contributing to Token 277 (SPACE) becoming argmax:
 * - Softmax probability p(277)
 * - Argmax status (is 277 the predicted token?)
 * - Loss contribution when 277 is the target
 * 
 * @param logits       [num_tokens, vocab_size] - current logits
 * @param targets      [num_tokens] - target token IDs
 * @param valid_mask   [num_tokens] - mask for valid positions (optional)
 * @param grad_logits  [num_tokens, vocab_size] - gradients if available (optional)
 * @param num_tokens   Number of tokens
 * @param vocab_size   Vocabulary size
 * @param batch_idx    Current batch index
 * @param step_idx     Current training step
 * @param stream       CUDA stream
 */
void launchToken277DiagnosticActual(
    const float* logits,
    const int* targets,
    const float* valid_mask,
    const float* grad_logits,
    int num_tokens,
    int vocab_size,
    int batch_idx,
    int step_idx,
    cudaStream_t stream,
    int tracked_token
);

// Issue #142: cross_entropy_loss() DELETED (Rule 26: dead code).
// Was a thin wrapper calling unified_loss() with hardcoded plain CE config.
// All callers now use unified_loss() directly with real config from model.

}  // namespace autograd
}  // namespace GRIM
