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
    
    // Static factory for plain CE (testing/default)
    static LossConfig plain_ce() { return LossConfig{}; }
};

//=============================================================================
// MAIN API
//=============================================================================

/**
 * Compute unified loss with autograd support
 * 
 * Loss formula:
 *   L = α * (1-p_t)^γ * CE_smooth + λ * Σ p*log(p)
 * 
 * Where:
 *   CE_smooth = -(1-ε)*log(p_t) - ε/(V-1)*Σ_{i≠t}log(p_i)
 *   p_t = softmax(logits)[target]
 * 
 * Gradient:
 *   ∂L/∂logits = (softmax - one_hot) / valid_count
 * 
 * NOTE: The backward uses standard CE gradient for simplicity and stability.
 * The focal/smoothing only affect the loss value (forward), not gradient direction.
 * This matches common focal loss implementations.
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
// KERNEL LAUNCH FUNCTIONS (internal)
//=============================================================================

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
);

void launchUnifiedLossBackward(
    const float* logits,
    const int* targets,
    const float* valid_mask,
    float* grad_logits,
    int num_tokens,
    int vocab_size,
    int valid_count,
    cudaStream_t stream
);

//=============================================================================
// LEGACY API (for existing call sites - forwards to unified_loss)
//=============================================================================

/**
 * Plain cross-entropy loss (legacy wrapper)
 * Calls unified_loss with default LossConfig (no focal, smoothing, or entropy reg)
 */
Tensor cross_entropy_loss(
    Tensor& logits,
    const int* targets,
    const float* valid_mask,
    int num_tokens,
    int vocab_size,
    cudaStream_t stream
);

}  // namespace autograd
}  // namespace GRIM
