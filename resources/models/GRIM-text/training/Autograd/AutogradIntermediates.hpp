//======================================================//
//  AutogradIntermediates.hpp
//  
//  Owns ALL intermediate tensors during forward→backward cycle.
//  Stored in TrainingState. Persists from forward() through backward().
//  Cleared after backward to free GPU memory.
//  
//  Separated from AutogradTraining.hpp to avoid circular includes
//  (TrainingState_GPU.hpp←→AutogradTraining.hpp).
//  
//  Rule 20: This is the SINGLE owner of autograd intermediates.
//  No other struct should store these tensors.
//======================================================//

#pragma once

#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../Shared/TensorContract/ForwardIntermediates.hpp"

#include <vector>

namespace GRIM {
namespace Autograd {

/**
 * AutogradIntermediates - Owns ALL intermediate tensors during forward-backward cycle
 * 
 * Stored in TrainingState. Persists from forward() through backward().
 * Cleared after backward completes to free GPU memory.
 * 
 * This replaces the old pattern of storing intermediates in AutogradContext.
 * TrainingState owns the lifecycle, making ownership unambiguous.
 */
struct AutogradIntermediates {
    // ═══════════════════════════════════════════════════════════════════════════
    // PER-LAYER INTERMEDIATES (Issue #56: keeps autograd graph alive)
    // ═══════════════════════════════════════════════════════════════════════════
    AllLayerIntermediates layer_intermediates;
    
    // ═══════════════════════════════════════════════════════════════════════════
    // CROSS-LAYER INTERMEDIATE TENSORS
    // These preserve the autograd graph so backward() propagates through all ops
    // ═══════════════════════════════════════════════════════════════════════════
    Tensor embedding_tensor;           // [total_tokens, d_model] - embedding output
    std::vector<Tensor> encoder_layer_outputs;  // One per encoder layer
    Tensor encoder_output_tensor;      // [total_tokens, d_model] - after final RMSNorm
    Tensor centered_encoder_output;    // [total_tokens, d_model] - Issue #127
    Tensor logits_tensor;              // [total_tokens, vocab_size] - autograd wrapper
    Tensor numeric_head_output;        // [total_tokens, 2] - (log_magnitude, sign_logit)
    Tensor loss_tensor;                // Scalar loss driving backward
    
    // ═══════════════════════════════════════════════════════════════════════════
    // LIFECYCLE
    // ═══════════════════════════════════════════════════════════════════════════
    
    /** Clear all tensors. Call ONLY after backward completes. */
    void clear() {
        layer_intermediates.clear();
        embedding_tensor = Tensor();
        encoder_layer_outputs.clear();
        encoder_output_tensor = Tensor();
        centered_encoder_output = Tensor();
        logits_tensor = Tensor();
        numeric_head_output = Tensor();
        loss_tensor = Tensor();
    }
    
    /** Check if intermediates are populated (forward has run) */
    bool hasLogits() const { return logits_tensor.data != nullptr; }
    bool hasLoss() const { return loss_tensor.data != nullptr; }
};

}  // namespace Autograd
}  // namespace GRIM
