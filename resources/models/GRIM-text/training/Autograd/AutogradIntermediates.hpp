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

#include "../../Shared/Forward/ModelForwardOutputs.hpp"
#include "../../Layers/ReasoningHead/reasoning_head_GPU.hpp"
#include "../../Layers/ExecutionBlock/execution_block_GPU.hpp"
#include "../../Layers/DecodeTimeSlotSelector/decode_time_slot_selector_GPU.hpp"

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
struct AutogradIntermediates : public Forward::ModelForwardOutputs {
    // Cat 1 (graph-owned, transient): the autograd Tensor wrapper for the LM
    // head output lives on the shared forward-owned base struct. Training owns
    // the broader forward+loss+backward window through this derived type.

    Tensor loss_tensor;                // Scalar loss driving backward
    bool exec_op_ce_added = false;                            // true iff an active op-path execution loss was added to loss_tensor
    bool exec_arg_ce_added = false;                           // true iff an active arg-path execution loss was added to loss_tensor
    bool exec_write_ce_added = false;                         // true iff active write selection CE was added to loss_tensor
    bool exec_transition_added = false;                       // true iff active transition/value execution loss was added to loss_tensor

    // Selector supervision — SelectorForwardResult owns intermediate Tensors
    // (q, slot_keys) whose .data is cached by MatMulGradFn nodes. MUST stay
    // alive from computeAutogradLoss() through executeAutogradBackward().
    // The selector input tensors below own copies of h_t and slot_features;
    // selector grad fns need those buffers for W_q/W_k gradients, so borrowed
    // views into ExecutionMemory/DecodeTimeNumPolicy are forbidden here.
    std::vector<Tensor> selector_h_t_inputs;
    std::vector<Tensor> selector_slot_feature_inputs;
    std::vector<SelectorForwardResult> selector_fwd_results;

    // NOTE (Rule 20 — Ownership Taxonomy): The cross-attention read-gate
    // accumulator (Category 3 workspace) and its host snapshot (Category 2
    // telemetry) live on TrainingState, NOT here. This struct is Category 1
    // (graph-owned, transient). See TrainingState::read_gate_accum_tensor /
    // h_read_gate_mean.

    // ═══════════════════════════════════════════════════════════════════════════
    // LIFECYCLE
    // ═══════════════════════════════════════════════════════════════════════════
    
    /** Clear all tensors. Call ONLY after backward completes. */
    void clear() {
        Forward::ModelForwardOutputs::clear();
        loss_tensor = Tensor();
        exec_op_ce_added = false;
        exec_arg_ce_added = false;
        exec_write_ce_added = false;
        exec_transition_added = false;
        selector_h_t_inputs.clear();
        selector_slot_feature_inputs.clear();
        selector_fwd_results.clear();
        // Rule 20 ownership taxonomy: this struct holds ONLY Category 1
        // (graph-owned, transient) state. There are no exception fields.
        // The cross-attention read-gate buffer/scalar live on TrainingState.
    }
    
    /** Check if intermediates are populated (forward has run) */
    bool hasLoss() const { return loss_tensor.data != nullptr; }
};

}  // namespace Autograd
}  // namespace GRIM
