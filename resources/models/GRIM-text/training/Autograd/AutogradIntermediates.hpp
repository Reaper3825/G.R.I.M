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
    Tensor embedding_structured_state; // [total_tokens, d_model] - all-token z for learned vector gate
    Tensor embedding_gate_concat;      // [total_tokens, 2*d_model] - [e; z]
    Tensor embedding_gate_logits;      // [total_tokens, d_model] - [e; z] @ Wg
    Tensor embedding_gate_values;      // [total_tokens, d_model] - sigmoid(logits)
    Tensor embedding_gate_delta;       // [total_tokens, d_model] - gate ⊙ z
    std::vector<Tensor> encoder_layer_outputs;  // One per encoder layer
    Tensor encoder_output_tensor;      // [total_tokens, d_model] - after final RMSNorm
    Tensor centered_encoder_output;    // [total_tokens, d_model] - Issue #127
    // Cat 1 (graph-owned, transient): the autograd Tensor wrapper for the LM
    // head output. Used INSIDE the active autograd/forward boundary (loss
    // assembly, backward, boundary-safe diagnostics). Any consumer that needs
    // logits must observe this tensor explicitly before AutogradStepScope
    // clears the boundary; there is no durable TrainingState logits snapshot.
    Tensor logits_tensor;              // [total_tokens, vocab_size] - autograd wrapper
    Tensor loss_tensor;                // Scalar loss driving backward
    std::vector<Tensor> mtp_logits_tensors;  // MTP head logits (one per k) — kept alive for backward
    Tensor mtp_input_tensor;           // A1: MTP preprocessed input (RMSNorm only path) — kept alive for backward

    // ReasoningHead — canonical owner of per-forward atom embeddings + output
    Tensor scratch_atom_embeddings;    // [num_atoms, atom_embedding_dim] - copy-first from ScratchBlock
    ReasoningHeadOutput reasoning_output;  // op_logits, arg1_logits, arg2_logits

    // ExecutionBlock — per-row memory and per-(row,step) outputs
    std::vector<ExecutionMemory> exec_memories;               // [batch_size] per-row isolation
    std::vector<ExecutionBlockOutput> exec_outputs_per_row;   // [batch_size][K steps]
    std::vector<Tensor> exec_expected_target_tensors;         // owned [1,1] teacher scalar tensors used while building ExecutionBlock grad nodes
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
        layer_intermediates.clear();
        embedding_tensor = Tensor();
        embedding_structured_state = Tensor();
        embedding_gate_concat = Tensor();
        embedding_gate_logits = Tensor();
        embedding_gate_values = Tensor();
        embedding_gate_delta = Tensor();
        encoder_layer_outputs.clear();
        encoder_output_tensor = Tensor();
        centered_encoder_output = Tensor();
        logits_tensor = Tensor();
        loss_tensor = Tensor();
        mtp_logits_tensors.clear();
        mtp_input_tensor = Tensor();
        scratch_atom_embeddings = Tensor();
        reasoning_output = ReasoningHeadOutput{};
        exec_memories.clear();
        exec_outputs_per_row.clear();
        exec_expected_target_tensors.clear();
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
    bool hasLogits() const { return logits_tensor.data != nullptr; }
    bool hasLoss() const { return loss_tensor.data != nullptr; }
};

}  // namespace Autograd
}  // namespace GRIM
