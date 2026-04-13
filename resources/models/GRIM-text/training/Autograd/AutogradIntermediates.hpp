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
    std::vector<Tensor> encoder_layer_outputs;  // One per encoder layer
    Tensor encoder_output_tensor;      // [total_tokens, d_model] - after final RMSNorm
    Tensor centered_encoder_output;    // [total_tokens, d_model] - Issue #127
    Tensor logits_tensor;              // [total_tokens, vocab_size] - autograd wrapper
    Tensor numeric_head_output;        // [total_tokens, 2] - (log_magnitude, sign_logit)
    Tensor loss_tensor;                // Scalar loss driving backward
    std::vector<Tensor> mtp_logits_tensors;  // MTP head logits (one per k) — kept alive for backward
    std::vector<Tensor> mtp_shifted_targets_gpu;  // Per-head GPU target buffers — kept alive for NLLLossGradFn backward
    Tensor mtp_input_tensor;           // A1: MTP preprocessed input (RMSNorm only path) — kept alive for backward

    // ReasoningHead — canonical owner of per-forward atom embeddings + output
    Tensor scratch_atom_embeddings;    // [num_atoms, atom_embedding_dim] - copy-first from ScratchBlock
    ReasoningHeadOutput reasoning_output;  // op_logits, arg1_logits, arg2_logits

    // ExecutionBlock — per-row memory and per-(row,step) outputs
    std::vector<ExecutionMemory> exec_memories;               // [batch_size] per-row isolation
    std::vector<ExecutionBlockOutput> exec_outputs_per_row;   // [batch_size][K steps]

    // Selector supervision — SelectorForwardResult owns intermediate Tensors
    // (q, slot_keys) whose .data is cached by MatMulGradFn nodes. MUST stay
    // alive from computeAutogradLoss() through executeAutogradBackward().
    std::vector<SelectorForwardResult> selector_fwd_results;

    // ═══════════════════════════════════════════════════════════════════════════
    // INJECTION DIAGNOSTICS — accumulated during forward for telemetry
    // ═══════════════════════════════════════════════════════════════════════════
    float* d_read_gate_accum = nullptr;   // [2] device: [sum_of_gate_values, total_token_count]
    float  h_read_gate_mean  = 0.0f;     // host: mean gate after forward-pass readback

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
        mtp_logits_tensors.clear();
        mtp_shifted_targets_gpu.clear();
        mtp_input_tensor = Tensor();
        scratch_atom_embeddings = Tensor();
        reasoning_output = ReasoningHeadOutput{};
        exec_memories.clear();
        exec_outputs_per_row.clear();
        selector_fwd_results.clear();
        // Note: d_read_gate_accum is NOT freed here — it's a persistent buffer
        // owned by TrainingState, zeroed before each forward pass.
        h_read_gate_mean = 0.0f;
    }
    
    /** Check if intermediates are populated (forward has run) */
    bool hasLogits() const { return logits_tensor.data != nullptr; }
    bool hasLoss() const { return loss_tensor.data != nullptr; }
};

}  // namespace Autograd
}  // namespace GRIM
