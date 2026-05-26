//======================================================//
//  ModelForwardOutputs.hpp
//
//  Mode-neutral live outputs owned by one shared forward call.
//  This is Category 1 graph-owned state: callers may retain it only for the
//  active forward/loss/backward or forward/sample window and must clear it at
//  the orchestration boundary.
//======================================================//

#pragma once

#ifdef USE_CUDA

#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../Shared/TensorContract/ForwardIntermediates.hpp"
#include "../../Layers/ReasoningHead/reasoning_head_GPU.hpp"
#include "../../Layers/ExecutionBlock/execution_block_GPU.hpp"

#include <vector>

namespace GRIM {
namespace Forward {

struct ModelForwardOutputs {
    // ═══════════════════════════════════════════════════════════════════════════
    // PER-LAYER INTERMEDIATES
    // ═══════════════════════════════════════════════════════════════════════════
    AllLayerIntermediates layer_intermediates;

    // ═══════════════════════════════════════════════════════════════════════════
    // CROSS-LAYER LIVE TENSORS
    // ═══════════════════════════════════════════════════════════════════════════
    Tensor embedding_tensor;
    Tensor embedding_structured_state;
    Tensor embedding_gate_concat;
    Tensor embedding_gate_logits;
    Tensor embedding_gate_values;
    Tensor embedding_gate_delta;
    std::vector<Tensor> encoder_layer_outputs;
    Tensor encoder_output_tensor;
    Tensor centered_encoder_output;
    Tensor logits_tensor;

    // Optional reasoning / execution forward-owned state.
    Tensor scratch_atom_embeddings;
    ReasoningHeadOutput reasoning_output;
    std::vector<ExecutionMemory> exec_memories;
    std::vector<ExecutionBlockOutput> exec_outputs_per_row;
    std::vector<Tensor> exec_expected_target_tensors;

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
        scratch_atom_embeddings = Tensor();
        reasoning_output = ReasoningHeadOutput{};
        exec_memories.clear();
        exec_outputs_per_row.clear();
        exec_expected_target_tensors.clear();
    }

    bool hasLogits() const { return logits_tensor.data != nullptr; }
};

}  // namespace Forward
}  // namespace GRIM

#endif  // USE_CUDA