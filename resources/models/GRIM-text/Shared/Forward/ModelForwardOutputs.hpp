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
#include "../../Layers/ExecutionBlock/execution_block_GPU.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace GRIM {
namespace Forward {

// ═══════════════════════════════════════════════════════════════════════════
// EXECUTION-LAYER FORWARD OUTPUTS
// Category 1 graph-owned payloads retained on the shared forward sink.
// ExecutionBlockLayer populates them, but ModelForwardOutputs owns their
// lifetime alongside every other forward-pass retained tensor.
// ═══════════════════════════════════════════════════════════════════════════
struct ExecStepMetrics {
    float arg1_entropy   = 0.0f;
    float arg2_entropy   = 0.0f;
    float op_entropy     = 0.0f;
    float write_entropy  = 0.0f;
    float max_p_write    = 0.0f;
    int   div_clamp_count = 0;
    float op_distribution[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float inject_gate_value = 0.0f;  // sigmoid output of result injection gate
};

struct ExecutionRecord {
    int arg1_slot = -1;
    int arg2_slot = -1;
    int op_id = -1;
    float value_before_1 = 0.0f;
    float value_before_2 = 0.0f;
    float value_after = 0.0f;
};

struct ExecutionBlockStepOutput {
    Tensor p_arg1;      // [1, V_val] softmax over value slots [S..V-1] only (detached)
    Tensor p_arg2;      // [1, V_val]
    Tensor p_op;        // [1, num_ops]
    Tensor p_write;     // [1, V]
    Tensor v_out;       // [1, 1] scalar result (hard forward selection; soft backward path)
    Tensor result_emb;  // [1, d_model]
    Tensor state_before_values;  // [V, 1] M.values snapshot before this step
    Tensor state_before_valid;   // [V]    M.valid_mask snapshot before this step
    Tensor state_after_values;   // [V, 1] M.values snapshot after this step
    Tensor state_after_valid;    // [V]    M.valid_mask snapshot after this step
    ExecutionRecord record;      // filled when diag_out != nullptr (host copy at step sync)
    ExecStepMetrics metrics;     // populated when debug_mode is enabled

    // Retained live tensors for later loss assembly (Category 1 graph-owned)
    Tensor arg1_logits_tensor;     // [1, V_val] live logits for arg1 CE / REINFORCE
    Tensor arg2_logits_tensor;     // [1, V_val] live logits for arg2 CE / REINFORCE
    Tensor op_logits_tensor;       // [1, num_ops] live logits for op CE / div penalty
    Tensor write_logits_tensor;    // [1, V] live logits for write-slot CE
    Tensor v_out_tensor;           // [1, 1] live scalar result for transition/div-magnitude loss
    float selection_temperature = 0.0f;
    bool div_was_clamped = false;
};

struct ExecutionBlockOutput {
    std::vector<ExecutionBlockStepOutput> steps;
};

struct ModelForwardOutputs {
private:
    static int countGradFns(const std::vector<Tensor>& tensors) {
        int count = 0;
        for (const auto& tensor : tensors) {
            if (tensor.grad_fn) count++;
        }
        return count;
    }

    static void resetTensorVectorPreserveGeometry(std::vector<Tensor>& tensors) {
        for (auto& tensor : tensors) {
            tensor = Tensor();
        }
    }

    static void clearTensorVector(std::vector<Tensor>& tensors) {
        resetTensorVectorPreserveGeometry(tensors);
        tensors.clear();
    }

    static void resetExecutionMemoryVectorPreserveGeometry(std::vector<ExecutionMemory>& memories) {
        for (auto& memory : memories) {
            memory = ExecutionMemory();
        }
    }

    static void resetExecutionOutputVectorPreserveGeometry(std::vector<ExecutionBlockOutput>& outputs) {
        for (auto& output : outputs) {
            for (auto& step : output.steps) {
                step = ExecutionBlockStepOutput();
            }
            output.steps.clear();
        }
    }

    void requireConsistentLayerStorage(const char* caller) const {
        const size_t expected = ln1_out_per_layer.size();
        auto requireSize = [&](const std::vector<Tensor>& tensors, const char* name) {
            if (tensors.size() != expected) {
                throw std::runtime_error(std::string(caller) + ": per-layer tensor vector size mismatch for " +
                                         name + " expected=" + std::to_string(expected) +
                                         " actual=" + std::to_string(tensors.size()));
            }
        };

        requireSize(ln2_out_per_layer, "ln2_out_per_layer");
        requireSize(qkv_out_per_layer, "qkv_out_per_layer");
        requireSize(Q_bhsd_per_layer, "Q_bhsd_per_layer");
        requireSize(K_bhsd_per_layer, "K_bhsd_per_layer");
        requireSize(V_bhsd_per_layer, "V_bhsd_per_layer");
        requireSize(attn_out_bhsd_per_layer, "attn_out_bhsd_per_layer");
        requireSize(attn_out_per_layer, "attn_out_per_layer");
        requireSize(proj_out_per_layer, "proj_out_per_layer");
        requireSize(scaled_proj_per_layer, "scaled_proj_per_layer");
        requireSize(residual1_per_layer, "residual1_per_layer");
        requireSize(ffn_out_per_layer, "ffn_out_per_layer");
        requireSize(scaled_ffn_per_layer, "scaled_ffn_per_layer");
        requireSize(output_per_layer, "output_per_layer");
        requireSize(ffn_gate_out_per_layer, "ffn_gate_out_per_layer");
        requireSize(ffn_silu_out_per_layer, "ffn_silu_out_per_layer");
        requireSize(ffn_linear1_out_per_layer, "ffn_linear1_out_per_layer");
        requireSize(ffn_swiglu_out_per_layer, "ffn_swiglu_out_per_layer");
    }

public:

    // ═══════════════════════════════════════════════════════════════════════════
    // PER-LAYER RETAINED TENSORS
    // Stored directly on ModelForwardOutputs so this header declares only the
    // single forward-owned sink type.
    // ═══════════════════════════════════════════════════════════════════════════
    std::vector<Tensor> ln1_out_per_layer;
    std::vector<Tensor> ln2_out_per_layer;

    std::vector<Tensor> qkv_out_per_layer;

    std::vector<Tensor> Q_bhsd_per_layer;
    std::vector<Tensor> K_bhsd_per_layer;
    std::vector<Tensor> V_bhsd_per_layer;

    std::vector<Tensor> attn_out_bhsd_per_layer;
    std::vector<Tensor> attn_out_per_layer;

    std::vector<Tensor> proj_out_per_layer;
    std::vector<Tensor> scaled_proj_per_layer;

    std::vector<Tensor> residual1_per_layer;

    std::vector<Tensor> ffn_out_per_layer;
    std::vector<Tensor> scaled_ffn_per_layer;

    std::vector<Tensor> output_per_layer;

    std::vector<Tensor> ffn_gate_out_per_layer;
    std::vector<Tensor> ffn_silu_out_per_layer;
    std::vector<Tensor> ffn_linear1_out_per_layer;
    std::vector<Tensor> ffn_swiglu_out_per_layer;

    void reserveLayerOutputs(size_t num_layers) {
        ln1_out_per_layer.reserve(num_layers);
        ln2_out_per_layer.reserve(num_layers);
        qkv_out_per_layer.reserve(num_layers);
        Q_bhsd_per_layer.reserve(num_layers);
        K_bhsd_per_layer.reserve(num_layers);
        V_bhsd_per_layer.reserve(num_layers);
        attn_out_bhsd_per_layer.reserve(num_layers);
        attn_out_per_layer.reserve(num_layers);
        proj_out_per_layer.reserve(num_layers);
        scaled_proj_per_layer.reserve(num_layers);
        residual1_per_layer.reserve(num_layers);
        ffn_out_per_layer.reserve(num_layers);
        scaled_ffn_per_layer.reserve(num_layers);
        output_per_layer.reserve(num_layers);
        ffn_gate_out_per_layer.reserve(num_layers);
        ffn_silu_out_per_layer.reserve(num_layers);
        ffn_linear1_out_per_layer.reserve(num_layers);
        ffn_swiglu_out_per_layer.reserve(num_layers);
    }

    void pushLayerOutputs() {
        ln1_out_per_layer.emplace_back();
        ln2_out_per_layer.emplace_back();
        qkv_out_per_layer.emplace_back();
        Q_bhsd_per_layer.emplace_back();
        K_bhsd_per_layer.emplace_back();
        V_bhsd_per_layer.emplace_back();
        attn_out_bhsd_per_layer.emplace_back();
        attn_out_per_layer.emplace_back();
        proj_out_per_layer.emplace_back();
        scaled_proj_per_layer.emplace_back();
        residual1_per_layer.emplace_back();
        ffn_out_per_layer.emplace_back();
        scaled_ffn_per_layer.emplace_back();
        output_per_layer.emplace_back();
        ffn_gate_out_per_layer.emplace_back();
        ffn_silu_out_per_layer.emplace_back();
        ffn_linear1_out_per_layer.emplace_back();
        ffn_swiglu_out_per_layer.emplace_back();
        requireConsistentLayerStorage("ModelForwardOutputs::pushLayerOutputs");
    }

    size_t layerCount() const {
        requireConsistentLayerStorage("ModelForwardOutputs::layerCount");
        return ln1_out_per_layer.size();
    }

    void validateLayerIndex(size_t layer_idx, const char* caller) const {
        requireConsistentLayerStorage(caller);
        if (layer_idx >= ln1_out_per_layer.size()) {
            throw std::runtime_error(std::string(caller) + ": layer_idx=" + std::to_string(layer_idx) +
                                     " out of range for retained layer tensor count=" +
                                     std::to_string(ln1_out_per_layer.size()));
        }
    }

    void clearRetainedLayerOutputs() {
        clearTensorVector(ln1_out_per_layer);
        clearTensorVector(ln2_out_per_layer);
        clearTensorVector(qkv_out_per_layer);
        clearTensorVector(Q_bhsd_per_layer);
        clearTensorVector(K_bhsd_per_layer);
        clearTensorVector(V_bhsd_per_layer);
        clearTensorVector(attn_out_bhsd_per_layer);
        clearTensorVector(attn_out_per_layer);
        clearTensorVector(proj_out_per_layer);
        clearTensorVector(scaled_proj_per_layer);
        clearTensorVector(residual1_per_layer);
        clearTensorVector(ffn_out_per_layer);
        clearTensorVector(scaled_ffn_per_layer);
        clearTensorVector(output_per_layer);
        clearTensorVector(ffn_gate_out_per_layer);
        clearTensorVector(ffn_silu_out_per_layer);
        clearTensorVector(ffn_linear1_out_per_layer);
        clearTensorVector(ffn_swiglu_out_per_layer);
    }

    int totalGradFnCount() const {
        requireConsistentLayerStorage("ModelForwardOutputs::totalGradFnCount");

        int count = 0;
        count += countGradFns(ln1_out_per_layer);
        count += countGradFns(ln2_out_per_layer);
        count += countGradFns(qkv_out_per_layer);
        count += countGradFns(Q_bhsd_per_layer);
        count += countGradFns(K_bhsd_per_layer);
        count += countGradFns(V_bhsd_per_layer);
        count += countGradFns(attn_out_bhsd_per_layer);
        count += countGradFns(attn_out_per_layer);
        count += countGradFns(proj_out_per_layer);
        count += countGradFns(scaled_proj_per_layer);
        count += countGradFns(residual1_per_layer);
        count += countGradFns(ffn_out_per_layer);
        count += countGradFns(scaled_ffn_per_layer);
        count += countGradFns(output_per_layer);
        count += countGradFns(ffn_gate_out_per_layer);
        count += countGradFns(ffn_silu_out_per_layer);
        count += countGradFns(ffn_linear1_out_per_layer);
        count += countGradFns(ffn_swiglu_out_per_layer);
        return count;
    }

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
    Tensor lm_head_input_tensor;
    Tensor logits_tensor;
    std::vector<Tensor> mtp_logits_tensors;
    Tensor latent_preset_mtp_hidden;        // [total_tokens, mtp_k * d_model] predicted hidden trajectory (shared hidden-trajectory projection of h[t])
    Tensor latent_preset_future_fused;      // [total_tokens, fuse_dim]
    Tensor latent_preset_future_entropy;    // [total_tokens, mtp_k] optional
    Tensor latent_preset_z;                 // [total_tokens, preset_dim]
    Tensor latent_preset_vec;               // [total_tokens, d_model]
    Tensor latent_preset_gate_pre;          // [total_tokens, 1] scalar gate logits
    Tensor latent_preset_gate;              // [total_tokens, 1] scalar gate values
    Tensor latent_preset_injected;          // [total_tokens, d_model]
    Tensor latent_preset_h_enhanced;        // [total_tokens, d_model]
    // Arg/option selector head: [total_tokens, num_pool_atoms] selection logits over
    // the candidate atom-entry pool (out-of-row-window candidates masked to -inf).
    // Empty unless the forward graph policy requested emit_selector_logits.
    Tensor selector_logits;

    // Optional reasoning / execution forward-owned state.
    Tensor scratch_atom_embeddings;
    std::vector<ExecutionMemory> exec_memories;
    std::vector<ExecutionBlockOutput> exec_outputs_per_row;

    void ensureExecutionBatchGeometry(size_t batch_size, const char* caller) {
        if (batch_size == 0) {
            throw std::runtime_error(std::string(caller) + ": execution batch_size must be > 0");
        }

        const bool memories_uninitialized = exec_memories.empty();
        const bool outputs_uninitialized = exec_outputs_per_row.empty();
        if (memories_uninitialized != outputs_uninitialized) {
            throw std::runtime_error(std::string(caller) +
                                     ": execution forward sink geometry is inconsistent (exec_memories size=" +
                                     std::to_string(exec_memories.size()) +
                                     ", exec_outputs_per_row size=" +
                                     std::to_string(exec_outputs_per_row.size()) + ")");
        }

        if (memories_uninitialized) {
            exec_memories.resize(batch_size);
            exec_outputs_per_row.resize(batch_size);
            return;
        }

        if (exec_memories.size() != batch_size || exec_outputs_per_row.size() != batch_size) {
            throw std::runtime_error(std::string(caller) +
                                     ": execution forward sink batch geometry mismatch expected=" +
                                     std::to_string(batch_size) +
                                     " exec_memories=" + std::to_string(exec_memories.size()) +
                                     " exec_outputs_per_row=" + std::to_string(exec_outputs_per_row.size()));
        }
    }

    Tensor* liveLmHeadInputOrNull() {
        if (lm_head_input_tensor.data) {
            return &lm_head_input_tensor;
        }
        if (latent_preset_h_enhanced.data) {
            return &latent_preset_h_enhanced;
        }
        if (encoder_output_tensor.data) {
            return &encoder_output_tensor;
        }
        return nullptr;
    }

    const Tensor* liveLmHeadInputOrNull() const {
        if (lm_head_input_tensor.data) {
            return &lm_head_input_tensor;
        }
        if (latent_preset_h_enhanced.data) {
            return &latent_preset_h_enhanced;
        }
        if (encoder_output_tensor.data) {
            return &encoder_output_tensor;
        }
        return nullptr;
    }

    void clear() {
        clearRetainedLayerOutputs();
        embedding_tensor = Tensor();
        embedding_structured_state = Tensor();
        embedding_gate_concat = Tensor();
        embedding_gate_logits = Tensor();
        embedding_gate_values = Tensor();
        embedding_gate_delta = Tensor();
        clearTensorVector(encoder_layer_outputs);
        encoder_output_tensor = Tensor();
        lm_head_input_tensor = Tensor();
        logits_tensor = Tensor();
        clearTensorVector(mtp_logits_tensors);
        latent_preset_mtp_hidden = Tensor();
        latent_preset_future_fused = Tensor();
        latent_preset_future_entropy = Tensor();
        latent_preset_z = Tensor();
        latent_preset_vec = Tensor();
        latent_preset_gate_pre = Tensor();
        latent_preset_gate = Tensor();
        latent_preset_injected = Tensor();
        latent_preset_h_enhanced = Tensor();
        selector_logits = Tensor();
        scratch_atom_embeddings = Tensor();
        resetExecutionMemoryVectorPreserveGeometry(exec_memories);
        resetExecutionOutputVectorPreserveGeometry(exec_outputs_per_row);
    }

    bool hasLogits() const { return logits_tensor.data != nullptr; }
};

}  // namespace Forward

}  // namespace GRIM

#endif  // USE_CUDA