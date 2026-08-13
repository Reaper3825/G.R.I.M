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

#include "../Goal/Goal.hpp"
#include "../Goal/GoalSpanView.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../Layers/ExecutionBlock/execution_block_GPU.hpp"

#include <cstddef>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace GRIM {
namespace Forward {

struct ModelForwardOutputs {
private:
    std::vector<std::shared_ptr<const Goal>> row_goals_;

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
        requireSize(attention_residual_gate_logits_per_layer, "attention_residual_gate_logits_per_layer");
        requireSize(attention_residual_gate_multiplier_per_layer, "attention_residual_gate_multiplier_per_layer");
        requireSize(attention_residual_branch_per_layer, "attention_residual_branch_per_layer");
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

    void setGoalMetadata(
        std::size_t batch_size,
        const std::vector<std::shared_ptr<const Goal>>& goals) {
        if (batch_size == 0) {
            throw std::runtime_error(
                "ModelForwardOutputs::setGoalMetadata: batch_size must be > 0");
        }
        if (goals.empty()) {
            row_goals_.assign(batch_size, nullptr);
            return;
        }
        if (goals.size() != batch_size) {
            throw std::runtime_error(
                "ModelForwardOutputs::setGoalMetadata: goals.size()=" +
                std::to_string(goals.size()) + " != batch_size=" +
                std::to_string(batch_size));
        }
        row_goals_ = goals;
    }

    std::size_t goalRowCount() const noexcept { return row_goals_.size(); }

    GoalSpanView goalSpansForRow(std::size_t row) const {
        if (row >= row_goals_.size()) {
            throw std::out_of_range(
                "ModelForwardOutputs::goalSpansForRow: row=" +
                std::to_string(row) + " is outside goalRowCount=" +
                std::to_string(row_goals_.size()));
        }
        const Goal* goal = row_goals_[row].get();
        if (!goal) {
            return GoalSpanView{};
        }
        const GoalTokenSpan* target_state = goal->target_state.has_value()
            ? &goal->target_state->span
            : nullptr;
        const SuccessCriteria* success_criteria =
            goal->success_criteria.has_value()
                ? &*goal->success_criteria
                : nullptr;
        const Constraints* constraints = goal->constraints.has_value()
            ? &*goal->constraints
            : nullptr;
        return GoalSpanView(target_state, success_criteria, constraints);
    }

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
    // Token-wise learned attention residual gate. Logits and multiplier remain
    // live because SigmoidGradFn and BroadcastRowMulGradFn borrow their data
    // through the complete backward window.
    std::vector<Tensor> attention_residual_gate_logits_per_layer;
    std::vector<Tensor> attention_residual_gate_multiplier_per_layer;
    // Actual attention branch after optional token gate and fixed depth scale,
    // before optional LayerScale.
    std::vector<Tensor> attention_residual_branch_per_layer;
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
        attention_residual_gate_logits_per_layer.reserve(num_layers);
        attention_residual_gate_multiplier_per_layer.reserve(num_layers);
        attention_residual_branch_per_layer.reserve(num_layers);
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
        attention_residual_gate_logits_per_layer.emplace_back();
        attention_residual_gate_multiplier_per_layer.emplace_back();
        attention_residual_branch_per_layer.emplace_back();
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
        clearTensorVector(attention_residual_gate_logits_per_layer);
        clearTensorVector(attention_residual_gate_multiplier_per_layer);
        clearTensorVector(attention_residual_branch_per_layer);
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
        count += countGradFns(attention_residual_gate_logits_per_layer);
        count += countGradFns(attention_residual_gate_multiplier_per_layer);
        count += countGradFns(attention_residual_branch_per_layer);
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
    // Final-layer hidden states after the optional pre-LM-head RMSNorm. This
    // lives on the forward sink so downstream forward operations can consume
    // it without borrowing a function-local tensor.
    Tensor final_normalized_hidden_states;
    // Generic pooled hidden-state output. This is not target-state Goal metadata;
    // target-state production belongs to the frozen target model.
    Tensor mean_pool;
    Tensor lm_head_input_tensor;
    // LM-head residual SwiGLU adapter (config.lm_head_mlp_enabled) retained
    // intermediates. gate/silu/up must survive until backward: SiluGradFn and
    // ElementwiseMulGradFn hold non-owning pointers into their input buffers
    // (same contract as the encoder FFN per-layer retained tensors).
    Tensor lm_head_mlp_gate_out;      // [total_tokens, mlp_d_ff] z @ W_gate (SiLU backward cache)
    Tensor lm_head_mlp_silu_out;      // [total_tokens, mlp_d_ff] SiLU(gate_out)
    Tensor lm_head_mlp_up_out;        // [total_tokens, mlp_d_ff] z @ W_up
    Tensor lm_head_mlp_swiglu_out;    // [total_tokens, mlp_d_ff] silu ⊙ up
    Tensor lm_head_mlp_residual_out;  // [total_tokens, d_model] u = z + alpha * (swiglu @ W_down)
    Tensor logits_tensor;
    // Candidate keys supplied by the independent selector pipeline. Core model
    // forward does not derive these from NumberEncoder.
    Tensor selector_candidate_keys; // [num_pool_atoms, d_model]
    // Arg/option selector head: [total_tokens, num_pool_atoms] selection logits over
    // the candidate atom-entry pool (out-of-row-window candidates masked to -inf).
    // Empty until the independent selector pipeline materializes keys and logits.
    Tensor selector_logits;

    // Contextual numeric-placeholder slot-seed state. ModelForwardOutputs is
    // the sole owner of every buffer referenced by SlotSeedEncoder backward.
    Tensor slot_seed_contextual_input;       // routed context + optional type embedding
    Tensor slot_seed_hidden_pre_activation;  // MLP input projection; SiLU cache
    Tensor slot_seed_hidden_activation;      // SiLU(hidden_pre_activation)
    Tensor slot_seed_residual_delta;         // hidden @ W_seed_out (+ bias)
    Tensor slot_seed_unmasked;               // contextual_input + residual_delta
    Tensor slot_seeds;                       // authored-slot-gated dense slot seeds

    // Optional reasoning forward-owned state.
    Tensor scratch_atom_embeddings;

    Tensor* liveLmHeadInputOrNull() {
        if (lm_head_input_tensor.data) {
            return &lm_head_input_tensor;
        }
        if (lm_head_mlp_residual_out.data) {
            return &lm_head_mlp_residual_out;
        }
        if (final_normalized_hidden_states.data) {
            return &final_normalized_hidden_states;
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
        if (lm_head_mlp_residual_out.data) {
            return &lm_head_mlp_residual_out;
        }
        if (final_normalized_hidden_states.data) {
            return &final_normalized_hidden_states;
        }
        if (encoder_output_tensor.data) {
            return &encoder_output_tensor;
        }
        return nullptr;
    }

    void clear() {
        row_goals_.clear();
        clearRetainedLayerOutputs();
        embedding_tensor = Tensor();
        embedding_structured_state = Tensor();
        embedding_gate_concat = Tensor();
        embedding_gate_logits = Tensor();
        embedding_gate_values = Tensor();
        embedding_gate_delta = Tensor();
        clearTensorVector(encoder_layer_outputs);
        encoder_output_tensor = Tensor();
        final_normalized_hidden_states = Tensor();
        mean_pool = Tensor();
        lm_head_input_tensor = Tensor();
        lm_head_mlp_gate_out = Tensor();
        lm_head_mlp_silu_out = Tensor();
        lm_head_mlp_up_out = Tensor();
        lm_head_mlp_swiglu_out = Tensor();
        lm_head_mlp_residual_out = Tensor();
        logits_tensor = Tensor();
        selector_candidate_keys = Tensor();
        selector_logits = Tensor();
        // Reverse graph order keeps non-owning backward caches alive until
        // their consumer GradFns have been released.
        slot_seeds = Tensor();
        slot_seed_unmasked = Tensor();
        slot_seed_residual_delta = Tensor();
        slot_seed_hidden_activation = Tensor();
        slot_seed_hidden_pre_activation = Tensor();
        slot_seed_contextual_input = Tensor();
        scratch_atom_embeddings = Tensor();
    }

    bool hasLogits() const { return logits_tensor.data != nullptr; }

    // ═══════════════════════════════════════════════════════════════════════════
    // FORWARD-OUTPUT SIZE REPORT (diagnostic)
    // Emits one line per live retained tensor with element count and byte size,
    // plus a total. Only tensors with allocated device storage are listed.
    // ═══════════════════════════════════════════════════════════════════════════
    std::string describeRetainedSizes(const std::string& tag) const {
        std::ostringstream body;
        body << std::fixed;
        body.precision(2);

        size_t total_bytes = 0;
        size_t live_tensors = 0;

        auto mib = [](size_t bytes) {
            return static_cast<double>(bytes) / (1024.0 * 1024.0);
        };

        auto reportTensor = [&](const std::string& name, const Tensor& t) {
            if (!t.data) return;
            const size_t bytes = t.size_bytes();
            total_bytes += bytes;
            ++live_tensors;
            body << "\n  " << name
                 << " numel=" << t.numel()
                 << " bytes=" << bytes
                 << " MiB=" << mib(bytes);
        };

        auto reportVector = [&](const std::string& name, const std::vector<Tensor>& v) {
            for (size_t i = 0; i < v.size(); ++i) {
                reportTensor(name + "[" + std::to_string(i) + "]", v[i]);
            }
        };

        // Per-layer retained tensors
        reportVector("ln1_out_per_layer", ln1_out_per_layer);
        reportVector("ln2_out_per_layer", ln2_out_per_layer);
        reportVector("qkv_out_per_layer", qkv_out_per_layer);
        reportVector("Q_bhsd_per_layer", Q_bhsd_per_layer);
        reportVector("K_bhsd_per_layer", K_bhsd_per_layer);
        reportVector("V_bhsd_per_layer", V_bhsd_per_layer);
        reportVector("attn_out_bhsd_per_layer", attn_out_bhsd_per_layer);
        reportVector("attn_out_per_layer", attn_out_per_layer);
        reportVector("proj_out_per_layer", proj_out_per_layer);
        reportVector("attention_residual_gate_logits_per_layer", attention_residual_gate_logits_per_layer);
        reportVector("attention_residual_gate_multiplier_per_layer", attention_residual_gate_multiplier_per_layer);
        reportVector("attention_residual_branch_per_layer", attention_residual_branch_per_layer);
        reportVector("scaled_proj_per_layer", scaled_proj_per_layer);
        reportVector("residual1_per_layer", residual1_per_layer);
        reportVector("ffn_out_per_layer", ffn_out_per_layer);
        reportVector("scaled_ffn_per_layer", scaled_ffn_per_layer);
        reportVector("output_per_layer", output_per_layer);
        reportVector("ffn_gate_out_per_layer", ffn_gate_out_per_layer);
        reportVector("ffn_silu_out_per_layer", ffn_silu_out_per_layer);
        reportVector("ffn_linear1_out_per_layer", ffn_linear1_out_per_layer);
        reportVector("ffn_swiglu_out_per_layer", ffn_swiglu_out_per_layer);

        // Cross-layer live tensors
        reportTensor("embedding_tensor", embedding_tensor);
        reportTensor("embedding_structured_state", embedding_structured_state);
        reportTensor("embedding_gate_concat", embedding_gate_concat);
        reportTensor("embedding_gate_logits", embedding_gate_logits);
        reportTensor("embedding_gate_values", embedding_gate_values);
        reportTensor("embedding_gate_delta", embedding_gate_delta);
        reportVector("encoder_layer_outputs", encoder_layer_outputs);
        reportTensor("encoder_output_tensor", encoder_output_tensor);
        reportTensor("final_normalized_hidden_states", final_normalized_hidden_states);
        reportTensor("mean_pool", mean_pool);
        reportTensor("lm_head_input_tensor", lm_head_input_tensor);
        reportTensor("lm_head_mlp_gate_out", lm_head_mlp_gate_out);
        reportTensor("lm_head_mlp_silu_out", lm_head_mlp_silu_out);
        reportTensor("lm_head_mlp_up_out", lm_head_mlp_up_out);
        reportTensor("lm_head_mlp_swiglu_out", lm_head_mlp_swiglu_out);
        reportTensor("lm_head_mlp_residual_out", lm_head_mlp_residual_out);
        reportTensor("logits_tensor", logits_tensor);
        reportTensor("selector_candidate_keys", selector_candidate_keys);
        reportTensor("selector_logits", selector_logits);
        reportTensor("slot_seed_contextual_input", slot_seed_contextual_input);
        reportTensor(
            "slot_seed_hidden_pre_activation",
            slot_seed_hidden_pre_activation);
        reportTensor(
            "slot_seed_hidden_activation",
            slot_seed_hidden_activation);
        reportTensor("slot_seed_residual_delta", slot_seed_residual_delta);
        reportTensor("slot_seed_unmasked", slot_seed_unmasked);
        reportTensor("slot_seeds", slot_seeds);
        reportTensor("scratch_atom_embeddings", scratch_atom_embeddings);

        std::ostringstream out;
        out << std::fixed;
        out.precision(2);
        out << "[ForwardOutputSizes] " << tag
            << " live_tensors=" << live_tensors
            << " total_bytes=" << total_bytes
            << " total_MiB=" << mib(total_bytes)
            << body.str();
        return out.str();
    }
};

}  // namespace Forward

}  // namespace GRIM

#endif  // USE_CUDA
