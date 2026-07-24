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
#include "../CudaAllocUtils.hpp"
#include "../../Layers/ExecutionBlock/execution_block_GPU.hpp"

#include <memory>
#include <sstream>
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
    int write_slot = -1;
    float value_before_1 = 0.0f;
    float value_before_2 = 0.0f;
    float value_after = 0.0f;
};

struct ExecutionGateOutput {
    Tensor logits;         // [1, 2], class 0=NOOP, class 1=EXECUTE
    Tensor probabilities;  // [1, 2]
    int predicted_class = -1;
    float noop_probability = 0.0f;
    float execute_probability = 0.0f;
};

struct ExecutionBlockStepOutput {
    Tensor p_arg1;      // [1, V_val] softmax over value slots [S..V-1] only (detached)
    Tensor p_arg2;      // [1, V_val]
    Tensor p_op;        // [1, num_ops]
    Tensor p_write;     // [1, V]
    Tensor v_out;       // [1, 1] detached scalar result from hard op selection
    Tensor result_emb;  // [1, d_model]
    Tensor state_before_values;  // [V, 1] M.values snapshot before this step
    Tensor state_before_valid;   // [V]    M.valid_mask snapshot before this step
    Tensor state_after_values;   // [V, 1] M.values snapshot after this step
    Tensor state_after_valid;    // [V]    M.valid_mask snapshot after this step
    ExecutionRecord record;      // filled from the execution diagnostics payload at step sync
    ExecStepMetrics metrics;     // populated when debug_mode is enabled

    // Retained live tensors for later loss assembly (Category 1 graph-owned)
    Tensor arg1_logits_tensor;     // [1, V_val] live logits for arg1 CE
    Tensor arg2_logits_tensor;     // [1, V_val] live logits for arg2 CE
    Tensor op_logits_tensor;       // [1, num_ops] live logits for op CE / div penalty
    Tensor write_logits_tensor;    // [1, V] live logits for write-slot CE
    Tensor stop_logits_tensor;      // [1, 2], class 0=CONTINUE, class 1=STOP
    Tensor stop_probabilities;      // [1, 2]

    // Forward-authored scalar injection gate retained for telemetry and
    // ExecutionBlockInjectGradFn backward. ModelForwardOutputs owns the
    // storage for the complete forward/loss/backward window.
    Tensor inject_gate_tensor;       // [1, 1], sigmoid injection gate

    // Category 1 execution-decoder activation retained specifically for
    // SiluGradFn's non-owning backward cache. The execution step moves the
    // pre-activation here before its local working set is destroyed; the
    // batch-boundary ModelForwardOutputs::clear() remains the sole teardown.
    Tensor decoder_silu_input_tensor;  // [1, value_decode_hidden_dim]

    int stop_predicted_class = -1;
    float continue_probability = 0.0f;
    float stop_probability = 0.0f;
    float selection_temperature = 0.0f;
    bool div_was_clamped = false;
    // True when Training mode materialized the hard arg/op/write transition
    // from BatchPayload.teacher_steps. Model logits/probabilities above remain
    // the Category 1 owners used by structured CE.
    bool teacher_forced_transition = false;
};

struct ExecutionBlockOutput {
    ExecutionGateOutput gate;
    std::vector<ExecutionBlockStepOutput> steps;
    bool execution_suppressed_no_bootstrap = false;
    bool stopped_by_model = false;
    bool stopped_at_max_steps = false;
};

// Loss-time Category 1 staging for one normalized-entropy term. The forward
// sink owns both buffers through the complete loss/backward window; the
// corresponding GradFn holds non-owning views only.
struct NormalizedEntropyBackwardStaging {
    Tensor saved_probs;
    Tensor grad_probs;
};

// Forward-owned saved record data for one live RecordEncodeGradFn. IDs are
// packed as [slot1 records][slot2 records][op records]. The GradFn borrows raw
// views only; this staging object is the sole owner through backward.
struct RecordEncodeBackwardStaging {
    std::shared_ptr<int> saved_ids;
    std::shared_ptr<float> saved_scalars;
    int record_count = 0;
};

// Owning storage for one row's execution register file. Shared execution math
// receives only the non-owning ExecutionMemory view bound by this owner.
struct ExecutionMemoryOwnedStorage {
    Tensor values;
    Tensor atom_embeds;
    Tensor state_embeds;
    Tensor valid_mask;
    Tensor usage;
    Tensor key_embeds;
    Tensor type_embed;
    Tensor recent_write_mask;

    void bind(ExecutionMemory& memory) {
        memory.bind(
            values,
            atom_embeds,
            state_embeds,
            valid_mask,
            usage,
            key_embeds,
            type_embed,
            recent_write_mask);
    }
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

    static void resetExecutionMemoryStorageVectorPreserveGeometry(
        std::vector<ExecutionMemoryOwnedStorage>& storage_by_row)
    {
        for (auto& storage : storage_by_row) {
            storage = ExecutionMemoryOwnedStorage();
        }
    }

    static void resetExecutionOutputVectorPreserveGeometry(std::vector<ExecutionBlockOutput>& outputs) {
        for (auto& output : outputs) {
            output.gate = ExecutionGateOutput();
            output.execution_suppressed_no_bootstrap = false;
            output.stopped_by_model = false;
            output.stopped_at_max_steps = false;
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
    // NumberEncoder-derived candidate keys shared by the token-level selector
    // and ExecutionBlock authored-slot bootstrap. Prepared before the encoder
    // loop whenever either consumer needs them.
    Tensor selector_candidate_keys; // [num_pool_atoms, d_model]
    // Arg/option selector head: [total_tokens, num_pool_atoms] selection logits over
    // the candidate atom-entry pool (out-of-row-window candidates masked to -inf).
    // Empty unless the forward graph policy requested emit_selector_logits.
    Tensor selector_logits;

    // Contextual numeric-placeholder slot-seed state. ModelForwardOutputs is
    // the sole owner of every buffer referenced by SlotSeedEncoder backward.
    Tensor slot_seed_contextual_input;       // routed context + optional type embedding
    Tensor slot_seed_hidden_pre_activation;  // MLP input projection; SiLU cache
    Tensor slot_seed_hidden_activation;      // SiLU(hidden_pre_activation)
    Tensor slot_seed_residual_delta;         // hidden @ W_seed_out (+ bias)
    Tensor slot_seed_unmasked;               // contextual_input + residual_delta
    Tensor slot_seeds;                       // authored-slot-gated dense slot seeds

    // Optional reasoning / execution forward-owned state.
    Tensor scratch_atom_embeddings;
    std::vector<ExecutionMemoryOwnedStorage> exec_memory_storage;
    std::vector<ExecutionMemory> exec_memories;
    std::vector<ExecutionBlockOutput> exec_outputs_per_row;
    std::vector<NormalizedEntropyBackwardStaging> normalized_entropy_backward_staging;
    std::vector<RecordEncodeBackwardStaging> record_encode_backward_staging;

    NormalizedEntropyBackwardStaging& appendNormalizedEntropyBackwardStaging(
        const TensorContract::TensorShape& shape,
        cudaStream_t stream)
    {
        normalized_entropy_backward_staging.emplace_back();
        auto& staging = normalized_entropy_backward_staging.back();
        staging.saved_probs = Tensor::empty(
            shape,
            false,
            stream,
            "normalized_entropy_saved_probs");
        staging.grad_probs = Tensor::zeros(
            shape,
            false,
            stream,
            "normalized_entropy_grad_probs");
        return staging;
    }

    RecordEncodeBackwardStaging& appendRecordEncodeBackwardStaging(
        int record_count,
        cudaStream_t stream)
    {
        if (record_count <= 0) {
            throw std::runtime_error(
                "appendRecordEncodeBackwardStaging: record_count must be > 0");
        }
        if (!stream) {
            throw std::runtime_error(
                "appendRecordEncodeBackwardStaging: stream is NULL");
        }

        record_encode_backward_staging.emplace_back();
        auto& staging = record_encode_backward_staging.back();
        staging.record_count = record_count;

        int* saved_ids = nullptr;
        CudaAlloc::cudaMallocOrThrow(
            reinterpret_cast<void**>(&saved_ids),
            static_cast<size_t>(record_count) * 3 * sizeof(int),
            "record_encode_saved_ids");
        staging.saved_ids.reset(saved_ids, [](int* ptr) {
            queueForDeferredCleanup(ptr);
        });

        float* saved_scalars = nullptr;
        CudaAlloc::cudaMallocOrThrow(
            reinterpret_cast<void**>(&saved_scalars),
            static_cast<size_t>(record_count) * 3 * sizeof(float),
            "record_encode_saved_scalars");
        staging.saved_scalars.reset(saved_scalars, [](float* ptr) {
            queueForDeferredCleanup(ptr);
        });
        return staging;
    }

    void ensureExecutionBatchGeometry(size_t batch_size, const char* caller) {
        if (batch_size == 0) {
            throw std::runtime_error(std::string(caller) + ": execution batch_size must be > 0");
        }

        const bool storage_uninitialized = exec_memory_storage.empty();
        const bool memories_uninitialized = exec_memories.empty();
        const bool outputs_uninitialized = exec_outputs_per_row.empty();
        if (storage_uninitialized != memories_uninitialized ||
            memories_uninitialized != outputs_uninitialized) {
            throw std::runtime_error(std::string(caller) +
                                     ": execution forward sink geometry is inconsistent (exec_memory_storage size=" +
                                     std::to_string(exec_memory_storage.size()) +
                                     ", exec_memories size=" +
                                     std::to_string(exec_memories.size()) +
                                     ", exec_outputs_per_row size=" +
                                     std::to_string(exec_outputs_per_row.size()) + ")");
        }

        if (memories_uninitialized) {
            exec_memory_storage.resize(batch_size);
            exec_memories.resize(batch_size);
            exec_outputs_per_row.resize(batch_size);
            return;
        }

        if (exec_memory_storage.size() != batch_size ||
            exec_memories.size() != batch_size ||
            exec_outputs_per_row.size() != batch_size) {
            throw std::runtime_error(std::string(caller) +
                                     ": execution forward sink batch geometry mismatch expected=" +
                                     std::to_string(batch_size) +
                                     " exec_memory_storage=" + std::to_string(exec_memory_storage.size()) +
                                     " exec_memories=" + std::to_string(exec_memories.size()) +
                                     " exec_outputs_per_row=" + std::to_string(exec_outputs_per_row.size()));
        }
    }

    void resetExecutionMemoryRow(size_t row, const char* caller) {
        if (row >= exec_memory_storage.size() || row >= exec_memories.size()) {
            throw std::runtime_error(std::string(caller) + ": execution row is out of range");
        }
        exec_memories[row] = ExecutionMemory();
        exec_memory_storage[row] = ExecutionMemoryOwnedStorage();
    }

    void allocateExecutionMemoryRow(
        size_t row,
        int V,
        int atom_dim,
        int d_model,
        int d_key,
        int d_type,
        cudaStream_t stream,
        const char* caller)
    {
        if (!stream) {
            throw std::runtime_error(std::string(caller) + ": stream is NULL");
        }
        if (V <= 0 || atom_dim <= 0 || d_model <= 0 || d_key <= 0 || d_type <= 0) {
            throw std::runtime_error(std::string(caller) + ": execution-memory dimensions must be positive");
        }
        resetExecutionMemoryRow(row, caller);

        auto& storage = exec_memory_storage[row];
        storage.values = Tensor::zeros({V, 1}, stream, "exec_memory_values");
        storage.atom_embeds = Tensor::zeros({V, atom_dim}, stream, "exec_memory_atom_embeds");
        storage.state_embeds = Tensor::zeros({V, d_model}, stream, "exec_memory_state_embeds");
        storage.valid_mask = Tensor::zeros({1, V}, stream, "exec_memory_valid_mask");
        storage.usage = Tensor::zeros({1, V}, stream, "exec_memory_usage");
        storage.key_embeds = Tensor::zeros({V, d_key}, stream, "exec_memory_key_embeds");
        storage.type_embed = Tensor::zeros({V, d_type}, stream, "exec_memory_type_embed");
        storage.recent_write_mask = Tensor::zeros({1, V}, stream, "exec_memory_recent_write_mask");
        storage.bind(exec_memories[row]);
    }

    Tensor* liveLmHeadInputOrNull() {
        if (lm_head_input_tensor.data) {
            return &lm_head_input_tensor;
        }
        if (lm_head_mlp_residual_out.data) {
            return &lm_head_mlp_residual_out;
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
        for (auto& staging : normalized_entropy_backward_staging) {
            staging.saved_probs = Tensor();
            staging.grad_probs = Tensor();
        }
        normalized_entropy_backward_staging.clear();
        for (auto& staging : record_encode_backward_staging) {
            staging.saved_ids.reset();
            staging.saved_scalars.reset();
            staging.record_count = 0;
        }
        record_encode_backward_staging.clear();
        resetExecutionMemoryVectorPreserveGeometry(exec_memories);
        resetExecutionMemoryStorageVectorPreserveGeometry(exec_memory_storage);
        resetExecutionOutputVectorPreserveGeometry(exec_outputs_per_row);
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

        auto reportBuffer = [&](const std::string& name, const void* data, size_t bytes) {
            if (!data) return;
            total_bytes += bytes;
            ++live_tensors;
            body << "\n  " << name
                 << " bytes=" << bytes
                 << " MiB=" << mib(bytes);
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
        for (size_t row = 0; row < exec_memory_storage.size(); ++row) {
            const auto& storage = exec_memory_storage[row];
            const std::string prefix =
                "exec_memory_storage[" + std::to_string(row) + "].";
            reportTensor(prefix + "values", storage.values);
            reportTensor(prefix + "atom_embeds", storage.atom_embeds);
            reportTensor(prefix + "state_embeds", storage.state_embeds);
            reportTensor(prefix + "valid_mask", storage.valid_mask);
            reportTensor(prefix + "usage", storage.usage);
            reportTensor(prefix + "key_embeds", storage.key_embeds);
            reportTensor(prefix + "type_embed", storage.type_embed);
            reportTensor(prefix + "recent_write_mask", storage.recent_write_mask);
        }
        for (size_t i = 0; i < normalized_entropy_backward_staging.size(); ++i) {
            reportTensor(
                "normalized_entropy_backward_staging[" + std::to_string(i) + "].saved_probs",
                normalized_entropy_backward_staging[i].saved_probs);
            reportTensor(
                "normalized_entropy_backward_staging[" + std::to_string(i) + "].grad_probs",
                normalized_entropy_backward_staging[i].grad_probs);
        }
        for (size_t i = 0; i < record_encode_backward_staging.size(); ++i) {
            const auto& staging = record_encode_backward_staging[i];
            const size_t element_count = static_cast<size_t>(staging.record_count) * 3;
            reportBuffer(
                "record_encode_backward_staging[" + std::to_string(i) + "].saved_ids",
                staging.saved_ids.get(),
                element_count * sizeof(int));
            reportBuffer(
                "record_encode_backward_staging[" + std::to_string(i) + "].saved_scalars",
                staging.saved_scalars.get(),
                element_count * sizeof(float));
        }
        for (size_t row = 0; row < exec_outputs_per_row.size(); ++row) {
            const auto& execution_output = exec_outputs_per_row[row];
            for (size_t step = 0; step < execution_output.steps.size(); ++step) {
                reportTensor(
                    "exec_outputs_per_row[" + std::to_string(row) + "].steps[" +
                        std::to_string(step) + "].decoder_silu_input_tensor",
                    execution_output.steps[step].decoder_silu_input_tensor);
                reportTensor(
                    "exec_outputs_per_row[" + std::to_string(row) + "].steps[" +
                        std::to_string(step) + "].inject_gate_tensor",
                    execution_output.steps[step].inject_gate_tensor);
            }
        }

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
