//======================================================//
//  execution_block_GPU.hpp
//  Differentiable Register Machine — GPU
//
//  Declares: ExecutionMemory, ExecutionBlockLayer.
//  Forward-owned execution output payload types live in
//  Shared/Forward/ModelForwardOutputs.hpp.
//
//  No CUDA kernels here. No orchestration logic.
//  No serialization code.
//======================================================//

#pragma once

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <vector>

#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../Shared/HyperParameters/HyperparameterGroupings.hpp"

namespace GRIM {

// Durable ExecutionBlockLayer parameter ownership.
// These tensors are persistent layer state, not Category 1 forward outputs.
struct ExecutionBlockParameterTensors {
    // Value decode MLP (atom embedding dims 16-39 -> scalar)
    Tensor w_decode_1;     // [24, 16]
    Tensor b_decode_1;     // [16]
    Tensor w_decode_2;     // [16, 1]

    // Arg selection: decision_input [1,3*dm] @ [3*dm,dm] → query [1,dm] → @ cand^T → [1,V_val]
    Tensor w_arg1_select;  // [3 * d_model, d_model]
    Tensor w_arg2_select;  // [3 * d_model, d_model]

    // Op selection: decision_input [1,3*dm] → [1,nop] (detached from arg selection)
    Tensor W_op_select;    // [3 * d_model, num_ops]

    // Key generation from result embedding
    Tensor W_key_proj;     // [d_model, d_key]

    // Write-head: write_ctx [1,4*dm] = (ctx,result_emb,trace_state,step_emb) → d_key query
    Tensor W_write_query;  // [4 * d_model, d_key]
    Tensor W_write_key;    // [d_key, d_key]
    Tensor alpha;          // [1] learned content score scalar (init 1.0)
    Tensor beta;           // [1] learned usage penalty scalar (init 1.0)

    // Step encoding
    Tensor step_embeddings; // [K, d_model]

    // Type embedding
    Tensor type_num_embed;  // [d_type]

    // Linear value embedding (replaces sinusoidal re_embed)
    Tensor W_value_to_emb; // [1, d_model]
    Tensor b_value_to_emb; // [1, d_model]

    // Injection gate
    Tensor w_inject_gate;  // [d_model, 1]

    // Cross-attention read (gated + sharpened)
    Tensor W_Q_read;       // [d_model, head_dim]
    Tensor W_K_read;       // [d_key, head_dim]
    Tensor W_V_read;       // [d_model, head_dim]
    Tensor W_O_read;       // [head_dim, d_model]
    Tensor W_gate_read;    // [d_model, 1] per-token read gate
    Tensor tau;            // [1] learnable temperature (init 1.0)

    // Trace encoding weights
    Tensor E_slot;          // [num_slots, d_model] slot embedding for record encoding
    Tensor E_op;            // [num_ops, d_model]   op embedding for record encoding
    Tensor W_scal;          // [3, d_model]         scalar projection for (v1, v2, v_out)
    Tensor b_scal;          // [1, d_model]         scalar projection bias
    Tensor W_trace;         // [K * d_model, d_model] flattened history → d_model
    Tensor b_trace;         // [1, d_model]         trace projection bias

    // Reasoning state update: candidate + gate
    Tensor W_reason_gate;   // [2 * d_model, d_model] concat(trace_state, cur_enc) → candidate
    Tensor W_trace_gate;    // [2 * d_model, d_model] concat(trace_state, cur_enc) → gate logits
};

namespace Batching { struct BatchPayload; struct BatchDeviceBindings; }
namespace Forward {
struct ExecStepMetrics;
struct ExecutionRecord;
struct ExecutionBlockStepOutput;
struct ExecutionBlockOutput;
}
namespace ExecutionBlockInternal {
struct LayerAccess;
}

//======================================================//
//  ExecutionMemory — addressable register file
//
//  Each instance represents ONE batch row's register file [V, …].
//  Per-row isolation: AutogradIntermediates stores a vector<ExecutionMemory>
//  of size batch_size, and materializeTrainingGraphActivations processes each
//  row with its own M, using token_offset/row_tokens to scope H access.
//======================================================//
struct ExecutionMemory {
    Tensor values;            // [V, 1]       scalar ground truth per slot
    Tensor atom_embeds;       // [V, 64]      ScratchBlock-format encoding
    Tensor state_embeds;      // [V, d_model] value projection for cross-attn V
    Tensor valid_mask;        // [V]          1.0 if filled, 0.0 if empty
    Tensor usage;             // [V]          decayed cross-attn read weight
    Tensor key_embeds;        // [V, d_key]   addressing keys
    Tensor type_embed;        // [V, d_type]  type tag per slot
    Tensor recent_write_mask; // [V]          one-hot mask of the most recent hard write

    void clear(cudaStream_t stream);
    void allocate(int V, int atom_dim, int d_model, int d_key, int d_type, cudaStream_t stream);
};

//======================================================//
//  ExecutionBlockLayer
//======================================================//
class ExecutionBlockLayer {
public:
    ExecutionBlockLayer() = delete;

    explicit ExecutionBlockLayer(const HyperParameters::ExecutionBlockConstructionHP& hp,
                                uint64_t seed,
                                cudaStream_t init_stream);

    ~ExecutionBlockLayer();

    ExecutionBlockLayer(ExecutionBlockLayer&& other) noexcept;
    ExecutionBlockLayer& operator=(ExecutionBlockLayer&& other) noexcept;

    ExecutionBlockLayer(const ExecutionBlockLayer&) = delete;
    ExecutionBlockLayer& operator=(const ExecutionBlockLayer&) = delete;

    //--------------------------------------------------//
    // Forward: one execution step — mutates H and M
    // token_offset / row_tokens enable per-batch-row processing:
    //   context = reduce_mean(H[token_offset : token_offset + row_tokens])
    //   injection at H[token_offset + row_tokens - 1]
    //
    // trace_state:     [1, d_model] running accumulator for this row (mutated:
    //                  trace_state = autograd::add(trace_state, step_emb)).
    // prior_records:   host-side ExecutionRecord history for this row (read-only).
    //                  Used to build trace_vec = f(encoded history).
    //--------------------------------------------------//
    void executeStep(
        Tensor& H,                          // [total_tokens, d_model] mutated in place
        ExecutionMemory& M,
        const int* atom_positions,          // row-local [max(1, num_atoms)] positions relative to current row [0, row_tokens)
        int num_atoms,
        const Batching::BatchPayload& payload,
        const Batching::BatchDeviceBindings& bindings,
        int batch_row,
        int step,
        float temperature,
        cudaStream_t stream,
        Forward::ExecutionBlockStepOutput* diag_out,
        Tensor& trace_state,
        const std::vector<Forward::ExecutionRecord>& prior_records
    );

    //--------------------------------------------------//
    // Bootstrap: copy literal values into M.values via slot map (detached, no grad)
    //--------------------------------------------------//
    void bootstrapMemoryFromSlotMap(
        ExecutionMemory& M,
        const float* device_numeric_values,  // [row_tokens]
        const int32_t* device_slot_map,      // [row_tokens]
        int row_tokens,
        cudaStream_t stream
    );

    //--------------------------------------------------//
    // Forward runtime preparation: reset caller-owned execution state for
    // one shared-forward execution layer boundary.
    //--------------------------------------------------//
    void prepareForwardRuntime(
        const Batching::BatchPayload& payload,
        bool connect_parameter_graph,
        cudaStream_t stream,
        std::vector<ExecutionMemory>& exec_memories,
        std::vector<Forward::ExecutionBlockOutput>& exec_outputs_per_row,
        std::vector<std::vector<Forward::ExecutionRecord>>& execution_trace_by_row,
        std::vector<Tensor>& trace_state_by_row
    ) const;

    //--------------------------------------------------//
    // Cross-attention read: H = H + g * W_O(R)
    // token_offset / row_tokens enable per-batch-row processing.
    //--------------------------------------------------//
    Tensor crossAttentionRead(
        const Tensor& hidden_states,
        ExecutionMemory& M,
        int total_tokens,
        cudaStream_t stream,
        int token_offset = 0,
        int row_tokens = -1,
        float* d_gate_accum = nullptr  // [2] device: [sum, count] for telemetry
    );

    //--------------------------------------------------//
    // Entropy loss over arg/op/write distributions
    //--------------------------------------------------//
    Tensor computeEntropyLoss(
        const std::vector<const Forward::ExecutionBlockStepOutput*>& steps,
        float weight,
        cudaStream_t stream
    ) const;

    //--------------------------------------------------//
    // Validation (hard-fail)
    //--------------------------------------------------//
    void validateConfigOrThrow() const;
    void validateMemoryOrThrow(const ExecutionMemory& M) const;
    // Forward-declared in this header (see top): namespace Batching { struct BatchDeviceBindings; }
    void validateExecuteStepInputsOrThrow(
        const Tensor& H,
        const int* atom_positions,
        int num_atoms,
        const Batching::BatchPayload& payload,
        const Batching::BatchDeviceBindings& bindings,
        int batch_row,
        const ExecutionMemory& M,
        int step) const;
    void validateCrossAttentionInputsOrThrow(
        const Tensor& hidden_states,
        const ExecutionMemory& M,
        int total_tokens) const;

    //--------------------------------------------------//
    // Parameter access (for registration + serialization)
    //--------------------------------------------------//
    Tensor& w_decode_1();
    Tensor& b_decode_1();
    Tensor& w_decode_2();
    Tensor& w_arg1_select();
    Tensor& w_arg2_select();
    Tensor& W_op_select();
    Tensor& W_key_proj();
    Tensor& W_write_query();
    Tensor& W_write_key();
    Tensor& alpha();
    Tensor& beta();
    Tensor& step_embeddings();
    Tensor& type_num_embed();
    Tensor& W_value_to_emb();
    Tensor& b_value_to_emb();
    Tensor& w_inject_gate();
    Tensor& W_Q_read();
    Tensor& W_K_read();
    Tensor& W_V_read();
    Tensor& W_O_read();
    Tensor& W_gate_read();
    Tensor& tau();
    Tensor& E_slot();
    Tensor& E_op();
    Tensor& W_scal();
    Tensor& b_scal();
    Tensor& W_trace();
    Tensor& b_trace();
    Tensor& W_reason_gate();
    Tensor& W_trace_gate();

    const Tensor& w_decode_1() const;
    const Tensor& b_decode_1() const;
    const Tensor& w_decode_2() const;
    const Tensor& w_arg1_select() const;
    const Tensor& w_arg2_select() const;
    const Tensor& W_op_select() const;
    const Tensor& W_key_proj() const;
    const Tensor& W_write_query() const;
    const Tensor& W_write_key() const;
    const Tensor& alpha() const;
    const Tensor& beta() const;
    const Tensor& step_embeddings() const;
    const Tensor& type_num_embed() const;
    const Tensor& W_value_to_emb() const;
    const Tensor& b_value_to_emb() const;
    const Tensor& w_inject_gate() const;
    const Tensor& W_Q_read() const;
    const Tensor& W_K_read() const;
    const Tensor& W_V_read() const;
    const Tensor& W_O_read() const;
    const Tensor& W_gate_read() const;
    const Tensor& tau() const;
    const Tensor& E_slot() const;
    const Tensor& E_op() const;
    const Tensor& W_scal() const;
    const Tensor& b_scal() const;
    const Tensor& W_trace() const;
    const Tensor& b_trace() const;
    const Tensor& W_reason_gate() const;
    const Tensor& W_trace_gate() const;

    const HyperParameters::ExecutionBlockConstructionHP& hp() const { return hp_; }
    float* reinforceBaselineBuffer() { return d_reinforce_baseline_; }

private:
    friend struct ExecutionBlockInternal::LayerAccess;

    HyperParameters::ExecutionBlockConstructionHP hp_;

    // Production hardening: persistent device-side error tracking
    int* d_numeric_error_flag_ = nullptr;  // atomicMax stage-id: numeric, softmax, collapse
    int* d_div_clamp_count_    = nullptr;  // atomicAdd on division clamp
    int* d_div_invalid_flag_   = nullptr;  // [1] per-step: 1 if division was clamped, 0 otherwise
    int* d_exec_idx_           = nullptr;  // [4] arg1_rel, arg2_rel, op_id, write_slot (abs)
    int* d_exec_record_i_      = nullptr;  // [3] packed for ExecutionRecord ints
    float* d_exec_record_f_    = nullptr;  // [3] value_before_1, value_before_2, value_after
    float* d_reinforce_baseline_ = nullptr; // [1] EMA of transition_err for REINFORCE variance reduction
    ExecutionBlockParameterTensors& parametersOrThrow();
    const ExecutionBlockParameterTensors& parametersOrThrow() const;
    std::unique_ptr<ExecutionBlockParameterTensors> parameters_;
};

}  // namespace GRIM

#endif  // USE_CUDA
