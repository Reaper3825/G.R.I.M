//======================================================//
//  execution_block_GPU.hpp
//  Differentiable Register Machine — GPU
//
//  Declares: ExecutionMemory, ExecutionBlockConfig,
//  ExecutionBlockStepOutput, ExecutionBlockOutput,
//  ExecutionBlockLayer.
//
//  No CUDA kernels here. No orchestration logic.
//  No serialization code.
//======================================================//

#pragma once

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

#include "../../Shared/TensorContract/TensorContract_GPU.hpp"

namespace GRIM {

namespace ExecutionBlockInternal {
struct LayerAccess;
}

//======================================================//
//  ExecutionMemory — addressable register file
//
//  Each instance represents ONE batch row's register file [V, …].
//  Per-row isolation: AutogradIntermediates stores a vector<ExecutionMemory>
//  of size batch_size, and executeAutogradForward processes each row with its
//  own M, using token_offset/row_tokens to scope H access.
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
//  ExecutionBlockConfig
//======================================================//
struct ExecutionBlockConfig {
    int d_model              = 0;
    int atom_embedding_dim   = 0;   // from ScratchBlock (default 64)
    int num_ops              = 4;   // +, -, *, /
    int num_slots            = 4;   // V — max memory slots
    int num_scratch_slots    = 0;   // S — scratch-only slots [0..S-1]; value slots are [S..V-1]
    int num_exec_steps       = 2;   // K — execution steps per forward
    int value_decode_input_dim  = 24;
    int value_decode_hidden_dim = 16;
    int d_key                = 64;
    int d_type               = 8;
    int cross_attn_head_dim  = 64;
    int cross_attn_topk      = 1;
    float usage_decay        = 0.9f;
    float empty_slot_bonus   = 10.0f;
    float diversity_kappa    = 2.0f;
    float inject_gate_temp   = 0.5f;
    int   result_slot_mode   = 0;     // 0 = last token, 1 = fixed index
    int   result_slot_index  = -1;    // used when result_slot_mode == 1
    bool  debug_mode         = true;  // extra diagnostics only; does not relax validation
    float entropy_collapse_threshold = 0.01f;
    float write_collapse_threshold   = 0.98f;
    float magnitude_limit            = 1e6f;

    // Causal state loss (plan: persistantExecutionMemory)
    float transition_hard_threshold  = 0.0f;  // Fix 1: hard gate threshold (0 = disabled)
};

//======================================================//
//  ExecStepMetrics — lightweight runtime monitoring
//======================================================//
struct ExecStepMetrics {
    float arg1_entropy   = 0.0f;
    float arg2_entropy   = 0.0f;
    float op_entropy     = 0.0f;
    float write_entropy  = 0.0f;
    float max_p_write    = 0.0f;
    int   div_clamp_count = 0;
    float op_distribution[4] = {0.0f, 0.0f, 0.0f, 0.0f};
};

//======================================================//
//  ExecutionRecord — discrete step trace (state₀ → op → v_out)
//======================================================//
struct ExecutionRecord {
    int arg1_slot = -1;
    int arg2_slot = -1;
    int op_id = -1;
    float value_before_1 = 0.0f;
    float value_before_2 = 0.0f;
    float value_after = 0.0f;
};

//======================================================//
//  ExecutionBlockStepOutput — per-step diagnostics + auxiliary-loss tensors
//  p_arg1/p_arg2/p_op/p_write are detached copies for diagnostics / entropy loss.
//  state_before_values / state_after_values enable transition validity checks.
//======================================================//
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
    ExecutionRecord record;   // filled when diag_out != nullptr (host copy at step sync)
    ExecStepMetrics metrics;  // populated when debug_mode is enabled

    // Causal state loss tensors (Fix 1-9, all [1,1] scalars)
    Tensor transition_error_hard;    // |v_out - expected_internal| (hard gate)
    Tensor transition_loss;          // |v_soft - target| (autograd via L1ScalarLossGradFn)
    bool   used_expected_target = false;  // whether teacher target was used for transition_loss
};

//======================================================//
//  ExecutionBlockOutput — all steps
//======================================================//
struct ExecutionBlockOutput {
    std::vector<ExecutionBlockStepOutput> steps;
};

//======================================================//
//  ExecutionBlockLayer
//======================================================//
class ExecutionBlockLayer {
public:
    ExecutionBlockLayer() = delete;

    explicit ExecutionBlockLayer(const ExecutionBlockConfig& config,
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
        const int32_t* token_to_slot_map,   // row-local [row_tokens] slot_id per token position (-1 = non-state-bearing)
        int num_atoms,
        int total_tokens,
        int step,
        float temperature,
        cudaStream_t stream,
        ExecutionBlockStepOutput* diag_out,
        int token_offset,
        int row_tokens,
        Tensor& trace_state,
        const std::vector<ExecutionRecord>& prior_records,
        const float* expected_target = nullptr      // Fix 6: optional teacher scalar (device [1])
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
    // Cross-attention read: H = H + g * W_O(R)
    // token_offset / row_tokens enable per-batch-row processing.
    //--------------------------------------------------//
    void crossAttentionRead(
        Tensor& hidden_states,
        ExecutionMemory& M,
        int total_tokens,
        cudaStream_t stream,
        int token_offset = 0,
        int row_tokens = -1
    );

    //--------------------------------------------------//
    // Entropy loss over arg/op/write distributions
    //--------------------------------------------------//
    Tensor computeEntropyLoss(
        const std::vector<ExecutionBlockStepOutput>& steps,
        float weight,
        cudaStream_t stream
    ) const;

    //--------------------------------------------------//
    // Validation (hard-fail)
    //--------------------------------------------------//
    void validateConfigOrThrow() const;
    void validateMemoryOrThrow(const ExecutionMemory& M) const;
    void validateExecuteStepInputsOrThrow(
        const Tensor& H,
        const int* atom_positions,
        const int32_t* token_to_slot_map,
        int num_atoms,
        int total_tokens,
        const ExecutionMemory& M,
        int step,
        int token_offset,
        int row_tokens) const;
    void validateCrossAttentionInputsOrThrow(
        const Tensor& hidden_states,
        const ExecutionMemory& M,
        int total_tokens) const;

    //--------------------------------------------------//
    // Parameter access (for registration + serialization)
    //--------------------------------------------------//
    Tensor& w_decode_1()      { return w_decode_1_; }
    Tensor& b_decode_1()      { return b_decode_1_; }
    Tensor& w_decode_2()      { return w_decode_2_; }
    Tensor& w_arg1_select()   { return w_arg1_select_; }
    Tensor& w_arg2_select()   { return w_arg2_select_; }
    Tensor& W_op_select()     { return W_op_select_; }
    Tensor& W_key_proj()      { return W_key_proj_; }
    Tensor& W_write_query()   { return W_write_query_; }
    Tensor& W_write_key()     { return W_write_key_; }
    Tensor& alpha()           { return alpha_; }
    Tensor& beta()            { return beta_; }
    Tensor& step_embeddings() { return step_embeddings_; }
    Tensor& type_num_embed()  { return type_num_embed_; }
    Tensor& W_value_to_emb()  { return W_value_to_emb_; }
    Tensor& b_value_to_emb()  { return b_value_to_emb_; }
    Tensor& w_inject_gate()   { return w_inject_gate_; }
    Tensor& W_Q_read()        { return W_Q_read_; }
    Tensor& W_K_read()        { return W_K_read_; }
    Tensor& W_V_read()        { return W_V_read_; }
    Tensor& W_O_read()        { return W_O_read_; }
    Tensor& W_gate_read()     { return W_gate_read_; }
    Tensor& tau()             { return tau_; }
    Tensor& E_slot()          { return E_slot_; }
    Tensor& E_op()            { return E_op_; }
    Tensor& W_scal()          { return W_scal_; }
    Tensor& b_scal()          { return b_scal_; }
    Tensor& W_trace()         { return W_trace_; }
    Tensor& b_trace()         { return b_trace_; }
    Tensor& W_reason_gate()   { return W_reason_gate_; }

    const Tensor& w_decode_1()    const { return w_decode_1_; }
    const Tensor& b_decode_1()    const { return b_decode_1_; }
    const Tensor& w_decode_2()    const { return w_decode_2_; }
    const Tensor& w_arg1_select() const { return w_arg1_select_; }
    const Tensor& w_arg2_select() const { return w_arg2_select_; }
    const Tensor& W_op_select()   const { return W_op_select_; }
    const Tensor& W_key_proj()    const { return W_key_proj_; }
    const Tensor& W_write_query() const { return W_write_query_; }
    const Tensor& W_write_key()   const { return W_write_key_; }
    const Tensor& alpha()         const { return alpha_; }
    const Tensor& beta()          const { return beta_; }
    const Tensor& step_embeddings() const { return step_embeddings_; }
    const Tensor& type_num_embed()  const { return type_num_embed_; }
    const Tensor& W_value_to_emb()  const { return W_value_to_emb_; }
    const Tensor& b_value_to_emb()  const { return b_value_to_emb_; }
    const Tensor& w_inject_gate()   const { return w_inject_gate_; }
    const Tensor& W_Q_read()      const { return W_Q_read_; }
    const Tensor& W_K_read()      const { return W_K_read_; }
    const Tensor& W_V_read()      const { return W_V_read_; }
    const Tensor& W_O_read()      const { return W_O_read_; }
    const Tensor& W_gate_read()   const { return W_gate_read_; }
    const Tensor& tau()           const { return tau_; }
    const Tensor& E_slot()        const { return E_slot_; }
    const Tensor& E_op()          const { return E_op_; }
    const Tensor& W_scal()        const { return W_scal_; }
    const Tensor& b_scal()        const { return b_scal_; }
    const Tensor& W_trace()       const { return W_trace_; }
    const Tensor& b_trace()       const { return b_trace_; }
    const Tensor& W_reason_gate() const { return W_reason_gate_; }

    const ExecutionBlockConfig& config() const { return config_; }

private:
    friend struct ExecutionBlockInternal::LayerAccess;

    ExecutionBlockConfig config_;

    // Production hardening: persistent device-side error tracking
    int* d_numeric_error_flag_ = nullptr;  // atomicMax stage-id: numeric, softmax, collapse
    int* d_div_clamp_count_    = nullptr;  // atomicAdd on division clamp
    int* d_exec_idx_           = nullptr;  // [4] arg1_rel, arg2_rel, op_id, write_slot (abs)
    int* d_exec_record_i_      = nullptr;  // [3] packed for ExecutionRecord ints
    float* d_exec_record_f_    = nullptr;  // [3] value_before_1, value_before_2, value_after

    // Value decode MLP (atom embedding dims 16-39 -> scalar)
    Tensor w_decode_1_;     // [24, 16]
    Tensor b_decode_1_;     // [16]
    Tensor w_decode_2_;     // [16, 1]

    // Arg selection: decision_input [1,3*dm] @ [3*dm,dm] → query [1,dm] → @ cand^T → [1,V_val]
    Tensor w_arg1_select_;  // [3 * d_model, d_model]
    Tensor w_arg2_select_;  // [3 * d_model, d_model]

    // Op selection: pool [1,5*dm] = (h_arg1,h_arg2,context,trace_state,step_emb) → [1,nop]
    Tensor W_op_select_;    // [5 * d_model, num_ops]

    // Key generation from result embedding
    Tensor W_key_proj_;     // [d_model, d_key]

    // Write-head: write_ctx [1,6*dm] = (h1,h2,ctx,result_emb,trace_state,step_emb) → d_key query
    Tensor W_write_query_;  // [6 * d_model, d_key]
    Tensor W_write_key_;    // [d_key, d_key]
    Tensor alpha_;          // [1] learned content score scalar (init 1.0)
    Tensor beta_;           // [1] learned usage penalty scalar (init 1.0)

    // Step encoding
    Tensor step_embeddings_; // [K, d_model]

    // Type embedding
    Tensor type_num_embed_;  // [d_type]

    // Linear value embedding (replaces sinusoidal re_embed)
    Tensor W_value_to_emb_; // [1, d_model]
    Tensor b_value_to_emb_; // [1, d_model]

    // Injection gate
    Tensor w_inject_gate_;  // [d_model, 1]

    // Cross-attention read (gated + sharpened)
    Tensor W_Q_read_;       // [d_model, head_dim]
    Tensor W_K_read_;       // [d_key, head_dim]
    Tensor W_V_read_;       // [d_model, head_dim]
    Tensor W_O_read_;       // [head_dim, d_model]
    Tensor W_gate_read_;    // [d_model, 1] per-token read gate
    Tensor tau_;            // [1] learnable temperature (init 1.0)

    // Trace encoding weights (Pattern B: owned by layer)
    Tensor E_slot_;          // [num_slots, d_model] slot embedding for record encoding
    Tensor E_op_;            // [num_ops, d_model]   op embedding for record encoding
    Tensor W_scal_;          // [3, d_model]          scalar projection for (v1, v2, v_out)
    Tensor b_scal_;          // [1, d_model]          scalar projection bias
    Tensor W_trace_;         // [K * d_model, d_model] flattened history → d_model
    Tensor b_trace_;         // [1, d_model]           trace projection bias

    // Reasoning state update gate (learned residual transform)
    Tensor W_reason_gate_;   // [2 * d_model, d_model] concat(trace_state, cur_enc) → update
};

}  // namespace GRIM

#endif  // USE_CUDA
