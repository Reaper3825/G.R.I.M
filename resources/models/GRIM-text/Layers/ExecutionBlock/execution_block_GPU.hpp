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
#include <cublas_v2.h>
#include <cstdint>
#include <vector>

#include "../../Shared/TensorContract/TensorContract_GPU.hpp"

namespace GRIM {

//======================================================//
//  ExecutionMemory — addressable register file
//======================================================//
struct ExecutionMemory {
    Tensor values;            // [V, 1]       scalar ground truth per slot
    Tensor atom_embeds;       // [V, 64]      ScratchBlock-format encoding
    Tensor state_embeds;      // [V, d_model] value projection for cross-attn V
    Tensor valid_mask;        // [V]          1.0 if filled, 0.0 if empty
    Tensor usage;             // [V]          decayed cross-attn read weight
    Tensor write_score;       // [V]          learned overwrite preference bias
    Tensor key_embeds;        // [V, d_key]   addressing keys (from result_emb, NOT state_embeds)
    Tensor type_embed;        // [V, d_type]  type tag per slot
    Tensor recent_write_mask; // [V]          1.0 if written in most recent step
    int num_filled = 0;

    void clear(cudaStream_t stream);
    void allocate(int V, int atom_dim, int d_model, int d_key, int d_type, cudaStream_t stream);
};

//======================================================//
//  ExecutionBlockConfig
//======================================================//
struct ExecutionBlockConfig {
    int d_model              = 0;
    int atom_embedding_dim   = 0;   // from ScratchBlock (default 64)
    int num_ops              = 8;   // Add,Sub,Mul,Div,Mod,Pow,Min,Max
    int num_slots            = 4;   // V — max memory slots
    int num_exec_steps       = 2;   // K — execution steps per forward
    int execution_block_layer= -1;  // encoder layer to run after (-1 = num_layers - 2)
    int value_decode_input_dim  = 24;
    int value_decode_hidden_dim = 16;
    int d_key                = 64;
    int d_type               = 8;
    int cross_attn_head_dim  = 64;
    int cross_attn_topk      = 1;   // 1 = one slot per query (default)
    float usage_decay        = 0.9f;
    float empty_slot_bonus   = 10.0f;
    float diversity_kappa    = 2.0f;
    float memory_slot_bias   = 0.5f; // initial bias penalty for memory slot candidates
    cudaStream_t stream      = nullptr;
    cublasHandle_t cublas_handle = nullptr;
};

//======================================================//
//  ExecutionBlockStepOutput — per-step diagnostics
//======================================================//
struct ExecutionBlockStepOutput {
    int   selected_op   = -1;
    int   selected_arg1 = -1;
    int   selected_arg2 = -1;
    int   selected_slot = -1;
    float decoded_v1    = 0.0f;
    float decoded_v2    = 0.0f;
    float computed_result = 0.0f;
    Tensor op_logits;       // [1, num_ops]
    Tensor arg1_scores;     // [1, C]
    Tensor arg2_scores;     // [1, C]
    Tensor write_logits;    // [1, V]
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

    ~ExecutionBlockLayer() = default;

    ExecutionBlockLayer(ExecutionBlockLayer&& other) noexcept;
    ExecutionBlockLayer& operator=(ExecutionBlockLayer&& other) noexcept;

    ExecutionBlockLayer(const ExecutionBlockLayer&) = delete;
    ExecutionBlockLayer& operator=(const ExecutionBlockLayer&) = delete;

    //--------------------------------------------------//
    // Forward: one execution step
    //--------------------------------------------------//
    ExecutionBlockStepOutput executeStep(
        const Tensor& hidden_states,        // [total_tokens, d_model] read-only
        const float* atom_embeddings,       // [num_atoms, atom_embedding_dim]
        const int* atom_positions,          // [num_atoms]
        int num_atoms,
        int total_tokens,
        ExecutionMemory& M,
        int step,
        cudaStream_t stream
    );

    //--------------------------------------------------//
    // Cross-attention read: H = H + g * W_O(R)
    //--------------------------------------------------//
    void crossAttentionRead(
        Tensor& hidden_states,              // [total_tokens, d_model] modified in place
        ExecutionMemory& M,                 // non-const: updates M.usage (decayed)
        int total_tokens,
        cudaStream_t stream
    );

    //--------------------------------------------------//
    // Validation (hard-fail)
    //--------------------------------------------------//
    void validateConfigOrThrow() const;
    void validateMemoryOrThrow(const ExecutionMemory& M) const;
    void validateExecuteStepInputsOrThrow(
        const Tensor& hidden_states,
        const float* atom_embeddings,
        const int* atom_positions,
        int num_atoms,
        int total_tokens,
        const ExecutionMemory& M,
        int step) const;
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
    Tensor& W_state()         { return W_state_; }
    Tensor& W_key_base()      { return W_key_base_; }
    Tensor& W_write_query()   { return W_write_query_; }
    Tensor& W_write_key()     { return W_write_key_; }
    Tensor& alpha()           { return alpha_; }
    Tensor& beta()            { return beta_; }
    Tensor& gamma()           { return gamma_; }
    Tensor& step_embeddings() { return step_embeddings_; }
    Tensor& type_num_embed()  { return type_num_embed_; }
    Tensor& W_Q_read()        { return W_Q_read_; }
    Tensor& W_K_read()        { return W_K_read_; }
    Tensor& W_V_read()        { return W_V_read_; }
    Tensor& W_O_read()        { return W_O_read_; }
    Tensor& W_gate_read()     { return W_gate_read_; }
    Tensor& tau()             { return tau_; }

    const Tensor& w_decode_1()    const { return w_decode_1_; }
    const Tensor& b_decode_1()    const { return b_decode_1_; }
    const Tensor& w_decode_2()    const { return w_decode_2_; }
    const Tensor& w_arg1_select() const { return w_arg1_select_; }
    const Tensor& w_arg2_select() const { return w_arg2_select_; }
    const Tensor& W_op_select()   const { return W_op_select_; }
    const Tensor& W_state()       const { return W_state_; }
    const Tensor& W_key_base()    const { return W_key_base_; }
    const Tensor& W_write_query() const { return W_write_query_; }
    const Tensor& W_write_key()   const { return W_write_key_; }
    const Tensor& alpha()         const { return alpha_; }
    const Tensor& beta()          const { return beta_; }
    const Tensor& gamma()         const { return gamma_; }
    const Tensor& step_embeddings() const { return step_embeddings_; }
    const Tensor& type_num_embed()  const { return type_num_embed_; }
    const Tensor& W_Q_read()      const { return W_Q_read_; }
    const Tensor& W_K_read()      const { return W_K_read_; }
    const Tensor& W_V_read()      const { return W_V_read_; }
    const Tensor& W_O_read()      const { return W_O_read_; }
    const Tensor& W_gate_read()   const { return W_gate_read_; }
    const Tensor& tau()           const { return tau_; }

    void setStream(cudaStream_t s)       { config_.stream = s; }
    void setCublasHandle(cublasHandle_t h){ config_.cublas_handle = h; }

    const ExecutionBlockConfig& config() const { return config_; }

private:
    ExecutionBlockConfig config_;

    // Value decode MLP (atom embedding dims 16-39 -> scalar)
    Tensor w_decode_1_;     // [24, 16]
    Tensor b_decode_1_;     // [16]
    Tensor w_decode_2_;     // [16, 1]

    // Arg selection
    Tensor w_arg1_select_;  // [d_model, 1]
    Tensor w_arg2_select_;  // [d_model, 1]

    // Context-aware op selection
    Tensor W_op_select_;    // [3 * d_model, num_ops]

    // Memory write: atom_embed -> state_embed
    Tensor W_state_;        // [atom_embedding_dim, d_model]

    // Key generation (from result_emb, separate from state)
    Tensor W_key_base_;     // [atom_embedding_dim, d_key]

    // Write-head (normalized logits)
    Tensor W_write_query_;  // [d_model, d_key]
    Tensor W_write_key_;    // [d_key, d_key]
    Tensor alpha_;          // [1] learned content score scalar (init 1.0)
    Tensor beta_;           // [1] learned usage penalty scalar (init 1.0)
    Tensor gamma_;          // [1] learned write score scalar (init 1.0)

    // Step encoding
    Tensor step_embeddings_; // [K, d_model]

    // Type embedding
    Tensor type_num_embed_;  // [d_type]

    // Cross-attention read (gated + sharpened)
    Tensor W_Q_read_;       // [d_model, head_dim]
    Tensor W_K_read_;       // [d_key, head_dim]
    Tensor W_V_read_;       // [d_model, head_dim]
    Tensor W_O_read_;       // [head_dim, d_model]
    Tensor W_gate_read_;    // [d_model, 1] per-token read gate
    Tensor tau_;            // [1] learnable temperature (init 1.0)
};

}  // namespace GRIM

#endif  // USE_CUDA
