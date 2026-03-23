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
    Tensor key_embeds;        // [V, d_key]   addressing keys
    Tensor type_embed;        // [V, d_type]  type tag per slot
    Tensor recent_write_mask; // [V]          last-step write probability distribution
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
    int num_ops              = 4;   // +, -, *, /
    int num_slots            = 4;   // V — max memory slots
    int num_exec_steps       = 2;   // K — execution steps per forward
    int execution_block_layer= -1;  // encoder layer to run after (-1 = num_layers - 2)
    int value_decode_input_dim  = 24;
    int value_decode_hidden_dim = 16;
    int d_key                = 64;
    int d_type               = 8;
    int cross_attn_head_dim  = 64;
    int cross_attn_topk      = 1;
    float usage_decay        = 0.9f;
    float empty_slot_bonus   = 10.0f;
    float diversity_kappa    = 2.0f;
    float memory_slot_bias   = 0.5f;
    float inject_gate_temp   = 0.5f;
    float temp_start         = 2.0f;
    float temp_end           = 0.5f;
    int   temp_schedule      = 0;   // 0=linear, 1=cosine
    float entropy_weight     = 0.01f;
    int   result_slot_mode   = 0;     // 0 = last token, 1 = fixed index
    int   result_slot_index  = -1;    // used when result_slot_mode == 1
    bool  diag_logging       = false;
    bool  deterministic      = false;
    bool  debug_mode         = false;  // extra diagnostics only; does not relax validation
    float entropy_collapse_threshold = 0.01f;
    float write_collapse_threshold   = 0.98f;
    float magnitude_limit            = 1e6f;
    cudaStream_t stream      = nullptr;
    cublasHandle_t cublas_handle = nullptr;
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
//  ExecutionBlockStepOutput — per-step diagnostics ONLY
//  Must NOT drive forward computation.
//======================================================//
struct ExecutionBlockStepOutput {
    Tensor p_arg1;      // [1, C] softmax probabilities (detached copy)
    Tensor p_arg2;      // [1, C]
    Tensor p_op;        // [1, num_ops]
    Tensor p_write;     // [1, V]
    Tensor v_out;       // [1, 1] scalar result
    Tensor result_emb;  // [1, d_model]
    ExecStepMetrics metrics;  // populated when debug_mode is enabled
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
    //--------------------------------------------------//
    void executeStep(
        Tensor& H,                          // [total_tokens, d_model] mutated in place
        ExecutionMemory& M,
        const float* atom_embeddings,       // [num_atoms, atom_embedding_dim]
        const int* atom_positions,          // [num_atoms]
        int num_atoms,
        int total_tokens,
        int step,
        float temperature,
        cudaStream_t stream,
        ExecutionBlockStepOutput* diag_out = nullptr
    );

    //--------------------------------------------------//
    // Cross-attention read: H = H + g * W_O(R)
    //--------------------------------------------------//
    void crossAttentionRead(
        Tensor& hidden_states,
        ExecutionMemory& M,
        int total_tokens,
        cudaStream_t stream
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
    Tensor& W_key_proj()      { return W_key_proj_; }
    Tensor& W_write_query()   { return W_write_query_; }
    Tensor& W_write_key()     { return W_write_key_; }
    Tensor& alpha()           { return alpha_; }
    Tensor& beta()            { return beta_; }
    Tensor& gamma()           { return gamma_; }
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
    const Tensor& gamma()         const { return gamma_; }
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

    void setStream(cudaStream_t s)       { config_.stream = s; }
    void setCublasHandle(cublasHandle_t h){ config_.cublas_handle = h; }

    const ExecutionBlockConfig& config() const { return config_; }

    int lastDivClampCount(cudaStream_t stream) const;

private:
    ExecutionBlockConfig config_;

    // Production hardening: persistent device-side error tracking
    int* d_numeric_error_flag_ = nullptr;  // atomicMax stage-id: numeric, softmax, collapse
    int* d_div_clamp_count_    = nullptr;  // atomicAdd on division clamp

    // Value decode MLP (atom embedding dims 16-39 -> scalar)
    Tensor w_decode_1_;     // [24, 16]
    Tensor b_decode_1_;     // [16]
    Tensor w_decode_2_;     // [16, 1]

    // Arg selection (transposed for matmul: [1,dm] @ [dm,C]^T = [1,C])
    Tensor w_arg1_select_;  // [1, d_model]
    Tensor w_arg2_select_;  // [1, d_model]

    // Context-aware op selection (4 ops only: +, -, *, /)
    Tensor W_op_select_;    // [3 * d_model, 4]

    // Key generation from result embedding
    Tensor W_key_proj_;     // [d_model, d_key]

    // Write-head (write_context = concat(h1,h2,ctx,result_emb) -> d_key query)
    Tensor W_write_query_;  // [4 * d_model, d_key]
    Tensor W_write_key_;    // [d_key, d_key]
    Tensor alpha_;          // [1] learned content score scalar (init 1.0)
    Tensor beta_;           // [1] learned usage penalty scalar (init 1.0)
    Tensor gamma_;          // [1] learned write score scalar (init 1.0)

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
};

}  // namespace GRIM

#endif  // USE_CUDA
