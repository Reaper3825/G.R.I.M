//======================================================//
//  execution_block_GPU.cu
//  Differentiable Register Machine — thin public coordinator
//
//  Owns: validation helpers, construction/move lifecycle,
//  and public wrappers that dispatch into the split
//  memory-stream and data-stream implementations.
//======================================================//

#include "execution_block_internal.hpp"
#include "execution_block_memory_stream_GPU.hpp"
#include "execution_block_data_stream_GPU.hpp"
#include "../../Shared/Batching/BatchPayload.hpp"
#include "../../Shared/Batching/BatchDeviceBindings.hpp"
#include "../../Shared/CudaAllocUtils.hpp"

using GRIM::CudaAlloc::cudaMallocOrThrow;

namespace GRIM {
using namespace ExecutionBlockInternal;

//======================================================//
//  Validation helpers
//======================================================//
void ExecutionBlockLayer::validateConfigOrThrow() const {
    EXEC_CHECK(hp_.d_model > 0,            "d_model must be positive");
    EXEC_CHECK(hp_.atom_embedding_dim > 0,  "atom_embedding_dim must be positive");
    EXEC_CHECK(hp_.num_ops > 0,            "num_ops must be positive");
    EXEC_CHECK(hp_.num_slots > 0,          "num_slots must be positive");
    EXEC_CHECK(hp_.num_exec_steps > 0,     "num_exec_steps must be positive");
    EXEC_CHECK(hp_.d_key > 0,              "d_key must be positive");
    EXEC_CHECK(hp_.d_type > 0,             "d_type must be positive");
    EXEC_CHECK(hp_.cross_attn_head_dim > 0,"cross_attn_head_dim must be positive");
    EXEC_CHECK(hp_.value_decode_input_dim > 0,    "value_decode_input_dim must be positive");
    EXEC_CHECK(hp_.value_decode_hidden_dim > 0,   "value_decode_hidden_dim must be positive");
    EXEC_CHECK(hp_.value_decode_input_dim + 16 <= hp_.atom_embedding_dim,
               "value_decode_input_dim + 16 must fit within atom_embedding_dim (decode slice out of bounds)");
    EXEC_CHECK(hp_.d_key <= 64,                    "d_key must be <= 64");
    EXEC_CHECK(hp_.num_scratch_slots >= 0, "num_scratch_slots must be non-negative");
    EXEC_CHECK(hp_.num_scratch_slots < hp_.num_slots,
               "num_scratch_slots must be < num_slots (need at least one value slot)");
}

void ExecutionBlockLayer::validateMemoryOrThrow(const ExecutionMemory& M) const {
    const int V = hp_.num_slots;
    const int ae = hp_.atom_embedding_dim;
    const int dm = hp_.d_model;
    const int dk = hp_.d_key;
    const int dt = hp_.d_type;

    EXEC_CHECK_SHAPE2(M.values,       "M.values",       V, 1);
    EXEC_CHECK_SHAPE2(M.atom_embeds,   "M.atom_embeds",   V, ae);
    EXEC_CHECK_SHAPE2(M.state_embeds,  "M.state_embeds",  V, dm);
    EXEC_CHECK_SHAPE1(M.valid_mask,    "M.valid_mask",    V);
    EXEC_CHECK_SHAPE1(M.usage,         "M.usage",         V);
    EXEC_CHECK_SHAPE2(M.key_embeds,    "M.key_embeds",    V, dk);
    EXEC_CHECK_SHAPE2(M.type_embed,    "M.type_embed",    V, dt);
    EXEC_CHECK_SHAPE1(M.recent_write_mask, "M.recent_write_mask", V);
}

void ExecutionBlockLayer::validateExecuteStepInputsOrThrow(
    const Tensor& H,
    const int* atom_positions,
    int num_atoms, const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    int batch_row,
    const ExecutionMemory& M, int step) const
{
    const int dm = hp_.d_model;
    EXEC_CHECK_SHAPE2(H, "H (executeStep)", payload.total_tokens, dm);
    EXEC_CHECK(atom_positions != nullptr,
               "atom_positions is null - caller MUST provide a row-local atom view (empty buffer allowed)");
    EXEC_CHECK(bindings.d_token_to_slot_map != nullptr, "bindings.d_token_to_slot_map is null");
    EXEC_CHECK(num_atoms >= 0, "num_atoms must be non-negative");
    EXEC_CHECK(payload.total_tokens > 0, "payload.total_tokens must be positive");
    EXEC_CHECK(step >= 0 && step < hp_.num_exec_steps, "step out of range");
    EXEC_CHECK(batch_row >= 0, "batch_row must be non-negative");
    EXEC_CHECK(payload.max_seq_len > 0, "payload.max_seq_len must be positive");
    EXEC_CHECK(batch_row < payload.batch_size, "batch_row out of range for payload.batch_size");
    EXEC_CHECK(static_cast<int>(payload.seq_lengths.size()) == payload.batch_size,
               "payload.seq_lengths size must equal payload.batch_size");
    const int row_tokens = payload.seq_lengths[static_cast<size_t>(batch_row)];
    EXEC_CHECK(row_tokens > 0, "payload.seq_lengths[batch_row] must be positive");
    EXEC_CHECK(row_tokens <= payload.max_seq_len, "payload.seq_lengths[batch_row] exceeds payload.max_seq_len");
    EXEC_CHECK(batch_row * payload.max_seq_len + row_tokens <= payload.total_tokens,
               "valid row-local span exceeds total token extent");
    validateMemoryOrThrow(M);
}

void ExecutionBlockLayer::validateCrossAttentionInputsOrThrow(
    const Tensor& hidden_states, const ExecutionMemory& M, int total_tokens) const
{
    const int dm = hp_.d_model;
    EXEC_CHECK_SHAPE2(hidden_states, "hidden_states (cross-attn)", total_tokens, dm);
    validateMemoryOrThrow(M);
}

__global__ void kernelFillConstant(
    float* __restrict__ out, float val, int N
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = val;
}

//======================================================//
//  Thin-coordinator local kernel
//======================================================//

//======================================================//
//  Constructor
//======================================================//
ExecutionBlockLayer::~ExecutionBlockLayer() {
    if (d_numeric_error_flag_) cudaFree(d_numeric_error_flag_);
    if (d_div_clamp_count_)    cudaFree(d_div_clamp_count_);
    if (d_div_invalid_flag_)   cudaFree(d_div_invalid_flag_);
    if (d_exec_idx_)           cudaFree(d_exec_idx_);
    if (d_exec_record_i_)      cudaFree(d_exec_record_i_);
    if (d_exec_record_f_)      cudaFree(d_exec_record_f_);
    if (d_reinforce_baseline_) cudaFree(d_reinforce_baseline_);
}

ExecutionBlockLayer::ExecutionBlockLayer(const HyperParameters::ExecutionBlockConstructionHP& hp,
                                       uint64_t seed,
                                       cudaStream_t init_stream)
    : hp_(hp)
{
    validateConfigOrThrow();
    EXEC_CHECK(init_stream != nullptr, "init_stream is NULL");

    cudaMallocOrThrow(reinterpret_cast<void**>(&d_numeric_error_flag_), sizeof(int), "exec_numeric_error_flag");
    CUDA_CHECK(cudaMemsetAsync(d_numeric_error_flag_, 0, sizeof(int), init_stream));
    cudaMallocOrThrow(reinterpret_cast<void**>(&d_div_clamp_count_), sizeof(int), "exec_div_clamp_count");
    CUDA_CHECK(cudaMemsetAsync(d_div_clamp_count_, 0, sizeof(int), init_stream));
    cudaMallocOrThrow(reinterpret_cast<void**>(&d_div_invalid_flag_), sizeof(int), "exec_div_invalid_flag");
    CUDA_CHECK(cudaMemsetAsync(d_div_invalid_flag_, 0, sizeof(int), init_stream));
    cudaMallocOrThrow(reinterpret_cast<void**>(&d_exec_idx_), 4 * sizeof(int), "exec_idx");
    cudaMallocOrThrow(reinterpret_cast<void**>(&d_exec_record_i_), 3 * sizeof(int), "exec_record_i");
    cudaMallocOrThrow(reinterpret_cast<void**>(&d_exec_record_f_), 3 * sizeof(float), "exec_record_f");
    cudaMallocOrThrow(reinterpret_cast<void**>(&d_reinforce_baseline_), sizeof(float), "exec_reinforce_baseline");
    CUDA_CHECK(cudaMemsetAsync(d_reinforce_baseline_, 0, sizeof(float), init_stream));

    const int dm  = hp_.d_model;
    const int dk  = hp_.d_key;
    const int dt  = hp_.d_type;
    const int hd  = hp_.cross_attn_head_dim;
    const int nop = hp_.num_ops;   // 4
    const int V   = hp_.num_slots;
    const int K   = hp_.num_exec_steps;
    const int vid = hp_.value_decode_input_dim;
    const int vhd = hp_.value_decode_hidden_dim;

    auto make_param = [&](int rows, int cols, uint64_t s, const char* name) -> Tensor {
        auto t = Tensor::zeros(TensorContract::TensorShape::make_BSM(rows, cols),
                               true, init_stream, name);
        t.requires_grad_();
        t.ensure_grad();
        Tensor::xavier_uniform_(t, s, init_stream);
        return t;
    };
    auto make_bias = [&](int cols, const char* name) -> Tensor {
        auto t = Tensor::zeros(TensorContract::TensorShape::make_BSM(1, cols),
                               true, init_stream, name);
        t.requires_grad_();
        t.ensure_grad();
        return t;
    };
    auto make_scalar = [&](float init_val, const char* name) -> Tensor {
        auto t = Tensor::zeros(TensorContract::TensorShape::make_BSM(1, 1),
                               true, init_stream, name);
        t.requires_grad_();
        t.ensure_grad();
        cudaMemcpyAsync(t.data, &init_val, sizeof(float), cudaMemcpyHostToDevice, init_stream);
        return t;
    };

    // Value decode MLP
    w_decode_1_ = make_param(vid, vhd, seed,     "exec_block.w_decode_1");
    b_decode_1_ = make_bias(vhd,                 "exec_block.b_decode_1");
    w_decode_2_ = make_param(vhd, 1, seed + 1,   "exec_block.w_decode_2");

    // Arg selection: decision_input [1, 3*dm] → query [1, dm] via w_arg_select [3*dm, dm]
    w_arg1_select_ = make_param(3 * dm, dm, seed + 2, "exec_block.w_arg1_select");
    w_arg2_select_ = make_param(3 * dm, dm, seed + 3, "exec_block.w_arg2_select");

    // Op selection: decision_input [1, 3*dm] → logits [1, nop]
    // Detached from arg selection: op sees (context, trace, step_emb) only.
    W_op_select_ = make_param(3 * dm, nop, seed + 4, "exec_block.W_op_select");

    // Key projection from result embedding
    W_key_proj_ = make_param(dm, dk, seed + 5, "exec_block.W_key_proj");

    // Write-head (write_context = 4*d_model -> d_key query)
    // Detached from arg selection: sees (context, result, trace, step) only.
    W_write_query_ = make_param(4 * dm, dk, seed + 7, "exec_block.W_write_query");
    W_write_key_   = make_param(dk, dk, seed + 8, "exec_block.W_write_key");

    // Learned scalars (init 1.0)
    alpha_ = make_scalar(1.0f, "exec_block.alpha");
    beta_  = make_scalar(1.0f, "exec_block.beta");

    // Step encoding
    step_embeddings_ = make_param(K, dm, seed + 9, "exec_block.step_embeddings");

    // Type embedding
    type_num_embed_ = make_param(1, dt, seed + 10, "exec_block.type_num_embed");

    // Linear value embedding (scalar -> d_model)
    W_value_to_emb_ = make_param(1, dm, seed + 15, "exec_block.W_value_to_emb");
    b_value_to_emb_ = make_bias(dm,                "exec_block.b_value_to_emb");

    // Injection gate: init to -2.0 so gate starts at sigmoid(-2) ≈ 0.12
    w_inject_gate_ = Tensor::zeros(TensorContract::TensorShape::make_BSM(dm, 1),
                                   true, init_stream, "exec_block.w_inject_gate");
    w_inject_gate_.requires_grad_();
    w_inject_gate_.ensure_grad();
    kernelFillConstant<<<(dm + kBlockSize - 1) / kBlockSize, kBlockSize, 0, init_stream>>>(
        w_inject_gate_.data, -2.0f, dm);
    CUDA_CHECK_KERNEL();

    // Cross-attention read
    W_Q_read_    = make_param(dm, hd, seed + 11, "exec_block.W_Q_read");
    W_K_read_    = make_param(dk, hd, seed + 12, "exec_block.W_K_read");
    W_V_read_    = make_param(dm, hd, seed + 13, "exec_block.W_V_read");
    W_O_read_    = make_param(hd, dm, seed + 14, "exec_block.W_O_read");
    W_gate_read_ = Tensor::zeros(TensorContract::TensorShape::make_BSM(dm, 1),
                                 true, init_stream, "exec_block.W_gate_read");
    W_gate_read_.requires_grad_();
    W_gate_read_.ensure_grad();

    // Temperature (init 1.0)
    tau_ = make_scalar(1.0f, "exec_block.tau");

    // Trace encoding weights
    E_slot_  = make_param(V, dm, seed + 16, "exec_block.E_slot");
    E_op_    = make_param(nop, dm, seed + 17, "exec_block.E_op");
    W_scal_  = make_param(3, dm, seed + 18, "exec_block.W_scal");
    b_scal_  = make_bias(dm,                "exec_block.b_scal");
    W_trace_ = make_param(K * dm, dm, seed + 19, "exec_block.W_trace");
    b_trace_ = make_bias(dm,                "exec_block.b_trace");

    // Reasoning state update: candidate + gated interpolation
    W_reason_gate_ = make_param(2 * dm, dm, seed + 20, "exec_block.W_reason_gate");
    W_trace_gate_  = make_param(2 * dm, dm, seed + 21, "exec_block.W_trace_gate");
}

//======================================================//
//  Move semantics
//======================================================//
ExecutionBlockLayer::ExecutionBlockLayer(ExecutionBlockLayer&& other) noexcept
    : hp_(other.hp_),
      d_numeric_error_flag_(other.d_numeric_error_flag_),
      d_div_clamp_count_(other.d_div_clamp_count_),
      d_div_invalid_flag_(other.d_div_invalid_flag_),
      d_exec_idx_(other.d_exec_idx_),
      d_exec_record_i_(other.d_exec_record_i_),
      d_exec_record_f_(other.d_exec_record_f_),
      d_reinforce_baseline_(other.d_reinforce_baseline_),
      w_decode_1_(std::move(other.w_decode_1_)),
      b_decode_1_(std::move(other.b_decode_1_)),
      w_decode_2_(std::move(other.w_decode_2_)),
      w_arg1_select_(std::move(other.w_arg1_select_)),
      w_arg2_select_(std::move(other.w_arg2_select_)),
      W_op_select_(std::move(other.W_op_select_)),
      W_key_proj_(std::move(other.W_key_proj_)),
      W_write_query_(std::move(other.W_write_query_)),
      W_write_key_(std::move(other.W_write_key_)),
      alpha_(std::move(other.alpha_)),
      beta_(std::move(other.beta_)),
      step_embeddings_(std::move(other.step_embeddings_)),
      type_num_embed_(std::move(other.type_num_embed_)),
      W_value_to_emb_(std::move(other.W_value_to_emb_)),
      b_value_to_emb_(std::move(other.b_value_to_emb_)),
      w_inject_gate_(std::move(other.w_inject_gate_)),
      W_Q_read_(std::move(other.W_Q_read_)),
      W_K_read_(std::move(other.W_K_read_)),
      W_V_read_(std::move(other.W_V_read_)),
      W_O_read_(std::move(other.W_O_read_)),
      W_gate_read_(std::move(other.W_gate_read_)),
      tau_(std::move(other.tau_)),
      E_slot_(std::move(other.E_slot_)),
      E_op_(std::move(other.E_op_)),
      W_scal_(std::move(other.W_scal_)),
      b_scal_(std::move(other.b_scal_)),
      W_trace_(std::move(other.W_trace_)),
      b_trace_(std::move(other.b_trace_)),
      W_reason_gate_(std::move(other.W_reason_gate_)),
      W_trace_gate_(std::move(other.W_trace_gate_))
{
    other.d_numeric_error_flag_ = nullptr;
    other.d_div_clamp_count_ = nullptr;
    other.d_div_invalid_flag_ = nullptr;
    other.d_exec_idx_ = nullptr;
    other.d_exec_record_i_ = nullptr;
    other.d_exec_record_f_ = nullptr;
    other.d_reinforce_baseline_ = nullptr;
}

ExecutionBlockLayer& ExecutionBlockLayer::operator=(ExecutionBlockLayer&& other) noexcept {
    if (this != &other) {
        if (d_numeric_error_flag_) cudaFree(d_numeric_error_flag_);
        if (d_div_clamp_count_)    cudaFree(d_div_clamp_count_);
        if (d_div_invalid_flag_)   cudaFree(d_div_invalid_flag_);
        if (d_exec_idx_)           cudaFree(d_exec_idx_);
        if (d_exec_record_i_)     cudaFree(d_exec_record_i_);
        if (d_exec_record_f_)     cudaFree(d_exec_record_f_);
        if (d_reinforce_baseline_) cudaFree(d_reinforce_baseline_);
        hp_ = other.hp_;
        d_numeric_error_flag_ = other.d_numeric_error_flag_;
        d_div_clamp_count_    = other.d_div_clamp_count_;
        d_div_invalid_flag_   = other.d_div_invalid_flag_;
        d_exec_idx_           = other.d_exec_idx_;
        d_exec_record_i_      = other.d_exec_record_i_;
        d_exec_record_f_      = other.d_exec_record_f_;
        d_reinforce_baseline_ = other.d_reinforce_baseline_;
        other.d_numeric_error_flag_ = nullptr;
        other.d_div_clamp_count_    = nullptr;
        other.d_div_invalid_flag_   = nullptr;
        other.d_exec_idx_           = nullptr;
        other.d_exec_record_i_      = nullptr;
        other.d_exec_record_f_      = nullptr;
        other.d_reinforce_baseline_ = nullptr;
        w_decode_1_    = std::move(other.w_decode_1_);
        b_decode_1_    = std::move(other.b_decode_1_);
        w_decode_2_    = std::move(other.w_decode_2_);
        w_arg1_select_ = std::move(other.w_arg1_select_);
        w_arg2_select_ = std::move(other.w_arg2_select_);
        W_op_select_   = std::move(other.W_op_select_);
        W_key_proj_    = std::move(other.W_key_proj_);
        W_write_query_ = std::move(other.W_write_query_);
        W_write_key_   = std::move(other.W_write_key_);
        alpha_         = std::move(other.alpha_);
        beta_          = std::move(other.beta_);
        step_embeddings_= std::move(other.step_embeddings_);
        type_num_embed_ = std::move(other.type_num_embed_);
        W_value_to_emb_= std::move(other.W_value_to_emb_);
        b_value_to_emb_= std::move(other.b_value_to_emb_);
        w_inject_gate_ = std::move(other.w_inject_gate_);
        W_Q_read_      = std::move(other.W_Q_read_);
        W_K_read_      = std::move(other.W_K_read_);
        W_V_read_      = std::move(other.W_V_read_);
        W_O_read_      = std::move(other.W_O_read_);
        W_gate_read_   = std::move(other.W_gate_read_);
        tau_           = std::move(other.tau_);
        E_slot_        = std::move(other.E_slot_);
        E_op_          = std::move(other.E_op_);
        W_scal_        = std::move(other.W_scal_);
        b_scal_        = std::move(other.b_scal_);
        W_trace_       = std::move(other.W_trace_);
        b_trace_       = std::move(other.b_trace_);
        W_reason_gate_ = std::move(other.W_reason_gate_);
        W_trace_gate_  = std::move(other.W_trace_gate_);
    }
    return *this;
}

//======================================================//
//  Thin public wrappers
//======================================================//
void ExecutionBlockLayer::executeStep(
    Tensor& H,
    ExecutionMemory& M,
    const int* atom_positions,
    int num_atoms,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    int batch_row,
    int step,
    float temperature,
    cudaStream_t stream,
    ExecutionBlockStepOutput* diag_out,
    Tensor& trace_state,
    const std::vector<ExecutionRecord>& prior_records,
    const float* expected_target,
    const TeacherSelectionTargets* selection_targets)
{
    validateExecuteStepInputsOrThrow(H, atom_positions,
                                     num_atoms, payload, bindings, batch_row, M, step);
    executeStepCoordinatorImpl(
        *this,
        H,
        M,
        atom_positions,
        num_atoms,
        payload,
        bindings,
        batch_row,
        step,
        temperature,
        stream,
        diag_out,
        trace_state,
        prior_records,
        expected_target,
        selection_targets);
}

//======================================================//
//  crossAttentionRead — thin wrapper
//======================================================//
Tensor ExecutionBlockLayer::crossAttentionRead(
    const Tensor& hidden_states,
    ExecutionMemory& M,
    int total_tokens,
    cudaStream_t stream,
    int token_offset,
    int row_tokens,
    float* d_gate_accum)
{
    validateCrossAttentionInputsOrThrow(hidden_states, M, total_tokens);
    if (row_tokens < 0) row_tokens = total_tokens;
    EXEC_CHECK(token_offset >= 0, "token_offset must be non-negative");
    EXEC_CHECK(row_tokens > 0, "row_tokens must be positive");
    EXEC_CHECK(token_offset + row_tokens <= total_tokens,
               "crossAttentionRead row-local span exceeds total token extent");
    return crossAttentionReadImpl(*this, hidden_states, M, stream, token_offset, row_tokens, d_gate_accum);
}

}  // namespace GRIM
