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
#include "../../Shared/Forward/ModelForwardOutputs.hpp"
#include "../../Shared/CudaAllocUtils.hpp"

using GRIM::CudaAlloc::cudaMallocOrThrow;

namespace GRIM {
using namespace ExecutionBlockInternal;
using Forward::ExecutionBlockOutput;
using Forward::ExecutionBlockStepOutput;
using Forward::ExecutionRecord;

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
        : hp_(hp),
            parameters_(std::make_unique<ExecutionBlockParameterTensors>())
{
    validateConfigOrThrow();
    EXEC_CHECK(init_stream != nullptr, "init_stream is NULL");
        auto& params = parametersOrThrow();

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
    params.w_decode_1 = make_param(vid, vhd, seed,     "exec_block.w_decode_1");
    params.b_decode_1 = make_bias(vhd,                 "exec_block.b_decode_1");
    params.w_decode_2 = make_param(vhd, 1, seed + 1,   "exec_block.w_decode_2");

    // Arg selection: decision_input [1, 3*dm] → query [1, dm] via w_arg_select [3*dm, dm]
    params.w_arg1_select = make_param(3 * dm, dm, seed + 2, "exec_block.w_arg1_select");
    params.w_arg2_select = make_param(3 * dm, dm, seed + 3, "exec_block.w_arg2_select");

    // Op selection: decision_input [1, 3*dm] → logits [1, nop]
    // Detached from arg selection: op sees (context, trace, step_emb) only.
    params.W_op_select = make_param(3 * dm, nop, seed + 4, "exec_block.W_op_select");

    // Key projection from result embedding
    params.W_key_proj = make_param(dm, dk, seed + 5, "exec_block.W_key_proj");

    // Write-head (write_context = 4*d_model -> d_key query)
    // Detached from arg selection: sees (context, result, trace, step) only.
    params.W_write_query = make_param(4 * dm, dk, seed + 7, "exec_block.W_write_query");
    params.W_write_key   = make_param(dk, dk, seed + 8, "exec_block.W_write_key");

    // Learned scalars (init 1.0)
    params.alpha = make_scalar(1.0f, "exec_block.alpha");
    params.beta  = make_scalar(1.0f, "exec_block.beta");

    // Step encoding
    params.step_embeddings = make_param(K, dm, seed + 9, "exec_block.step_embeddings");

    // Type embedding
    params.type_num_embed = make_param(1, dt, seed + 10, "exec_block.type_num_embed");

    // Linear value embedding (scalar -> d_model)
    params.W_value_to_emb = make_param(1, dm, seed + 15, "exec_block.W_value_to_emb");
    params.b_value_to_emb = make_bias(dm,                "exec_block.b_value_to_emb");

    // Injection gate: init to -2.0 so gate starts at sigmoid(-2) ≈ 0.12
    params.w_inject_gate = Tensor::zeros(TensorContract::TensorShape::make_BSM(dm, 1),
                                   true, init_stream, "exec_block.w_inject_gate");
    params.w_inject_gate.requires_grad_();
    params.w_inject_gate.ensure_grad();
    kernelFillConstant<<<(dm + kBlockSize - 1) / kBlockSize, kBlockSize, 0, init_stream>>>(
        params.w_inject_gate.data, -2.0f, dm);
    CUDA_CHECK_KERNEL();

    // Cross-attention read
    params.W_Q_read    = make_param(dm, hd, seed + 11, "exec_block.W_Q_read");
    params.W_K_read    = make_param(dk, hd, seed + 12, "exec_block.W_K_read");
    params.W_V_read    = make_param(dm, hd, seed + 13, "exec_block.W_V_read");
    params.W_O_read    = make_param(hd, dm, seed + 14, "exec_block.W_O_read");
    params.W_gate_read = Tensor::zeros(TensorContract::TensorShape::make_BSM(dm, 1),
                                 true, init_stream, "exec_block.W_gate_read");
    params.W_gate_read.requires_grad_();
    params.W_gate_read.ensure_grad();

    // Temperature (init 1.0)
    params.tau = make_scalar(1.0f, "exec_block.tau");

    // Trace encoding weights
    params.E_slot  = make_param(V, dm, seed + 16, "exec_block.E_slot");
    params.E_op    = make_param(nop, dm, seed + 17, "exec_block.E_op");
    params.W_scal  = make_param(3, dm, seed + 18, "exec_block.W_scal");
    params.b_scal  = make_bias(dm,                "exec_block.b_scal");
    params.W_trace = make_param(K * dm, dm, seed + 19, "exec_block.W_trace");
    params.b_trace = make_bias(dm,                "exec_block.b_trace");

    // Reasoning state update: candidate + gated interpolation
    params.W_reason_gate = make_param(2 * dm, dm, seed + 20, "exec_block.W_reason_gate");
    params.W_trace_gate  = make_param(2 * dm, dm, seed + 21, "exec_block.W_trace_gate");
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
    parameters_(std::move(other.parameters_))
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
        parameters_          = std::move(other.parameters_);
    }
    return *this;
}

ExecutionBlockParameterTensors& ExecutionBlockLayer::parametersOrThrow() {
    if (!parameters_) {
        throw std::runtime_error("ExecutionBlockLayer: parameter storage is NULL");
    }
    return *parameters_;
}

const ExecutionBlockParameterTensors& ExecutionBlockLayer::parametersOrThrow() const {
    if (!parameters_) {
        throw std::runtime_error("ExecutionBlockLayer: parameter storage is NULL");
    }
    return *parameters_;
}

Tensor& ExecutionBlockLayer::w_decode_1() { return parametersOrThrow().w_decode_1; }
Tensor& ExecutionBlockLayer::b_decode_1() { return parametersOrThrow().b_decode_1; }
Tensor& ExecutionBlockLayer::w_decode_2() { return parametersOrThrow().w_decode_2; }
Tensor& ExecutionBlockLayer::w_arg1_select() { return parametersOrThrow().w_arg1_select; }
Tensor& ExecutionBlockLayer::w_arg2_select() { return parametersOrThrow().w_arg2_select; }
Tensor& ExecutionBlockLayer::W_op_select() { return parametersOrThrow().W_op_select; }
Tensor& ExecutionBlockLayer::W_key_proj() { return parametersOrThrow().W_key_proj; }
Tensor& ExecutionBlockLayer::W_write_query() { return parametersOrThrow().W_write_query; }
Tensor& ExecutionBlockLayer::W_write_key() { return parametersOrThrow().W_write_key; }
Tensor& ExecutionBlockLayer::alpha() { return parametersOrThrow().alpha; }
Tensor& ExecutionBlockLayer::beta() { return parametersOrThrow().beta; }
Tensor& ExecutionBlockLayer::step_embeddings() { return parametersOrThrow().step_embeddings; }
Tensor& ExecutionBlockLayer::type_num_embed() { return parametersOrThrow().type_num_embed; }
Tensor& ExecutionBlockLayer::W_value_to_emb() { return parametersOrThrow().W_value_to_emb; }
Tensor& ExecutionBlockLayer::b_value_to_emb() { return parametersOrThrow().b_value_to_emb; }
Tensor& ExecutionBlockLayer::w_inject_gate() { return parametersOrThrow().w_inject_gate; }
Tensor& ExecutionBlockLayer::W_Q_read() { return parametersOrThrow().W_Q_read; }
Tensor& ExecutionBlockLayer::W_K_read() { return parametersOrThrow().W_K_read; }
Tensor& ExecutionBlockLayer::W_V_read() { return parametersOrThrow().W_V_read; }
Tensor& ExecutionBlockLayer::W_O_read() { return parametersOrThrow().W_O_read; }
Tensor& ExecutionBlockLayer::W_gate_read() { return parametersOrThrow().W_gate_read; }
Tensor& ExecutionBlockLayer::tau() { return parametersOrThrow().tau; }
Tensor& ExecutionBlockLayer::E_slot() { return parametersOrThrow().E_slot; }
Tensor& ExecutionBlockLayer::E_op() { return parametersOrThrow().E_op; }
Tensor& ExecutionBlockLayer::W_scal() { return parametersOrThrow().W_scal; }
Tensor& ExecutionBlockLayer::b_scal() { return parametersOrThrow().b_scal; }
Tensor& ExecutionBlockLayer::W_trace() { return parametersOrThrow().W_trace; }
Tensor& ExecutionBlockLayer::b_trace() { return parametersOrThrow().b_trace; }
Tensor& ExecutionBlockLayer::W_reason_gate() { return parametersOrThrow().W_reason_gate; }
Tensor& ExecutionBlockLayer::W_trace_gate() { return parametersOrThrow().W_trace_gate; }

const Tensor& ExecutionBlockLayer::w_decode_1() const { return parametersOrThrow().w_decode_1; }
const Tensor& ExecutionBlockLayer::b_decode_1() const { return parametersOrThrow().b_decode_1; }
const Tensor& ExecutionBlockLayer::w_decode_2() const { return parametersOrThrow().w_decode_2; }
const Tensor& ExecutionBlockLayer::w_arg1_select() const { return parametersOrThrow().w_arg1_select; }
const Tensor& ExecutionBlockLayer::w_arg2_select() const { return parametersOrThrow().w_arg2_select; }
const Tensor& ExecutionBlockLayer::W_op_select() const { return parametersOrThrow().W_op_select; }
const Tensor& ExecutionBlockLayer::W_key_proj() const { return parametersOrThrow().W_key_proj; }
const Tensor& ExecutionBlockLayer::W_write_query() const { return parametersOrThrow().W_write_query; }
const Tensor& ExecutionBlockLayer::W_write_key() const { return parametersOrThrow().W_write_key; }
const Tensor& ExecutionBlockLayer::alpha() const { return parametersOrThrow().alpha; }
const Tensor& ExecutionBlockLayer::beta() const { return parametersOrThrow().beta; }
const Tensor& ExecutionBlockLayer::step_embeddings() const { return parametersOrThrow().step_embeddings; }
const Tensor& ExecutionBlockLayer::type_num_embed() const { return parametersOrThrow().type_num_embed; }
const Tensor& ExecutionBlockLayer::W_value_to_emb() const { return parametersOrThrow().W_value_to_emb; }
const Tensor& ExecutionBlockLayer::b_value_to_emb() const { return parametersOrThrow().b_value_to_emb; }
const Tensor& ExecutionBlockLayer::w_inject_gate() const { return parametersOrThrow().w_inject_gate; }
const Tensor& ExecutionBlockLayer::W_Q_read() const { return parametersOrThrow().W_Q_read; }
const Tensor& ExecutionBlockLayer::W_K_read() const { return parametersOrThrow().W_K_read; }
const Tensor& ExecutionBlockLayer::W_V_read() const { return parametersOrThrow().W_V_read; }
const Tensor& ExecutionBlockLayer::W_O_read() const { return parametersOrThrow().W_O_read; }
const Tensor& ExecutionBlockLayer::W_gate_read() const { return parametersOrThrow().W_gate_read; }
const Tensor& ExecutionBlockLayer::tau() const { return parametersOrThrow().tau; }
const Tensor& ExecutionBlockLayer::E_slot() const { return parametersOrThrow().E_slot; }
const Tensor& ExecutionBlockLayer::E_op() const { return parametersOrThrow().E_op; }
const Tensor& ExecutionBlockLayer::W_scal() const { return parametersOrThrow().W_scal; }
const Tensor& ExecutionBlockLayer::b_scal() const { return parametersOrThrow().b_scal; }
const Tensor& ExecutionBlockLayer::W_trace() const { return parametersOrThrow().W_trace; }
const Tensor& ExecutionBlockLayer::b_trace() const { return parametersOrThrow().b_trace; }
const Tensor& ExecutionBlockLayer::W_reason_gate() const { return parametersOrThrow().W_reason_gate; }
const Tensor& ExecutionBlockLayer::W_trace_gate() const { return parametersOrThrow().W_trace_gate; }

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
    const std::vector<ExecutionRecord>& prior_records)
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
        prior_records);
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
