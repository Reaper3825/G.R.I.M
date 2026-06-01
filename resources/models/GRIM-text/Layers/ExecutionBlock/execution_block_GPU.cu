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
using Forward::ExecutionBlockStepOutput;
using Forward::ExecutionRecord;

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
                                         cudaStream_t init_stream)
        : hp_(hp)
{
    // ExecutionBlockConstructionHP is already validated upstream by
    // validateRootConfigDocument() (root config) and
    // validateExecutionBlockConstructionHP() (startup registration) before this
    // layer is constructed; a redundant config re-check here is forbidden.
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
    d_reinforce_baseline_(other.d_reinforce_baseline_)
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
    }
    return *this;
}

//======================================================//
//  Thin public wrappers
//======================================================//
void ExecutionBlockLayer::executeStep(
    Tensor& H,
    ExecutionMemory& M,
    ExecutionBlockParameterTensors& parameters,
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
    // BatchPayload owns/validates per-batch geometry (batch_size, seq_lengths,
    // max_seq_len, total_tokens) via BatchPayload::validate() +
    // validateExecutionPayload(). Only the local call-boundary contract that
    // BatchPayload does not own is checked here.
    EXEC_CHECK_SHAPE2(H, "H (executeStep)", payload.total_tokens, hp_.d_model);
    EXEC_CHECK(atom_positions != nullptr,
               "executeStep: atom_positions is null - caller MUST provide a row-local atom view");
    EXEC_CHECK(num_atoms >= 0, "executeStep: num_atoms must be non-negative");
    EXEC_CHECK(bindings.d_token_to_slot_map != nullptr,
               "executeStep: bindings.d_token_to_slot_map is null");
    EXEC_CHECK(step >= 0 && step < hp_.num_exec_steps, "executeStep: step out of range");
    EXEC_CHECK(batch_row >= 0 && batch_row < payload.batch_size,
               "executeStep: batch_row out of range for payload.batch_size");
    executeStepCoordinatorImpl(
        *this,
        parameters,
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
    ExecutionBlockParameterTensors& parameters,
    int total_tokens,
    cudaStream_t stream,
    int token_offset,
    int row_tokens,
    float* d_gate_accum)
{
    EXEC_CHECK_SHAPE2(hidden_states, "hidden_states (cross-attn)", total_tokens, hp_.d_model);
    if (row_tokens < 0) row_tokens = total_tokens;
    EXEC_CHECK(token_offset >= 0, "token_offset must be non-negative");
    EXEC_CHECK(row_tokens > 0, "row_tokens must be positive");
    EXEC_CHECK(token_offset + row_tokens <= total_tokens,
               "crossAttentionRead row-local span exceeds total token extent");
    return crossAttentionReadImpl(*this, parameters, hidden_states, M, stream, token_offset, row_tokens, d_gate_accum);
}

}  // namespace GRIM
