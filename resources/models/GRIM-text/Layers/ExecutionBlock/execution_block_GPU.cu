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

void executionBlockPredictGate(
    const HyperParameters::ExecutionBlockConstructionHP& hp,
    Tensor& H,
    ExecutionBlockParameterTensors& parameters,
    const Batching::BatchPayload& payload,
    int batch_row,
    cudaStream_t stream,
    Forward::ExecutionGateOutput* output)
{
    EXEC_CHECK_SHAPE2(H, "H (execution gate)", payload.total_tokens, hp.d_model);
    EXEC_CHECK(batch_row >= 0 && batch_row < payload.batch_size,
               "execution gate batch_row out of range");
    EXEC_CHECK(output != nullptr, "execution gate output is NULL");
    predictExecutionGateImpl(hp, parameters, H, payload, batch_row, stream, output);
}

//======================================================//
//  Thin-coordinator local kernel
//======================================================//

//======================================================//
//  ExecutionBlockDiagnosticsBuffers — persistent device buffers
//======================================================//
void ExecutionBlockDiagnosticsBuffers::allocate(cudaStream_t stream) {
    EXEC_CHECK(stream != nullptr, "ExecutionBlockDiagnosticsBuffers::allocate: stream is NULL");
    EXEC_CHECK(!allocated(), "ExecutionBlockDiagnosticsBuffers::allocate: already allocated");

    cudaMallocOrThrow(reinterpret_cast<void**>(&d_numeric_error_flag), sizeof(int), "exec_numeric_error_flag");
    CUDA_CHECK(cudaMemsetAsync(d_numeric_error_flag, 0, sizeof(int), stream));
    cudaMallocOrThrow(reinterpret_cast<void**>(&d_div_clamp_count), sizeof(int), "exec_div_clamp_count");
    CUDA_CHECK(cudaMemsetAsync(d_div_clamp_count, 0, sizeof(int), stream));
    cudaMallocOrThrow(reinterpret_cast<void**>(&d_div_invalid_flag), sizeof(int), "exec_div_invalid_flag");
    CUDA_CHECK(cudaMemsetAsync(d_div_invalid_flag, 0, sizeof(int), stream));
    cudaMallocOrThrow(reinterpret_cast<void**>(&d_exec_idx), 4 * sizeof(int), "exec_idx");
    cudaMallocOrThrow(reinterpret_cast<void**>(&d_exec_record_i), 3 * sizeof(int), "exec_record_i");
    cudaMallocOrThrow(reinterpret_cast<void**>(&d_exec_record_f), 3 * sizeof(float), "exec_record_f");
}

void ExecutionBlockDiagnosticsBuffers::destroy() {
    if (d_numeric_error_flag) cudaFree(d_numeric_error_flag);
    if (d_div_clamp_count)    cudaFree(d_div_clamp_count);
    if (d_div_invalid_flag)   cudaFree(d_div_invalid_flag);
    if (d_exec_idx)           cudaFree(d_exec_idx);
    if (d_exec_record_i)      cudaFree(d_exec_record_i);
    if (d_exec_record_f)      cudaFree(d_exec_record_f);
    d_numeric_error_flag = nullptr;
    d_div_clamp_count    = nullptr;
    d_div_invalid_flag   = nullptr;
    d_exec_idx           = nullptr;
    d_exec_record_i      = nullptr;
    d_exec_record_f      = nullptr;
}

//======================================================//
//  Execution-block free operations
//======================================================//
void executionBlockStep(
    const HyperParameters::ExecutionBlockConstructionHP& hp,
    ExecutionBlockDiagnosticsBuffers& diag,
    Tensor& H,
    ExecutionMemory& M,
    ExecutionBlockParameterTensors& parameters,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    int batch_row,
    int step,
    bool teacher_force_transition,
    float temperature,
    cudaStream_t stream,
    ExecutionBlockStepOutput& forward_output,
    Forward::RecordEncodeBackwardStaging& record_encode_backward_staging,
    Tensor& trace_state,
    const std::vector<ExecutionRecord>& prior_records,
    const Tensor* selector_candidate_keys,
    const Tensor* slot_seeds)
{
    // BatchPayload owns/validates per-batch geometry (batch_size, seq_lengths,
    // max_seq_len, total_tokens) via BatchPayload::validate() +
    // validateExecutionPayload(). Only the local call-boundary contract that
    // BatchPayload does not own is checked here.
    EXEC_CHECK_SHAPE2(H, "H (executeStep)", payload.total_tokens, hp.d_model);
    EXEC_CHECK(bindings.d_token_to_slot_map != nullptr,
               "executeStep: bindings.d_token_to_slot_map is null");
    EXEC_CHECK(bindings.d_atom_mask != nullptr,
               "executeStep: bindings.d_atom_mask is null - execution sources atom "
               "positions directly from the global atom mask");
    EXEC_CHECK(step >= 0 && step < hp.num_exec_steps, "executeStep: step out of range");
    EXEC_CHECK(batch_row >= 0 && batch_row < payload.batch_size,
               "executeStep: batch_row out of range for payload.batch_size");
    executeStepCoordinatorImpl(
        hp,
        diag,
        parameters,
        H,
        M,
        payload,
        bindings,
        batch_row,
        step,
        teacher_force_transition,
        temperature,
        stream,
        forward_output,
        record_encode_backward_staging,
        trace_state,
        prior_records,
        selector_candidate_keys,
        slot_seeds);
}

//======================================================//
//  executionBlockCrossAttentionRead
//======================================================//
Tensor executionBlockCrossAttentionRead(
    const HyperParameters::ExecutionBlockConstructionHP& hp,
    const Tensor& hidden_states,
    ExecutionMemory& M,
    ExecutionBlockParameterTensors& parameters,
    int total_tokens,
    cudaStream_t stream,
    int token_offset,
    int row_tokens,
    float* d_gate_accum)
{
    EXEC_CHECK_SHAPE2(hidden_states, "hidden_states (cross-attn)", total_tokens, hp.d_model);
    if (row_tokens < 0) row_tokens = total_tokens;
    EXEC_CHECK(token_offset >= 0, "token_offset must be non-negative");
    EXEC_CHECK(row_tokens > 0, "row_tokens must be positive");
    EXEC_CHECK(token_offset + row_tokens <= total_tokens,
               "crossAttentionRead row-local span exceeds total token extent");
    return crossAttentionReadImpl(hp, parameters, hidden_states, M, stream, token_offset, row_tokens, d_gate_accum);
}

}  // namespace GRIM
