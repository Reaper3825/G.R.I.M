//======================================================//
//  execution_block_GPU.hpp
//  Differentiable Register Machine — GPU
//
//  Declares: ExecutionMemory, ExecutionBlockDiagnosticsBuffers, execution-block free operations.
//  Durable trainable parameter ownership lives in
//  training/Phases/Startup/Model/ParameterRegistry.hpp.
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
#include "../../Shared/Batching/BatchPayload.hpp"
#include "../../Shared/Batching/BatchDeviceBindings.hpp"
#include "../../training/Phases/Startup/Model/ParameterRegistry.hpp"

namespace GRIM {
namespace Forward {
struct ExecStepMetrics;
struct ExecutionRecord;
struct ExecutionGateOutput;
struct ExecutionBlockStepOutput;
struct ExecutionBlockOutput;
}

//======================================================//
//  ExecutionMemory — addressable register file
//
//  Each instance represents ONE batch row's register file [V, …].
//  Per-row isolation: the active forward sink stores a vector<ExecutionMemory>
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
//  ExecutionBlockDiagnosticsBuffers
//
//  Persistent device-side diagnostic / hardening buffers for the execution
//  step. Allocated once, reused every step. The per-step flags/records are
//  Category 3 workspace (stale contents between steps); reinforce_baseline is a
//  durable EMA. Owned by one long-lived owner; the math ops borrow it.
//======================================================//
struct ExecutionBlockDiagnosticsBuffers {
    int* d_numeric_error_flag = nullptr;  // atomicMax stage-id: numeric, softmax, collapse
    int* d_div_clamp_count    = nullptr;  // atomicAdd on division clamp
    int* d_div_invalid_flag   = nullptr;  // [1] per-step: 1 if division was clamped, 0 otherwise
    int* d_exec_idx           = nullptr;  // [4] arg1_rel, arg2_rel, op_id, write_slot (abs)
    int* d_exec_record_i      = nullptr;  // [3] packed for ExecutionRecord ints
    float* d_exec_record_f    = nullptr;  // [3] value_before_1, value_before_2, value_after
    float* d_reinforce_baseline = nullptr; // [1] EMA of transition_err for REINFORCE variance reduction

    ExecutionBlockDiagnosticsBuffers() = default;
    ~ExecutionBlockDiagnosticsBuffers() { destroy(); }

    ExecutionBlockDiagnosticsBuffers(const ExecutionBlockDiagnosticsBuffers&) = delete;
    ExecutionBlockDiagnosticsBuffers& operator=(const ExecutionBlockDiagnosticsBuffers&) = delete;
    ExecutionBlockDiagnosticsBuffers(ExecutionBlockDiagnosticsBuffers&& other) noexcept { moveFrom(other); }
    ExecutionBlockDiagnosticsBuffers& operator=(ExecutionBlockDiagnosticsBuffers&& other) noexcept {
        if (this != &other) { destroy(); moveFrom(other); }
        return *this;
    }

    bool allocated() const { return d_numeric_error_flag != nullptr; }
    void allocate(cudaStream_t stream);  // defined in execution_block_GPU.cu
    void destroy();                      // defined in execution_block_GPU.cu

    int* numericErrorFlag() const { return d_numeric_error_flag; }
    int* divClampCount() const    { return d_div_clamp_count; }
    int* divInvalidFlag() const   { return d_div_invalid_flag; }
    int* execIndices() const      { return d_exec_idx; }
    int* execRecordI() const      { return d_exec_record_i; }
    float* execRecordF() const    { return d_exec_record_f; }
    float* reinforceBaseline() const { return d_reinforce_baseline; }

private:
    void moveFrom(ExecutionBlockDiagnosticsBuffers& other) noexcept {
        d_numeric_error_flag = other.d_numeric_error_flag;
        d_div_clamp_count    = other.d_div_clamp_count;
        d_div_invalid_flag   = other.d_div_invalid_flag;
        d_exec_idx           = other.d_exec_idx;
        d_exec_record_i      = other.d_exec_record_i;
        d_exec_record_f      = other.d_exec_record_f;
        d_reinforce_baseline = other.d_reinforce_baseline;
        other.d_numeric_error_flag = nullptr;
        other.d_div_clamp_count    = nullptr;
        other.d_div_invalid_flag   = nullptr;
        other.d_exec_idx           = nullptr;
        other.d_exec_record_i      = nullptr;
        other.d_exec_record_f      = nullptr;
        other.d_reinforce_baseline = nullptr;
    }
};

//======================================================//
//  Execution-block free operations
//
//  The execution math no longer hangs off ExecutionBlockLayer state. Callers
//  pass the construction hyperparameters explicitly, plus a runtime-owned
//  ExecutionBlockDiagnosticsBuffers (durable REINFORCE baseline + per-step
//  workspace) where the step machinery needs one.
//======================================================//

//--------------------------------------------------//
// Activation control head. Reads only the row-relative planner query token.
//--------------------------------------------------//
void executionBlockPredictGate(
    const HyperParameters::ExecutionBlockConstructionHP& hp,
    Tensor& H,
    ExecutionBlockParameterTensors& parameters,
    const Batching::BatchPayload& payload,
    int batch_row,
    cudaStream_t stream,
    Forward::ExecutionGateOutput* output);

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
//
// Atom positions are NOT passed in: the step sources them directly from the
// global atom mask (bindings.d_atom_mask) for this batch row.
//--------------------------------------------------//
void executionBlockStep(
    const HyperParameters::ExecutionBlockConstructionHP& hp,
    ExecutionBlockDiagnosticsBuffers& diag,
    Tensor& H,                          // [total_tokens, d_model] mutated in place
    ExecutionMemory& M,
    ExecutionBlockParameterTensors& parameters,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    int batch_row,
    int step,
    float temperature,
    cudaStream_t stream,
    Forward::ExecutionBlockStepOutput& forward_output,
    Tensor& trace_state,
    const std::vector<Forward::ExecutionRecord>& prior_records
);

//--------------------------------------------------//
// Bootstrap: copy literal values into M.values via slot map (detached, no grad)
//--------------------------------------------------//
void executionBlockBootstrapMemoryFromSlotMap(
    const HyperParameters::ExecutionBlockConstructionHP& hp,
    ExecutionMemory& M,
    ExecutionBlockParameterTensors& parameters,
    const float* device_numeric_values,  // [row_tokens]
    const int32_t* device_slot_map,      // [row_tokens]
    int row_tokens,
    cudaStream_t stream
);

//--------------------------------------------------//
// Cross-attention read: H = H + g * W_O(R)
// token_offset / row_tokens enable per-batch-row processing.
//--------------------------------------------------//
Tensor executionBlockCrossAttentionRead(
    const HyperParameters::ExecutionBlockConstructionHP& hp,
    const Tensor& hidden_states,
    ExecutionMemory& M,
    ExecutionBlockParameterTensors& parameters,
    int total_tokens,
    cudaStream_t stream,
    int token_offset = 0,
    int row_tokens = -1,
    float* d_gate_accum = nullptr  // [2] device: [sum, count] for telemetry
);

}  // namespace GRIM

#endif  // USE_CUDA
