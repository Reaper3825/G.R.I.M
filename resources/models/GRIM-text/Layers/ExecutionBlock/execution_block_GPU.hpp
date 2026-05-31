//======================================================//
//  execution_block_GPU.hpp
//  Differentiable Register Machine — GPU
//
//  Declares: ExecutionMemory, ExecutionBlockLayer.
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

namespace GRIM {

struct ExecutionBlockParameterTensors;

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
//  ExecutionBlockLayer
//======================================================//
class ExecutionBlockLayer {
public:
    ExecutionBlockLayer() = delete;

    explicit ExecutionBlockLayer(const HyperParameters::ExecutionBlockConstructionHP& hp,
                                ExecutionBlockParameterTensors& parameters,
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
    ExecutionBlockParameterTensors* parameters_ = nullptr;
};

}  // namespace GRIM

#endif  // USE_CUDA
