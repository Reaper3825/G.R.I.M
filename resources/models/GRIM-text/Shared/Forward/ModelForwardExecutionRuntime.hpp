//======================================================//
//  ModelForwardExecutionRuntime.hpp
//
//  Typed execution-trace runtime owned by one caller of shared forward.
//  Training and inference each own their own instance; shared forward only
//  receives an explicit pointer to the active owner for the current call.
//======================================================//

#pragma once

#ifdef USE_CUDA

#include <vector>
#include <stdexcept>

#include "ModelForwardOutputs.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../Layers/ExecutionBlock/execution_block_GPU.hpp"

namespace GRIM {
namespace Forward {

struct ModelForwardExecutionRuntime {
    std::vector<std::vector<ExecutionRecord>> execution_trace_by_row;
    std::vector<Tensor> trace_state_by_row;

    // Durable home of the execution-step diagnostic / hardening buffers. The
    // per-step flags/records are Category 3 workspace (re-written each step);
    // reinforce_baseline is a Category 2 durable EMA. Allocated once via
    // ensureDiagnostics() and intentionally NOT touched by clear() so the EMA
    // survives across steps.
    ExecutionBlockDiagnosticsBuffers execution_diag;

    void ensureDiagnostics(cudaStream_t stream) {
        if (!stream) {
            throw std::runtime_error("ModelForwardExecutionRuntime::ensureDiagnostics: stream is NULL");
        }
        if (!execution_diag.allocated()) {
            execution_diag.allocate(stream);
        }
    }

    void ensureBatchGeometry(size_t batch_size, const char* caller) {
        if (batch_size == 0) {
            throw std::runtime_error(std::string(caller) + ": execution runtime batch_size must be > 0");
        }

        const bool traces_uninitialized = execution_trace_by_row.empty();
        const bool states_uninitialized = trace_state_by_row.empty();
        if (traces_uninitialized != states_uninitialized) {
            throw std::runtime_error(std::string(caller) +
                                     ": execution runtime geometry is inconsistent (execution_trace_by_row size=" +
                                     std::to_string(execution_trace_by_row.size()) +
                                     ", trace_state_by_row size=" +
                                     std::to_string(trace_state_by_row.size()) + ")");
        }

        if (traces_uninitialized) {
            execution_trace_by_row.resize(batch_size);
            trace_state_by_row.resize(batch_size);
            return;
        }

        if (execution_trace_by_row.size() != batch_size || trace_state_by_row.size() != batch_size) {
            throw std::runtime_error(std::string(caller) +
                                     ": execution runtime batch geometry mismatch expected=" +
                                     std::to_string(batch_size) +
                                     " execution_trace_by_row=" + std::to_string(execution_trace_by_row.size()) +
                                     " trace_state_by_row=" + std::to_string(trace_state_by_row.size()));
        }
    }

    void clear() {
        for (auto& row_trace : execution_trace_by_row) {
            row_trace.clear();
        }
        for (auto& trace_state : trace_state_by_row) {
            trace_state = Tensor();
        }
    }
};

//======================================================//
//  provisionExecutionForwardRuntime
//
//  Shared-forward-owned per-row reset + allocation for one execution-layer
//  boundary. Allocation of ExecutionMemory and the per-row trace_state lives
//  here (the runtime owner), not on any layer object: active rows get a fresh
//  zeroed register file and a [1, d_model] trace accumulator; inactive rows are
//  reset to empty. Row geometry must already be provisioned by
//  ModelForwardOutputs::ensureExecutionBatchGeometry and
//  ModelForwardExecutionRuntime::ensureBatchGeometry.
//======================================================//
inline void provisionExecutionForwardRuntime(
    const std::vector<bool>& execution_active,
    int batch_size,
    int num_slots,
    int atom_embedding_dim,
    int d_model,
    int d_key,
    int d_type,
    bool connect_parameter_graph,
    cudaStream_t stream,
    ModelForwardOutputs& forward_outputs,
    ModelForwardExecutionRuntime& execution_runtime)
{
    if (!stream) {
        throw std::runtime_error("provisionExecutionForwardRuntime: stream is NULL");
    }
    if (batch_size <= 0) {
        throw std::runtime_error("provisionExecutionForwardRuntime: batch_size must be positive");
    }
    if (!execution_active.empty() &&
        static_cast<int>(execution_active.size()) != batch_size) {
        throw std::runtime_error("provisionExecutionForwardRuntime: execution_active size must equal batch_size");
    }
    if (static_cast<int>(forward_outputs.exec_memories.size()) != batch_size ||
        static_cast<int>(forward_outputs.exec_outputs_per_row.size()) != batch_size ||
        static_cast<int>(execution_runtime.execution_trace_by_row.size()) != batch_size ||
        static_cast<int>(execution_runtime.trace_state_by_row.size()) != batch_size) {
        throw std::runtime_error("provisionExecutionForwardRuntime: execution runtime geometry must equal batch_size "
                                 "(call ensureExecutionBatchGeometry/ensureBatchGeometry first)");
    }

    for (int b = 0; b < batch_size; ++b) {
        const size_t row = static_cast<size_t>(b);
        execution_runtime.execution_trace_by_row[row].clear();
        forward_outputs.exec_outputs_per_row[row].steps.clear();
        auto& row_memory = forward_outputs.exec_memories[row];

        const bool row_exec_active = !execution_active.empty() && execution_active[row];
        if (!row_exec_active) {
            row_memory = ExecutionMemory();
            execution_runtime.trace_state_by_row[row] = Tensor();
            continue;
        }

        row_memory = ExecutionMemory();
        row_memory.allocate(num_slots, atom_embedding_dim, d_model, d_key, d_type, stream);
        row_memory.clear(stream);

        execution_runtime.trace_state_by_row[row] =
            Tensor::zeros({1, d_model}, stream, "trace_state_row");
        if (connect_parameter_graph) {
            execution_runtime.trace_state_by_row[row].requires_grad_();
            execution_runtime.trace_state_by_row[row].ensure_grad();
        } else {
            execution_runtime.trace_state_by_row[row].requires_grad = false;
        }
    }
}

}  // namespace Forward
}  // namespace GRIM

#endif  // USE_CUDA