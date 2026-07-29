//======================================================//
//  ModelForwardExecutionBlock_GPU.cu
//  ExecutionBlock portions of the shared model forward
//======================================================//

#ifndef USE_CUDA
#define USE_CUDA
#endif

#include "ModelForwardExecutionBlock_GPU.hpp"

#include "../HyperParameters/HyperparameterGroupings.hpp"
#include "../../Layers/ExecutionBlock/execution_block_GPU.hpp"
#include "../../Layers/SlotSeedEncoder/SlotSeedEncoder_GPU.hpp"
#include "../../training/Phases/Startup/Model/ParameterRegistry.hpp"

#include <array>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace GRIM {
namespace Forward {
namespace {

int requirePayloadRowLength(const Batching::BatchPayload& payload,
                            int row,
                            const char* caller) {
    if (row < 0 || row >= payload.batch_size) {
        throw std::runtime_error(std::string(caller) + ": row index " +
                                 std::to_string(row) + " out of range for batch_size=" +
                                 std::to_string(payload.batch_size));
    }
    if (static_cast<int>(payload.seq_lengths.size()) != payload.batch_size) {
        throw std::runtime_error(std::string(caller) + ": payload.seq_lengths size (" +
                                 std::to_string(payload.seq_lengths.size()) +
                                 ") != batch_size (" + std::to_string(payload.batch_size) + ")");
    }
    const int row_len = payload.seq_lengths[static_cast<size_t>(row)];
    if (row_len <= 0 || row_len > payload.max_seq_len) {
        throw std::runtime_error(std::string(caller) + ": invalid seq_lengths[" +
                                 std::to_string(row) + "]=" + std::to_string(row_len) +
                                 " for payload.max_seq_len=" + std::to_string(payload.max_seq_len));
    }
    return row_len;
}

std::array<float, 2> readBinaryControlProbabilities(
    const Tensor& probabilities,
    cudaStream_t stream,
    const char* caller)
{
    probabilities.require(caller);
    if (probabilities.numel() != 2) {
        throw std::runtime_error(std::string(caller) + ": expected exactly two probabilities");
    }
    std::array<float, 2> host{};
    cudaError_t copy_err = cudaMemcpyAsync(
        host.data(), probabilities.data, 2 * sizeof(float),
        cudaMemcpyDeviceToHost, stream);
    if (copy_err != cudaSuccess) {
        throw std::runtime_error(std::string(caller) + ": cudaMemcpyAsync failed: " +
                                 cudaGetErrorString(copy_err));
    }
    cudaError_t sync_err = cudaStreamSynchronize(stream);
    if (sync_err != cudaSuccess) {
        throw std::runtime_error(std::string(caller) + ": cudaStreamSynchronize failed: " +
                                 cudaGetErrorString(sync_err));
    }
    return host;
}

GRIM::SlotSeedEncoderParameterTensors detachSlotSeedEncoderParameters(
    const GRIM::SlotSeedEncoderParameterTensors& parameters,
    const HyperParameters::SlotSeedEncoderConstructionHP& hp,
    cudaStream_t stream)
{
    GRIM::SlotSeedEncoderParameterTensors detached{};
    detached.W_seed_in = parameters.W_seed_in.detach(stream);
    detached.W_seed_out = parameters.W_seed_out.detach(stream);
    if (hp.bias_enabled) {
        detached.b_seed_in = parameters.b_seed_in.detach(stream);
        detached.b_seed_out = parameters.b_seed_out.detach(stream);
    }
    if (hp.type_embedding_enabled) {
        detached.type_embeddings = parameters.type_embeddings.detach(stream);
    }
    return detached;
}

void materializeForwardSlotSeeds(
    const ModelForwardRequest& request,
    const HyperParameters::SlotSeedEncoderConstructionHP& hp,
    const Tensor& contextual_hidden_states,
    const Batching::BatchPayload& payload,
    int num_slots,
    ModelForwardOutputs& forward_outputs)
{
    if (!hp.enabled) {
        return;
    }

    const auto& registered =
        request.parameter_registry->requireSlotSeedEncoderParameters(
            "executeModelForward(slot_seed_encoder)");
    const GRIM::SlotSeedEncoderParameterTensors* active = &registered;
    GRIM::SlotSeedEncoderParameterTensors detached{};
    if (!request.graph.connect_parameter_graph) {
        detached = detachSlotSeedEncoderParameters(registered, hp, request.stream);
        active = &detached;
    }

    SlotSeedEncoder::forward(
        hp,
        *active,
        contextual_hidden_states,
        payload,
        *request.bindings,
        num_slots,
        request.stream,
        forward_outputs);
}

}  // namespace

bool applyExecutionBlockReadback(
    const ModelForwardRequest& request,
    const HyperParameters::ExecutionBlockConstructionHP& execution_hp,
    ExecutionBlockParameterTensors* execution_block_parameters,
    int total_tokens,
    int layer_idx,
    int exec_layer,
    bool execution_block_active,
    const std::vector<bool>& execution_active_by_row,
    const Tensor& layer_input,
    Tensor& execution_read_augmented_input,
    ModelForwardRuntimePayload& runtime,
    ModelForwardOutputs& forward_outputs)
{
    if (exec_layer < 0 || layer_idx <= exec_layer || !execution_block_active) {
        return false;
    }

    const auto& payload = *request.payload;
    if (payload.isInferenceDecode()) {
        if (!runtime.persistent_execution_memory) {
            return false;
        }
        if (payload.batch_size != 1) {
            throw std::runtime_error(
                "ModelForward: persistent decode execution memory requires batch_size == 1");
        }
        const int row_len = requirePayloadRowLength(
            payload, 0, "ModelForward persistent decode readback");
        Tensor row_delta = GRIM::executionBlockCrossAttentionRead(
            execution_hp, layer_input, *runtime.persistent_execution_memory,
            *execution_block_parameters, total_tokens, request.stream,
            /*token_offset=*/0, row_len,
            runtime.read_gate_accum_tensor
                ? runtime.read_gate_accum_tensor->data
                : nullptr);
        execution_read_augmented_input = autograd::add(
            layer_input, row_delta, request.stream);
        runtime.persistent_execution_memory_was_read = true;
        return true;
    }

    if (forward_outputs.exec_memories.empty()) {
        return false;
    }

    bool has_execution_readback = false;
    for (int b = 0; b < payload.batch_size; ++b) {
        const bool row_exec_active = !execution_active_by_row.empty()
            && execution_active_by_row[static_cast<size_t>(b)];
        if (!row_exec_active) continue;
        if (payload.execution_prompt_end_positions.empty()) {
            throw std::runtime_error(
                "ModelForward ExecutionBlock readback requires "
                "execution_prompt_end_positions");
        }
        const Tensor& read_source = has_execution_readback
            ? execution_read_augmented_input
            : layer_input;
        const int row_len = requirePayloadRowLength(
            payload, b, "ModelForward ExecutionBlock readback");
        const int prompt_end =
            payload.execution_prompt_end_positions[static_cast<size_t>(b)];
        if (prompt_end < 0 || prompt_end >= row_len) {
            throw std::runtime_error(
                "ModelForward ExecutionBlock readback prompt_end=" +
                std::to_string(prompt_end) + " is outside row " +
                std::to_string(b) + " length " + std::to_string(row_len));
        }
        const int readback_token_offset =
            b * payload.max_seq_len + prompt_end;
        Tensor row_delta = GRIM::executionBlockCrossAttentionRead(
            execution_hp, read_source, forward_outputs.exec_memories[b],
            *execution_block_parameters, total_tokens, request.stream,
            readback_token_offset, 1,
            runtime.read_gate_accum_tensor
                ? runtime.read_gate_accum_tensor->data
                : nullptr);
        Tensor padded = autograd::zero_pad(
            row_delta, readback_token_offset, total_tokens, request.stream);
        execution_read_augmented_input = autograd::add(
            read_source, padded, request.stream);
        has_execution_readback = true;
    }
    return has_execution_readback;
}

void runExecutionBlockNoGraph(
    const ModelForwardRequest& request,
    const HyperParameters::ExecutionBlockConstructionHP& execution_hp,
    const HyperParameters::SlotSeedEncoderConstructionHP& slot_seed_encoder_hp,
    ExecutionBlockParameterTensors* execution_block_parameters,
    int layer_idx,
    int exec_layer,
    int exec_step_count,
    bool execution_block_active,
    bool execution_selector_bridge_requested,
    Tensor& layer_output,
    ModelForwardRuntimePayload& runtime,
    ModelForwardOutputs& forward_outputs,
    std::vector<bool>& execution_active_by_row)
{
    const auto& payload = *request.payload;
    if (layer_idx == exec_layer &&
        execution_block_active &&
        slot_seed_encoder_hp.enabled) {
        materializeForwardSlotSeeds(
            request,
            slot_seed_encoder_hp,
            layer_output,
            payload,
            execution_hp.num_slots,
            forward_outputs);
    }

    if (!payload.isInferencePrefill() ||
        layer_idx != exec_layer ||
        !execution_block_active) {
        return;
    }

    std::vector<bool> provision_rows(
        static_cast<size_t>(payload.batch_size), true);
    auto& execution_runtime = *runtime.execution_runtime;
    Forward::provisionExecutionForwardRuntime(
        provision_rows,
        payload.batch_size,
        execution_hp.num_slots,
        execution_hp.atom_embedding_dim,
        execution_hp.d_model,
        execution_hp.d_key,
        execution_hp.d_type,
        false,
        request.stream,
        forward_outputs,
        execution_runtime);
    execution_runtime.ensureDiagnostics(request.stream);

    for (int b = 0; b < payload.batch_size; ++b) {
        auto& row_output = forward_outputs.exec_outputs_per_row[b];
        GRIM::executionBlockPredictGate(
            execution_hp,
            layer_output,
            *execution_block_parameters,
            payload,
            b,
            request.stream,
            &row_output.gate);
        const auto gate_probs = readBinaryControlProbabilities(
            row_output.gate.probabilities,
            request.stream,
            "ModelForward(no_grad) execution gate");
        row_output.gate.noop_probability = gate_probs[0];
        row_output.gate.execute_probability = gate_probs[1];
        row_output.gate.predicted_class = gate_probs[1] > gate_probs[0] ? 1 : 0;

        const int row_len = requirePayloadRowLength(
            payload, b, "ModelForward(no_grad) execution bootstrap");
        const int row_offset = b * payload.max_seq_len;
        bool has_bootstrap_slot = false;
        for (int t = 0; t < row_len; ++t) {
            if (payload.token_to_slot_index_map[static_cast<size_t>(row_offset + t)] >= 0) {
                has_bootstrap_slot = true;
                break;
            }
        }
        const bool execute_row = row_output.gate.predicted_class == 1
            && has_bootstrap_slot;
        row_output.execution_suppressed_no_bootstrap =
            row_output.gate.predicted_class == 1 && !has_bootstrap_slot;
        execution_active_by_row[static_cast<size_t>(b)] = execute_row;
        if (!execute_row) continue;

        if (!request.bindings || !request.bindings->d_token_to_slot_index_map
            || !request.bindings->d_atom_entry_ids
            || !request.bindings->d_pool_numeric_float_values
            || !request.bindings->d_pool_numeric_int_values
            || !request.bindings->d_pool_numeric_kinds
            || !request.bindings->d_bootstrap_slot_to_pool_index
            || request.bindings->num_pool_atoms <= 0) {
            throw std::runtime_error(
                "ModelForward(no_grad): execution decision has no atom-entry-pool "
                "bootstrap bindings");
        }
        auto& memory = forward_outputs.exec_memories[b];
        if (!forward_outputs.selector_candidate_keys.data) {
            throw std::runtime_error(
                "ModelForward(no_grad): execution bootstrap has selector bridge "
                "metadata but no candidate-key tensor");
        }
        GRIM::executionBlockBootstrapMemoryFromSlotMap(
            execution_hp,
            memory,
            *execution_block_parameters,
            payload,
            *request.bindings,
            b,
            forward_outputs.selector_candidate_keys.data,
            request.stream);

        bool stopped = false;
        for (int step = 0; step < exec_step_count; ++step) {
            ExecutionBlockStepOutput step_output;
            auto& record_encode_backward_staging =
                forward_outputs.appendRecordEncodeBackwardStaging(
                    1, request.stream);
            GRIM::executionBlockStep(
                execution_hp,
                execution_runtime.execution_diag,
                layer_output,
                memory,
                *execution_block_parameters,
                payload,
                *request.bindings,
                b,
                step,
                execution_hp.temp_start,
                request.stream,
                step_output,
                record_encode_backward_staging,
                execution_runtime.trace_state_by_row[b],
                execution_runtime.execution_trace_by_row[b],
                execution_selector_bridge_requested
                    ? &forward_outputs.selector_candidate_keys
                    : nullptr,
                slot_seed_encoder_hp.enabled
                    ? &forward_outputs.slot_seeds
                    : nullptr);
            execution_runtime.execution_trace_by_row[b].push_back(step_output.record);

            const auto stop_probs = readBinaryControlProbabilities(
                step_output.stop_probabilities,
                request.stream,
                "ModelForward(no_grad) stop control");
            step_output.continue_probability = stop_probs[0];
            step_output.stop_probability = stop_probs[1];
            step_output.stop_predicted_class = stop_probs[1] >= stop_probs[0] ? 1 : 0;
            stopped = step_output.stop_predicted_class == 1;
            row_output.steps.push_back(std::move(step_output));
            if (stopped) {
                row_output.stopped_by_model = true;
                break;
            }
        }
        if (!stopped) {
            row_output.stopped_at_max_steps = true;
        }
    }
}

void runExecutionBlockConnectedGraph(
    const ModelForwardRequest& request,
    const HyperParameters::ExecutionBlockConstructionHP& execution_hp,
    const HyperParameters::SlotSeedEncoderConstructionHP& slot_seed_encoder_hp,
    ExecutionBlockParameterTensors* execution_block_parameters,
    int layer_idx,
    int exec_layer,
    int exec_step_count,
    bool execution_block_active,
    bool execution_selector_bridge_requested,
    Tensor& layer_output,
    ModelForwardRuntimePayload& runtime,
    ModelForwardOutputs& forward_outputs)
{
    const auto& payload = *request.payload;
    if (layer_idx == exec_layer &&
        execution_block_active &&
        slot_seed_encoder_hp.enabled) {
        materializeForwardSlotSeeds(
            request,
            slot_seed_encoder_hp,
            layer_output,
            payload,
            execution_hp.num_slots,
            forward_outputs);
    }

    if (layer_idx != exec_layer || !execution_block_active) {
        return;
    }

    const float T = execution_hp.temp_start;

    auto& execution_runtime = *runtime.execution_runtime;
    Forward::provisionExecutionForwardRuntime(
        payload.execution_active,
        payload.batch_size,
        execution_hp.num_slots,
        execution_hp.atom_embedding_dim,
        execution_hp.d_model,
        execution_hp.d_key,
        execution_hp.d_type,
        true,
        request.stream,
        forward_outputs,
        execution_runtime);
    execution_runtime.ensureDiagnostics(request.stream);

    for (int b = 0; b < payload.batch_size; ++b) {
        const bool row_exec_active = !payload.execution_active.empty()
            && payload.execution_active[b];

        const bool gate_supervised = !payload.execution_gate_targets.empty()
            && payload.execution_gate_targets[b]
                != Execution::ExecutionGateTarget::UNSUPERVISED;
        if (gate_supervised) {
            GRIM::executionBlockPredictGate(
                execution_hp,
                layer_output,
                *execution_block_parameters,
                payload,
                b,
                request.stream,
                &forward_outputs.exec_outputs_per_row[b].gate);
        }

        if (!row_exec_active) continue;

        auto& M_b = forward_outputs.exec_memories[b];

        requirePayloadRowLength(
            payload, b, "ModelForward ExecutionBlock bootstrap");

        if (!request.bindings || !request.bindings->d_token_to_slot_index_map
            || !request.bindings->d_atom_entry_ids
            || !request.bindings->d_pool_numeric_float_values
            || !request.bindings->d_pool_numeric_int_values
            || !request.bindings->d_pool_numeric_kinds
            || !request.bindings->d_bootstrap_slot_to_pool_index
            || request.bindings->num_pool_atoms <= 0) {
            throw std::runtime_error(
                "ModelForward: execution-active row " + std::to_string(b)
                + " has no slot map or atom-entry-pool values for bootstrap; "
                "compiled payload marks row active but pool data is missing");
        }
        if (!forward_outputs.selector_candidate_keys.data) {
            throw std::runtime_error(
                "ModelForward: execution-active row has selector bridge metadata "
                "but no candidate-key tensor");
        }
        GRIM::executionBlockBootstrapMemoryFromSlotMap(
            execution_hp,
            M_b,
            *execution_block_parameters,
            payload,
            *request.bindings,
            b,
            forward_outputs.selector_candidate_keys.data,
            request.stream);

        if (payload.teacher_step_mask.empty()
            || static_cast<int>(payload.teacher_step_mask.size()) <= b) {
            throw std::runtime_error(
                "ModelForward: execution-active row has no teacher_step_mask");
        }
        const auto& step_mask = payload.teacher_step_mask[b];
        int real_step_count = 0;
        bool saw_padding = false;
        for (int step = 0; step < exec_step_count; ++step) {
            const bool is_real = step < static_cast<int>(step_mask.size())
                && step_mask[static_cast<size_t>(step)] != 0;
            if (!is_real) {
                saw_padding = true;
                continue;
            }
            if (saw_padding) {
                throw std::runtime_error(
                    "ModelForward: teacher_step_mask must contain a contiguous real-step prefix");
            }
            ++real_step_count;
        }
        if (real_step_count <= 0) {
            throw std::runtime_error(
                "ModelForward: execution-active row has zero real teacher steps");
        }

        for (int step = 0; step < real_step_count; ++step) {
            ExecutionBlockStepOutput step_diag;
            auto& record_encode_backward_staging =
                forward_outputs.appendRecordEncodeBackwardStaging(
                    1, request.stream);

            GRIM::executionBlockStep(
                execution_hp, execution_runtime.execution_diag,
                layer_output, M_b, *execution_block_parameters,
                payload, *request.bindings, b,
                step,
                T, request.stream,
                step_diag,
                record_encode_backward_staging,
                runtime.execution_runtime->trace_state_by_row[b],
                runtime.execution_runtime->execution_trace_by_row[b],
                execution_selector_bridge_requested
                    ? &forward_outputs.selector_candidate_keys
                    : nullptr,
                slot_seed_encoder_hp.enabled
                    ? &forward_outputs.slot_seeds
                    : nullptr);
            runtime.execution_runtime->execution_trace_by_row[b].push_back(step_diag.record);
            forward_outputs.exec_outputs_per_row[b].steps.push_back(std::move(step_diag));
        }
    }
}

}  // namespace Forward
}  // namespace GRIM
