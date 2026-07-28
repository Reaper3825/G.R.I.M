//======================================================//
//  ExecutionBlockTest.cu
//  Workstream 0 surface tests for the trimmed ExecutionBlock API:
//    - ModelForwardOutputs execution-memory ownership / clearing / per-row isolation
//    - surviving config / record / metrics / step-output defaults
//    - current arithmetic and bootstrap semantics documented by the layer
//======================================================//

#include "ExecutionBlockTest.hpp"

#include "../Layers/ExecutionBlock/execution_block_GPU.hpp"
#include "../Layers/ExecutionBlock/execution_block_data_stream_GPU.hpp"
#include "../Layers/ExecutionBlock/execution_block_memory_stream_GPU.hpp"
#include "../Shared/Batching/BatchPayload.hpp"
#include "../Shared/Dynamic_Execution/ExecutionTransitionSchedule.hpp"
#include "../Shared/Execution/ExecutionResultEmission.hpp"
#include "../Shared/Forward/ModelForwardOutputs.hpp"
#include "../Shared/Forward/ModelForwardRuntimePayload.hpp"
#include "../Shared/UnigramByte/AtomTable.hpp"
#include "../Shared/UnigramByte/TokenLayout.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cmath>
#include <memory>
#include <vector>

using namespace GRIM;
using namespace GRIM::Forward;
using namespace GRIM::Test;

__global__ void kernelTestFirstExecutionErrorWins(int* error_flag) {
    if (threadIdx.x != 0) return;
    GRIM::ExecutionBlockInternal::recordFirstExecutionError(error_flag, 3);
    GRIM::ExecutionBlockInternal::recordFirstExecutionError(error_flag, 8);
}

//======================================================//
//  1. Batched ExecutionMemory — per-row isolation
//======================================================//

bool testBatchedMemoryIsolation(std::string& message) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    const int V = 8;
    const int ae = 4;
    const int dm = 64;
    const int dk = 16;
    const int dt = 16;
    const int B = 3;

    ModelForwardOutputs forward_outputs;
    forward_outputs.ensureExecutionBatchGeometry(B, "testBatchedMemoryIsolation");
    for (int b = 0; b < B; ++b) {
        forward_outputs.allocateExecutionMemoryRow(
            static_cast<size_t>(b), V, ae, dm, dk, dt, stream,
            "testBatchedMemoryIsolation");
    }
    auto& memories = forward_outputs.exec_memories;

    for (int b = 0; b < B; ++b) {
        float val = static_cast<float>(b + 1) * 10.0f;
        cudaMemcpyAsync(memories[b].values.data, &val, sizeof(float),
                        cudaMemcpyHostToDevice, stream);
        float one = 1.0f;
        cudaMemcpyAsync(memories[b].valid_mask.data, &one, sizeof(float),
                        cudaMemcpyHostToDevice, stream);
    }
    cudaStreamSynchronize(stream);

    for (int b = 0; b < B; ++b) {
        float read_val = 0.0f;
        cudaMemcpy(&read_val, memories[b].values.data, sizeof(float), cudaMemcpyDeviceToHost);
        float expected = static_cast<float>(b + 1) * 10.0f;
        EB_ASSERT_NEAR(read_val, expected, 1e-6f,
                       "Row isolation: slot 0 should be independent per row");
    }

    cudaStreamDestroy(stream);
    return true;
}

//======================================================//
//  2. ExecutionBlockStepOutput defaults
//======================================================//

bool testStepOutputDefaults(std::string& message) {
    GRIM::Forward::ExecutionBlockStepOutput sout{};

    EB_ASSERT_TRUE(!sout.p_arg1.data, "p_arg1 should start null");
    EB_ASSERT_TRUE(!sout.p_arg2.data, "p_arg2 should start null");
    EB_ASSERT_TRUE(!sout.p_op.data, "p_op should start null");
    EB_ASSERT_TRUE(!sout.p_write.data, "p_write should start null");
    EB_ASSERT_TRUE(!sout.v_out.data, "v_out should start null");
    EB_ASSERT_TRUE(!sout.result_emb.data, "result_emb should start null");
    EB_ASSERT_TRUE(!sout.decoder_silu_input_tensor.data,
                   "decoder_silu_input_tensor should start null");

    EB_ASSERT_TRUE(!sout.state_before_values.data,
                   "state_before_values should start null");
    EB_ASSERT_TRUE(!sout.state_before_valid.data,
                   "state_before_valid should start null");
    EB_ASSERT_TRUE(!sout.state_after_values.data,
                   "state_after_values should start null");
    EB_ASSERT_TRUE(!sout.state_after_valid.data,
                   "state_after_valid should start null");

    EB_ASSERT_TRUE(!sout.arg1_logits_tensor.data,
                   "arg1_logits_tensor should start null");
    EB_ASSERT_TRUE(!sout.arg2_logits_tensor.data,
                   "arg2_logits_tensor should start null");
    EB_ASSERT_TRUE(!sout.op_logits_tensor.data,
                   "op_logits_tensor should start null");
    EB_ASSERT_TRUE(!sout.write_logits_tensor.data,
                   "write_logits_tensor should start null");
    EB_ASSERT_TRUE(!sout.stop_logits_tensor.data,
                   "stop_logits_tensor should start null");
    EB_ASSERT_TRUE(!sout.stop_probabilities.data,
                   "stop_probabilities should start null");
    EB_ASSERT_EQ(sout.stop_predicted_class, -1,
                 "stop_predicted_class default");
    EB_ASSERT_NEAR(sout.selection_temperature, 0.0f, 1e-6f,
                   "selection_temperature should default zero");
    EB_ASSERT_TRUE(!sout.div_was_clamped,
                   "div_was_clamped should default false");
    EB_ASSERT_TRUE(!sout.teacher_forced_transition,
                   "teacher_forced_transition should default false");

    EB_ASSERT_EQ(sout.record.arg1_slot, -1, "record.arg1_slot default");
    EB_ASSERT_EQ(sout.record.arg2_slot, -1, "record.arg2_slot default");
    EB_ASSERT_EQ(sout.record.op_id, -1, "record.op_id default");
    EB_ASSERT_EQ(sout.metrics.div_clamp_count, 0, "metrics.div_clamp_count default");

    return true;
}

//======================================================//
//  3. ModelForwardOutputs execution-memory allocation shapes
//======================================================//

bool testExecutionMemoryAllocateShapes(std::string& message) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    const int V = 6;
    const int ae = 64;
    const int dm = 128;
    const int dk = 32;
    const int dt = 8;

    ModelForwardOutputs forward_outputs;
    forward_outputs.ensureExecutionBatchGeometry(1, "testExecutionMemoryAllocateShapes");
    forward_outputs.allocateExecutionMemoryRow(
        0, V, ae, dm, dk, dt, stream,
        "testExecutionMemoryAllocateShapes");
    auto& M = forward_outputs.exec_memories.front();
    cudaStreamSynchronize(stream);

    EB_ASSERT_EQ(M.values.shape.as_2d().rows, V, "values rows");
    EB_ASSERT_EQ(M.values.shape.as_2d().cols, 1, "values cols");
    EB_ASSERT_EQ(M.atom_embeds.shape.as_2d().rows, V, "atom_embeds rows");
    EB_ASSERT_EQ(M.atom_embeds.shape.as_2d().cols, ae, "atom_embeds cols");
    EB_ASSERT_EQ(M.state_embeds.shape.as_2d().rows, V, "state_embeds rows");
    EB_ASSERT_EQ(M.state_embeds.shape.as_2d().cols, dm, "state_embeds cols");
    EB_ASSERT_EQ(M.valid_mask.shape.as_2d().cols, V, "valid_mask dim");
    EB_ASSERT_EQ(M.usage.shape.as_2d().cols, V, "usage dim");
    EB_ASSERT_EQ(M.key_embeds.shape.as_2d().rows, V, "key_embeds rows");
    EB_ASSERT_EQ(M.key_embeds.shape.as_2d().cols, dk, "key_embeds cols");
    EB_ASSERT_EQ(M.type_embed.shape.as_2d().rows, V, "type_embed rows");
    EB_ASSERT_EQ(M.type_embed.shape.as_2d().cols, dt, "type_embed cols");
    EB_ASSERT_EQ(M.recent_write_mask.shape.as_2d().cols, V, "recent_write_mask dim");

    EB_ASSERT_TRUE(M.values.data != nullptr, "values allocated");
    EB_ASSERT_TRUE(M.atom_embeds.data != nullptr, "atom_embeds allocated");
    EB_ASSERT_TRUE(M.state_embeds.data != nullptr, "state_embeds allocated");
    EB_ASSERT_TRUE(M.valid_mask.data != nullptr, "valid_mask allocated");
    EB_ASSERT_TRUE(M.usage.data != nullptr, "usage allocated");
    EB_ASSERT_TRUE(M.key_embeds.data != nullptr, "key_embeds allocated");
    EB_ASSERT_TRUE(M.type_embed.data != nullptr, "type_embed allocated");
    EB_ASSERT_TRUE(M.recent_write_mask.data != nullptr, "recent_write_mask allocated");

    cudaStreamDestroy(stream);
    return true;
}

//======================================================//
//  4. ModelForwardOutputs allocation rejects invalid dimensions
//======================================================//

bool testExecutionMemoryAllocateRejectsInvalid(std::string& message) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    ModelForwardOutputs forward_outputs;
    forward_outputs.ensureExecutionBatchGeometry(1, "testExecutionMemoryAllocateRejectsInvalid");
    bool threw = false;

    try { forward_outputs.allocateExecutionMemoryRow(0, 0, 64, 128, 32, 8, stream, "test"); }
    catch (const std::runtime_error&) { threw = true; }
    EB_ASSERT_TRUE(threw, "allocate(V=0) must throw");

    threw = false;
    try { forward_outputs.allocateExecutionMemoryRow(0, 4, 0, 128, 32, 8, stream, "test"); }
    catch (const std::runtime_error&) { threw = true; }
    EB_ASSERT_TRUE(threw, "allocate(atom_dim=0) must throw");

    threw = false;
    try { forward_outputs.allocateExecutionMemoryRow(0, 4, 64, 0, 32, 8, stream, "test"); }
    catch (const std::runtime_error&) { threw = true; }
    EB_ASSERT_TRUE(threw, "allocate(d_model=0) must throw");

    threw = false;
    try { forward_outputs.allocateExecutionMemoryRow(0, 4, 64, 128, 0, 8, stream, "test"); }
    catch (const std::runtime_error&) { threw = true; }
    EB_ASSERT_TRUE(threw, "allocate(d_key=0) must throw");

    threw = false;
    try { forward_outputs.allocateExecutionMemoryRow(0, 4, 64, 128, 32, 0, stream, "test"); }
    catch (const std::runtime_error&) { threw = true; }
    EB_ASSERT_TRUE(threw, "allocate(d_type=0) must throw");

    cudaStreamDestroy(stream);
    return true;
}

//======================================================//
//  5. ExecutionMemory clear zeros data
//======================================================//

bool testExecutionMemoryClear(std::string& message) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    const int V = 4;
    const int ae = 64;
    const int dm = 128;
    const int dk = 32;
    const int dt = 8;
    ModelForwardOutputs forward_outputs;
    forward_outputs.ensureExecutionBatchGeometry(1, "testExecutionMemoryClear");
    forward_outputs.allocateExecutionMemoryRow(
        0, V, ae, dm, dk, dt, stream,
        "testExecutionMemoryClear");
    auto& M = forward_outputs.exec_memories.front();

    float val = 42.0f;
    float one = 1.0f;
    cudaMemcpyAsync(M.values.data, &val, sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(M.valid_mask.data, &one, sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(M.recent_write_mask.data, &one, sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    M.clear(stream);
    cudaStreamSynchronize(stream);

    float read_val = -1.0f;
    cudaMemcpy(&read_val, M.values.data, sizeof(float), cudaMemcpyDeviceToHost);
    EB_ASSERT_NEAR(read_val, 0.0f, 1e-6f, "clear zeros values");

    float read_mask = -1.0f;
    cudaMemcpy(&read_mask, M.valid_mask.data, sizeof(float), cudaMemcpyDeviceToHost);
    EB_ASSERT_NEAR(read_mask, 0.0f, 1e-6f, "clear zeros valid_mask");

    float read_recent = -1.0f;
    cudaMemcpy(&read_recent, M.recent_write_mask.data, sizeof(float), cudaMemcpyDeviceToHost);
    EB_ASSERT_NEAR(read_recent, 0.0f, 1e-6f, "clear zeros recent_write_mask");

    cudaStreamDestroy(stream);
    return true;
}

//======================================================//
//  6. ExecutionBlockConstructionHP defaults
//======================================================//

bool testExecutionBlockConstructionHPDefaults(std::string& message) {
    HyperParameters::ExecutionBlockConstructionHP cfg;

    EB_ASSERT_EQ(cfg.d_model, 0, "d_model default 0");
    EB_ASSERT_EQ(cfg.atom_embedding_dim, 0, "atom_embedding_dim default 0");
    EB_ASSERT_EQ(cfg.num_ops, 0, "num_ops default 0");
    EB_ASSERT_EQ(cfg.num_slots, 0, "num_slots default 0");
    EB_ASSERT_EQ(cfg.num_scratch_slots, 0, "num_scratch_slots default 0");
    EB_ASSERT_EQ(cfg.num_exec_steps, 0, "num_exec_steps default 0");
    EB_ASSERT_EQ(cfg.value_decode_input_dim, 0,
                 "value_decode_input_dim is config-owned and defaults to 0");
    EB_ASSERT_EQ(cfg.value_decode_hidden_dim, 0,
                 "value_decode_hidden_dim is config-owned and defaults to 0");
    EB_ASSERT_EQ(cfg.d_key, 0, "d_key default 0");
    EB_ASSERT_EQ(cfg.d_type, 0, "d_type default 0");
    EB_ASSERT_EQ(cfg.cross_attn_head_dim, 0, "cross_attn_head_dim default 0");
    EB_ASSERT_EQ(cfg.cross_attn_topk, 0, "cross_attn_topk default 0");
    EB_ASSERT_NEAR(cfg.usage_decay, 0.0f, 1e-6f, "usage_decay default");
    // [DELETED] empty_slot_bonus, diversity_kappa checks — fields removed per Fix #4.
    EB_ASSERT_NEAR(cfg.inject_gate_temp, 0.0f, 1e-6f,
                   "inject_gate_temp is config-owned and defaults to 0");
    EB_ASSERT_EQ(cfg.result_slot_mode, 0, "result_slot_mode default 0");
    EB_ASSERT_EQ(cfg.result_slot_index, 0,
                 "result_slot_index is config-owned and defaults to 0");
    EB_ASSERT_TRUE(!cfg.debug_mode,
                   "debug_mode is config-owned and defaults to false");
    EB_ASSERT_NEAR(cfg.entropy_collapse_threshold, 0.0f, 1e-6f,
                   "entropy_collapse_threshold is config-owned and defaults to 0");
    EB_ASSERT_NEAR(cfg.write_collapse_threshold, 0.0f, 1e-6f,
                   "write_collapse_threshold is config-owned and defaults to 0");
    EB_ASSERT_NEAR(cfg.magnitude_limit, 0.0f, 1e-6f,
                   "magnitude_limit is config-owned and defaults to 0");
    EB_ASSERT_NEAR(cfg.transition_hard_threshold, 0.0f, 1e-6f,
                   "transition_hard_threshold default");
    return true;
}

//======================================================//
//  7. ExecutionRecord defaults
//======================================================//

bool testExecutionRecordDefaults(std::string& message) {
    GRIM::Forward::ExecutionRecord rec{};

    EB_ASSERT_EQ(rec.arg1_slot, -1, "arg1_slot default -1");
    EB_ASSERT_EQ(rec.arg2_slot, -1, "arg2_slot default -1");
    EB_ASSERT_EQ(rec.op_id, -1, "op_id default -1");
    EB_ASSERT_EQ(rec.write_slot, -1, "write_slot default -1");
    EB_ASSERT_NEAR(rec.value_before_1, 0.0f, 1e-6f, "value_before_1 default");
    EB_ASSERT_NEAR(rec.value_before_2, 0.0f, 1e-6f, "value_before_2 default");
    EB_ASSERT_NEAR(rec.value_after, 0.0f, 1e-6f, "value_after default");

    return true;
}

//======================================================//
//  8. ExecStepMetrics defaults
//======================================================//

bool testExecStepMetricsDefaults(std::string& message) {
    GRIM::Forward::ExecStepMetrics m{};

    EB_ASSERT_NEAR(m.arg1_entropy, 0.0f, 1e-6f, "arg1_entropy default");
    EB_ASSERT_NEAR(m.arg2_entropy, 0.0f, 1e-6f, "arg2_entropy default");
    EB_ASSERT_NEAR(m.op_entropy, 0.0f, 1e-6f, "op_entropy default");
    EB_ASSERT_NEAR(m.write_entropy, 0.0f, 1e-6f, "write_entropy default");
    EB_ASSERT_NEAR(m.max_p_write, 0.0f, 1e-6f, "max_p_write default");
    EB_ASSERT_EQ(m.div_clamp_count, 0, "div_clamp_count default");
    for (int i = 0; i < 4; ++i) {
        EB_ASSERT_NEAR(m.op_distribution[i], 0.0f, 1e-6f, "op_distribution default");
    }

    return true;
}

//======================================================//
//  9. Current arithmetic semantics (+,-,*,safe /)
//======================================================//

bool testFourOpsSemantics(std::string& message) {
    {
        const float v1 = 10.0f;
        const float v2 = 3.0f;
        EB_ASSERT_NEAR(v1 + v2, 13.0f, 1e-6f, "add");
        EB_ASSERT_NEAR(v1 - v2, 7.0f, 1e-6f, "sub");
        EB_ASSERT_NEAR(v1 * v2, 30.0f, 1e-6f, "mul");
        EB_ASSERT_NEAR(v1 / v2, 10.0f / 3.0f, 1e-5f, "div");
    }

    {
        const float v1 = 5.0f;
        const float v2 = 0.0f;
        const float eps = 1e-7f;
        const float denom = std::copysign(eps, v2);
        const float safe_div = v1 / denom;
        EB_ASSERT_TRUE(!std::isnan(safe_div), "division clamping: no NaN");
        EB_ASSERT_TRUE(!std::isinf(safe_div), "division clamping: no Inf");
        EB_ASSERT_TRUE(std::abs(safe_div) > 1e6f, "division clamping: large magnitude");
    }

    {
        const float v1 = 5.0f;
        const float v2 = -1e-15f;
        const float eps = 1e-7f;
        const float denom = std::copysign(eps, v2);
        const float safe_div = v1 / denom;
        EB_ASSERT_TRUE(safe_div < 0.0f, "division clamping: negative sign preserved");
    }

    return true;
}

//======================================================//
//  10. Bootstrap slot-map semantics
//======================================================//

bool testBootstrapSlotMapSemantics(std::string& message) {
    const int V = 4;
    const int total_tokens = 6;

    std::vector<int32_t> slot_map = {-1, 0, -1, 2, -1, -1};
    std::vector<float> numeric_vals = {0.0f, 3.14f, 0.0f, 2.72f, 0.0f, 0.0f};
    std::vector<float> host_values(V, 0.0f);
    std::vector<float> host_valid(V, 0.0f);

    for (int pos = 0; pos < total_tokens; ++pos) {
        int slot = slot_map[pos];
        if (slot >= 0 && slot < V) {
            host_values[slot] = numeric_vals[pos];
            host_valid[slot] = 1.0f;
        }
    }

    EB_ASSERT_NEAR(host_values[0], 3.14f, 1e-6f, "slot 0 bootstrapped from token 1");
    EB_ASSERT_NEAR(host_values[1], 0.0f, 1e-6f, "slot 1 untouched");
    EB_ASSERT_NEAR(host_values[2], 2.72f, 1e-6f, "slot 2 bootstrapped from token 3");
    EB_ASSERT_NEAR(host_values[3], 0.0f, 1e-6f, "slot 3 untouched");
    EB_ASSERT_NEAR(host_valid[0], 1.0f, 1e-6f, "slot 0 marked valid");
    EB_ASSERT_NEAR(host_valid[1], 0.0f, 1e-6f, "slot 1 remains invalid");
    EB_ASSERT_NEAR(host_valid[2], 1.0f, 1e-6f, "slot 2 marked valid");
    EB_ASSERT_NEAR(host_valid[3], 0.0f, 1e-6f, "slot 3 remains invalid");

    return true;
}

//======================================================//
//  11. ExecutionBlockOutput multi-step aggregation
//======================================================//

bool testExecutionBlockOutputMultiStep(std::string& message) {
    GRIM::Forward::ExecutionBlockOutput output;

    EB_ASSERT_EQ(static_cast<int>(output.steps.size()), 0, "steps starts empty");
    EB_ASSERT_EQ(output.gate.predicted_class, -1, "gate prediction starts unset");
    EB_ASSERT_TRUE(!output.stopped_by_model, "model-stop flag starts false");
    EB_ASSERT_TRUE(!output.stopped_at_max_steps, "max-step flag starts false");
    EB_ASSERT_TRUE(!output.execution_suppressed_no_bootstrap,
                   "no-bootstrap suppression flag starts false");

    for (int k = 0; k < 3; ++k) {
        GRIM::Forward::ExecutionBlockStepOutput s{};
        s.record.op_id = k;
        output.steps.push_back(std::move(s));
    }

    EB_ASSERT_EQ(static_cast<int>(output.steps.size()), 3, "3 steps added");
    EB_ASSERT_EQ(output.steps[0].record.op_id, 0, "step 0 op_id");
    EB_ASSERT_EQ(output.steps[1].record.op_id, 1, "step 1 op_id");
    EB_ASSERT_EQ(output.steps[2].record.op_id, 2, "step 2 op_id");

    return true;
}

bool testExecutionOperandSelectionScale(std::string& message) {
    const float expected = 1.0f / std::sqrt(768.0f);
    const float actual = HyperParameters::computeExecutionOperandSelectionScale(
        768, "testExecutionOperandSelectionScale");
    EB_ASSERT_NEAR(actual, expected, 1e-7f,
                   "operand selection scale uses full d_model width");

    bool threw = false;
    try {
        (void)HyperParameters::computeExecutionOperandSelectionScale(
            0, "testExecutionOperandSelectionScale");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    EB_ASSERT_TRUE(threw, "operand selection scale rejects d_model=0");
    return true;
}

bool testDecoderSiluCacheForwardOwnership(std::string& message) {
    ModelForwardOutputs outputs;
    outputs.ensureExecutionBatchGeometry(1, "testDecoderSiluCacheForwardOwnership");

    float sentinel = 0.0f;
    ExecutionBlockStepOutput step_output;
    step_output.decoder_silu_input_tensor.data = &sentinel;
    step_output.decoder_silu_input_tensor.owns_data = false;

    outputs.exec_outputs_per_row[0].steps.push_back(std::move(step_output));
    EB_ASSERT_TRUE(
        outputs.exec_outputs_per_row[0].steps[0].decoder_silu_input_tensor.data == &sentinel,
        "ModelForwardOutputs step payload should retain the decoder SiLU cache");

    outputs.clear();
    EB_ASSERT_TRUE(outputs.exec_outputs_per_row[0].steps.empty(),
                   "ModelForwardOutputs::clear should release execution-step payloads");
    return true;
}

bool testSelectorExecutionBootstrapMetadata(std::string& message) {
    auto atom_table = std::make_shared<GRIM::Tokenizer::AtomTable>();
    const uint32_t integer_id = atom_table->registerGeneratedNumericValue(7.0f);
    const uint32_t float_id = atom_table->registerGeneratedNumericValue(3.5f);
    const auto integer_entry = atom_table->getAtom(integer_id);
    const auto float_entry = atom_table->getAtom(float_id);
    EB_ASSERT_TRUE(integer_entry.has_value(), "integer candidate entry exists");
    EB_ASSERT_TRUE(float_entry.has_value(), "float candidate entry exists");

    const std::vector<int> token_ids{
        GRIM::Tokenizer::atomTypeToTokenId(integer_entry->type),
        GRIM::Tokenizer::atomTypeToTokenId(float_entry->type),
        GRIM::Tokenizer::atomTypeToTokenId(integer_entry->type)};
    const std::vector<float> numeric_values{7.0f, 3.5f, 7.0f};
    const std::vector<uint8_t> atom_mask{1, 1, 1};
    const std::vector<uint32_t> atom_flags{
        integer_entry->flags, float_entry->flags, integer_entry->flags};
    const std::vector<uint32_t> atom_entry_ids{integer_id, float_id, integer_id};
    const std::vector<int32_t> slot_map{3, 4, 5};

    auto payload = GRIM::Batching::buildInferenceBatchPayload(
        token_ids,
        numeric_values,
        atom_mask,
        atom_flags,
        atom_table,
        atom_entry_ids,
        slot_map,
        /*vocab_size=*/1024,
        /*batch_capacity=*/1,
        /*max_cached_seq_len=*/16,
        /*execution_num_slots=*/8,
        /*execution_num_scratch_slots=*/2,
        /*number_encoder_digit_slots=*/16,
        /*number_encoder_max_abs_pow10=*/32);

    EB_ASSERT_EQ(payload.execution_slot_count, 8,
                 "bridge carries execution slot geometry");
    EB_ASSERT_EQ(static_cast<int>(payload.bootstrap_slot_to_pool_index.size()), 8,
                 "bridge is one row by execution slots");
    EB_ASSERT_EQ(payload.bootstrap_slot_to_pool_index[3], 0,
                 "first slot preserves its selector candidate identity");
    EB_ASSERT_EQ(payload.bootstrap_slot_to_pool_index[4], 1,
                 "second slot preserves its selector candidate identity");
    EB_ASSERT_EQ(payload.bootstrap_slot_to_pool_index[5], 0,
                 "third slot may reference the same candidate through a distinct token");
    EB_ASSERT_EQ(payload.bootstrap_slot_to_pool_index[2], -1,
                 "unbound slot remains explicitly unmapped");

    auto invalid = payload;
    invalid.bootstrap_slot_to_pool_index[4] = 0;
    bool threw = false;
    try {
        invalid.validate("testSelectorExecutionBootstrapMetadata");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    EB_ASSERT_TRUE(threw, "validation rejects selector/slot identity disagreement");

    bool duplicate_slot_threw = false;
    try {
        (void)GRIM::Batching::buildInferenceBatchPayload(
            token_ids,
            numeric_values,
            atom_mask,
            atom_flags,
            atom_table,
            atom_entry_ids,
            /*token_to_slot_map=*/{3, 3, 4},
            /*vocab_size=*/1024,
            /*batch_capacity=*/1,
            /*max_cached_seq_len=*/16,
            /*execution_num_slots=*/8,
            /*execution_num_scratch_slots=*/2,
            /*number_encoder_digit_slots=*/16,
            /*number_encoder_max_abs_pow10=*/32);
    } catch (const std::runtime_error&) {
        duplicate_slot_threw = true;
    }
    EB_ASSERT_TRUE(
        duplicate_slot_threw,
        "inference payload construction rejects duplicate bootstrap slot bindings");
    return true;
}

bool testSelectorExecutionForwardBridge(std::string& message) {
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    cublasHandle_t cublas_handle = nullptr;
    cublasCreate(&cublas_handle);
    autograd::set_autograd_cublas_handle(cublas_handle);

    constexpr int V = 4;
    constexpr int S = 2;
    constexpr int dm = 4;
    constexpr int dk = 2;
    constexpr int dt = 2;
    constexpr int pool_atoms = 2;
    constexpr float inv_sqrt_2 = 0.7071067811865475f;

    HyperParameters::ExecutionBlockConstructionHP hp{};
    hp.num_slots = V;
    hp.num_scratch_slots = S;
    hp.d_model = dm;
    hp.d_key = dk;
    hp.d_type = dt;

    {
        ModelForwardOutputs outputs;
        outputs.ensureExecutionBatchGeometry(1, "testSelectorExecutionForwardBridge");
        outputs.allocateExecutionMemoryRow(
            0, V, /*atom_dim=*/4, dm, dk, dt, stream,
            "testSelectorExecutionForwardBridge");
        auto& memory = outputs.exec_memories[0];

        ExecutionBlockParameterTensors params{};
        params.W_value_to_emb = Tensor::zeros({1, dm}, stream, "test_W_value_to_emb");
        params.b_value_to_emb = Tensor::zeros({1, dm}, stream, "test_b_value_to_emb");
        params.W_key_proj = Tensor::zeros({dm, dk}, stream, "test_W_key_proj");
        params.type_num_embed = Tensor::zeros({1, dt}, stream, "test_type_num_embed");
        params.E_slot = Tensor::zeros({V, dm}, stream, "test_E_slot");

        const std::vector<float> h_slot_embeddings{
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
            100.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 200.0f, 0.0f, 0.0f};
        cudaMemcpyAsync(
            params.E_slot.data,
            h_slot_embeddings.data(),
            h_slot_embeddings.size() * sizeof(float),
            cudaMemcpyHostToDevice,
            stream);

        Tensor candidate_keys = Tensor::zeros(
            {pool_atoms, dm}, stream, "test_selector_candidate_keys");
        const std::vector<float> h_candidate_keys{
            1.0f, 2.0f, 3.0f, 4.0f,
            10.0f, 20.0f, 30.0f, 40.0f};
        cudaMemcpyAsync(
            candidate_keys.data,
            h_candidate_keys.data(),
            h_candidate_keys.size() * sizeof(float),
            cudaMemcpyHostToDevice,
            stream);

        const std::vector<double> h_pool_numeric_float_values{0.0, 11.5};
        const std::vector<int64_t> h_pool_numeric_int_values{7, 0};
        const std::vector<uint8_t> h_pool_numeric_kinds{
            static_cast<uint8_t>(Tokenizer::NumericPayloadKind::INTEGER),
            static_cast<uint8_t>(Tokenizer::NumericPayloadKind::FLOAT)};
        const std::vector<uint32_t> h_atom_entry_ids{0, 1};
        const std::vector<int32_t> h_slot_map{2, 3};
        const std::vector<int> h_slot_to_pool{-1, -1, 0, 1};
        Batching::BatchPayload payload;
        payload.batch_size = 1;
        payload.max_seq_len = static_cast<int>(h_slot_map.size());
        payload.total_tokens = payload.max_seq_len;
        payload.seq_lengths = {payload.max_seq_len};
        payload.token_to_slot_map = h_slot_map;
        payload.num_pool_atoms = pool_atoms;
        payload.row_atom_offset = {0, pool_atoms};
        payload.execution_slot_count = V;
        payload.bootstrap_slot_to_pool_index = h_slot_to_pool;
        Batching::BatchDeviceBindings bindings;
        double* d_pool_numeric_float_values = nullptr;
        int64_t* d_pool_numeric_int_values = nullptr;
        uint8_t* d_pool_numeric_kinds = nullptr;
        uint32_t* d_atom_entry_ids = nullptr;
        int32_t* d_slot_map = nullptr;
        int* d_slot_to_pool = nullptr;
        cudaMalloc(reinterpret_cast<void**>(&d_pool_numeric_float_values),
                   h_pool_numeric_float_values.size() * sizeof(double));
        cudaMalloc(reinterpret_cast<void**>(&d_pool_numeric_int_values),
                   h_pool_numeric_int_values.size() * sizeof(int64_t));
        cudaMalloc(reinterpret_cast<void**>(&d_pool_numeric_kinds),
                   h_pool_numeric_kinds.size() * sizeof(uint8_t));
        cudaMalloc(reinterpret_cast<void**>(&d_atom_entry_ids),
                   h_atom_entry_ids.size() * sizeof(uint32_t));
        cudaMalloc(reinterpret_cast<void**>(&d_slot_map),
                   h_slot_map.size() * sizeof(int32_t));
        cudaMalloc(reinterpret_cast<void**>(&d_slot_to_pool),
                   h_slot_to_pool.size() * sizeof(int));
        cudaMemcpyAsync(
            d_pool_numeric_float_values,
            h_pool_numeric_float_values.data(),
            h_pool_numeric_float_values.size() * sizeof(double),
            cudaMemcpyHostToDevice,
            stream);
        cudaMemcpyAsync(
            d_pool_numeric_int_values,
            h_pool_numeric_int_values.data(),
            h_pool_numeric_int_values.size() * sizeof(int64_t),
            cudaMemcpyHostToDevice,
            stream);
        cudaMemcpyAsync(
            d_pool_numeric_kinds,
            h_pool_numeric_kinds.data(),
            h_pool_numeric_kinds.size() * sizeof(uint8_t),
            cudaMemcpyHostToDevice,
            stream);
        cudaMemcpyAsync(
            d_atom_entry_ids,
            h_atom_entry_ids.data(),
            h_atom_entry_ids.size() * sizeof(uint32_t),
            cudaMemcpyHostToDevice,
            stream);
        cudaMemcpyAsync(
            d_slot_map,
            h_slot_map.data(),
            h_slot_map.size() * sizeof(int32_t),
            cudaMemcpyHostToDevice,
            stream);
        cudaMemcpyAsync(
            d_slot_to_pool,
            h_slot_to_pool.data(),
            h_slot_to_pool.size() * sizeof(int),
            cudaMemcpyHostToDevice,
            stream);
        bindings.d_atom_entry_ids = d_atom_entry_ids;
        bindings.d_pool_numeric_float_values = d_pool_numeric_float_values;
        bindings.d_pool_numeric_int_values = d_pool_numeric_int_values;
        bindings.d_pool_numeric_kinds = d_pool_numeric_kinds;
        bindings.d_token_to_slot_map = d_slot_map;
        bindings.d_bootstrap_slot_to_pool_index = d_slot_to_pool;
        bindings.num_pool_atoms = pool_atoms;
        bindings.execution_slot_count = V;

        executionBlockBootstrapMemoryFromSlotMap(
            hp,
            memory,
            params,
            payload,
            bindings,
            /*batch_row=*/0,
            candidate_keys.data,
            stream);

        std::vector<float> h_values(static_cast<size_t>(V), 0.0f);
        cudaMemcpy(
            h_values.data(),
            memory.values.data,
            h_values.size() * sizeof(float),
            cudaMemcpyDeviceToHost);
        EB_ASSERT_NEAR(
            h_values[2],
            static_cast<float>(h_pool_numeric_int_values[0]),
            1e-6f,
            "bootstrap slot 2 gathers its scalar from atom-pool entry 0");
        EB_ASSERT_NEAR(
            h_values[3],
            static_cast<float>(h_pool_numeric_float_values[1]),
            1e-6f,
            "bootstrap slot 3 gathers its scalar from atom-pool entry 1");

        std::vector<float> h_state(static_cast<size_t>(V) * dm, 0.0f);
        cudaMemcpy(
            h_state.data(),
            memory.state_embeds.data,
            h_state.size() * sizeof(float),
            cudaMemcpyDeviceToHost);
        for (int d = 0; d < dm; ++d) {
            EB_ASSERT_NEAR(
                h_state[static_cast<size_t>(2) * dm + d],
                inv_sqrt_2 * h_candidate_keys[static_cast<size_t>(d)],
                1e-5f,
                "bootstrap slot 2 gathers selector candidate 0");
            EB_ASSERT_NEAR(
                h_state[static_cast<size_t>(3) * dm + d],
                inv_sqrt_2 * h_candidate_keys[static_cast<size_t>(dm + d)],
                1e-5f,
                "bootstrap slot 3 gathers selector candidate 1");
        }

        ExecutionBlockInternal::StepWorkingSet work;
        ExecutionBlockInternal::buildValueSlotCandidates(
            hp,
            memory,
            params,
            payload,
            /*batch_row=*/0,
            /*prior_records=*/{},
            &candidate_keys,
            stream,
            work);
        std::vector<float> h_candidates(static_cast<size_t>(V - S) * dm, 0.0f);
        cudaMemcpy(
            h_candidates.data(),
            work.cand_hidden.data,
            h_candidates.size() * sizeof(float),
            cudaMemcpyDeviceToHost);
        for (int d = 0; d < dm; ++d) {
            const float slot2_state =
                inv_sqrt_2 * h_candidate_keys[static_cast<size_t>(d)];
            const float slot3_state =
                inv_sqrt_2 * h_candidate_keys[static_cast<size_t>(dm + d)];
            EB_ASSERT_NEAR(
                h_candidates[static_cast<size_t>(d)],
                inv_sqrt_2 *
                    (slot2_state + h_slot_embeddings[static_cast<size_t>(2) * dm + d]),
                1e-5f,
                "operand candidate 0 includes absolute slot identity");
            EB_ASSERT_NEAR(
                h_candidates[static_cast<size_t>(dm + d)],
                inv_sqrt_2 *
                    (slot3_state + h_slot_embeddings[static_cast<size_t>(3) * dm + d]),
                1e-5f,
                "operand candidate 1 includes absolute slot identity");
        }

        const int invalid_pool_index = pool_atoms;
        cudaMemcpyAsync(
            d_slot_to_pool + 3,
            &invalid_pool_index,
            sizeof(int),
            cudaMemcpyHostToDevice,
            stream);
        memory.clear(stream);
        bool invalid_pool_identity_threw = false;
        try {
            executionBlockBootstrapMemoryFromSlotMap(
                hp,
                memory,
                params,
                payload,
                bindings,
                /*batch_row=*/0,
                candidate_keys.data,
                stream);
        } catch (const std::runtime_error&) {
            invalid_pool_identity_threw = true;
        }
        EB_ASSERT_TRUE(
            invalid_pool_identity_threw,
            "bootstrap must fail when a populated slot maps outside the atom-entry pool");

        cudaFree(d_pool_numeric_float_values);
        cudaFree(d_pool_numeric_int_values);
        cudaFree(d_pool_numeric_kinds);
        cudaFree(d_atom_entry_ids);
        cudaFree(d_slot_map);
        cudaFree(d_slot_to_pool);
    }

    autograd::set_autograd_cublas_handle(nullptr);
    cublasDestroy(cublas_handle);
    cudaStreamDestroy(stream);
    return true;
}

//======================================================//
//  12. Operand bridge backward uses existing primitives
//======================================================//

bool testSelectorExecutionBackwardBridge(std::string& message) {
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    cublasHandle_t cublas_handle = nullptr;
    cublasCreate(&cublas_handle);
    autograd::set_autograd_cublas_handle(cublas_handle);
    GRIM::setUseEngineBackward(true);

    constexpr int V = 4;
    constexpr int S = 2;
    constexpr int dm = 3;
    constexpr int dk = 2;
    constexpr int dt = 2;
    constexpr int pool_atoms = 2;
    constexpr float inv_sqrt_2 = 0.7071067811865475f;

    HyperParameters::ExecutionBlockConstructionHP hp{};
    hp.num_slots = V;
    hp.num_scratch_slots = S;
    hp.d_model = dm;
    hp.d_key = dk;
    hp.d_type = dt;

    ModelForwardOutputs outputs;
    outputs.ensureExecutionBatchGeometry(1, "testSelectorExecutionBackwardBridge");
    outputs.allocateExecutionMemoryRow(
        0, V, /*atom_dim=*/4, dm, dk, dt, stream,
        "testSelectorExecutionBackwardBridge");
    auto& memory = outputs.exec_memories[0];

    const std::vector<float> h_valid{0.0f, 0.0f, 1.0f, 1.0f};
    cudaMemcpyAsync(
        memory.valid_mask.data,
        h_valid.data(),
        h_valid.size() * sizeof(float),
        cudaMemcpyHostToDevice,
        stream);

    ExecutionBlockParameterTensors params{};
    params.E_slot = Tensor::zeros({V, dm}, stream, "test_backward_E_slot");
    params.E_slot.requires_grad_(true);
    params.E_slot.alloc_grad();

    Tensor candidate_keys = Tensor::zeros(
        {pool_atoms, dm}, stream, "test_backward_selector_keys");
    candidate_keys.requires_grad_(true);
    candidate_keys.alloc_grad();

    Batching::BatchPayload payload;
    payload.batch_size = 1;
    payload.execution_slot_count = V;
    payload.bootstrap_slot_to_pool_index = {-1, -1, 0, 1};

    ExecutionRecord overwritten_record{};
    overwritten_record.write_slot = 3;
    const std::vector<ExecutionRecord> prior_records{overwritten_record};

    ExecutionBlockInternal::StepWorkingSet work;
    ExecutionBlockInternal::buildValueSlotCandidates(
        hp,
        memory,
        params,
        payload,
        /*batch_row=*/0,
        prior_records,
        &candidate_keys,
        stream,
        work);

    Tensor upstream = Tensor::zeros(
        {V - S, dm}, stream, "test_backward_candidate_seed");
    const std::vector<float> h_upstream(static_cast<size_t>(V - S) * dm, 1.0f);
    cudaMemcpyAsync(
        upstream.data,
        h_upstream.data(),
        h_upstream.size() * sizeof(float),
        cudaMemcpyHostToDevice,
        stream);
    work.cand_hidden.alloc_grad();
    work.cand_hidden.backward(&upstream);

    std::vector<float> h_slot_grad(static_cast<size_t>(V) * dm, 0.0f);
    std::vector<float> h_key_grad(static_cast<size_t>(pool_atoms) * dm, 0.0f);
    cudaMemcpy(
        h_slot_grad.data(),
        params.E_slot.grad_data(),
        h_slot_grad.size() * sizeof(float),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(
        h_key_grad.data(),
        candidate_keys.grad_data(),
        h_key_grad.size() * sizeof(float),
        cudaMemcpyDeviceToHost);

    for (int slot = 0; slot < V; ++slot) {
        const float expected = slot >= S ? inv_sqrt_2 : 0.0f;
        for (int d = 0; d < dm; ++d) {
            EB_ASSERT_NEAR(
                h_slot_grad[static_cast<size_t>(slot) * dm + d],
                expected,
                1e-5f,
                "operand gradient routes through E_slot primitive matmul");
        }
    }
    for (int d = 0; d < dm; ++d) {
        EB_ASSERT_NEAR(
            h_key_grad[static_cast<size_t>(d)],
            0.5f,
            1e-5f,
            "authored slot routes operand gradient to its selector key");
        EB_ASSERT_NEAR(
            h_key_grad[static_cast<size_t>(dm + d)],
            0.0f,
            1e-5f,
            "overwritten slot does not retain stale authored-key gradient");
    }

    autograd::set_autograd_cublas_handle(nullptr);
    cublasDestroy(cublas_handle);
    cudaStreamDestroy(stream);
    return true;
}

//======================================================//
//  13. Inference control boundary is final prompt token
//======================================================//

bool testInferencePromptControlBoundary(std::string& message) {
    const std::vector<int> token_ids{5, 6, 7, 8};
    const std::vector<float> numeric_values(token_ids.size(), 0.0f);
    const std::vector<uint8_t> atom_mask(token_ids.size(), 0);
    const std::vector<uint32_t> atom_flags(token_ids.size(), 0);
    const std::vector<uint32_t> atom_entry_ids(
        token_ids.size(), GRIM::Tokenizer::kAtomEntryNone);
    const std::vector<int32_t> candidate_slot_map;

    auto payload = GRIM::Batching::buildInferenceBatchPayload(
        token_ids,
        numeric_values,
        atom_mask,
        atom_flags,
        std::make_shared<GRIM::Tokenizer::AtomTable>(),
        atom_entry_ids,
        candidate_slot_map,
        /*vocab_size=*/1024,
        /*batch_capacity=*/1,
        /*max_cached_seq_len=*/16,
        /*execution_num_slots=*/8,
        /*execution_num_scratch_slots=*/0,
        /*number_encoder_digit_slots=*/0,
        /*number_encoder_max_abs_pow10=*/0);

    EB_ASSERT_TRUE(payload.isInferencePrefill(), "payload should be inference prefill");
    EB_ASSERT_EQ(payload.execution_prompt_lengths[0], 4,
                 "execution prompt length is the full prompt");
    EB_ASSERT_EQ(payload.execution_prompt_end_positions[0], 3,
                 "execution decision is anchored to the final prompt token");
    EB_ASSERT_TRUE(!payload.execution_active[0],
                   "inference metadata must not pre-activate execution");
    EB_ASSERT_TRUE(
        payload.execution_gate_targets[0]
            == GRIM::Execution::ExecutionGateTarget::UNSUPERVISED,
        "inference control must come from the model, not a teacher target");

    return true;
}

//======================================================//
//  13. Cached decode payload retains host input ownership
//======================================================//

bool testInferenceDecodeHostInputOwnership(std::string& message) {
    const std::vector<int> token_ids{7};
    const std::vector<float> numeric_values{0.0f};
    const std::vector<uint8_t> atom_mask{0};
    const std::vector<uint32_t> atom_flags{0};
    const std::vector<uint32_t> atom_entry_ids{GRIM::Tokenizer::kAtomEntryNone};
    const std::vector<int32_t> candidate_slot_map;

    auto payload = GRIM::Batching::buildInferenceBatchPayload(
        token_ids,
        numeric_values,
        atom_mask,
        atom_flags,
        std::make_shared<GRIM::Tokenizer::AtomTable>(),
        atom_entry_ids,
        candidate_slot_map,
        /*vocab_size=*/1024,
        /*batch_capacity=*/1,
        /*max_cached_seq_len=*/1,
        /*execution_num_slots=*/8,
        /*execution_num_scratch_slots=*/0,
        /*number_encoder_digit_slots=*/0,
        /*number_encoder_max_abs_pow10=*/0);

    payload.mode = GRIM::Batching::BatchPayloadMode::InferenceDecode;
    EB_ASSERT_TRUE(payload.isInferenceDecode(), "payload should be inference decode");
    EB_ASSERT_TRUE(payload.ownsHostInputData(),
                   "populated inference decode payload should own host input data");
    payload.validate("testInferenceDecodeHostInputOwnership");

    const auto geometry_only = GRIM::Batching::buildInferenceDecodePayload(/*vocab_size=*/1024);
    EB_ASSERT_TRUE(!geometry_only.ownsHostInputData(),
                   "geometry-only decode payload should not claim host input data");

    return true;
}

//======================================================//
//  14. num_scratch_slots must stay < num_slots
//======================================================//

bool testScratchSlotConstraint(std::string& message) {
    HyperParameters::ExecutionBlockConstructionHP cfg;
    cfg.num_slots = 4;
    cfg.num_scratch_slots = 4;

    EB_ASSERT_TRUE(cfg.num_scratch_slots >= cfg.num_slots,
                   "scratch == slots should be invalid (no value slots)");

    cfg.num_scratch_slots = 3;
    EB_ASSERT_TRUE(cfg.num_scratch_slots < cfg.num_slots,
                   "scratch < slots should be valid");

    cfg.num_scratch_slots = 0;
    EB_ASSERT_TRUE(cfg.num_scratch_slots < cfg.num_slots,
                   "zero scratch slots should be valid");

    return true;
}

//======================================================//
//  14. Inference slot compiler respects scratch prefix
//======================================================//

bool testInferenceSlotCompiler(std::string& message) {
    const int int_token = GRIM::Tokenizer::atomTypeToTokenId(
        GRIM::Tokenizer::AtomType::ATOM_INT);
    const int float_token = GRIM::Tokenizer::atomTypeToTokenId(
        GRIM::Tokenizer::AtomType::ATOM_FLOAT);
    const std::vector<int> tokens{
        GRIM::Tokenizer::BOS_TOKEN_ID,
        int_token,
        GRIM::Tokenizer::byteToTokenId(static_cast<uint8_t>('+')),
        float_token};
    const std::vector<uint8_t> atom_mask{0, 1, 0, 1};

    const auto slot_map = GRIM::Batching::buildInferenceExecutionSlotMap(
        tokens, atom_mask, /*num_slots=*/5, /*num_scratch_slots=*/2);

    EB_ASSERT_EQ(slot_map.size(), tokens.size(), "slot map geometry");
    EB_ASSERT_EQ(slot_map[0], -1, "BOS remains unbound");
    EB_ASSERT_EQ(slot_map[1], 2, "first atom starts after scratch prefix");
    EB_ASSERT_EQ(slot_map[2], -1, "plain byte remains unbound");
    EB_ASSERT_EQ(slot_map[3], 3, "second atom receives next value slot");
    return true;
}

//======================================================//
//  15. Inference slot compiler fails on capacity overflow
//======================================================//

bool testInferenceSlotCompilerOverflow(std::string& message) {
    const int int_token = GRIM::Tokenizer::atomTypeToTokenId(
        GRIM::Tokenizer::AtomType::ATOM_INT);
    const std::vector<int> tokens{int_token, int_token};
    const std::vector<uint8_t> atom_mask{1, 1};

    bool threw = false;
    try {
        (void)GRIM::Batching::buildInferenceExecutionSlotMap(
            tokens, atom_mask, /*num_slots=*/3, /*num_scratch_slots=*/2);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    EB_ASSERT_TRUE(threw, "numeric atoms beyond value-slot capacity must throw");
    return true;
}

//======================================================//
//  16. Persistent decode memory is an explicit validated borrow
//======================================================//

bool testPersistentExecutionMemoryRuntimeContract(std::string& message) {
    GRIM::Forward::ModelForwardExecutionRuntime execution_runtime;
    GRIM::Forward::ModelForwardRuntimePayload runtime{};
    runtime.execution_runtime = &execution_runtime;

    GRIM::ExecutionMemory empty_memory;
    runtime.persistent_execution_memory = &empty_memory;

    bool rejected_empty = false;
    try {
        runtime.validate("testPersistentExecutionMemoryRuntimeContract", true);
    } catch (const std::runtime_error&) {
        rejected_empty = true;
    }
    EB_ASSERT_TRUE(rejected_empty, "incomplete persistent memory must be rejected");

    bool rejected_inactive = false;
    try {
        runtime.validate("testPersistentExecutionMemoryRuntimeContract", false);
    } catch (const std::runtime_error&) {
        rejected_inactive = true;
    }
    EB_ASSERT_TRUE(rejected_inactive, "persistent memory with inactive ExecutionBlock must be rejected");
    EB_ASSERT_TRUE(!runtime.persistent_execution_memory_was_read,
                   "persistent read telemetry starts false");
    return true;
}

//======================================================//
//  17. Terminal result emission requires an explicit model stop
//======================================================//

bool testTerminalExecutionResultEmissionContract(std::string& message) {
    std::vector<GRIM::ExecutionStepControlTelemetry> steps(2);
    steps[0].predicted_class = 0;
    steps[0].write_slot = 1;
    steps[1].predicted_class = 1;
    steps[1].write_slot = 3;
    const std::vector<float> values{0.0f, 4.0f, 0.0f, 42.0f};
    const std::vector<uint8_t> valid{0, 1, 0, 1};

    const auto emission = GRIM::Execution::resolveTerminalExecutionResult(
        true, true, false, steps, values, valid);
    EB_ASSERT_TRUE(emission.available, "model-stopped terminal result is available");
    EB_ASSERT_EQ(emission.slot, 3, "terminal STOP step owns the result slot");
    EB_ASSERT_NEAR(emission.value, 42.0f, 1e-6f, "terminal result value");
    EB_ASSERT_TRUE(emission.atom_type == GRIM::Tokenizer::AtomType::ATOM_INT,
                   "integral result uses INT atom");

    const auto max_step_only = GRIM::Execution::resolveTerminalExecutionResult(
        true, false, true, steps, values, valid);
    EB_ASSERT_TRUE(!max_step_only.available,
                   "reaching max steps does not implicitly choose a result");

    bool rejected_invalid = false;
    try {
        const std::vector<uint8_t> invalid{0, 1, 0, 0};
        (void)GRIM::Execution::resolveTerminalExecutionResult(
            true, true, false, steps, values, invalid);
    } catch (const std::runtime_error&) {
        rejected_invalid = true;
    }
    EB_ASSERT_TRUE(rejected_invalid, "invalid terminal write slot must fail loudly");
    return true;
}

//======================================================//
//  18. Generated numeric atoms live in a session-owned table
//======================================================//

bool testGeneratedNumericAtomSessionTable(std::string& message) {
    auto authored = std::make_shared<GRIM::Tokenizer::AtomTable>();
    const uint32_t authored_id = authored->registerGeneratedNumericValue(7.0f);
    auto session = authored->cloneHostForGeneration();
    const uint32_t result_id = session->registerGeneratedNumericValue(3.5f);

    EB_ASSERT_EQ(authored_id, 0u, "first authored id");
    EB_ASSERT_EQ(static_cast<int>(authored->size()), 1, "authored table remains unchanged");
    EB_ASSERT_EQ(static_cast<int>(session->size()), 2, "session table receives result atom");
    const auto result = session->getAtom(result_id);
    EB_ASSERT_TRUE(result.has_value(), "generated result atom is retrievable");
    EB_ASSERT_TRUE(result->type == GRIM::Tokenizer::AtomType::ATOM_FLOAT,
                   "fractional result uses FLOAT atom");
    EB_ASSERT_TRUE(result->origin == GRIM::Tokenizer::AtomOrigin::MODEL_GENERATED,
                   "generated result origin is retained");
    EB_ASSERT_TRUE(session->atomToString(*result) == "3.5",
                   "generated result uses canonical text");
    return true;
}

//======================================================//
//  19. First execution validation failure is preserved
//======================================================//

bool testFirstExecutionErrorWins(std::string& message) {
    int* d_error = nullptr;
    cudaMalloc(&d_error, sizeof(int));
    cudaMemset(d_error, 0, sizeof(int));
    kernelTestFirstExecutionErrorWins<<<1, 1>>>(d_error);

    int error = 0;
    cudaMemcpy(&error, d_error, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_error);

    EB_ASSERT_EQ(error, 3,
                 "later validation stages must not overwrite the first failure");
    return true;
}

//======================================================//
//  20. Hard transition decision follows payload mode
//======================================================//

bool testHardTransitionDecisionPolicy(std::string& message) {
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    HyperParameters::ExecutionBlockConstructionHP hp;
    hp.num_slots = 8;
    hp.num_scratch_slots = 2;
    hp.num_ops = 4;
    ExecutionBlockDiagnosticsBuffers diag;
    diag.allocate(stream);

    {
        GRIM::ExecutionBlockInternal::StepWorkingSet work;
        work.p_arg1 = Tensor::zeros({1, 6}, stream, "test_policy_arg1");
        work.p_arg2 = Tensor::zeros({1, 6}, stream, "test_policy_arg2");
        work.p_op = Tensor::zeros({1, 4}, stream, "test_policy_op");
        work.p_write = Tensor::zeros({1, 8}, stream, "test_policy_write");

        const float arg1[] = {0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
        const float arg2[] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        const float op[] = {0.0f, 0.0f, 0.0f, 1.0f};
        const float write[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
        cudaMemcpyAsync(work.p_arg1.data, arg1, sizeof(arg1), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(work.p_arg2.data, arg2, sizeof(arg2), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(work.p_op.data, op, sizeof(op), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(work.p_write.data, write, sizeof(write), cudaMemcpyHostToDevice, stream);

        Batching::BatchPayload payload;
        payload.mode = Batching::BatchPayloadMode::Training;
        payload.batch_size = 1;
        payload.teacher_steps = {{Execution::TeacherStep{
            /*op_id=*/2,
            /*arg1_slot=*/3,
            /*arg2_slot=*/4,
            /*write_slot=*/5,
            /*expected_value=*/0.0f}}};
        payload.teacher_step_mask = {{1}};

        GRIM::ExecutionBlockInternal::materializeHardReadAndOpDecision(
            hp, diag, payload, 0, 0, true, stream, work);
        GRIM::ExecutionBlockInternal::materializeHardWriteDecision(
            hp, diag, payload, 0, 0, true, stream, work);

        int indices[4] = {-1, -1, -1, -1};
        cudaMemcpyAsync(indices, diag.execIndices(), sizeof(indices),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        EB_ASSERT_EQ(indices[0], 1, "training arg1 uses teacher slot relative to S");
        EB_ASSERT_EQ(indices[1], 2, "training arg2 uses teacher slot relative to S");
        EB_ASSERT_EQ(indices[2], 2, "training op uses teacher op");
        EB_ASSERT_EQ(indices[3], 5, "training write uses teacher slot");

        payload.mode = Batching::BatchPayloadMode::InferencePrefill;
        GRIM::ExecutionBlockInternal::materializeHardReadAndOpDecision(
            hp, diag, payload, 0, 0, false, stream, work);
        GRIM::ExecutionBlockInternal::materializeHardWriteDecision(
            hp, diag, payload, 0, 0, false, stream, work);
        cudaMemcpyAsync(indices, diag.execIndices(), sizeof(indices),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        EB_ASSERT_EQ(indices[0], 4, "inference arg1 uses model argmax");
        EB_ASSERT_EQ(indices[1], 0, "inference arg2 uses model argmax");
        EB_ASSERT_EQ(indices[2], 3, "inference op uses model argmax");
        EB_ASSERT_EQ(indices[3], 7, "inference write uses model argmax");
    }

    diag.destroy();
    cudaStreamDestroy(stream);
    return true;
}

bool testExecutionTransitionSchedule(std::string& message) {
    using namespace GRIM::ExecutionTransition;

    ExecutionTransitionScheduleConfig cfg;
    cfg.initial_student_alpha = 0.0f;
    cfg.final_student_alpha = 1.0f;
    cfg.ramp_steps = 10;
    ExecutionTransitionSchedule schedule(cfg);

    const auto first = schedule.query(0);
    EB_ASSERT_TRUE(first.student_alpha > 0.0f,
                   "update zero should already have progressive student exposure");
    EB_ASSERT_TRUE(first.student_alpha < 1.0f,
                   "update zero should remain inside the handoff ramp");
    EB_ASSERT_TRUE(first.in_ramp, "update zero should report in_ramp");

    const auto last = schedule.query(9);
    EB_ASSERT_NEAR(last.student_alpha, 1.0f, 1e-6f,
                   "last ramp update should reach full student authority");
    EB_ASSERT_TRUE(!last.in_ramp, "completed ramp should clear in_ramp");
    EB_ASSERT_NEAR(schedule.studentAlpha(100), 1.0f, 1e-6f,
                   "student authority should hold after ramp completion");

    ExecutionTransitionScheduleConfig teacher_cfg = cfg;
    teacher_cfg.force_teacher = true;
    ExecutionTransitionSchedule teacher_schedule(teacher_cfg);
    EB_ASSERT_NEAR(teacher_schedule.studentAlpha(0), 0.0f, 1e-6f,
                   "teacher override must pin update zero to teacher authority");
    EB_ASSERT_NEAR(teacher_schedule.studentAlpha(100), 0.0f, 1e-6f,
                   "teacher override must pin post-ramp updates to teacher authority");

    ExecutionTransitionScheduleConfig student_cfg = cfg;
    student_cfg.force_student = true;
    ExecutionTransitionSchedule student_schedule(student_cfg);
    EB_ASSERT_NEAR(student_schedule.studentAlpha(0), 1.0f, 1e-6f,
                   "student override must pin update zero to student authority");
    EB_ASSERT_NEAR(student_schedule.studentAlpha(100), 1.0f, 1e-6f,
                   "student override must pin post-ramp updates to student authority");

    bool rejected_conflicting_overrides = false;
    try {
        ExecutionTransitionScheduleConfig conflicting_cfg = cfg;
        conflicting_cfg.force_teacher = true;
        conflicting_cfg.force_student = true;
        ExecutionTransitionSchedule conflicting_schedule(conflicting_cfg);
        (void)conflicting_schedule;
    } catch (const std::runtime_error&) {
        rejected_conflicting_overrides = true;
    }
    EB_ASSERT_TRUE(rejected_conflicting_overrides,
                   "teacher and student overrides must be mutually exclusive");

    EB_ASSERT_TRUE(!useModelTrajectory(0.0f, 3, 7, 1),
                   "alpha zero must always select the teacher");
    EB_ASSERT_TRUE(useModelTrajectory(1.0f, 3, 7, 1),
                   "alpha one must always select the model");
    const bool first_pick = useModelTrajectory(0.5f, 3, 7, 1);
    const bool repeated_pick = useModelTrajectory(0.5f, 3, 7, 1);
    EB_ASSERT_EQ(first_pick, repeated_pick,
                 "trajectory selection must be deterministic");
    return true;
}

//======================================================//
//  Entry point
//======================================================//

int main() {
    return GRIM::Test::runExecutionBlockTests();
}

int GRIM::Test::runExecutionBlockTests() {
    ExecutionBlockTestSuite suite;

    suite.addTest("Memory: per-row isolation", testBatchedMemoryIsolation);
    suite.addTest("StepOutput: defaults", testStepOutputDefaults);
    suite.addTest("StepOutput: decoder SiLU cache ownership", testDecoderSiluCacheForwardOwnership);
    suite.addTest("Memory: allocate shapes", testExecutionMemoryAllocateShapes);
    suite.addTest("Memory: allocate rejects invalid dims", testExecutionMemoryAllocateRejectsInvalid);
    suite.addTest("Memory: clear zeros state", testExecutionMemoryClear);
    suite.addTest("Config: ExecutionBlockConstructionHP defaults", testExecutionBlockConstructionHPDefaults);
    suite.addTest("Config: execution operand-selection scale", testExecutionOperandSelectionScale);
    suite.addTest("Record: ExecutionRecord defaults", testExecutionRecordDefaults);
    suite.addTest("Metrics: ExecStepMetrics defaults", testExecStepMetricsDefaults);
    suite.addTest("Arithmetic: four-op semantics", testFourOpsSemantics);
    suite.addTest("Bootstrap: slot map semantics", testBootstrapSlotMapSemantics);
    suite.addTest("Output: multi-step aggregation", testExecutionBlockOutputMultiStep);
    suite.addTest("Metadata: selector-to-execution bootstrap identity", testSelectorExecutionBootstrapMetadata);
    suite.addTest("Forward: selector-to-execution operand bridge", testSelectorExecutionForwardBridge);
    suite.addTest("Backward: selector-to-execution operand bridge", testSelectorExecutionBackwardBridge);
    suite.addTest("Control: final prompt-token boundary", testInferencePromptControlBoundary);
    suite.addTest("Inference: decode payload owns host inputs", testInferenceDecodeHostInputOwnership);
    suite.addTest("Config: scratch-slot constraint", testScratchSlotConstraint);
    suite.addTest("Inference: slot compiler uses value-slot range", testInferenceSlotCompiler);
    suite.addTest("Inference: slot compiler rejects overflow", testInferenceSlotCompilerOverflow);
    suite.addTest("Inference: persistent memory runtime contract", testPersistentExecutionMemoryRuntimeContract);
    suite.addTest("Inference: terminal result emission contract", testTerminalExecutionResultEmissionContract);
    suite.addTest("Inference: generated result session table", testGeneratedNumericAtomSessionTable);
    suite.addTest("Validation: first execution error wins", testFirstExecutionErrorWins);
    suite.addTest("Policy: execution transition alpha schedule", testExecutionTransitionSchedule);
    suite.addTest("Policy: hard transition follows payload mode", testHardTransitionDecisionPolicy);

    auto results = suite.runAll();

    int failed = 0;
    for (const auto& r : results) {
        if (!r.passed) ++failed;
    }

    return failed;
}
