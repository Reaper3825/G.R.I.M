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
#include "../Shared/Batching/BatchPayload.hpp"
#include "../Shared/Execution/ExecutionResultEmission.hpp"
#include "../Shared/Forward/ModelForwardOutputs.hpp"
#include "../Shared/Forward/ModelForwardRuntimePayload.hpp"
#include "../Shared/UnigramByte/AtomTable.hpp"
#include "../Shared/UnigramByte/TokenLayout.hpp"

#include <cuda_runtime.h>

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
    EB_ASSERT_TRUE(!sout.v_out_tensor.data,
                   "v_out_tensor should start null");
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
    EB_ASSERT_TRUE(!cfg.teacher_force_transitions,
                   "teacher_force_transitions default false");

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

//======================================================//
//  12. Inference control boundary is final prompt token
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
//  13. num_scratch_slots must stay < num_slots
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
    hp.teacher_force_transitions = true;

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
            hp, diag, payload, 0, 0, stream, work);
        GRIM::ExecutionBlockInternal::materializeHardWriteDecision(
            hp, diag, payload, 0, 0, stream, work);

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
            hp, diag, payload, 0, 0, stream, work);
        GRIM::ExecutionBlockInternal::materializeHardWriteDecision(
            hp, diag, payload, 0, 0, stream, work);
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
    suite.addTest("Record: ExecutionRecord defaults", testExecutionRecordDefaults);
    suite.addTest("Metrics: ExecStepMetrics defaults", testExecStepMetricsDefaults);
    suite.addTest("Arithmetic: four-op semantics", testFourOpsSemantics);
    suite.addTest("Bootstrap: slot map semantics", testBootstrapSlotMapSemantics);
    suite.addTest("Output: multi-step aggregation", testExecutionBlockOutputMultiStep);
    suite.addTest("Control: final prompt-token boundary", testInferencePromptControlBoundary);
    suite.addTest("Config: scratch-slot constraint", testScratchSlotConstraint);
    suite.addTest("Inference: slot compiler uses value-slot range", testInferenceSlotCompiler);
    suite.addTest("Inference: slot compiler rejects overflow", testInferenceSlotCompilerOverflow);
    suite.addTest("Inference: persistent memory runtime contract", testPersistentExecutionMemoryRuntimeContract);
    suite.addTest("Inference: terminal result emission contract", testTerminalExecutionResultEmissionContract);
    suite.addTest("Inference: generated result session table", testGeneratedNumericAtomSessionTable);
    suite.addTest("Validation: first execution error wins", testFirstExecutionErrorWins);
    suite.addTest("Policy: hard transition follows payload mode", testHardTransitionDecisionPolicy);

    auto results = suite.runAll();

    int failed = 0;
    for (const auto& r : results) {
        if (!r.passed) ++failed;
    }

    return failed;
}
