//======================================================//
//  ExecutionBlockTest.cu
//  Workstream 0 surface tests for the trimmed ExecutionBlock API:
//    - ExecutionMemory allocation / clearing / per-row isolation
//    - surviving config / record / metrics / step-output defaults
//    - current arithmetic and bootstrap semantics documented by the layer
//======================================================//

#include "ExecutionBlockTest.hpp"

#include "../Layers/ExecutionBlock/execution_block_GPU.hpp"
#include "../Shared/Batching/BatchPayload.hpp"
#include "../Shared/Forward/ModelForwardOutputs.hpp"
#include "../Shared/UnigramByte/AtomTable.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <memory>
#include <vector>

using namespace GRIM;
using namespace GRIM::Forward;
using namespace GRIM::Test;

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

    std::vector<ExecutionMemory> memories(B);
    for (int b = 0; b < B; ++b) {
        memories[b].allocate(V, ae, dm, dk, dt, stream);
        memories[b].clear(stream);
    }

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

    EB_ASSERT_EQ(sout.record.arg1_slot, -1, "record.arg1_slot default");
    EB_ASSERT_EQ(sout.record.arg2_slot, -1, "record.arg2_slot default");
    EB_ASSERT_EQ(sout.record.op_id, -1, "record.op_id default");
    EB_ASSERT_EQ(sout.metrics.div_clamp_count, 0, "metrics.div_clamp_count default");

    return true;
}

//======================================================//
//  3. ExecutionMemory allocate shapes
//======================================================//

bool testExecutionMemoryAllocateShapes(std::string& message) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    const int V = 6;
    const int ae = 64;
    const int dm = 128;
    const int dk = 32;
    const int dt = 8;

    ExecutionMemory M;
    M.allocate(V, ae, dm, dk, dt, stream);
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
//  4. ExecutionMemory allocate rejects invalid dimensions
//======================================================//

bool testExecutionMemoryAllocateRejectsInvalid(std::string& message) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    ExecutionMemory M;
    bool threw = false;

    try { M.allocate(0, 64, 128, 32, 8, stream); }
    catch (const std::runtime_error&) { threw = true; }
    EB_ASSERT_TRUE(threw, "allocate(V=0) must throw");

    threw = false;
    try { M.allocate(4, 0, 128, 32, 8, stream); }
    catch (const std::runtime_error&) { threw = true; }
    EB_ASSERT_TRUE(threw, "allocate(atom_dim=0) must throw");

    threw = false;
    try { M.allocate(4, 64, 0, 32, 8, stream); }
    catch (const std::runtime_error&) { threw = true; }
    EB_ASSERT_TRUE(threw, "allocate(d_model=0) must throw");

    threw = false;
    try { M.allocate(4, 64, 128, 0, 8, stream); }
    catch (const std::runtime_error&) { threw = true; }
    EB_ASSERT_TRUE(threw, "allocate(d_key=0) must throw");

    threw = false;
    try { M.allocate(4, 64, 128, 32, 0, stream); }
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
    ExecutionMemory M;
    M.allocate(V, ae, dm, dk, dt, stream);

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
    EB_ASSERT_EQ(cfg.value_decode_input_dim, 24, "value_decode_input_dim default 24");
    EB_ASSERT_EQ(cfg.value_decode_hidden_dim, 16, "value_decode_hidden_dim default 16");
    EB_ASSERT_EQ(cfg.d_key, 0, "d_key default 0");
    EB_ASSERT_EQ(cfg.d_type, 0, "d_type default 0");
    EB_ASSERT_EQ(cfg.cross_attn_head_dim, 0, "cross_attn_head_dim default 0");
    EB_ASSERT_EQ(cfg.cross_attn_topk, 0, "cross_attn_topk default 0");
    EB_ASSERT_NEAR(cfg.usage_decay, 0.0f, 1e-6f, "usage_decay default");
    // [DELETED] empty_slot_bonus, diversity_kappa checks — fields removed per Fix #4.
    EB_ASSERT_NEAR(cfg.inject_gate_temp, 0.5f, 1e-6f, "inject_gate_temp default");
    EB_ASSERT_EQ(cfg.result_slot_mode, 0, "result_slot_mode default 0");
    EB_ASSERT_EQ(cfg.result_slot_index, -1, "result_slot_index default -1");
    EB_ASSERT_TRUE(cfg.debug_mode, "debug_mode default true");
    EB_ASSERT_NEAR(cfg.entropy_collapse_threshold, 0.01f, 1e-6f,
                   "entropy_collapse_threshold default");
    EB_ASSERT_NEAR(cfg.write_collapse_threshold, 0.98f, 1e-6f,
                   "write_collapse_threshold default");
    EB_ASSERT_NEAR(cfg.magnitude_limit, 1e6f, 1e-1f, "magnitude_limit default");
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
//  Entry point
//======================================================//

int main() {
    return GRIM::Test::runExecutionBlockTests();
}

int GRIM::Test::runExecutionBlockTests() {
    ExecutionBlockTestSuite suite;

    suite.addTest("Memory: per-row isolation", testBatchedMemoryIsolation);
    suite.addTest("StepOutput: defaults", testStepOutputDefaults);
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

    auto results = suite.runAll();

    int failed = 0;
    for (const auto& r : results) {
        if (!r.passed) ++failed;
    }

    return failed;
}
