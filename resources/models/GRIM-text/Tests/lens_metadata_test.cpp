// Standalone host-only tests: no model, CUDA runtime, or training execution.
#include "../Shared/Lenses/LensMetadata.hpp"
#include "../Shared/Batching/BatchPayload.hpp"
#include "../training/Diagnostics/DiagnosticLens.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>

using namespace GRIM;
using namespace GRIM::Lenses;

template<class F> void mustThrow(F f) {
    bool threw = false;
    try { f(); } catch (const std::exception&) { threw = true; }
    assert(threw);
}

int main() {
    Batching::BatchPayload payload;
    payload.batch_size = 1;
    payload.max_seq_len = 6;
    payload.total_tokens = 6;
    payload.vocab_size = 20;
    payload.seq_lengths = {4};
    payload.seq_ids = {42};
    payload.input_ids = {3, 4, 5, 6, 0, 0};
    payload.target_ids = {-1, 5, 6, -1, -1, -1};
    payload.prompt_lengths = {2};
    payload.prompt_end_positions = {1};
    payload.atom_aux_target_mask = {0, 0, 1, 0, 0, 0};
    auto spans = std::make_shared<NamedConceptSpans>();
    spans->entries = {
        {"prompt", {0, 2}, kNoNamedConceptSpanEntry, {}},
        {"answer", {2, 4}, kNoNamedConceptSpanEntry, {2}},
        {"value", {2, 3}, 1, {}}
    };
    payload.named_concept_spans = {spans};
    // Regression: structured inference must accept its row-aligned tree.
    for (const auto mode : {Batching::BatchPayloadMode::Training,
                           Batching::BatchPayloadMode::InferencePrefill,
                           Batching::BatchPayloadMode::InferenceDecode}) {
        auto checked = payload;
        checked.mode = mode;
        checked.validateNamedConceptSpanMetadata("span regression");
        checked.named_concept_spans.push_back(spans);
        mustThrow([&] { checked.validateNamedConceptSpanMetadata("extra row"); });
        checked.named_concept_spans = {spans};
        checked.seq_lengths[0] = 3;
        mustThrow([&] { checked.validateNamedConceptSpanMetadata("out of bounds"); });
        checked.named_concept_spans.clear();
        if (checked.isTraining())
            mustThrow([&] { checked.validateNamedConceptSpanMetadata("missing training row"); });
        else
            checked.validateNamedConceptSpanMetadata("plain inference");
    }
    NamedConceptSpanDefinition prompt;
    prompt.name = "prompt";
    prompt.supervision = ConceptSpanSupervision::Context;
    NamedConceptSpanDefinition answer;
    answer.name = "answer";
    answer.supervision = ConceptSpanSupervision::Supervised;
    NamedConceptSpanDefinition value;
    value.name = "value";
    answer.children = {value};
    NamedConceptSpanDefinitions definitions{prompt, answer};
    LensCaptureRequest request;
    request.identity = {"session", "checkpoint", "config", "tokenizer", 7};
    auto capture = [&](bool causal = true, int prefix = 0) {
        return makeCaptureMetadata(request, payload, definitions, 12, 768, prefix, causal, false, 91);
    };
    auto metadata = capture();
    assert(metadata.layer_index == 11 && metadata.forward_invocation_id == 91);
    assert(metadata.sequence_id == 42u && metadata.token_position == 1);
    assert(metadata.supervision.is_prompt_token == true);
    assert(metadata.supervision.is_lm_supervised_prediction == true);
    assert(metadata.supervision.target_token_id == 5);
    assert(metadata.supervision.input_span_policy == ConceptSpanSupervision::Context);
    assert(metadata.supervision.input_span_entries == std::vector<std::uint32_t>{0});
    assert((metadata.supervision.target_span_entries == std::vector<std::uint32_t>{1, 2}));
    assert(metadata.resolved_span_policies[2] == ConceptSpanSupervision::Supervised);
    assert(metadata.attention.visible_keys.end == 2);
    assert(capture(false).attention.visible_keys.end == 4); // excludes padding

    request.token_position = 2;
    payload.target_ids[2] = -1;
    metadata = capture();
    assert(metadata.supervision.input_span_policy == ConceptSpanSupervision::Supervised);
    assert(metadata.supervision.is_lm_supervised_prediction == false);
    assert(metadata.supervision.atom_aux_target_mask == true);
    assert(metadata.supervision.target_span_entries.empty());
    request.token_position = 4;
    mustThrow([&] { capture(); }); // padded position
    request.token_position = -1;
    payload.EnableAtomIdentification = true;
    mustThrow([&] { capture(); });
    payload.EnableAtomIdentification = false;
    request.replay_identity_control = true;
    request.max_replay_rows = 5;
    assert(capture().token_position == 1); // row limit now caps chunks, not the rectangle
    request.max_replay_rows = 0;
    mustThrow([&] { capture(); });
    request.max_replay_rows = 1024;
    request.replay_identity_control = false;

    request.all_positions = true;
    request.token_position = 99;
    assert(capture().token_position == 0); // enumerate real positions independently of selection
    request.all_positions = false;
    request.token_position = -1;
    std::uint64_t planned = 0;
    LensCaptureRequest chunk_request;
    chunk_request.max_replay_rows = 3;
    assert(planCaptureChunkRows(chunk_request, 8, 16, 20, 32, true, true, false, planned) == 3);
    const auto three_rows = planned;
    chunk_request.temporary_memory_budget_bytes = three_rows;
    assert(planCaptureChunkRows(chunk_request, 8, 16, 20, 32, true, true, false, planned) == 3);
    chunk_request.temporary_memory_budget_bytes = three_rows - 1;
    assert(planCaptureChunkRows(chunk_request, 8, 16, 20, 32, true, true, false, planned) == 2);
    chunk_request.temporary_memory_budget_bytes = 1;
    mustThrow([&] { planCaptureChunkRows(chunk_request, 8, 16, 20, 32, true, true, false, planned); });
    chunk_request.temporary_memory_budget_bytes = 64 * 1024 * 1024;
    assert(planCaptureChunkRows(chunk_request, 1, 16, 20, 0, false, false, true, planned) == 1);
    assert(planned == 20 * (2 * sizeof(float) + sizeof(int)) + 20 * 16 * sizeof(float) + 20 * sizeof(float));
    mustThrow([&] { planCaptureChunkRows(chunk_request, 0, 16, 20, 0, false, false, false, planned); });
    chunk_request.temporary_memory_budget_bytes = std::numeric_limits<std::uint64_t>::max();
    mustThrow([&] { planCaptureChunkRows(chunk_request, 1,
        std::numeric_limits<int>::max(), std::numeric_limits<int>::max(),
        0, false, false, true, planned); });
    request.temporary_memory_budget_bytes = 0;
    mustThrow([&] { capture(); });
    request.temporary_memory_budget_bytes = 64 * 1024 * 1024;

    payload.mode = Batching::BatchPayloadMode::InferencePrefill;
    payload.prompt_lengths.clear();
    payload.prompt_end_positions.clear();
    metadata = capture();
    assert(metadata.token_position == 1 && metadata.supervision.is_prompt_token == true);
    assert(!metadata.supervision.target_token_id.has_value());
    request.token_position = 3;
    assert(capture().supervision.is_prompt_token == false); // prefill is not all prompt
    request.token_position = -1;
    request.prompt_span_name = "missing";
    mustThrow([&] { capture(); });
    request.prompt_span_name = "prompt";
    request.prompt_span_entry_index = 1;
    mustThrow([&] { capture(); }); // entry/name mismatch
    request.prompt_span_entry_index = 0;
    assert(capture().token_position == 1);
    request.prompt_span_entry_index.reset();
    payload.named_concept_spans.clear();
    spans.reset();
    assert(metadata.spans->entries.size() == 3); // retained immutable metadata
    payload.mode = Batching::BatchPayloadMode::InferenceDecode;
    mustThrow([&] { capture(); }); // no default prompt position in decode
    request.token_position = 0;
    metadata = capture(true, 8);
    assert(metadata.absolute_token_position == 8 && metadata.attention.visible_keys.end == 9);
    assert(!metadata.supervision.is_prompt_token.has_value());
    request.identity.checkpoint_fingerprint.clear();
    mustThrow([&] { capture(); });

    // A second batch row uses padded storage strides but only its real length.
    Batching::BatchPayload mixed;
    mixed.mode = Batching::BatchPayloadMode::InferencePrefill;
    mixed.batch_size = 2; mixed.max_seq_len = 8; mixed.total_tokens = 16;
    mixed.vocab_size = 20; mixed.seq_lengths = {2, 7};
    mixed.input_ids = {1, 2, 0, 0, 0, 0, 0, 0, 3, 4, 5, 6, 7, 8, 9, 0};
    LensCaptureRequest selected;
    selected.identity = {"session", "checkpoint", "config", "tokenizer", 7};
    selected.batch_row = 1; selected.token_position = 0;
    selected.all_layers = true; selected.all_positions = true;
    selected.max_replay_rows = 3;
    assert(planCaptureChunkRows(selected, mixed.seq_lengths[1], 16, 20, 0, false, false, false, planned) == 3);
    for (int i = 0; i < mixed.seq_lengths[1]; ++i) {
        selected.all_positions = false; selected.token_position = i;
        const auto m = makeCaptureMetadata(selected, mixed, {}, 5, 16, 0, true, false, 92);
        assert(m.supervision.input_token_id == i + 3);
        assert(m.token_position == i && m.layer_count == 5);
        assert(m.attention.visible_keys.end == i + 1);
    }
    selected.token_position = 7;
    mustThrow([&] { makeCaptureMetadata(selected, mixed, {}, 5, 16, 0, true, false, 92); });

    // Compatibility views retain exactly the canonical collection's ownership.
    auto collection = std::make_shared<LensCaptureResult>();
    collection->snapshots.resize(2);
    collection->snapshots.back().metadata.token_position = 6;
    std::shared_ptr<const LensSnapshot> alias(collection, &collection->snapshots.back());
    collection.reset();
    assert(alias->metadata.token_position == 6);

    auto scores = summarizeLogits({1000, 1000, 1000, 1000}, 2, ReadoutKind::ActualFinal);
    assert(scores.top_tokens[0].token_id == 0 && scores.top_tokens[1].token_id == 1);
    assert(std::abs(scores.top_tokens[0].probability - 0.25) < 1e-12);
    assert(std::abs(scores.entropy_nats - std::log(4.0)) < 1e-12);
    auto shifted = summarizeLogits({-1000, -1000, -1000, -1000}, 2, ReadoutKind::Direct);
    assert(shifted.top_tokens[0].probability == scores.top_tokens[0].probability);
    mustThrow([] { summarizeLogits({std::numeric_limits<float>::quiet_NaN()}, 1, ReadoutKind::Direct); });
    mustThrow([] { summarizeLogits({1}, 2, ReadoutKind::Direct); });
    GRIMText::Training::DiagnosticLensOptions configured;
    auto options = [&] {
        return GRIMText::Training::validateDiagnosticLensOptions(configured);
    };
    assert(!options().enabled);
    configured.validate = true;
    assert(options().enabled && options().validate && options().validation_rounds == 3);
    configured.absolute_tolerance = std::numeric_limits<double>::quiet_NaN();
    mustThrow([&] { options(); });
    configured.absolute_tolerance = 0.001;
    configured.validation_rounds = 1;
    mustThrow([&] { options(); });
    configured.validation_rounds = 2;
    assert(options().absolute_tolerance == 0.001);
    LensSnapshot snapshot;
    snapshot.metadata = metadata;
    snapshot.readouts = {scores};
    auto report = [&] {
        return GRIMText::Training::diagnosticLensReport(snapshot, 1e-4,
            [](int) { return std::string("quoted\"\n"); });
    };
    assert(report()["identity_status"] == "unavailable");
    assert(report()["target_token_id"].is_null());
    assert(report()["lm_supervised_prediction"].is_null());
    snapshot.identity_max_abs_logit_error = 1e-5;
    assert(report()["identity_status"] == "pass");
    snapshot.identity_max_abs_logit_error = 0.1;
    assert(report()["identity_status"] == "fail");
    const auto encoded_report = report().dump();
    assert(encoded_report.find('\n') == std::string::npos);
    assert(nlohmann::json::parse(encoded_report)["readouts"][0]["top_tokens"][0]["text"] == "quoted\"\n");
    std::cout << "lens metadata tests passed\n";
    const auto validation = GRIMText::Training::diagnosticLensValidationReport(
        true, true, false, 3, {100, 100, 100}, "live-step");
    assert(validation["status"] == "fail");
    assert(validation["memory_verdict"] == "observation_only");
    assert(validation["device_used_bytes_after_cleanup"].size() == 3);
}
