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
    mustThrow([&] { capture(); });
    request.replay_identity_control = false;

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
