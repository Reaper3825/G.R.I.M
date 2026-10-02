#include "DiagnosticLens.hpp"

namespace GRIMText::Training {
nlohmann::json diagnosticLensValidationReport(
    bool token_match, bool text_match, bool identity_pass, int rounds,
    const std::vector<std::size_t>& used_bytes, const std::string& weights) {
    return {
        {"status", token_match && text_match && identity_pass ? "pass" : "fail"},
        {"greedy_token_ids_match", token_match}, {"decoded_text_match", text_match},
        {"identity_pass", identity_pass}, {"capture_rounds", rounds},
        {"device_used_bytes_after_cleanup", used_bytes},
        {"memory_verdict", "observation_only"}, {"weights", weights}
    };
}

nlohmann::json diagnosticLensReport(
    const GRIM::Lenses::LensSnapshot& snapshot, double tolerance,
    const std::function<std::string(int)>& decode) {
    const auto& m = snapshot.metadata;
    const auto& s = m.supervision;
    nlohmann::json report = {
        {"schema_version", GRIM::Lenses::LensCaptureMetadata::schema_version}, {"session", m.identity.execution_session_id},
        {"weights", m.identity.checkpoint_fingerprint}, {"step", m.identity.parameter_revision},
        {"config", m.identity.compiled_config_fingerprint}, {"tokenizer", m.identity.tokenizer_fingerprint},
        {"forward_invocation_id", m.forward_invocation_id}, {"capture_sequence", m.capture_sequence},
        {"timestamp_unix_ns", m.capture_timestamp_unix_ns}, {"monotonic_ns", m.capture_monotonic_ns},
        {"layer_index", m.layer_index}, {"layer_count", m.layer_count},
        {"position", m.absolute_token_position}, {"token_id", s.input_token_id},
        {"token", decode(s.input_token_id)}, {"boundary", m.target_boundary},
        {"visible_keys", {m.attention.visible_keys.begin, m.attention.visible_keys.end}},
        {"causal", m.attention.kind == GRIM::Lenses::VisibilityKind::CausalPrefix},
        {"input_span_entries", s.input_span_entries}, {"target_span_entries", s.target_span_entries},
        {"span_metadata_available", static_cast<bool>(m.spans)},
        {"identity_atol", tolerance}, {"readouts", nlohmann::json::array()}
    };
    report["is_prompt_token"] = s.is_prompt_token ? nlohmann::json(*s.is_prompt_token) : nlohmann::json(nullptr);
    report["target_token_id"] = s.target_token_id ? nlohmann::json(*s.target_token_id) : nlohmann::json(nullptr);
    report["lm_supervised_prediction"] = s.is_lm_supervised_prediction ? nlohmann::json(*s.is_lm_supervised_prediction) : nlohmann::json(nullptr);
    report["atom_aux_target_mask"] = s.atom_aux_target_mask ? nlohmann::json(*s.atom_aux_target_mask) : nlohmann::json(nullptr);
    auto policy_name = [](GRIM::ConceptSpanSupervision policy) {
        switch (policy) {
            case GRIM::ConceptSpanSupervision::Context: return "context";
            case GRIM::ConceptSpanSupervision::Supervised: return "supervised";
            case GRIM::ConceptSpanSupervision::Ignore: return "ignore";
            default: return "inherit";
        }
    };
    report["input_span_policy"] = s.input_span_policy ? nlohmann::json(policy_name(*s.input_span_policy)) : nlohmann::json(nullptr);
    report["spans"] = nlohmann::json::array();
    if (m.spans) {
        for (std::size_t i = 0; i < m.spans->entries.size(); ++i) {
            const auto& entry = m.spans->entries[i];
            report["spans"].push_back({{"entry", i}, {"name", entry.name},
                {"range", {entry.span.begin, entry.span.end}},
                {"policy", policy_name(m.resolved_span_policies.at(i))}});
        }
    }
    report["identity_max_abs_logit_error"] = snapshot.identity_max_abs_logit_error
        ? nlohmann::json(*snapshot.identity_max_abs_logit_error) : nlohmann::json(nullptr);
    report["identity_status"] = !snapshot.identity_max_abs_logit_error ? "unavailable"
        : (std::isfinite(*snapshot.identity_max_abs_logit_error) && *snapshot.identity_max_abs_logit_error <= tolerance ? "pass" : "fail");
    for (const auto& readout : snapshot.readouts) {
        const char* kind = readout.kind == GRIM::Lenses::ReadoutKind::ActualFinal ? "actual_final"
            : readout.kind == GRIM::Lenses::ReadoutKind::Direct ? "direct" : "jacobian_identity_control";
        nlohmann::json entries = nlohmann::json::array();
        for (const auto& token : readout.top_tokens)
            entries.push_back({{"id", token.token_id}, {"text", decode(token.token_id)},
                {"logit", token.logit}, {"probability", token.probability}});
        report["readouts"].push_back({{"kind", kind}, {"entropy_nats", readout.entropy_nats}, {"top_tokens", entries}});
    }
    return report;
}
} // namespace GRIMText::Training
