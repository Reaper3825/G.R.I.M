#pragma once

// Host-only diagnostic policy and reporting; no model execution or ownership.
#include "../../Shared/Lenses/LensMetadata.hpp"
#include <cmath>
#include <functional>
#include <stdexcept>

namespace GRIMText::Training {
struct DiagnosticLensOptions {
    bool enabled = false;
    bool validate = false;
    int top_k = 10;
    int validation_rounds = 3;
    int validation_tokens = 16;
    int max_replay_rows = 1024;
    double absolute_tolerance = 1e-4;
};

inline DiagnosticLensOptions validateDiagnosticLensOptions(DiagnosticLensOptions options) {
    options.enabled = options.enabled || options.validate;
    if (options.top_k < 1 || options.top_k > 100 ||
        options.validation_rounds < 2 || options.validation_rounds > 20 ||
        options.validation_tokens < 1 || options.validation_tokens > 256 ||
        options.max_replay_rows < 1 || options.max_replay_rows > 1048576 ||
        !std::isfinite(options.absolute_tolerance) || options.absolute_tolerance < 0)
        throw std::runtime_error("Invalid generation_lens_* diagnostic configuration");
    return options;
}

inline nlohmann::json diagnosticLensReport(
    const GRIM::Lenses::LensSnapshot& snapshot, double tolerance,
    const std::function<std::string(int)>& decode) {
    const auto& m = snapshot.metadata;
    const auto& s = m.supervision;
    nlohmann::json report = {
        {"schema_version", m.schema_version}, {"session", m.identity.execution_session_id},
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
