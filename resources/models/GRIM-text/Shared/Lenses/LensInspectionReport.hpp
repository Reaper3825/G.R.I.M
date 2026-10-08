#pragma once
// HTTP presentation adapter over the canonical host-owned capture. This is not
// the compact .grimlens persistence format and never reruns readout math.
#include "LensMetadata.hpp"
#include "../ModelConfig/CompiledModelConfig.hpp"
#include <nlohmann/json.hpp>
#include <map>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace GRIM::Lenses {
inline std::string inspectionConfigDigest(const Config::CompiledModelConfigSnapshot& config) {
    std::ostringstream out;
    for (auto b : config.integrity.semantic_sha256)
        out << std::hex << std::setw(2) << std::setfill('0') << unsigned(b);
    return out.str();
}
inline const char* inspectionReadoutName(ReadoutKind kind) {
    switch (kind) {
        case ReadoutKind::ActualFinal: return "actual_final";
        case ReadoutKind::Direct: return "direct";
        case ReadoutKind::JacobianIdentity: return "identity_control";
    }
    throw std::invalid_argument("Unknown lens readout kind");
}
template<class DecodeToken>
nlohmann::json inspectionReport(const LensCaptureResult& capture,
    const Config::CompiledModelConfigSnapshot& config,
    const std::string& checkpoint, DecodeToken decode) {
    using nlohmann::json;
    if (capture.snapshots.empty()) throw std::runtime_error("Inspection returned no captures");
    const auto& first = capture.snapshots.front().metadata;
    json report = {{"schema_version", 1}, {"config_sha256", inspectionConfigDigest(config)},
        {"model_store_entry", config.source_path.parent_path().string()},
        {"checkpoint", checkpoint}, {"layer_count", first.layer_count},
        {"d_model", first.d_model}, {"vocab_size", first.vocab_size},
        {"mode", first.is_training ? "training_example" : "inference"},
        {"attention_visibility", first.attention.kind == VisibilityKind::CausalPrefix ? "causal" : "full_sequence"},
        {"positions", json::array()}, {"chunk_rows", capture.chunk_rows},
        {"planned_temporary_bytes", capture.planned_temporary_bytes}};
    std::map<int, json> positions;
    std::map<int, std::string> text;
    auto tokenText = [&](int id) -> const std::string& {
        auto it = text.find(id);
        if (it == text.end()) it = text.emplace(id, decode(id)).first;
        return it->second;
    };
    for (const auto& snapshot : capture.snapshots) {
        const auto& m = snapshot.metadata;
        if (m.layer_count != first.layer_count || m.d_model != first.d_model ||
            m.vocab_size != first.vocab_size || m.is_training != first.is_training)
            throw std::runtime_error("Inspection metadata changed within one capture");
        auto [it, added] = positions.try_emplace(m.token_position);
        if (added) it->second = {{"position", m.token_position},
            {"input_text", tokenText(m.supervision.input_token_id)}, {"layers", json::array()}};
        json layer = {{"layer", m.layer_index}, {"readouts", json::array()}};
        for (const auto& readout : snapshot.readouts) {
            json row = {{"kind", inspectionReadoutName(readout.kind)},
                {"entropy_nats", readout.entropy_nats}, {"candidates", json::array()}};
            for (const auto& token : readout.top_tokens)
                row["candidates"].push_back({{"token_id", token.token_id},
                    {"text", tokenText(token.token_id)}, {"logit", token.logit},
                    {"probability", token.probability}});
            layer["readouts"].push_back(std::move(row));
        }
        if (snapshot.identity_max_abs_logit_error)
            layer["identity_max_abs_logit_error"] = *snapshot.identity_max_abs_logit_error;
        it->second["layers"].push_back(std::move(layer));
    }
    for (auto& [position, value] : positions) report["positions"].push_back(std::move(value));
    return report;
}
} // namespace GRIM::Lenses
