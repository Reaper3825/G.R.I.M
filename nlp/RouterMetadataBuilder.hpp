// RouterMetadataBuilder — builds the structured metadata payload
// sent to grim-text (the router model).
//
// Takes NlpAnnotation + ContextSnapshot + ToolRegistry summary
// and produces a JSON-serializable RouterMetadata envelope.
//
// This is the body's canonical interface to the router:
// the router receives metadata, not raw NLP state.
//======================================================//
#pragma once

#include "NlpAnnotation.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace GRIM {

// Forward
struct ContextSnapshot;

// =========================================================
// RouterMetadata — the envelope sent to grim-text
// =========================================================
struct RouterMetadata {
    // User input
    std::string raw_input;
    std::string normalized_input;

    // NLP annotation (serialized subset)
    nlohmann::json nlp_annotation;

    // Context
    nlohmann::json context_snapshot;

    // Memory retrieval hints
    std::vector<std::string> memory_tags;
    std::vector<std::string> memory_query_hints;

    // Tool surface (compact summary from ToolRegistry)
    nlohmann::json tool_summary;

    // Visual context placeholders
    nlohmann::json visual_context_physical;
    nlohmann::json visual_context_digital;

    // Action policy hints
    std::vector<std::string> risk_tags;
    nlohmann::json action_policy_hints;

    // Confidence
    nlohmann::json confidence_snapshot;

    // Serialize to JSON for sending to the router model
    nlohmann::json toJson() const;
};

// =========================================================
// RouterMetadataBuilder — constructs RouterMetadata
// =========================================================
class RouterMetadataBuilder {
public:
    // Set the NLP annotation (required)
    RouterMetadataBuilder& setAnnotation(const NlpAnnotation& ann);

    // Set context snapshot (required)
    RouterMetadataBuilder& setContext(const ContextSnapshot& ctx);

    // Set tool surface summary (called from ToolRegistry)
    RouterMetadataBuilder& setToolSummary(const std::string& compact_prompt);

    // Set visual context (optional, when perception is active)
    RouterMetadataBuilder& setPhysicalVisualContext(const nlohmann::json& ctx);
    RouterMetadataBuilder& setDigitalVisualContext(const nlohmann::json& ctx);

    // Set action policy hints (optional)
    RouterMetadataBuilder& setActionPolicyHints(const nlohmann::json& hints);

    // Build the final metadata envelope
    RouterMetadata build() const;

private:
    const NlpAnnotation*   annotation_ = nullptr;
    const ContextSnapshot* context_    = nullptr;
    std::string            tool_summary_;
    nlohmann::json         phys_visual_ = nlohmann::json::object();
    nlohmann::json         digi_visual_ = nlohmann::json::object();
    nlohmann::json         policy_hints_ = nlohmann::json::object();
};

} // namespace GRIM
