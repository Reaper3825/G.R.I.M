// RouterMetadataBuilder.cpp — builds the structured metadata payload
// sent to grim-text (the router model).
//======================================================//

#include "RouterMetadataBuilder.hpp"
#include "../memory/context_snapshot.hpp"

namespace GRIM {

// ─── Serialization helpers ────────────────────────────────

static nlohmann::json serializeAnnotation(const NlpAnnotation& ann) {
    nlohmann::json j;
    j["normalized_text"] = ann.normalized_text;
    j["language"]        = ann.language;

    // Utterance priors
    j["utterance_priors"] = {
        {"command",  ann.utterance_priors.command},
        {"question", ann.utterance_priors.question},
        {"banter",   ann.utterance_priors.banter},
        {"unknown",  ann.utterance_priors.unknown}
    };

    // Entities
    nlohmann::json ents = nlohmann::json::array();
    for (const auto& e : ann.entities) {
        nlohmann::json ej;
        ej["text"]  = e.text;
        ej["type"]  = e.type;
        ej["start"] = e.start_offset;
        ej["end"]   = e.end_offset;
        ej["confidence"] = e.confidence;
        if (!e.attributes.empty()) ej["attributes"] = e.attributes;
        ents.push_back(std::move(ej));
    }
    j["entities"] = std::move(ents);

    // Tags
    j["action_affordances"]    = ann.action_affordances;
    j["candidate_tool_tokens"] = ann.candidate_tool_tokens;
    j["tool_context_hints"]    = ann.tool_context_hints;
    j["memory_tags"]           = ann.memory_tags;
    j["router_tags"]           = ann.router_tags;
    j["context_tags"]          = ann.context_tags;
    j["risk_tags"]             = ann.risk_tags;
    j["references"]            = ann.references;

    // Ambiguities
    nlohmann::json ambs = nlohmann::json::array();
    for (const auto& a : ann.ambiguities) {
        nlohmann::json aj;
        aj["span"]             = a.span;
        aj["interpretations"]  = a.interpretations;
        aj["confidence_gap"]   = a.confidence_gap;
        ambs.push_back(std::move(aj));
    }
    j["ambiguities"] = std::move(ambs);

    // Confidence
    j["confidence"] = {
        {"entity",  ann.confidence_summary.entity_confidence},
        {"intent",  ann.confidence_summary.intent_confidence},
        {"prior",   ann.confidence_summary.prior_confidence},
        {"overall", ann.confidence_summary.overall}
    };

    // Legacy fields (available during migration)
    if (!ann.legacy_intent_name.empty()) {
        j["legacy_intent"]   = ann.legacy_intent_name;
        j["legacy_category"] = ann.legacy_category;
        j["legacy_matched"]  = ann.legacy_matched;
    }

    return j;
}

static nlohmann::json serializeContext(const ContextSnapshot& ctx) {
    nlohmann::json j;
    j["recent_intents"]       = ctx.recentIntents;
    j["recent_commands"]      = ctx.recentCommands;
    j["current_mood"]         = ctx.currentMood;
    j["conversation_depth"]   = ctx.conversationDepth;
    j["last_nlp_category"]    = ctx.lastNlpCategory;
    j["consecutive_commands"] = ctx.consecutiveCommands;
    j["last_command_time"]    = ctx.lastCommandTime;
    return j;
}

// ─── RouterMetadata ───────────────────────────────────────

nlohmann::json RouterMetadata::toJson() const {
    nlohmann::json j;
    j["user_input"] = {
        {"raw",        raw_input},
        {"normalized", normalized_input}
    };
    j["nlp_annotation"]      = nlp_annotation;
    j["context_snapshot"]     = context_snapshot;
    j["memory_tags"]          = memory_tags;
    j["memory_query_hints"]   = memory_query_hints;
    j["tool_summary"]         = tool_summary;
    j["visual_context"] = {
        {"physical_semantics", visual_context_physical},
        {"digital_visual",     visual_context_digital}
    };
    j["risk_tags"]            = risk_tags;
    j["action_policy_hints"]  = action_policy_hints;
    j["confidence_snapshot"]  = confidence_snapshot;
    return j;
}

// ─── RouterMetadataBuilder ────────────────────────────────

RouterMetadataBuilder& RouterMetadataBuilder::setAnnotation(const NlpAnnotation& ann) {
    annotation_ = &ann;
    return *this;
}

RouterMetadataBuilder& RouterMetadataBuilder::setContext(const ContextSnapshot& ctx) {
    context_ = &ctx;
    return *this;
}

RouterMetadataBuilder& RouterMetadataBuilder::setToolSummary(const std::string& compact_prompt) {
    tool_summary_ = compact_prompt;
    return *this;
}

RouterMetadataBuilder& RouterMetadataBuilder::setPhysicalVisualContext(const nlohmann::json& ctx) {
    phys_visual_ = ctx;
    return *this;
}

RouterMetadataBuilder& RouterMetadataBuilder::setDigitalVisualContext(const nlohmann::json& ctx) {
    digi_visual_ = ctx;
    return *this;
}

RouterMetadataBuilder& RouterMetadataBuilder::setActionPolicyHints(const nlohmann::json& hints) {
    policy_hints_ = hints;
    return *this;
}

RouterMetadata RouterMetadataBuilder::build() const {
    if (!annotation_) throw std::runtime_error("RouterMetadataBuilder: annotation is NULL — caller MUST provide NlpAnnotation");
    if (!context_)    throw std::runtime_error("RouterMetadataBuilder: context is NULL — caller MUST provide ContextSnapshot");

    RouterMetadata meta;
    meta.raw_input        = annotation_->raw_text;
    meta.normalized_input = annotation_->normalized_text;
    meta.nlp_annotation   = serializeAnnotation(*annotation_);
    meta.context_snapshot = serializeContext(*context_);
    meta.memory_tags      = annotation_->memory_tags;
    meta.risk_tags        = annotation_->risk_tags;
    meta.action_policy_hints  = policy_hints_;
    meta.visual_context_physical = phys_visual_;
    meta.visual_context_digital  = digi_visual_;

    // Tool summary as JSON
    if (!tool_summary_.empty()) {
        meta.tool_summary = tool_summary_;
    } else {
        meta.tool_summary = nlohmann::json::object();
    }

    // Confidence snapshot
    meta.confidence_snapshot = {
        {"entity",  annotation_->confidence_summary.entity_confidence},
        {"intent",  annotation_->confidence_summary.intent_confidence},
        {"prior",   annotation_->confidence_summary.prior_confidence},
        {"overall", annotation_->confidence_summary.overall}
    };

    return meta;
}

} // namespace GRIM
