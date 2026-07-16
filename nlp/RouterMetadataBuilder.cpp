// RouterMetadataBuilder.cpp — builds the structured metadata payload
// sent to grim-text (the router model).
//======================================================//

#include "RouterMetadataBuilder.hpp"
#include "../memory/context_snapshot.hpp"
#include "../MMO/Core/SessionContextManager.hpp"
#include "../MMO/Shared/MMD.hpp"

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
    j["subject_tags"]         = subject_tags;
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
    has_v2_  = false;
    return *this;
}

RouterMetadataBuilder& RouterMetadataBuilder::setContextV2(const GRIM::MMO::ContextSnapshotV2& v2) {
    // Project V2 → V1 (owned copy) so build() can still use serializeContext(V1)
    owned_v1_.currentMood         = v2.current_mood;
    owned_v1_.lastNlpCategory     = v2.lastNlpCategory;
    owned_v1_.consecutiveCommands = v2.consecutiveCommands;
    owned_v1_.conversationDepth   = static_cast<int>(v2.recent_turn_summaries.size());
    owned_v1_.recentIntents       = v2.utterance_priors;
    context_ = &owned_v1_;

    // Serialize V2-only rich fields for the router
    context_v2_json_ = nlohmann::json::object();
    context_v2_json_["session_id"]  = v2.session_id;
    context_v2_json_["turn_id"]     = v2.turn_id;
    context_v2_json_["recent_turn_summaries"] = v2.recent_turn_summaries;
    context_v2_json_["latest_nlp_summary"]    = v2.latest_nlp_summary;
    context_v2_json_["memory_breadcrumbs"]    = v2.memory_breadcrumbs;
    context_v2_json_["resource_pressure"]     = v2.resource_pressure;
    context_v2_json_["current_mood"]          = v2.current_mood;

    // Active referents
    nlohmann::json refs = nlohmann::json::array();
    for (const auto& r : v2.active_referents) {
        refs.push_back({
            {"canonical_id", r.canonical_id},
            {"value",        r.value},
            {"entity_type",  r.entity_type},
            {"confidence",   r.confidence}
        });
    }
    context_v2_json_["active_referents"] = std::move(refs);

    // Visual context (populate builder's visual fields too)
    {
        nlohmann::json dv;
        dv["active_window"]  = v2.visual_context.digital.active_window;
        dv["active_process"] = v2.visual_context.digital.active_process;
        dv["ocr_text"]       = v2.visual_context.digital.ocr_text;
        dv["ocr_regions"]    = v2.visual_context.digital.ocr_regions;
        dv["ui_elements"]    = v2.visual_context.digital.ui_elements;
        dv["scene"]          = v2.visual_context.digital.scene_classification;
        dv["monitor_id"]     = v2.visual_context.digital.monitor_id;
        dv["source_device_id"] = v2.visual_context.digital.source_device_id;
        dv["source_platform"] = v2.visual_context.digital.source_platform;
        dv["source_transport"] = v2.visual_context.digital.source_transport;
        dv["capture_status"] = v2.visual_context.digital.capture_status;
        dv["capture_error"]  = v2.visual_context.digital.capture_error;
        dv["capture_backend"] = v2.visual_context.digital.capture_backend;
        dv["ocr_status"]      = v2.visual_context.digital.ocr_status;
        dv["ocr_error"]       = v2.visual_context.digital.ocr_error;
        dv["ocr_provider"]    = v2.visual_context.digital.ocr_provider;
        dv["ocr_mean_confidence"] = v2.visual_context.digital.ocr_mean_confidence;
        dv["automation_status"] = v2.visual_context.digital.automation_status;
        dv["automation_error"] = v2.visual_context.digital.automation_error;
        dv["automation_provider"] = v2.visual_context.digital.automation_provider;
        dv["automation_target_window"] = v2.visual_context.digital.automation_target_window;
        dv["automation_target_matches_capture"] =
            v2.visual_context.digital.automation_target_matches_capture;
        dv["automation_target_changed"] = v2.visual_context.digital.automation_target_changed;
        dv["preferred_grounding_source"] =
            v2.visual_context.digital.preferred_grounding_source;
        dv["capture_frame_counter"] = v2.visual_context.digital.provenance_frame_counter;
        dv["primitive_frame_counter"] =
            v2.visual_context.digital.primitive_provenance_frame_counter;
        digi_visual_ = std::move(dv);
    }
    {
        nlohmann::json pv;
        pv["scene_summary"]    = v2.visual_context.physical.scene_summary;
        pv["detected_objects"] = v2.visual_context.physical.detected_objects;
        phys_visual_ = std::move(pv);
    }

    has_v2_ = true;
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
    if (!context_)    throw std::runtime_error("RouterMetadataBuilder: context is NULL — caller MUST provide ContextSnapshot (V1 or V2)");

    RouterMetadata meta;
    meta.raw_input        = annotation_->raw_text;
    meta.normalized_input = annotation_->normalized_text;
    meta.nlp_annotation   = serializeAnnotation(*annotation_);

    // Context: base V1 + V2 enrichment if available
    nlohmann::json ctx_json = serializeContext(*context_);
    if (has_v2_) {
        ctx_json["v2"] = context_v2_json_;
    }
    meta.context_snapshot = std::move(ctx_json);

    meta.memory_tags      = annotation_->memory_tags;
    meta.subject_tags     = GRIM::MMO::getSubjectTags(annotation_->raw_text);
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
