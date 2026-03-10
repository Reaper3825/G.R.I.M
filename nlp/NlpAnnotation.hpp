// NlpAnnotation — canonical NLP output payload
//
// Replaces Intent as the authoritative analysis product.
// Consumed by: memory tagging, router metadata builder, context manager,
//              tool selection, action policy evaluation.
//
// Intent remains as a legacy compatibility projection (Intent::from(NlpAnnotation)).
//
// Alignment: entity/atom categories mirror GRIM-text atom concepts
//   (numbers, URLs, emails, paths, dates, code-literals) so body
//   and router share one semantic vocabulary.
//======================================================//
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace GRIM {

// =========================================================
// Entity — a recognized semantic span in the input
// =========================================================
struct Entity {
    std::string text;               // raw surface form
    std::string type;               // "app", "file", "path", "url", "email",
                                    // "date", "time", "number", "location",
                                    // "person", "command", "code_literal"
    int         start_offset = 0;   // byte offset in normalized_text
    int         end_offset   = 0;
    float       confidence   = 1.0f;
    std::unordered_map<std::string, std::string> attributes; // type-specific kv
};

// =========================================================
// Ambiguity — an unresolved or competing interpretation
// =========================================================
struct Ambiguity {
    std::string span;               // text span that is ambiguous
    std::vector<std::string> interpretations; // competing readings
    float       confidence_gap = 0.0f; // delta between best two
};

// =========================================================
// UtterancePriors — cheap command/question/banter probabilities
//   Produced by IntentGate / FastClassifier.
// =========================================================
struct UtterancePriors {
    float command   = 0.0f;
    float question  = 0.0f;
    float banter    = 0.0f;
    float unknown   = 1.0f;
};

// =========================================================
// ConfidenceSummary — overall annotation quality signal
// =========================================================
struct ConfidenceSummary {
    float entity_confidence     = 0.0f; // aggregate entity recognition
    float intent_confidence     = 0.0f; // top intent match quality
    float prior_confidence      = 0.0f; // utterance classification sharpness
    float overall               = 0.0f; // blended summary
};

// =========================================================
// NlpAnnotation — the full annotation payload
// =========================================================
struct NlpAnnotation {
    // ─── Core text ────────────────────────────────────────
    std::string raw_text;               // original user input
    std::string normalized_text;        // lowered, whitespace-cleaned
    std::string language;               // ISO 639-1 (e.g. "en")

    // ─── Classification priors ────────────────────────────
    UtterancePriors utterance_priors;

    // ─── Entities ─────────────────────────────────────────
    std::vector<Entity> entities;

    // ─── Affordance / tool hints ──────────────────────────
    //   action_affordances: semantic action tags
    //     e.g. "open", "search", "navigate", "edit", "dangerous_delete"
    std::vector<std::string> action_affordances;

    //   candidate_tool_tokens: early tool candidates
    //     e.g. "<press_hold_drag>", "<open_app>", "<click_ui>", "<type_text>"
    std::vector<std::string> candidate_tool_tokens;

    //   tool_context_hints: app/UI/workflow context
    //     e.g. "app:blender", "selection:mesh", "viewport:3d"
    std::vector<std::string> tool_context_hints;

    // ─── Tags (for routing, memory, context) ──────────────
    std::vector<std::string> memory_tags;   // optimized for storage/retrieval
    std::vector<std::string> router_tags;   // optimized for subject-model selection
    std::vector<std::string> context_tags;  // task/domain/env: "coding", "filesystem", "web", "system", "ui", "voice"
    std::vector<std::string> risk_tags;     // "destructive", "system", "network", "credential_sensitive"

    // ─── References and ambiguities ───────────────────────
    std::vector<std::string> references;    // pronouns / deictics: "that file", "it", "same folder"
    std::vector<Ambiguity>   ambiguities;

    // ─── Confidence ───────────────────────────────────────
    ConfidenceSummary confidence_summary;

    // ─── Legacy compatibility ─────────────────────────────
    //   Populated during migration so old consumers can still work.
    std::string legacy_intent_name;
    std::string legacy_category;
    bool        legacy_matched = false;
};

// Forward
struct ContextSnapshot;

// Build a full NlpAnnotation from raw user input + context.
// Wraps NLP::parse, IntentGate::decide, FastClassifier::evaluate.
NlpAnnotation annotate(const std::string& raw_input,
                        const ContextSnapshot& context);

} // namespace GRIM

// Forward declare V2 for overload
namespace GRIM::MMO { struct ContextSnapshotV2; }

namespace GRIM {
// V2 overload — accepts rich ContextSnapshotV2, projects to V1 internally.
NlpAnnotation annotate(const std::string& raw_input,
                        const GRIM::MMO::ContextSnapshotV2& context_v2);
} // namespace GRIM
