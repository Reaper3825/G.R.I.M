//======================================================//
//  ConceptBlock — Atomic curriculum learning unit.
//
//  Sits between raw sequences and model assignment:
//    structured sequence -> id -> ConceptBlock -> id
//    -> name -> dataset
//
//  Each block represents a single thought, equation,
//  definition, proof step, etc. Models are assigned a
//  curriculum (ordered set of ConceptBlock IDs).
//======================================================//

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include "../resources/models/GRIM-text/Shared/Goal/GoalContract.hpp"

namespace GRIM {

struct ConceptBlock {
    std::string id;
    std::string name;

    std::string prompt;
    // Explicit information available to and missing from this concept. These
    // are top-level ConceptBlock fields, independent of the optional goal.
    std::vector<std::string> knowns;
    std::vector<std::string> unknowns;
    std::vector<std::string> intermediates;
    std::string determine;
    std::string define;
    std::string execute;
    std::string update;
    std::string answer;
    // Unstructured model-visible text. Raw blocks use this field instead of
    // overloading prompt/answer with artificial document segments.
    std::string raw;

    /// NL steps parallel to `intermediates`; JSON field `explanation`.
    std::vector<std::string> explanation;

    std::optional<ConceptBlockGoal> goal;

    int intermediate_count = 0;
    std::vector<int> step_index;

    std::string format_type = "chain_of_thought";
    std::string source_sequence_id;
    int64_t     timestamp = 0;

    void recomputeDerived() {
        if (explanation.empty() && !intermediates.empty())
            explanation = intermediates;
        else if (!explanation.empty() && intermediates.empty())
            intermediates = explanation;

        intermediate_count = static_cast<int>(intermediates.size());
        step_index.resize(intermediate_count);
        for (int i = 0; i < intermediate_count; ++i)
            step_index[i] = i;
    }
};

// ── Format presets ──────────────────────────────────────

struct ConceptFormatPreset {
    const char* key;
    const char* label;
    const char* questionLabel;
    const char* intermediatesLabel;
    const char* answerLabel;
    int         defaultIntermediateCount;
};

inline constexpr ConceptFormatPreset kConceptPresets[] = {
    { "qa",               "Q/A",              "Prompt",   nullptr,            "Answer",  0 },
    { "chain_of_thought", "Chain of Thought", "Prompt",   "Thought Steps",    "Answer",  3 },
    { "definition",       "Definition",       "Term",       "Examples / Notes", "Summary", 2 },
    { "proof",            "Proof",            "Theorem",    "Proof Steps",      "QED",     3 },
    { "derivation",       "Derivation",       "Expression", "Derivation Steps", "Result",  2 },
    { "conversation",     "Conversation",     "User",       "Turns",            "Response", 2 },
    { "raw",              "Raw",              "Raw",        nullptr,            nullptr,   0 },
};

inline constexpr int kConceptPresetCount = sizeof(kConceptPresets) / sizeof(kConceptPresets[0]);

inline int presetIndexForKey(const std::string& key) {
    for (int i = 0; i < kConceptPresetCount; ++i)
        if (key == kConceptPresets[i].key) return i;
    return 1;
}

inline std::vector<std::string> presetLabels() {
    std::vector<std::string> v;
    v.reserve(kConceptPresetCount);
    for (int i = 0; i < kConceptPresetCount; ++i)
        v.emplace_back(kConceptPresets[i].label);
    return v;
}

// Curriculum -> Course -> ConceptBlock

// Courses own membership; curriculum block IDs are a derived compatibility view.
struct Course {
    std::string id;
    std::string name;
    std::vector<std::string> concept_block_ids;
    // Persistent environment/game reference; empty means no environment assigned.
    // Resolved through World Lab's environment registry; independent of entity IDs.
    std::string environment_id;
};

struct Curriculum {
    std::vector<std::string> course_ids;
    std::string              id;
    std::string              name;
    std::vector<std::string> concept_block_ids;
    std::vector<std::string> plaintext_block_ids;
    // Training procedure metadata owned by the curriculum registry.
    // Valid persisted values are: pt, sft, dpo, rlhf, rl.
    std::string              training_stage = "sft";
    int64_t                  timestamp = 0;
    bool                     randomize_course_order = false;
    bool                     randomize_concept_block_order = false;
    bool                     format_as_concept = true;  // false = render curriculum rows as plain text

    bool containsBlock(const std::string& cb_id) const {
        for (const auto& bid : concept_block_ids)
            if (bid == cb_id) return true;
        return false;
    }


};

} // namespace GRIM
