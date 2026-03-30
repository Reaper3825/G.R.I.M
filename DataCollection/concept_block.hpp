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

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace GRIM {

// Structured register-style fields (optional). DEBUG: DataLoader may serialize this
// JSON into GRMT directly. Target: blocks are mainly id + pointer (e.g.
// source_sequence_id) to structured sequences in mass dataset / cache; prep resolves
// the pointer and encodes once — see ADDITION_SEQUENCES_AND_ARG_LEARNING.md.
struct ConceptBlockState0 {
    std::vector<double> atoms;
    std::string         type;
};

struct ConceptExecutionStep {
    std::string          op;
    std::vector<double>  args;
    std::vector<int>     arg_slots;  // Optional: canonical slot indices for args.
                                     // When present, slot identity is explicit —
                                     // no value-based resolution needed.
    double               result = 0.0;
};

struct ConceptBlockState1 {
    double result     = 0.0;
    bool   has_result = false;
};

struct ConceptBlock {
    std::string id;
    std::string name;

    std::string question;
    std::vector<std::string> intermediates;
    std::string answer;

    /// NL steps parallel to `intermediates`; JSON field `explanation`.
    std::vector<std::string> explanation;

    ConceptBlockState0              state_0;
    std::vector<ConceptExecutionStep> execution;
    ConceptBlockState1              state_1;

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
    { "qa",               "Q/A",              "Question",   nullptr,            "Answer",  0 },
    { "chain_of_thought", "Chain of Thought", "Question",   "Thought Steps",    "Answer",  3 },
    { "definition",       "Definition",       "Term",       "Examples / Notes", "Summary", 2 },
    { "proof",            "Proof",            "Theorem",    "Proof Steps",      "QED",     3 },
    { "derivation",       "Derivation",       "Expression", "Derivation Steps", "Result",  2 },
    { "conversation",     "Conversation",     "User",       "Turns",            "Response", 2 },
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

// ── Curriculum — Named group of ConceptBlock IDs ────────

struct Curriculum {
    std::string              id;
    std::string              name;
    std::vector<std::string> concept_block_ids;
    int64_t                  timestamp = 0;

    bool containsBlock(const std::string& cb_id) const {
        for (const auto& bid : concept_block_ids)
            if (bid == cb_id) return true;
        return false;
    }

    bool addBlock(const std::string& cb_id) {
        if (containsBlock(cb_id)) return false;
        concept_block_ids.push_back(cb_id);
        return true;
    }

    bool removeBlock(const std::string& cb_id) {
        auto it = std::find(concept_block_ids.begin(), concept_block_ids.end(), cb_id);
        if (it == concept_block_ids.end()) return false;
        concept_block_ids.erase(it);
        return true;
    }
};

} // namespace GRIM
