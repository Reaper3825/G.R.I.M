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
#include <string>
#include <vector>

namespace GRIM {

struct ConceptBlock {
    std::string id;
    std::string name;

    std::string question;
    std::vector<std::string> intermediates;
    std::string answer;

    int intermediate_count = 0;
    std::vector<int> step_index;

    std::string format_type = "chain_of_thought";
    std::string source_sequence_id;
    int64_t     timestamp = 0;

    void recomputeDerived() {
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

} // namespace GRIM
