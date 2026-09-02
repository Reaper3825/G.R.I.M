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
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace GRIM {

struct ConceptExecutionStep {
    std::string          op;
    std::vector<double>  args;
    std::vector<int>     arg_slots;  // Optional: canonical slot indices for args.
                                     // When present, slot identity is explicit —
                                     // no value-based resolution needed.
    double               result = 0.0;
};

// Optional evidence stays nested with its criterion so downstream verification
// never has to infer the association from parallel arrays or criterion text
// keys. An empty string means the criterion is waiting for evidence generation.
struct ConceptBlockSuccessCriterion {
    std::string criterion;
    std::string evidence;
};

struct ConceptBlockGoal {
    std::string target_state;
    std::vector<ConceptBlockSuccessCriterion> success_criteria;
    std::vector<std::string> constraints;
};

struct ConceptBlock {
    std::string id;
    std::string name;

    std::string prompt;
    // Explicit information available to and missing from this concept. These
    // are top-level ConceptBlock fields, independent of the optional goal.
    std::vector<std::string> knowns;
    std::vector<std::string> unknowns;
    std::vector<std::string> intermediates;
    std::string answer;
    // Unstructured model-visible text. Raw blocks use this field instead of
    // overloading prompt/answer with artificial document segments.
    std::string raw;

    /// NL steps parallel to `intermediates`; JSON field `explanation`.
    std::vector<std::string> explanation;

    std::vector<ConceptExecutionStep> execution;
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

// Returns an empty string when the block is safe to persist for the GRIM-text
// concept loader. The loader performs the same arithmetic checks again while
// compiling the training corpus; this provides immediate authoring feedback.
inline std::string validateConceptBlockExecutionControl(const ConceptBlock& cb) {
    for (size_t step_index = 0; step_index < cb.execution.size(); ++step_index) {
        const auto& step = cb.execution[step_index];
        const std::string prefix = "EXEC step " + std::to_string(step_index + 1) + ": ";
        if (step.args.size() < 2) return prefix + "requires two args";
        if (!step.arg_slots.empty() && step.arg_slots.size() != 2)
            return prefix + "requires exactly two arg_slots when provided";
        if (!std::isfinite(step.result)) return prefix + "result is not finite";
        if (!std::isfinite(step.args[0]) || !std::isfinite(step.args[1]))
            return prefix + "args must be finite";

        double computed = 0.0;
        if (step.op == "add") computed = step.args[0] + step.args[1];
        else if (step.op == "sub") computed = step.args[0] - step.args[1];
        else if (step.op == "mul") computed = step.args[0] * step.args[1];
        else if (step.op == "div") {
            if (step.args[1] == 0.0) return prefix + "division by zero";
            computed = step.args[0] / step.args[1];
        } else {
            return prefix + "unknown operation '" + step.op + "'";
        }

        const double tolerance = 1e-9 * std::max(1.0, std::fabs(step.result));
        if (std::fabs(computed - step.result) > tolerance) {
            return prefix + "args compute " + std::to_string(computed)
                + " but result is " + std::to_string(step.result);
        }
    }
    return {};
}

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

// ── Curriculum — Named group of ConceptBlock IDs ────────

struct Curriculum {
    std::string              id;
    std::string              name;
    std::vector<std::string> concept_block_ids;
    std::vector<std::string> plaintext_block_ids;
    // Training procedure metadata owned by the curriculum registry.
    // Valid persisted values are: pt, sft, dpo, rlhf.
    std::string              training_stage = "sft";
    int64_t                  timestamp = 0;
    bool                     format_as_concept = true;  // false = render curriculum rows as plain text

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
