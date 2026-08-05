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
#include <string>
#include <vector>

namespace GRIM {

enum class ConceptExecutionGateTarget : uint8_t {
    Unsupervised = 0,
    Noop = 1,
    Execute = 2
};

inline const char* conceptExecutionGateTargetJsonValue(ConceptExecutionGateTarget target) {
    switch (target) {
        case ConceptExecutionGateTarget::Noop: return "noop";
        case ConceptExecutionGateTarget::Execute: return "execute";
        case ConceptExecutionGateTarget::Unsupervised: return "ignore";
    }
    return "ignore";
}

inline ConceptExecutionGateTarget conceptExecutionGateTargetFromJsonValue(
    const std::string& value)
{
    if (value == "noop") return ConceptExecutionGateTarget::Noop;
    if (value == "execute") return ConceptExecutionGateTarget::Execute;
    return ConceptExecutionGateTarget::Unsupervised;
}

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

    std::string prompt;
    std::vector<std::string> intermediates;
    std::string answer;

    ConceptExecutionGateTarget execution_gate_target =
        ConceptExecutionGateTarget::Unsupervised;

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

// Returns an empty string when the block is safe to persist for the GRIM-text
// concept loader. The loader performs the same arithmetic checks again while
// compiling the training corpus; this provides immediate authoring feedback.
inline std::string validateConceptBlockExecutionControl(const ConceptBlock& cb) {
    const bool has_state = !cb.state_0.type.empty() || !cb.state_0.atoms.empty();
    const bool has_bootstrap = !cb.state_0.atoms.empty();
    const bool has_steps = !cb.execution.empty();

    if (cb.execution_gate_target != ConceptExecutionGateTarget::Unsupervised
        && cb.prompt.empty()) {
        return "supervised execution control requires a non-empty prompt";
    }

    if (cb.execution_gate_target == ConceptExecutionGateTarget::Noop) {
        if (has_state || has_steps) {
            return "NOOP blocks cannot contain STATE0 or EXEC data";
        }
        return {};
    }

    if (cb.execution_gate_target == ConceptExecutionGateTarget::Execute) {
        if (!has_bootstrap) return "EXECUTE blocks require at least one STATE0 atom";
        if (!has_steps) return "EXECUTE blocks require at least one EXEC step";
    } else if (has_state || has_steps) {
        return "blocks with STATE0 or EXEC data must use the EXECUTE gate target";
    } else {
        return {};
    }

    std::vector<double> slots = cb.state_0.atoms;
    for (size_t i = 0; i < slots.size(); ++i) {
        if (!std::isfinite(slots[i])) {
            return "STATE0 atom " + std::to_string(i) + " is not finite";
        }
    }

    for (size_t step_index = 0; step_index < cb.execution.size(); ++step_index) {
        const auto& step = cb.execution[step_index];
        const std::string prefix = "EXEC step " + std::to_string(step_index + 1) + ": ";
        if (step.args.size() < 2) return prefix + "requires two args";
        if (step.arg_slots.size() != 2) return prefix + "requires exactly two arg_slots";
        const int lhs_slot = step.arg_slots[0];
        const int rhs_slot = step.arg_slots[1];
        if (lhs_slot < 0 || lhs_slot >= static_cast<int>(slots.size())) {
            return prefix + "arg_slots[0] is outside the current slot range";
        }
        if (rhs_slot < 0 || rhs_slot >= static_cast<int>(slots.size())) {
            return prefix + "arg_slots[1] is outside the current slot range";
        }
        if (!std::isfinite(step.result)) return prefix + "result is not finite";

        const double lhs_tolerance = 1e-9 * std::max(1.0, std::fabs(slots[lhs_slot]));
        const double rhs_tolerance = 1e-9 * std::max(1.0, std::fabs(slots[rhs_slot]));
        if (!std::isfinite(step.args[0])
            || std::fabs(step.args[0] - slots[lhs_slot]) > lhs_tolerance) {
            return prefix + "args[0] does not match arg_slots[0]";
        }
        if (!std::isfinite(step.args[1])
            || std::fabs(step.args[1] - slots[rhs_slot]) > rhs_tolerance) {
            return prefix + "args[1] does not match arg_slots[1]";
        }

        double computed = 0.0;
        if (step.op == "add") computed = slots[lhs_slot] + slots[rhs_slot];
        else if (step.op == "sub") computed = slots[lhs_slot] - slots[rhs_slot];
        else if (step.op == "mul") computed = slots[lhs_slot] * slots[rhs_slot];
        else if (step.op == "div") {
            if (slots[rhs_slot] == 0.0) return prefix + "division by zero";
            computed = slots[lhs_slot] / slots[rhs_slot];
        } else {
            return prefix + "unknown operation '" + step.op + "'";
        }

        const double tolerance = 1e-9 * std::max(1.0, std::fabs(step.result));
        if (std::fabs(computed - step.result) > tolerance) {
            return prefix + "arg_slots compute " + std::to_string(computed)
                + " but result is " + std::to_string(step.result);
        }
        slots.push_back(step.result);
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
    bool                     format_as_concept = true;  // false = plain text / pretraining mode

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
