#pragma once
#include "concept_block.hpp"

namespace GRIM {
// Supplied by upstream state producers; excludes answers and execution output.
struct ReasoningState {
    std::vector<std::string> knowns;
    std::vector<std::string> unknowns;
    std::optional<ConceptBlockGoal> goal;

    ConceptBlock withPrompt(const std::string& prompt) const {
        ConceptBlock block;
        block.prompt = prompt;
        block.knowns = knowns;
        block.unknowns = unknowns;
        block.goal = goal;
        return block;
    }
};
} // namespace GRIM
