#pragma once
#include "concept_block.hpp"

namespace GRIM {
// Supplied task/observation context; excludes model-authored continuation.
struct ReasoningState {
    std::vector<std::string> knowns;
    std::vector<std::string> unknowns;
    std::optional<ConceptBlockGoal> goal;
    std::string environment_context;

    ConceptBlock withPrompt(const std::string& prompt) const {
        if(!environment_context.empty())throw std::invalid_argument("ConceptBlock cannot store recurrent environment context; use inference input directly");
        ConceptBlock block;
        block.prompt = prompt;
        block.knowns = knowns;
        block.unknowns = unknowns;
        block.goal = goal;
        return block;
    }
};
} // namespace GRIM
