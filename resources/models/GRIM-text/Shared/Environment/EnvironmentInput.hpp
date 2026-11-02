#pragma once

#include "../Goal/GoalContract.hpp"
#include <string>

namespace GRIM::Environment {

struct EnvironmentInferenceInput {
    std::string prompt;
    nlohmann::json reasoning_state;
    std::string episode_status; // Host control state; not included in model input.
};

// Project only player-visible text and Goal requirement text. Verifier facts,
// evidence, command suggestions, IDs and host envelopes stay private.
EnvironmentInferenceInput buildEnvironmentInferenceInput(
    const ConceptBlockGoal& goal, const nlohmann::json& response);

} // namespace GRIM::Environment
