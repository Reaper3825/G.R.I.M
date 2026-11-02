#pragma once
#include "Phase2_EnvironmentLoop.hpp"

namespace GRIMText::Training {
struct Phase2EnvironmentRunResult {
    Phase2EnvironmentResult interaction;
    std::string stop_reason; // episode_terminal or generation_terminal
};
// Run the current frozen Goal through span pauses, Answer submissions and real
// environment feedback. No schedule advancement or RL optimizer work.
Phase2EnvironmentRunResult executePhase2EnvironmentInteraction(TrainingContext& ctx);
}
