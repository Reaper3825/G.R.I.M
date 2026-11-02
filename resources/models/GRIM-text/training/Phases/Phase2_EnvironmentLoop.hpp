#pragma once

#include "Phase2_InferenceLoop.hpp"

namespace GRIMText::Training {

struct Phase2EnvironmentResult {
    std::string episode_status;
    // Absent if the prepared episode is already terminal. Otherwise contains
    // the first span pause or terminal inference return, never a Goal verdict.
    std::optional<Phase2TextInferenceResult> generation;
};

// Consume the prepared RL environment handoff and begin structured generation.
// Returns at its first span pause/EOS/limit, retaining the normal live paused
// GenerationState. No command dispatch, automatic resume, or optimizer work.
Phase2EnvironmentResult beginPhase2EnvironmentInteraction(TrainingContext& ctx);

} // namespace GRIMText::Training
