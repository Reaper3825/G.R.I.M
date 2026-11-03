#pragma once

#include "Phase2_InferenceLoop.hpp"

namespace GRIMText::Training {

struct Phase2EnvironmentResult {
    std::string episode_status;
    // First span pause or terminal inference return, never a Goal verdict.
    // A terminal world can still receive a final model Answer submission.
    std::optional<Phase2TextInferenceResult> generation;
};

struct Phase2EnvironmentRunResult {
    Phase2EnvironmentResult interaction;
    std::string stop_reason; // attempt_submitted or generation_terminal
};

struct Phase2EnvironmentLoopResult {
    std::size_t examples_completed = 0;
    std::size_t examples_succeeded = 0;
    std::size_t examples_failed = 0;
    std::size_t commands = 0;
};

// Run every selected course example in the loader's order. Each example gets
// a fresh world starting state and generation context. Failed model attempts
// advance normally; infrastructure errors stop with prior episode records intact.
Phase2EnvironmentLoopResult executePhase2EnvironmentLoop(TrainingContext& ctx);

// Phase2 environment entrypoint. Owns the current frozen Goal's interaction
// loop: dispatch Execute commands, insert permitted feedback, and resume until
// Answer submits the Goal attempt or generation terminates.
// Single-example helper; the outer entrypoint owns advancement. No RL updates.
Phase2EnvironmentRunResult executePhase2EnvironmentInteraction(TrainingContext& ctx);

// Consume the prepared RL environment handoff and begin structured generation.
// Returns at its first span pause/EOS/limit, retaining the normal live paused
// GenerationState. No command dispatch, automatic resume, or optimizer work.
Phase2EnvironmentResult beginPhase2EnvironmentInteraction(TrainingContext& ctx);

} // namespace GRIMText::Training
