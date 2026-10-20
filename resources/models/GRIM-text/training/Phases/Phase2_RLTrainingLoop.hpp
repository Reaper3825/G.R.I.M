#pragma once

// Phase 2 RL forward/backward/update scaffold.
// The implementation currently contains pseudocode and throws if invoked.
// Episode scheduling and retry collection belong to a separate sequence.

namespace GRIMText::Training {

struct TrainingContext;

// Preparation-owned handoff, not yet implemented. It must provide a complete
// 16-task / 64-episode group, recorded-prefix BatchPayload microbatches, causal
// sampled-action rows, detached attempt advantages, and policy/reference identity.
// This loop does not construct episodes or calculate rewards/advantages.
struct PreparedRLEpisodeGroup;

// Selected by TrainingStage::RL once startup and dispatch are wired.
// Executes one group's forwards/backwards and exactly one optimizer update.
// Returns true only after a successful update; failures throw. The calling
// sequence owns group consumption, policy-version publication, and retry EMA.
// This declaration does not enable an RL run; the current body always throws.
bool executePhase2RLTrainingLoop(
    TrainingContext& ctx,
    const PreparedRLEpisodeGroup& group);

} // namespace GRIMText::Training
