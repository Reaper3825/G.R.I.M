#pragma once

namespace GRIMText::Training {
struct TrainingContext;
// RL preparation only: no SFT corpus, batches, optimizer, or rollout execution.
void EnvironmentReady(TrainingContext& ctx);
}
