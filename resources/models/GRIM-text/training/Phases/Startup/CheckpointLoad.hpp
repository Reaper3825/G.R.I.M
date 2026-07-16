#pragma once

namespace GRIMText::Training {

struct TrainingContext;

void CheckpointPlanReady(TrainingContext& ctx);
void CheckpointLoaded(TrainingContext& ctx);

} // namespace GRIMText::Training
