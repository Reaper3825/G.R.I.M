#pragma once

namespace GRIMText::Training {

struct TrainingContext;

struct Phase2HandoffInputs {
    TrainingContext& ctx;
};

Phase2HandoffInputs makePhase2HandoffReady(TrainingContext& ctx);
void Phase2HandoffReady(TrainingContext& ctx);

} // namespace GRIMText::Training

