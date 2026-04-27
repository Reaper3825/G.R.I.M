#include "Phase2Handoff.hpp"

#include <chrono>

#include "../../../../../../../control/training_control_generated.h"
#include "../../Phase1_Startup.hpp"

namespace GRIMText::Training {

Phase2HandoffInputs makePhase2HandoffReady(TrainingContext& ctx) {
    ctx.logging.status_writer->writeStatus(
        GRIMText::Control::TrainingState_Training,
        0, static_cast<int>(ctx.epoch_batch_order.size()), 0, ctx.epoch_plan.total_batches,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        "Phase 1 complete - ready for training");
    ctx.start_time = std::chrono::steady_clock::now();
    return Phase2HandoffInputs{ctx};
}

void Phase2HandoffReady(TrainingContext& ctx) {
    (void)makePhase2HandoffReady(ctx);
}

} // namespace GRIMText::Training

