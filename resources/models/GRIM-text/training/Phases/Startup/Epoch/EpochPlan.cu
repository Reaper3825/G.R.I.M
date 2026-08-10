#include "EpochPlan.hpp"

#include "../../Phase1_Startup.hpp"

#include <stdexcept>

namespace GRIMText::Training {

void EpochPlanReady(TrainingContext& ctx) {
    const int authored_train_batches = static_cast<int>(ctx.train_payloads.size());
    ctx.epoch_plan = finalizeEpochPlanOrThrow(ctx.config, authored_train_batches);
    ctx.lr_schedule.emplace(ctx.epoch_plan.lr_config);
    if (!ctx.lr_schedule) {
        throw std::runtime_error("FATAL: lr_schedule not initialized during startup");
    }
}

} // namespace GRIMText::Training

