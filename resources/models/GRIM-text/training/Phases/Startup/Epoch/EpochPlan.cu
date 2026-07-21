#include "EpochPlan.hpp"

#include "../../Phase1_Startup.hpp"

#include <stdexcept>

namespace GRIMText::Training {

void EpochPlanReady(TrainingContext& ctx) {
    const int authored_train_batches = static_cast<int>(ctx.train_payloads.size());
    ctx.epoch_plan = finalizeEpochPlanOrThrow(ctx.config, authored_train_batches);
    ctx.estimated_total_steps = ctx.epoch_plan.estimated_total_steps;
    ctx.lr_schedule.emplace(ctx.epoch_plan.lr_config);
    if (!ctx.lr_schedule) {
        throw std::runtime_error("FATAL: lr_schedule not initialized during startup");
    }
    ctx.execution_transition_schedule.emplace(
        ctx.epoch_plan.execution_transition_config);
    if (!ctx.execution_transition_schedule) {
        throw std::runtime_error(
            "FATAL: execution_transition_schedule not initialized during startup");
    }
}

} // namespace GRIMText::Training

