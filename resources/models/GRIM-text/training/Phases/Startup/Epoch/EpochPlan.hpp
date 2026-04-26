#pragma once

#include "../Scheduling/SchedulerPreflight.hpp"

#include "../../../../Shared/Dynamic_LR/LRSchedule.hpp"
#include "../../../../Shared/HyperParameters/HyperparameterGroupings.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace GRIMText::Training {

struct EpochPlan {
    int total_batches = 0;
    int estimated_total_steps = 0;
    int steps_per_epoch = 0;
    int warmup_steps = 0;
    ::GRIM::LR::LRScheduleConfig lr_config;
};

inline EpochPlan finalizeEpochPlanOrThrow(
    ::GRIM::HyperParameters::StartupConfig& config,
    const SchedulerPreflightState& preflight)
{
    const auto& hp = config.hyperparameters;
    const int num_epochs = std::max(1, hp.epochs);
    const int accum = std::max(1, hp.gradient_accumulation_steps);

    EpochPlan plan;
    plan.total_batches = preflight.total_batches;
    if (plan.total_batches <= 0) {
        throw std::runtime_error("FATAL: scheduler produced 0 batches during startup epoch-plan finalization");
    }

    plan.estimated_total_steps = (num_epochs * plan.total_batches) / accum;
    if (plan.estimated_total_steps <= 0) {
        throw std::runtime_error("FATAL: estimated_total_steps computed as <= 0 during startup (epochs=" +
                                 std::to_string(num_epochs) + " batches=" +
                                 std::to_string(plan.total_batches) + " accum=" +
                                 std::to_string(accum) + ")");
    }

    ::GRIM::Config::deriveWarmupSteps(config.hyperparameters, plan.estimated_total_steps);
    plan.steps_per_epoch = plan.total_batches / accum;
    plan.warmup_steps = config.hyperparameters.warmup_steps;
    plan.lr_config = ::GRIM::HyperParameters::makeLRScheduleConfig(
        ::GRIM::HyperParameters::learningRateScheduleInputs(config.hyperparameters),
        plan.estimated_total_steps,
        plan.steps_per_epoch);

    return plan;
}

} // namespace GRIMText::Training

