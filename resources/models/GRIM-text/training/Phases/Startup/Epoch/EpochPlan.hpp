#pragma once

#include "../Scheduling/SchedulerPreflight.hpp"

#include "../../../../Shared/Dynamic_LR/LRSchedule.hpp"
#include "../../../../Shared/HyperParameters/HyperparameterGroupings.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace GRIMText::Training {

struct TrainingContext;

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
    if (hp.epochs <= 0) {
        throw std::runtime_error("FATAL: epochs must be > 0 during startup epoch-plan finalization (got " +
                                 std::to_string(hp.epochs) + ")");
    }
    if (hp.gradient_accumulation_steps <= 0) {
        throw std::runtime_error("FATAL: gradient_accumulation_steps must be > 0 during startup epoch-plan finalization (got " +
                                 std::to_string(hp.gradient_accumulation_steps) + ")");
    }
    const int num_epochs = hp.epochs;
    const int accum = hp.gradient_accumulation_steps;

    EpochPlan plan;
    plan.total_batches = preflight.total_batches;
    if (plan.total_batches <= 0) {
        throw std::runtime_error("FATAL: scheduler produced 0 batches during startup epoch-plan finalization");
    }

    const int64_t total_micro_batches =
        static_cast<int64_t>(num_epochs) * static_cast<int64_t>(plan.total_batches);
    if (total_micro_batches > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("FATAL: estimated_total_steps input overflow during startup (epochs=" +
                                 std::to_string(num_epochs) + " batches=" +
                                 std::to_string(plan.total_batches) + " product=" +
                                 std::to_string(total_micro_batches) + ")");
    }

    plan.estimated_total_steps = static_cast<int>(total_micro_batches / accum);
    if (plan.estimated_total_steps <= 0) {
        throw std::runtime_error("FATAL: estimated_total_steps computed as <= 0 during startup (epochs=" +
                                 std::to_string(num_epochs) + " batches=" +
                                 std::to_string(plan.total_batches) + " accum=" +
                                 std::to_string(accum) + ")");
    }

    ::GRIM::Config::deriveWarmupSteps(config.hyperparameters, plan.estimated_total_steps);
    plan.steps_per_epoch = plan.total_batches / accum;
    if (plan.steps_per_epoch <= 0) {
        throw std::runtime_error("FATAL: steps_per_epoch computed as <= 0 during startup (batches=" +
                                 std::to_string(plan.total_batches) + " accum=" +
                                 std::to_string(accum) + ")");
    }
    plan.warmup_steps = config.hyperparameters.warmup_steps;
    plan.lr_config = ::GRIM::HyperParameters::makeLRScheduleConfig(
        ::GRIM::HyperParameters::learningRateScheduleInputs(config.hyperparameters),
        plan.estimated_total_steps,
        plan.steps_per_epoch);

    return plan;
}

void EpochPlanReady(TrainingContext& ctx);

} // namespace GRIMText::Training

