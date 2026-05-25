#pragma once

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
    ::GRIM::HyperParameters::LanguageModelConfig& config,
    int authored_train_batches)
{
    const auto schedule_hp =
        ::GRIM::HyperParameters::trainingScheduleHP(config);

    if (schedule_hp.epochs <= 0) {
        throw std::runtime_error("FATAL: epochs must be > 0 during startup epoch-plan finalization (got " +
                                 std::to_string(schedule_hp.epochs) + ")");
    }
    if (schedule_hp.gradient_accumulation_steps <= 0) {
        throw std::runtime_error("FATAL: gradient_accumulation_steps must be > 0 during startup epoch-plan finalization (got " +
                                 std::to_string(schedule_hp.gradient_accumulation_steps) + ")");
    }
    const int num_epochs = schedule_hp.epochs;
    const int accum = schedule_hp.gradient_accumulation_steps;

    EpochPlan plan;
    if (authored_train_batches <= 0) {
        throw std::runtime_error("FATAL: PlannedBatchesReady authored 0 train batches before epoch-plan finalization");
    }
    if (schedule_hp.single_batch_overfit_enabled) {
        if (schedule_hp.single_batch_overfit_max_steps <= 0) {
            throw std::runtime_error("FATAL: single_batch_overfit_max_steps must be > 0 during startup when single_batch_overfit_enabled=true (got " +
                                     std::to_string(schedule_hp.single_batch_overfit_max_steps) + ")");
        }
        plan.total_batches = schedule_hp.single_batch_overfit_max_steps;
    } else {
        plan.total_batches = authored_train_batches;
    }

    const int64_t total_accumulation_slots =
        static_cast<int64_t>(num_epochs) * static_cast<int64_t>(plan.total_batches);
    if (total_accumulation_slots > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("FATAL: estimated_total_steps input overflow during startup (epochs=" +
                                 std::to_string(num_epochs) + " batches=" +
                                 std::to_string(plan.total_batches) + " product=" +
                                 std::to_string(total_accumulation_slots) + ")");
    }

    plan.estimated_total_steps = static_cast<int>(total_accumulation_slots / accum);
    if (plan.estimated_total_steps <= 0) {
        throw std::runtime_error("FATAL: estimated_total_steps computed as <= 0 during startup (epochs=" +
                                 std::to_string(num_epochs) + " batches=" +
                                 std::to_string(plan.total_batches) + " accum=" +
                                 std::to_string(accum) + ")");
    }

    if (config.warmup_fraction <= 0.0f ||
        config.warmup_fraction >= 1.0f) {
        throw std::runtime_error(
            "FATAL: warmup_fraction must be in (0, 1) during startup epoch-plan finalization, got " +
            std::to_string(config.warmup_fraction));
    }
    config.warmup_steps = std::max(
        1,
        static_cast<int>(config.warmup_fraction * plan.estimated_total_steps));
    config.mtp_alpha_warmup_steps = config.warmup_steps;
    config.telemetry_warmup_steps = config.warmup_steps;
    config.execution_block_gate_warmup_steps = config.warmup_steps;

    plan.steps_per_epoch = plan.total_batches / accum;
    if (plan.steps_per_epoch <= 0) {
        throw std::runtime_error("FATAL: steps_per_epoch computed as <= 0 during startup (batches=" +
                                 std::to_string(plan.total_batches) + " accum=" +
                                 std::to_string(accum) + ")");
    }
    plan.warmup_steps = config.warmup_steps;
    plan.lr_config = ::GRIM::HyperParameters::makeLRScheduleConfig(
        ::GRIM::HyperParameters::learningRateScheduleInputs(config),
        plan.estimated_total_steps,
        plan.steps_per_epoch);

    return plan;
}

void EpochPlanReady(TrainingContext& ctx);

} // namespace GRIMText::Training

