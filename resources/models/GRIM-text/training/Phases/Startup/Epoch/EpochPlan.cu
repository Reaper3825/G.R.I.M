#include "EpochPlan.hpp"

#include "../../Phase1_Startup.hpp"

#include <limits>
#include <stdexcept>
#include <string>

namespace GRIMText::Training {

namespace {

void resolveLearningRateResumePolicy(TrainingContext& ctx) {
    const auto checkpoint_hp = GRIM::HyperParameters::checkpointLoadHP(
        ctx.config,
        ctx.loaded_checkpoint_path,
        GRIM::HyperParameters::snapshotExecutionMode(ctx.config));
    const int optimizer_step = ctx.optimizer.optimizer_step.step;
    const std::string current_curriculum =
        GRIM::HyperParameters::snapshotTrainingConfigField<std::string>(
            ctx.config, "training_curriculum");

    bool restart_schedule =
        checkpoint_hp.checkpoint_resume_mode ==
        GRIM::HyperParameters::CheckpointResumeMode::RESTART;
    std::string reason = restart_schedule ? "config requested restart" : std::string{};

    if (!restart_schedule && !ctx.optimizer.optimizer_state_restored) {
        restart_schedule = true;
        reason = "optimizer state was not restored";
    }
    if (!restart_schedule && !ctx.checkpoint_latest_curriculum_completion) {
        restart_schedule = true;
        reason = "checkpoint has no latest curriculum completion metadata";
    }
    if (!ctx.current_curriculum_metadata) {
        throw std::runtime_error("LR resume policy has no current curriculum metadata");
    }

    std::uint64_t completed_epochs = 0;
    if (!restart_schedule) {
        const auto& latest = *ctx.checkpoint_latest_curriculum_completion;
        if (latest.curriculum.id != ctx.current_curriculum_metadata->id) {
            restart_schedule = true;
            reason = "curriculum changed from '" + latest.curriculum.name +
                "' to '" + current_curriculum + "'";
        } else {
            const auto previous_stage =
                GRIM::HyperParameters::parseTrainingStage(latest.curriculum.training_stage);
            if (previous_stage != checkpoint_hp.training_stage) {
                restart_schedule = true;
                reason = "training stage changed from " + latest.curriculum.training_stage +
                    " to " +
                    GRIM::HyperParameters::trainingStageToJsonString(checkpoint_hp.training_stage);
            } else {
                completed_epochs = latest.epochs_completed;
            }
        }
    }

    const auto max_schedule_step =
        static_cast<std::uint64_t>(std::numeric_limits<int>::max());
    if (!restart_schedule &&
        completed_epochs > max_schedule_step /
            static_cast<std::uint64_t>(ctx.epoch_plan.steps_per_epoch)) {
        throw std::runtime_error("LR resume schedule step exceeds supported integer range");
    }
    const std::uint64_t schedule_step_u64 = restart_schedule
        ? 0
        : completed_epochs * static_cast<std::uint64_t>(ctx.epoch_plan.steps_per_epoch);
    const int schedule_step = static_cast<int>(schedule_step_u64);
    ctx.lr_schedule_optimizer_step_at_start = optimizer_step;
    ctx.lr_schedule_step_at_start = schedule_step;
    ctx.curriculum_epochs_completed_at_start = completed_epochs;

    ctx.logging.logger->log(
        std::string("[LR_RESUME] mode=") +
        (restart_schedule ? "restart" : "resume") +
        " optimizer_step=" + std::to_string(optimizer_step) +
        " schedule_step=" + std::to_string(schedule_step) +
        " completed_epochs=" + std::to_string(completed_epochs) +
        (reason.empty() ? std::string{} : " reason=\"" + reason + "\""));
}

} // namespace

void EpochPlanReady(TrainingContext& ctx) {
    const int authored_train_batches = static_cast<int>(ctx.train_payloads.size());
    ctx.epoch_plan = finalizeEpochPlanOrThrow(ctx.config, authored_train_batches);
    ctx.lr_schedule.emplace(ctx.epoch_plan.lr_config);
    if (!ctx.lr_schedule) {
        throw std::runtime_error("FATAL: lr_schedule not initialized during startup");
    }
    resolveLearningRateResumePolicy(ctx);
}

} // namespace GRIMText::Training

