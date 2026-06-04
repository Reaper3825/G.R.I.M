#include "StartupValidation.hpp"

#include "../../Phase1_Startup.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace GRIMText::Training {

namespace {

void require(bool ok, const std::string& message) {
    if (!ok) {
        throw std::runtime_error("FATAL: StartupValidated failed: " + message);
    }
}

} // namespace

void validateStartupOrThrow(const StartupValidationInputs& inputs) {
    const TrainingContext& ctx = inputs.ctx;
    const auto fixed_shape = GRIM::HyperParameters::trainingFixedShapeHP(ctx.config);
    const auto schedule_hp =
                GRIM::HyperParameters::trainingScheduleHP(ctx.config);

    require(ctx.logging.logger != nullptr, "logging logger is null");
    require(ctx.logging.status_writer != nullptr, "status writer is null");

    require(fixed_shape.batch_size == GRIM::HyperParameters::snapshotTrainingConfigField<int>(ctx.config, "batch_size"),
            "trainingFixedShapeHP.batch_size does not match config.batch_size");
    require(fixed_shape.max_seq_len == GRIM::HyperParameters::snapshotTrainingConfigField<int>(ctx.config, "max_seq_len"),
            "trainingFixedShapeHP.max_seq_len does not match config.max_seq_len");

    require(ctx.data.vocab_size >= static_cast<std::uint32_t>(GRIM::Tokenizer::UNIGRAM_VOCAB_OFFSET),
            "SequenceData.vocab_size does not include special+byte+atom token ranges");
    require(static_cast<int>(ctx.data.vocab_size) == GRIM::HyperParameters::snapshotTrainingConfigField<int>(ctx.config, "vocab_size"),
            "SequenceData.vocab_size does not match config.vocab_size");

    require(ctx.model != nullptr, "model is null");
    require(ctx.model_allocation.model_max_tokens_per_batch == fixed_shape.max_tokens_per_batch,
            "model allocation token mirror does not match trainingFixedShapeHP");

    require(ctx.telemetry.lattice != nullptr, "telemetry lattice is null");
    require(ctx.telemetry.csv_logger != nullptr, "telemetry CSV logger is null");
    require(ctx.telemetry.control_config.reference_tokens == fixed_shape.max_tokens_per_batch,
            "telemetry reference token budget does not match trainingFixedShapeHP");
    require(ctx.telemetry.control_config.reference_seq_len == fixed_shape.max_seq_len,
            "telemetry reference seq len does not match trainingFixedShapeHP");

    require(!ctx.train_payloads.empty(), "train_payloads is empty");
    require(ctx.fixed_train_schedule.batches.size() == ctx.train_payloads.size(),
            "fixed_train_schedule batch count does not match train_payloads");
    if (schedule_hp.single_batch_overfit_enabled) {
        require(ctx.epoch_plan.total_batches == schedule_hp.single_batch_overfit_max_steps,
                "EpochPlan total_batches does not match single_batch_overfit_max_steps");
    } else {
        require(ctx.epoch_plan.total_batches == static_cast<int>(ctx.train_payloads.size()),
                "EpochPlan total_batches does not match authored train payload count");
    }
    require(ctx.epoch_plan.estimated_total_steps == ctx.estimated_total_steps,
            "EpochPlan estimated_total_steps does not match TrainingContext");
    require(ctx.epoch_plan.estimated_total_steps > 0, "estimated_total_steps <= 0");
    require(ctx.lr_schedule.has_value(), "lr_schedule is not initialized");

    require(static_cast<int>(ctx.epoch_batch_order.size()) == schedule_hp.epochs,
            "epoch_batch_order size does not match epochs");
    for (std::size_t epoch = 0; epoch < ctx.epoch_batch_order.size(); ++epoch) {
        const auto& order = ctx.epoch_batch_order[epoch];
        require(static_cast<int>(order.size()) == ctx.epoch_plan.total_batches,
                "epoch_batch_order entry size does not match EpochPlan total_batches");
        for (int batch_index : order) {
            require(batch_index >= 0 &&
                    batch_index < static_cast<int>(ctx.train_payloads.size()),
                    "epoch_batch_order contains out-of-range train payload index");
        }
    }
}

void StartupValidated(TrainingContext& ctx) {
    validateStartupOrThrow(StartupValidationInputs{ctx});
}

} // namespace GRIMText::Training

