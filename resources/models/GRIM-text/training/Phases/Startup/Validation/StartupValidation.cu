#include "StartupValidation.hpp"

#include "../../Phase1_Startup.hpp"

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

    require(ctx.logging.logger != nullptr, "logging logger is null");
    require(ctx.logging.status_writer != nullptr, "status writer is null");
    require(ctx.memory_snapshot.device >= 0, "memory snapshot not captured");
    require(ctx.memory_snapshot.total_bytes > 0, "memory snapshot total_bytes is zero");

    require(ctx.config.hyperparameters.batch_size > 0, "hyperparameters.batch_size <= 0");
    require(ctx.config.max_seq_len > 0, "config.max_seq_len <= 0");
    require(ctx.run_capacity.batch_rows == static_cast<std::uint32_t>(ctx.config.hyperparameters.batch_size),
            "RunCapacity.batch_rows does not match post-policy hyperparameters.batch_size");
    require(ctx.run_capacity.seq_cap == static_cast<std::uint32_t>(ctx.config.max_seq_len),
            "RunCapacity.seq_cap does not match config.max_seq_len");

    require(ctx.data_info.actual_vocab_size == ctx.config.actual_vocab_size,
            "DataInfo.actual_vocab_size does not match StartupConfig.actual_vocab_size");
    require(ctx.data_info.train_sequence_count == ctx.data.train_seqs.size(),
            "DataInfo train sequence count does not match SequenceData");
    require(ctx.data_info.val_sequence_count == ctx.data.val_seqs.size(),
            "DataInfo val sequence count does not match SequenceData");

    require(ctx.model != nullptr, "model is null");
    require(ctx.model_allocation.model_max_cached_batch == static_cast<int>(ctx.run_capacity.batch_rows),
            "model allocation batch mirror does not match RunCapacity");
    require(ctx.model_allocation.model_max_cached_seq_len == ctx.run_capacity.seq_cap,
            "model allocation seq mirror does not match RunCapacity");
    require(ctx.model_allocation.model_max_tokens_per_batch == static_cast<int>(ctx.run_capacity.max_tokens_per_batch),
            "model allocation token mirror does not match RunCapacity");

    require(ctx.telemetry.lattice != nullptr, "telemetry lattice is null");
    require(ctx.telemetry.csv_logger != nullptr, "telemetry CSV logger is null");
    require(static_cast<std::uint32_t>(ctx.telemetry.control_config.reference_tokens) == ctx.run_capacity.max_tokens_per_batch,
            "telemetry reference token budget does not match RunCapacity");
    require(static_cast<std::uint32_t>(ctx.telemetry.control_config.reference_seq_len) == ctx.run_capacity.seq_cap,
            "telemetry reference seq len does not match RunCapacity");

    require(ctx.scheduler_preflight.total_batches > 0, "scheduler preflight total_batches <= 0");
    require(ctx.epoch_plan.total_batches == ctx.scheduler_preflight.total_batches,
            "EpochPlan total_batches does not match SchedulerPreflightState");
    require(ctx.epoch_plan.estimated_total_steps == ctx.estimated_total_steps,
            "EpochPlan estimated_total_steps does not match TrainingContext");
    require(ctx.epoch_plan.estimated_total_steps > 0, "estimated_total_steps <= 0");
    require(ctx.lr_schedule.has_value(), "lr_schedule is not initialized");
}

} // namespace GRIMText::Training

