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
    require(ctx.model_allocation.model_max_tokens_per_batch == static_cast<int>(ctx.run_capacity.max_tokens_per_batch),
            "model allocation token mirror does not match RunCapacity");

    if (ctx.config.hyperparameters.guess_aux_enabled) {
        require(ctx.guess_cache_scope != nullptr,
                "guess cache scope is null while guess_aux_enabled=true");
        require(ctx.guess_cache_state.guess_cache_ready,
                "guess cache is not ready while guess_aux_enabled=true");
        require(ctx.guess_cache_state.batch_buffers != nullptr,
                "guess cache batch buffers are null while guess_aux_enabled=true");
    } else {
        require(!ctx.guess_cache_state.guess_cache_ready,
                "guess cache ready while guess_aux_enabled=false");
    }

    require(ctx.telemetry.lattice != nullptr, "telemetry lattice is null");
    require(ctx.telemetry.csv_logger != nullptr, "telemetry CSV logger is null");
    require(static_cast<std::uint32_t>(ctx.telemetry.control_config.reference_tokens) == ctx.run_capacity.max_tokens_per_batch,
            "telemetry reference token budget does not match RunCapacity");
    require(static_cast<std::uint32_t>(ctx.telemetry.control_config.reference_seq_len) == ctx.run_capacity.seq_cap,
            "telemetry reference seq len does not match RunCapacity");

    require(!ctx.train_payloads.empty(), "train_payloads is empty");
    require(ctx.fixed_train_schedule.batches.size() == ctx.train_payloads.size(),
            "fixed_train_schedule batch count does not match train_payloads");
    if (ctx.config.hyperparameters.single_batch_overfit_enabled) {
        require(ctx.epoch_plan.total_batches == ctx.config.hyperparameters.single_batch_overfit_max_steps,
                "EpochPlan total_batches does not match single_batch_overfit_max_steps");
    } else {
        require(ctx.epoch_plan.total_batches == static_cast<int>(ctx.train_payloads.size()),
                "EpochPlan total_batches does not match authored train payload count");
    }
    require(ctx.epoch_plan.estimated_total_steps == ctx.estimated_total_steps,
            "EpochPlan estimated_total_steps does not match TrainingContext");
    require(ctx.epoch_plan.estimated_total_steps > 0, "estimated_total_steps <= 0");
    require(ctx.lr_schedule.has_value(), "lr_schedule is not initialized");

    require(static_cast<int>(ctx.epoch_batch_order.size()) == ctx.config.hyperparameters.epochs,
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

