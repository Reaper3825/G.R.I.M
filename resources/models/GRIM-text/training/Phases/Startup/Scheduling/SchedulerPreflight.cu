#include "SchedulerPreflight.hpp"

#include <stdexcept>
#include <string>

namespace GRIMText::Training {

SchedulerPreflightState runSchedulerPreflightOrThrow(
    const SchedulerInputs& inputs,
    const ::GRIM::Batching::EpochBatchingLogFn& log_fn)
{
    if (inputs.train_catalog == nullptr) {
        throw std::runtime_error("FATAL: SchedulerPreflightReady received null train_catalog");
    }
    if (inputs.capacity.max_tokens_per_batch == 0 || inputs.capacity.batch_rows == 0) {
        throw std::runtime_error("FATAL: SchedulerPreflightReady received invalid RunCapacity (tokens=" +
                                 std::to_string(inputs.capacity.max_tokens_per_batch) +
                                 " rows=" + std::to_string(inputs.capacity.batch_rows) + ")");
    }

    const auto schedule = ::GRIM::Batching::buildEpochBatches(
        *inputs.train_catalog,
        inputs.capacity.max_tokens_per_batch,
        inputs.capacity.batch_rows,
        inputs.global_step,
        inputs.epoch,
        inputs.data_seed,
        log_fn);

    SchedulerPreflightState state;
    state.total_batches = static_cast<int>(schedule.batches.size());
    state.total_tokens = schedule.total_tokens;
    state.actual_tokens = schedule.actual_tokens;
    state.padding_tokens = schedule.padding_tokens;
    state.overflow_batches = schedule.overflow_batches;
    state.max_batch_size_observed = schedule.max_batch_size_observed;
    state.max_seq_len_observed = schedule.max_seq_len_observed;

    if (state.total_batches <= 0) {
        throw std::runtime_error("FATAL: scheduler produced 0 batches during startup scheduler preflight");
    }
    if (state.max_seq_len_observed > inputs.capacity.seq_cap) {
        throw std::runtime_error("FATAL: scheduler observed sequence length above RunCapacity (observed=" +
                                 std::to_string(state.max_seq_len_observed) +
                                 " cap=" + std::to_string(inputs.capacity.seq_cap) + ")");
    }

    return state;
}

} // namespace GRIMText::Training

