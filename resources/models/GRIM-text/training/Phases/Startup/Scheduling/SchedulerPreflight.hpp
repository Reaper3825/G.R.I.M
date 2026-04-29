#pragma once

#include "../Capacity/RunCapacity.hpp"
#include "../../../../Shared/Batching/EpochBatching.hpp"

#include <cstdint>
#include <vector>

namespace GRIMText::Training {

struct TrainingContext;

struct SchedulerInputs {
    const std::vector<std::uint32_t>* train_seq_lengths = nullptr;
    RunCapacity capacity;
    int global_step = 0;
    int epoch = 0;
    std::uint64_t data_seed = 0;
};

struct SchedulerPreflightState {
    int total_batches = 0;
    std::uint64_t total_tokens = 0;
    std::uint64_t actual_tokens = 0;
    std::uint64_t padding_tokens = 0;
    std::uint32_t overflow_batches = 0;
    std::uint32_t max_batch_size_observed = 0;
    std::uint32_t max_seq_len_observed = 0;
};

SchedulerPreflightState runSchedulerPreflightOrThrow(
    const SchedulerInputs& inputs,
    const ::GRIM::Batching::EpochBatchingLogFn& log_fn);

void SchedulerPreflightReady(TrainingContext& ctx);

} // namespace GRIMText::Training

