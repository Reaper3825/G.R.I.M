//======================================================//
//  EpochBatching.cu
//  Implementation of per-epoch BatchSchedule construction.
//======================================================//

#include "EpochBatching.hpp"

#include <algorithm>
#include <string>

namespace GRIM { namespace Batching {

void logBatchSchedule(
    const BatchSchedule& schedule,
    uint32_t max_tokens_per_batch,
    const PackerPolicy& policy,
    const EpochBatchingLogFn& log_fn)
{
    if (!log_fn) return;

    log_fn("Created " + std::to_string(schedule.batches.size()) + " dynamic batches");
    log_fn("[Batching] Token budget: " + std::to_string(max_tokens_per_batch));
    log_fn("[Batching] Strategy: " + std::to_string(static_cast<int>(policy.strategy)));
    log_fn("[Batching] Batch size range: " +
           std::to_string(schedule.min_batch_size_observed) + "-" +
           std::to_string(schedule.max_batch_size_observed));
    log_fn("[Batching] Packing efficiency: " +
           std::to_string(static_cast<int>(schedule.avg_packing_efficiency * 100)) + "%");
}

BatchSchedule buildEpochBatches(
    const std::vector<uint32_t>& sequence_lengths,
    uint32_t max_tokens_per_batch,
    uint32_t max_batch_size,
    int global_step,
    int epoch,
    uint64_t data_seed,
    const EpochBatchingLogFn& log_fn)
{
    // Gate flags — kept for parity with the previous orchestration-site
    // implementation. warmup_phase is unused below today but the gate is
    // preserved so future per-epoch warmup behaviour has a single home.
    (void)global_step;
    const bool curriculum_active = (epoch < kCurriculumEpochs);

    PackerPolicy policy;

    // Issue #90: GREEDY forced — SIMILARITY_GROUPED promotes mode collapse
    // (many small batches of similar sequences → unstable loss spikes).
    policy.strategy = PackingStrategy::GREEDY;

    policy.similarity_threshold = 0.30f;
    policy.prefer_short_first   = curriculum_active;
    policy.curriculum_progress  = curriculum_active
        ? static_cast<float>(epoch + 1) / kCurriculumEpochs
        : 1.0f;

    // Per-epoch deterministic shuffle.
    policy.rng_seed = data_seed + static_cast<uint64_t>(epoch);

    // RANDOM ordering avoids loss spikes at epoch end. LENGTH_ASCENDING
    // produces an easy→hard progression that plateaus then explodes.
    policy.batch_ordering      = BatchOrdering::RANDOM;
    policy.interleave_overflow = true;

    auto schedule = buildBatches(sequence_lengths, max_tokens_per_batch, max_batch_size, policy);

    logBatchSchedule(schedule, max_tokens_per_batch, policy, log_fn);

    return schedule;
}

} } // namespace GRIM::Batching
