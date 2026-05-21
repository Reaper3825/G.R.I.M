//======================================================//
//  EpochBatching.cu
//  Implementation of per-epoch BatchSchedule construction.
//======================================================//

#include "EpochBatching.hpp"
#include "PackerPolicy.hpp"

#include <algorithm>
#include <string>

namespace GRIM { namespace Batching {

void logBatchSchedule(
    const BatchSchedule& schedule,
    const EpochBatchingLogFn& log_fn)
{
    if (!log_fn) return;

    log_fn("Created " + std::to_string(schedule.batches.size()) + " fixed batches");
    log_fn("[Batching] Sequence cap: " + std::to_string(schedule.sequence_cap));
    log_fn("[Batching] Batch size: " + std::to_string(schedule.batch_size));
    log_fn("[Batching] Fixed token rectangle: " +
            std::to_string(static_cast<uint64_t>(schedule.sequence_cap) * schedule.batch_size));
    log_fn("[Batching] Packing efficiency: " +
            std::to_string(static_cast<int>(schedule.packing_efficiency * 100)) + "%");
        if (schedule.discarded_tail_sequences > 0) {
        log_fn("[Batching] Discarded " +
               std::to_string(schedule.discarded_tail_sequences) +
               " trailing sequence(s) that did not fill a full fixed batch");
        }
}

BatchSchedule buildEpochBatches(
    const std::vector<uint32_t>& sequence_lengths,
    uint32_t fixed_sequence_cap,
    uint32_t fixed_batch_size,
    int global_step,
    int epoch,
    uint64_t data_seed,
    const EpochBatchingLogFn& log_fn)
{
    (void)global_step;

    PackerPolicy policy;

    // Per-epoch deterministic shuffle.
    policy.rng_seed = data_seed + static_cast<uint64_t>(epoch) + 1ULL;

    // RANDOM ordering avoids loss spikes at epoch end. Length curricula are
    // forbidden here: rows are fixed-window padded, so length-sorted exposure
    // reintroduces boundary bias.
    policy.batch_ordering      = BatchOrdering::RANDOM;

    auto schedule = buildBatches(sequence_lengths, fixed_sequence_cap, fixed_batch_size, policy);

    logBatchSchedule(schedule, log_fn);

    return schedule;
}

} } // namespace GRIM::Batching
