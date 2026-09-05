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
    uint32_t fixed_sequence_cap,
    const EpochBatchingLogFn& log_fn)
{
    if (!log_fn) return;

    log_fn("Created " + std::to_string(schedule.batches.size()) + " fixed batches");
    log_fn("[Batching] Sequence cap: " + std::to_string(fixed_sequence_cap));
    log_fn("[Batching] Batch size: " + std::to_string(schedule.batch_size));
    log_fn("[Batching] Projected fixed token rectangle: " +
           std::to_string(static_cast<uint64_t>(fixed_sequence_cap) * schedule.batch_size));
    log_fn("[Batching] Projected packing efficiency: " +
           std::to_string(static_cast<int>(schedule.projected_packing_efficiency * 100)) + "%");
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
    const CurriculumMetadata& curriculum,
    const std::vector<std::string>& concept_block_ids,
    CurriculumOrdering ordering,
    const EpochBatchingLogFn& log_fn)
{
    (void)global_step;

    PackerPolicy policy;

    policy.sequence_order = orderedCourseSequences(
        sequence_lengths, concept_block_ids, curriculum, ordering,
        data_seed + static_cast<uint64_t>(epoch) + 1ULL);
    // The explicit block plan is authoritative; no later row/batch shuffle.
    policy.batch_ordering = BatchOrdering::PRESERVE;

    auto schedule = buildBatches(sequence_lengths, fixed_sequence_cap, fixed_batch_size, policy);

    logBatchSchedule(schedule, fixed_sequence_cap, log_fn);

    return schedule;
}

} } // namespace GRIM::Batching
