//======================================================//
//  EpochBatching.hpp
//  Per-epoch batching policy: builds the BatchSchedule for
//  one epoch by configuring PackerPolicy and calling
//  GRIM::Batching::buildBatches(...). Logs the resulting
//  schedule via a caller-supplied log callback.
//
//  This lives in Shared/Batching (next to Batching_GPU and
//  BatchPayload) rather than in the training-loop
//  orchestration file so the per-epoch packing policy is
//  co-located with the rest of the batch construction
//  logic. To avoid pulling training-layer types
//  (TrainingContext / TrainingLogger) into Shared, the API
//  accepts shared curriculum metadata and explicit row provenance.
//======================================================//

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "CurriculumOrdering.hpp"
#include "Batching_GPU.hpp"   // BatchSchedule

namespace GRIM { namespace Batching {

using EpochBatchingLogFn = std::function<void(const std::string&)>;

//======================================================//
// Build the BatchSchedule for one epoch.
//
//  sequence_lengths       — sequence lengths; index is the seq_id.
//  fixed_sequence_cap    — configured max_seq_len / run sequence cap.
//  fixed_batch_size      — configured batch_size.
//  global_step  — current optimizer step; accepted to keep the call boundary explicit.
//  epoch        — 0-based epoch index used for deterministic per-epoch RNG.
//  data_seed    — base data RNG seed; per-epoch seed = data_seed + epoch + 1.
//  curriculum   — flags and ordered course/block membership from the registry.
//  concept_block_ids — source block identity for each sequence_lengths row.
//  ordering     — CURRICULUM uses flags; PRESERVE/RANDOM override both flags.
//  log_fn       — optional callback for the [Batching] summary lines.
//                 Pass {} to suppress logging.
//======================================================//
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
    const EpochBatchingLogFn& log_fn);

//======================================================//
// Emit the standard [Batching] summary block for an
// already-built BatchSchedule. Called internally by
// buildEpochBatches when log_fn is non-null; exposed so
// callers can re-log a schedule (e.g. validation epochs).
//======================================================//
void logBatchSchedule(
    const BatchSchedule& schedule,
    uint32_t fixed_sequence_cap,
    const EpochBatchingLogFn& log_fn);

} } // namespace GRIM::Batching
