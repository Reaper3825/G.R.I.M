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
//  accepts only the primitives it actually needs.
//======================================================//

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "Batching_GPU.hpp"   // BatchSchedule
#include "PackerPolicy.hpp"

namespace GRIM { namespace Batching {

using EpochBatchingLogFn = std::function<void(const std::string&)>;

//======================================================//
// Per-epoch batching constants. Mirror what was previously
// declared inline in Phase2_TrainingLoop.hpp; kept here so
// the policy lives with the batching code that consumes
// them.
//======================================================//
inline constexpr int kWarmupTokenSteps = 2048;
inline constexpr int kCurriculumEpochs = 1;

//======================================================//
// Build the BatchSchedule for one epoch.
//
//  sequence_lengths       — sequence lengths; index is the seq_id.
//  max_tokens_per_batch — run capacity token rectangle (batch_rows * seq_cap).
//  max_batch_size       — run capacity batch rows.
//  global_step  — current optimizer step (used for warmup gating).
//  epoch        — 0-based epoch index (used for curriculum gating + RNG).
//  data_seed    — base data RNG seed; per-epoch seed = data_seed + epoch.
//  log_fn       — optional callback for the [Batching] summary lines.
//                 Pass {} to suppress logging.
//======================================================//
BatchSchedule buildEpochBatches(
    const std::vector<uint32_t>& sequence_lengths,
    uint32_t max_tokens_per_batch,
    uint32_t max_batch_size,
    int global_step,
    int epoch,
    uint64_t data_seed,
    const EpochBatchingLogFn& log_fn);

//======================================================//
// Emit the standard [Batching] summary block for an
// already-built BatchSchedule. Called internally by
// buildEpochBatches when log_fn is non-null; exposed so
// callers can re-log a schedule (e.g. validation epochs).
//======================================================//
void logBatchSchedule(
    const BatchSchedule& schedule,
    uint32_t max_tokens_per_batch,
    const PackerPolicy& policy,
    const EpochBatchingLogFn& log_fn);

} } // namespace GRIM::Batching
