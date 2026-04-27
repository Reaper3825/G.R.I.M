#pragma once
//======================================================//
//  Startup/Batching/PlannedBatches.hpp
//
//  Phase1-owned construction of the train/val BatchSchedules and the
//  full BatchPayload vectors that Phase2 indexes into.
//
//  CONTRACT
//  ========
//  Phase1 authors:
//    - ctx.fixed_train_schedule  : single training BatchSchedule
//    - ctx.train_payloads        : prebuilt host BatchPayloads (one per batch)
//    - ctx.fixed_val_schedule    : single validation BatchSchedule
//    - ctx.val_payloads          : prebuilt host BatchPayloads (one per val batch)
//    - ctx.epoch_batch_order     : per-epoch permutation of train batch indices
//                                  (deterministic from rng.data_seed)
//
//  Phase2 NEVER calls buildBatches / buildEpochBatches / buildBatchPayload.
//  It selects the active batch via:
//      ctx.train_payloads[ctx.epoch_batch_order[epoch][batch_i]]
//  and iterates ctx.val_payloads in order during validation.
//
//  PRECONDITIONS
//  =============
//  - DataInfoReady (train_catalog / val_catalog populated)
//  - ModelAllocated (ctx.model is non-null)
//  - CapacityStemReady (ctx.run_capacity has token rectangle + batch rows)
//  - PayloadBuildInputsReady (ctx.payload_build_inputs is authored)
//  - RngReady (ctx.rng.data_seed is set so epoch permutations are deterministic)
//
//  Failures are loud (Rule 20): empty schedules, capacity mismatches, and
//  individual BatchPayload validation errors all throw.
//======================================================//

#include "../../../../Shared/Batching/BatchPayload.hpp"
#include "../../../../Shared/Batching/Batching_GPU.hpp"

namespace GRIMText::Training {

struct TrainingContext;

//======================================================//
// Build a single BatchPayload from a BatchAssignment using the Phase1-authored
// payload_build_inputs. Routes through GRIM::Batching::buildBatchPayload, the
// SAME builder that ran in Phase2 before this refactor — just driven from
// startup so the result is immutable for the run.
//======================================================//
GRIM::Batching::BatchPayload buildTrainPayload(
    const TrainingContext& ctx,
    const GRIM::Batching::BatchAssignment& assignment);

GRIM::Batching::BatchPayload buildValPayload(
    const TrainingContext& ctx,
    const GRIM::Batching::BatchAssignment& assignment);

//======================================================//
// PlannedBatchesReady
//
// Authors ctx.fixed_train_schedule + ctx.train_payloads, ctx.fixed_val_schedule
// + ctx.val_payloads, and ctx.epoch_batch_order. Idempotent within a run; must
// run exactly once during Phase1.
//======================================================//
void PlannedBatchesReady(TrainingContext& ctx);

} // namespace GRIMText::Training
