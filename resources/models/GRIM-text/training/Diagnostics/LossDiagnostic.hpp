//======================================================//
//  LossDiagnostic.hpp
//  Loss equation/statistics, observed baseline tracking, and token validation.
//======================================================//

#pragma once

namespace GRIMText::Training {
    struct TrainingContext;
    struct TrainingLoopState;
    struct BatchResult;
}
namespace GRIM::Batching { struct BatchPayload; }

namespace GRIM::Diagnostics {

// Logs BATCH_LOSS and LossStats, including the observed baseline/minimum and
// the Phase1-selected model source when initializing the baseline.
// Mutates state.initial_loss, state.min_observed_loss, state.warmup_batches.
// Subsequent batches validate token IDs and throw on data corruption.
// Loss spike detection remains owned by TelemetryLattice.
void runLossDiagnostic(
    GRIMText::Training::TrainingContext& ctx,
    GRIMText::Training::TrainingLoopState& state,
    const GRIM::Batching::BatchPayload& payload,
    const GRIMText::Training::BatchResult& result,
    int batch_idx);

} // namespace GRIM::Diagnostics
