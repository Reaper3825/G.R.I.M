//======================================================//
//  LossBaselineDiagnostic.hpp
//  Adaptive loss baseline tracking + invalid-token
//  validation. Lifted from Phase2_TrainingLoop.cu
//  (processBatch, post-loss section).
//======================================================//

#pragma once

namespace GRIM::Batching { struct BatchPayload; }
namespace GRIMText { namespace Training {
    struct TrainingContext;
    struct TrainingLoopState;
} }

namespace GRIM::Diagnostics {

// On the very first batch, captures `state.initial_loss` and logs the
// baseline relative to ln(vocab_size). On subsequent batches:
//   - updates `state.min_observed_loss` and `state.warmup_batches`
//   - scans payload.input_ids and payload.target_ids for IDs outside
//     the vocab range; THROWS std::runtime_error on corruption (Rule 20)
//
// High-loss detection is owned exclusively by runLossSpikeDiagnostic
// (Diagnostics/LossSpikeDiagnostic.{hpp,cu}); this function does NOT
// emit a separate HIGH_LOSS log line.
//
// Mutates: state.initial_loss, state.min_observed_loss, state.warmup_batches.
void runLossBaselineAndTokenValidation(
    GRIMText::Training::TrainingContext& ctx,
    GRIMText::Training::TrainingLoopState& state,
    const GRIM::Batching::BatchPayload& payload,
    float loss,
    int batch_idx);

} // namespace GRIM::Diagnostics
