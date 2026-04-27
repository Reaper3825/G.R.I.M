#pragma once
//======================================================//
//  TelemetryUpdate.hpp
//  Centralized per-batch telemetry observation update
//======================================================//
//
//  Encapsulates ALL telemetry metric computation that was
//  previously inline in Phase2_TrainingLoop.cu. Call once
//  per batch via updateTelemetryObservations(), and once
//  per epoch via logTelemetrySummary().
//
//======================================================//

#include "TelemetryState_GPU.hpp"
#include "TelemetryLattice_GPU.hpp"
#include "TelemetryCsvLogger.hpp"
#include "../Batching/BatchPayload.hpp"
#include "../GradNorm/GradNormGPU.hpp"

#include <string>
#include <cstdint>

#include "../../GRIM/grim_language_model_cuda.hpp"

namespace GRIMText::Training {
struct BatchResult;
struct TrainingContext;
struct TrainingLoopState;
}

namespace GRIM::Telemetry {

//======================================================//
//  Batch-level telemetry input (all data needed to
//  populate last_obs[0..30] and run lattice->update())
//======================================================//

struct TelemetryBatchInput {
    // Core metrics (streams 0-4)
    float loss                  = 0.0f;
    float preclip_grad_rms      = 0.0f;
    float learning_rate         = 0.0f;
    int   total_tokens          = 0;

    // Gradient component for EB ratio (stream 15)
    float enc_rms_pre           = 0.0f;

    // Optimizer state (streams 9-13)
    int   optimizer_step        = 0;
    bool  should_step           = false;

    // Loss breakdown (stream 25)
    float total_loss_value      = 0.0f;
    float aux_loss              = 0.0f;

    // Batch geometry (stream 30)
    int   max_seq_len           = 0;

    // Identifiers for error messages
    int   batch_idx             = 0;
    int   global_step           = 0;

    // Config
    int   actual_vocab_size     = 0;
    int   d_model               = 0;
};

//======================================================//
//  Single-call batch telemetry update
//======================================================//

/// Populates ctx.telemetry.last_obs[0..30], calls lattice->update(),
/// exports CSV, and validates NaN/Inf. Throws on any anomaly (Rule 20).
///
/// @param ctx      Training context (owns telemetry state, logger, model)
/// @param input    Pre-computed metrics from processBatch
/// @param gm       Gradient metrics (for EB grad norm)
/// @param payload  Batch payload (for execution_active flags); may be nullptr
void updateTelemetryObservations(
    GRIMText::Training::TrainingContext& ctx,
    const TelemetryBatchInput& input,
    const GRIM::GradNorm::GradMetrics& gm,
    const GRIM::Batching::BatchPayload* payload);

/// Emits log-interval telemetry/monitoring derived from the latest batch:
/// step loss/lr, MTP per-head telemetry, and GuessCache telemetry.
void logIntervalTelemetry(
    GRIMText::Training::TrainingContext& ctx,
    GRIMText::Training::TrainingLoopState& state,
    const GRIMText::Training::BatchResult& batch_result);

//======================================================//
//  Epoch-level telemetry summary log
//======================================================//

/// Reads TelemetryVectors at L0 and L2, logs formatted summary.
/// Call once at end of each epoch.
void logTelemetrySummary(GRIMText::Training::TrainingContext& ctx);

} // namespace GRIM::Telemetry
