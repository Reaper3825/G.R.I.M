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

#include "../../training/Phases/Startup/Model/ModelGpuState.hpp"
#include "../../training/Phases/Startup/Model/ParameterRegistry.hpp"

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
//  populate telemetry raw-observation streams and run lattice->update())
//======================================================//

struct TelemetryBatchInput {
    // Core metrics (streams 0-4)
    float loss                  = 0.0f;
    float preclip_grad_rms      = 0.0f;
    float learning_rate         = 0.0f;
    int   total_tokens          = 0;

    // Optimizer state (streams 9-13 and optimizer_iteration stream 60)
    int   optimizer_step        = 0;
    bool  should_step           = false;

    // Explicit loss breakdown
    float text_loss             = 0.0f;
    float local_atom_retrieval_loss = 0.0f; // Weighted contribution to total loss

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

/// Populates ctx.telemetry.last_obs[], calls lattice->update(),
/// exports CSV, and validates NaN/Inf. Throws on any anomaly (Rule 20).
///
/// @param ctx      Training context (owns telemetry state, logger, model)
/// @param parameter_registry Explicit durable startup parameter owner
/// @param input    Pre-computed metrics from processBatch
void updateTelemetryObservations(
    GRIMText::Training::TrainingContext& ctx,
    const ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const TelemetryBatchInput& input);

/// Emits log-interval telemetry/monitoring derived from the latest batch:
/// step loss/lr and model-health telemetry.
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
