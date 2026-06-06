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
#include "../GradNorm/GradNormGPU.hpp"

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

    // Explicit loss breakdown (streams 25-26 currently consume execution/mtp loss fractions)
    float text_loss             = 0.0f;
    float mtp_loss              = 0.0f;
    float execution_loss        = 0.0f;
    float selector_loss         = 0.0f;

    // Batch geometry (stream 30)
    int   max_seq_len           = 0;

    // Reduced execution-block forward snapshots authored inside processBatch
    float exec_selection_entropy = 0.0f;
    float exec_op_entropy        = 0.0f;
    float exec_div_clamp_rate    = 0.0f;
    float exec_max_p_write       = 0.0f;
    float exec_active_ratio      = 0.0f;
    float inject_gate_mean       = 0.0f;

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
/// @param training_state Explicit durable training-state owner
/// @param gpu_model Explicit durable GPU topology owner
/// @param parameter_registry Explicit durable startup parameter owner
/// @param input    Pre-computed metrics from processBatch
/// @param gm       Gradient metrics (for EB grad norm)
void updateTelemetryObservations(
    GRIMText::Training::TrainingContext& ctx,
    const GRIM::TrainingState& training_state,
    const GRIMText::Training::Startup::GpuModelState& gpu_model,
    const ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const TelemetryBatchInput& input,
    const GRIM::GradNorm::GradMetrics& gm);

/// Emits log-interval telemetry/monitoring derived from the latest batch:
/// step loss/lr and MTP per-head telemetry.
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
