//======================================================//
//  Phase2_TrainingLoop.cu
//  Core training computation: epoch iteration, batch
//  processing, forward/backward, optimizer steps,
//  validation, checkpointing, auto-stop.
//======================================================//
#include "Phase2_TrainingLoop.hpp"
#include "../OptimizerCheckpoint.hpp"
#include "../Diagnostics/DiagnosticInference.hpp"
#include "../Diagnostics/RhoDiagnostic.hpp"
#include "../Diagnostics/LMHeadWeightStats.hpp"
#include "../Diagnostics/LogitScaleDiagnostic.hpp"
#include "../Diagnostics/BoundaryDiagnostic.hpp"
#include "../Diagnostics/SpecialTokenDiagnostic.hpp"
#include "../Diagnostics/AtomStatsDiagnostic.hpp"
#include "../Diagnostics/LossSpikeDiagnostic.hpp"
#include "../Diagnostics/LossBaselineDiagnostic.hpp"
#include "../Diagnostics/LossStatsDiagnostic.hpp"
#include "../Diagnostics/GradientNormDiagnostic.hpp"
#include "../Diagnostics/OptimizerStepGuards.hpp"
#include "../Diagnostics/TieVerifyDiagnostic.hpp"
#include "../Diagnostics/MtpDiagnostic.hpp"
#include "../Diagnostics/OptimizerMomentDiagnostic.hpp"
#include "../Diagnostics/PostOptimizerWeightTrace.hpp"
#include "../Diagnostics/PredictionDistributionDiagnostic.hpp"
#include "../Diagnostics/DiagnosticGates.hpp"
#include "../../Shared/LogRecorder/LogRecorder.hpp"
#include "../../Shared/CudaAllocUtils.hpp"
#include "../../Shared/Gradients/GradStatsCollector.hpp"

using GRIM::CudaAlloc::cudaMallocOrThrow;
#include "../../Shared/Gradients/GradientCC_GPU.hpp"       // GradClip::clipGradientNorms (registry-level clipping)
#include "../../Shared/Dynamic_LR/LRSchedule.hpp"          // GRIM::LR::LRSchedule (exposed LR curve)
#include "../../Shared/GradNorm/GradNormGPU.hpp"           // GradNorm::measureGradientNorms, GradMetrics
#include "../../Shared/TrainingState/TrainingState_GPU.hpp"
#include "../../Shared/LogRecorder/BatchLogTape.hpp"
#include "../../Shared/Telemetry/TelemetryUpdate.hpp"
#include "../Diagnostics/TrainingDiagnostics.hpp"
#include "../../Shared/UnigramByte/Unigram.hpp"
#include "../../Shared/UnigramByte/AtomTable.hpp"
#include "../../Shared/Batching/BatchPayload.hpp"
#include "../Autograd/AutogradTraining.hpp"  // autogradTrainingStep: unified forward+loss+backward
#include "../../Shared/Optimizers/AdamW/AdamW_Kernal_GPU.hpp"  // launchAdamWStep, resetAdamWMoments, scaleAdamWMoments
#include "../../Shared/Optimizers/RAdamW/RAdamW_Kernal_GPU.hpp"  // launchRAdamWStep — selectable via training.config.optimizer.kind
#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"  // single entry point; transitively pulls in control/ai_config_paths.hpp (resolveGrimRoot, etc.)
#include "../../Shared/HyperParameters/HyperparameterGroupings.hpp"  // CoreRunHP / LearningRateScheduleInputs — single source of truth for grouped HP reads
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <filesystem>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

// Module logging aliases
using GRIM::Logging::ModuleId;
using GRIM::Logging::EmitModuleInfo;
using GRIM::Logging::EmitModuleWarning;
using GRIM::Logging::EmitModuleError;

namespace GRIMText::Training {
namespace fs = std::filesystem;

TrainingLoopState::~TrainingLoopState() = default;

//======================================================//
//  Internal Helpers
//======================================================//

namespace Internal {

std::string formatScalar(float value, int precision) {
    std::ostringstream oss;
    if (std::isfinite(value)) {
         if (value == 0.0f) value = 0.0f; // kill -0.0
        oss << std::fixed << std::setprecision(precision) << value;
    } else if (std::isnan(value)) {
        oss << "nan";
    } else {
        oss << (value > 0 ? "inf" : "-inf");
    }
    return oss.str();
}

std::string formatMetric(std::string_view name, float value, int precision) {
    return std::string(name) + "=" + formatScalar(value, precision);
}

// Query LR schedule at a given step, honoring stability overrides.
float getScheduledLearningRate(
    const GRIM::LR::LRSchedule& schedule,
    int step,
    float base_lr,
    bool stability_overrides_enabled) {
    if (stability_overrides_enabled) {
        return base_lr;
    }
    return schedule.lr(step);
}

void evaluateAutoStop(
    TrainingContext& ctx,
    TrainingLoopState& state,
    EpochResult& result,
    int epoch_idx) {
    const auto& hp = ctx.config.hyperparameters;
    if (!hp.auto_stop_enabled || ctx.auto_stop_triggered) {
        return;
    }

    const float prev_best = ctx.best_val_loss;
    bool significant_improvement = true;
    if (std::isfinite(prev_best)) {
        significant_improvement = (prev_best - result.validation.loss) > hp.auto_stop_plateau_min_delta;
    }

    auto trip = [&](const char* reason) {
        ctx.auto_stop_triggered = true;
        ctx.auto_stop_reason = reason;
        ctx.auto_stop_epoch = epoch_idx + 1;
        ctx.auto_stop_metric = result.validation.loss;
        result.auto_stop_triggered = true;
        result.auto_stop_reason = reason;
    };

    // Plateau detection
    if (hp.auto_stop_plateau_patience > 0) {
        if (significant_improvement) {
            state.plateau_epochs_without_improvement = 0;
        } else {
            state.plateau_epochs_without_improvement++;
            if (state.plateau_epochs_without_improvement >= hp.auto_stop_plateau_patience) {
                trip("plateau");
            }
        }
    }

    // High-loss policy: LossSignalBus owns the patience counter; we honor
    // its validation_high flag. patience=0 disables the policy entirely.
    if (!ctx.auto_stop_triggered && hp.auto_stop_high_loss_patience > 0) {
        if (state.loss_signals->latest().validation_high) {
            trip("high_loss");
        }
    }
}

bool maybeSaveCheckpoint(
    TrainingContext& ctx,
    float val_loss,
    int epoch) {
    
    if (val_loss >= ctx.best_val_loss) {
        return false;
    }
    
    ctx.best_val_loss = val_loss;
    ctx.logging.logger->log("✓ New best! Saving checkpoint...");
    
    std::string checkpoint_path = ctx.config.paths.checkpoint_dir + 
                                  "/checkpoint_epoch_" + std::to_string(epoch + 1) + ".bin";
    // model->save() handles its own sync before reading weights.
    try {
        bool save_result = ctx.model->save(checkpoint_path);
        if (save_result) {
            ctx.logging.logger->log("  ✓ Checkpoint saved: " + checkpoint_path);
            if (fs::exists(checkpoint_path)) {
                auto file_size = fs::file_size(checkpoint_path);
                ctx.logging.logger->log("  File size: " + std::to_string(file_size / (1024*1024)) + " MB");
            }
            // Save optimizer sidecar (.opt) alongside model checkpoint
            try {
                std::string opt_path = optimizerSidecarPath(checkpoint_path);
                saveOptimizerState(ctx, opt_path);
            } catch (const std::exception& e) {
                ctx.logging.logger->log(std::string("  ⚠ Optimizer state save failed: ") + e.what());
            }
            return true;
        } else {
            ctx.logging.logger->log("  ✗ Save returned false");
        }
    } catch (const std::exception& e) {
        ctx.logging.logger->log(std::string("  ✗ Exception: ") + e.what());
    }
    
    return false;
}

} // namespace Internal

//======================================================//
//  CoreRunHP read path
//
//  All BatchPayloads are authored once in Phase1
//  (Startup/Batching/PlannedBatches.cu); Phase2 only INDEXES into
//  ctx.train_payloads / ctx.val_payloads via ctx.epoch_batch_order. The
//  per-batch payload builder, the per-epoch BatchSchedule construction, and
//  the train_views shuffle that used to live here have all moved to
//  PlannedBatchesReady.
//======================================================//
namespace {

// Single Phase2 entry point for CoreRunHP fields (epochs, accumulation,
// single-batch-overfit). Re-validates Phase1's invariants as defense-in-depth.
// Phase2 must read these fields ONLY via this helper — never via
// ctx.config.hyperparameters.* directly. See Shared/HyperParameters/
// HyperparameterGroupings.hpp for the grouping definition.
::GRIM::HyperParameters::CoreRunHP validatedCoreRunHP(const TrainingContext& ctx) {
    auto core = ::GRIM::HyperParameters::coreRunHP(ctx.config);
    if (core.epochs <= 0) {
        throw std::runtime_error("FATAL: epochs must be > 0 in Phase2 (got " +
                                 std::to_string(core.epochs) + ")");
    }
    if (core.gradient_accumulation_steps <= 0) {
        throw std::runtime_error("FATAL: gradient_accumulation_steps must be > 0 in Phase2 (got " +
                                 std::to_string(core.gradient_accumulation_steps) + ")");
    }
    return core;
}

} // namespace

//======================================================//
//  Validation Implementation
//======================================================//

ValidationResult runValidation(TrainingContext& ctx) {
    ValidationResult result;
    
    ctx.logging.logger->log("Running validation...");

    // PRE-VALIDATION SAFETY: drain any deferred CUDA errors. Without this they
    // manifest as SEH exceptions inside the val loop, bypassing C++ catch.
    {
        cudaError_t sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            ctx.logging.logger->log("[Val] WARNING: cudaDeviceSynchronize before validation returned: " +
                std::string(cudaGetErrorString(sync_err)));
        }
        cudaError_t deferred_err = cudaGetLastError();
        if (deferred_err != cudaSuccess) {
            ctx.logging.logger->log("[Val] WARNING: Cleared deferred CUDA error before validation: " +
                std::string(cudaGetErrorString(deferred_err)));
        }
    }

    // PRE-VALIDATION CHECKPOINT: a val crash via SEH would otherwise lose the
    // entire epoch of training.
    {
        std::string safety_path = ctx.config.paths.checkpoint_dir + "/checkpoint_pre_validation.bin";
        ctx.logging.logger->log("[Val] Saving pre-validation safety checkpoint...");
        try {
            bool saved = ctx.model->save(safety_path);
            if (saved) {
                ctx.logging.logger->log("[Val] Safety checkpoint saved: " + safety_path);
            } else {
                ctx.logging.logger->log("[Val] WARNING: Safety checkpoint save returned false");
            }
        } catch (const std::exception& e) {
            ctx.logging.logger->log("[Val] WARNING: Safety checkpoint failed: " + std::string(e.what()));
            // Non-fatal — continue with validation anyway
        }
    }

    // Match GPU state to "start of training step". Rule 20: pre-val reset is
    // one boundary; per-batch RAII handles the rest.
    {
        GRIM::Autograd::AutogradStepScope pre_val_scope(ctx.model->getTrainingState());
    }
    flushDeferredCleanup();
    cudaDeviceSynchronize();
    (void)cudaGetLastError();

    {
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        ctx.logging.logger->log("[Val] GPU memory: " +
            std::to_string(free_mem / (1024*1024)) + " MB free / " +
            std::to_string(total_mem / (1024*1024)) + " MB total");
    }

    const uint32_t batch_rows = ctx.run_capacity.batch_rows;
    const uint32_t seq_cap    = ctx.run_capacity.seq_cap;
    const uint32_t token_budget = ctx.run_capacity.max_tokens_per_batch;
    if (batch_rows == 0 || seq_cap == 0 || token_budget == 0) {
        throw std::runtime_error("[Val] FATAL: invalid RunCapacity (batch_rows=" +
                                 std::to_string(batch_rows) + " seq_cap=" +
                                 std::to_string(seq_cap) + " token_budget=" +
                                 std::to_string(token_budget) + ")");
    }

    ctx.logging.logger->log("[Val] Token budget: " + std::to_string(token_budget) +
        " (RunCapacity: batch_rows=" + std::to_string(batch_rows) + " x seq_cap=" + std::to_string(seq_cap) + ")");

    // Validation iterates the Phase1-authored ctx.val_payloads in order.
    // Phase2 MUST NOT call buildBatches / buildValPayload — that work is
    // owned by PlannedBatchesReady at startup (Rule 20: fixed batch
    // membership; per-step batch creation is forbidden in the hot loop).
    const int total_val_batches = static_cast<int>(ctx.val_payloads.size());
    ctx.logging.logger->log("[Val] Iterating " + std::to_string(total_val_batches) +
                            " Phase1-authored validation payloads");

    float val_loss = 0.0f;
    int val_sequences_processed = 0;

    const auto val_start_time = std::chrono::steady_clock::now();

    for (int val_idx = 0; val_idx < total_val_batches; ++val_idx) {
        const auto& val_payload = ctx.val_payloads[val_idx];

        // Progress logging every 50 batches
        if (val_idx % 50 == 0) {
            ctx.logging.logger->log("[Val] Processing batch " + std::to_string(val_idx + 1) +
                "/" + std::to_string(total_val_batches) +
                " (processed=" + std::to_string(val_sequences_processed) + ")");
        }

        if (val_payload.batch_size == 0) {
            // Rule 20: PlannedBatches.cu validates payloads at startup, so an
            // empty payload arriving here means startup invariants regressed.
            throw std::runtime_error(
                "[Val] empty payload at batch " + std::to_string(val_idx + 1) +
                "/" + std::to_string(total_val_batches) +
                " — Phase1 PlannedBatches authored an empty val payload");
        }

        // Rule 20 single-owner clear: AutogradStepScope covers this batch.
        // Sync slice: upload the host-only payload to device once per val batch,
        // then pass the resulting bindings to computeLossBatch. computeLossBatch
        // never authors device pointers itself.
        float batch_val_loss = 0.0f;
        {
            GRIM::Autograd::AutogradStepScope val_step_scope(ctx.model->getTrainingState());
            const auto val_bindings = ctx.model->uploadBatchToDevice(val_payload);
            batch_val_loss = ctx.model->computeLossBatch(val_payload, val_bindings, /*is_training=*/false);
        }
        flushDeferredCleanup();

        cudaError_t batch_err = cudaGetLastError();
        if (batch_err != cudaSuccess) {
            throw std::runtime_error(
                "[Val] CUDA error after batch " + std::to_string(val_idx + 1) +
                ": " + std::string(cudaGetErrorString(batch_err)));
        }
        if (!std::isfinite(batch_val_loss)) {
            throw std::runtime_error(
                "[Val] non-finite loss at batch " + std::to_string(val_idx + 1) +
                " (loss=" + std::to_string(batch_val_loss) + ") — fix the model/data, do not skip");
        }

        val_loss += batch_val_loss * val_payload.batch_size;
        val_sequences_processed += val_payload.batch_size;
    }
    
    auto val_duration = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - val_start_time);
    
    if (val_sequences_processed > 0) {
        result.loss = val_loss / val_sequences_processed;
    } else {
        // No validation data configured. +inf so best-val tracking ignores it.
        result.loss = std::numeric_limits<float>::infinity();
    }
    result.sequences_processed = val_sequences_processed;
    result.perplexity = (std::isfinite(result.loss) && result.loss < 50.0f)
        ? std::exp(result.loss)
        : std::numeric_limits<float>::infinity();
    result.is_best = (result.loss < ctx.best_val_loss);
    
    ctx.logging.logger->log("[Val] " + Internal::formatMetric("loss", result.loss) + " " +
                            Internal::formatMetric("ppl", result.perplexity, 3) +
                            " seqs=" + std::to_string(val_sequences_processed) +
                            " time=" + std::to_string(val_duration.count()) + "s");
    
    return result;
}

//======================================================//
//  Batch Processing Implementation
//======================================================//

BatchResult processBatch(
    TrainingContext& ctx,
    TrainingLoopState& state,
    const GRIM::Batching::BatchPayload& payload,
    int batch_idx,
    int epoch_idx) {
    
    BatchResult result;
    result.batch_idx = batch_idx;

    // Begin tape recording for this batch (clears prior entries, sets step/batch)
    if (ctx.logging.tape) {
        ctx.logging.tape->beginBatch(static_cast<int>(ctx.global_step), batch_idx);
    }

    // Issue #142b: drain deferred CUDA errors from prior ops (sample gen,
    // previous batch backward); without this they manifest deep inside
    // encoderForward as an SEH exception.
    {
        cudaError_t pre_err = cudaGetLastError();
        if (pre_err != cudaSuccess) {
            std::string err_msg = "[processBatch] CUDA error BEFORE batch " +
                std::to_string(batch_idx + 1) + ": " +
                std::string(cudaGetErrorString(pre_err)) +
                " (code=" + std::to_string(static_cast<int>(pre_err)) + ")";
            ctx.logging.logger->log(err_msg);
            fprintf(stderr, "%s\n", err_msg.c_str());
            cudaDeviceSynchronize();
            cudaGetLastError();
        }
    }

    const auto& hp = ctx.config.hyperparameters;

    // Step counter convention:
    //   batch_number = batch_idx + 1                     (every batch)
    //   ctx.global_step                                  (every batch, token counter)
    //   ctx.optimizer.optimizer_state.step               (every accum_steps)
    // Log batch_number during fwd/bwd, optimizer_step at the optimizer step.

    if (payload.batch_size == 0) {
        // Rule 20: scheduler MUST NOT emit empty batches.
        throw std::runtime_error(
            "[processBatch] empty payload at batch_idx=" + std::to_string(batch_idx) +
            " — scheduler produced batch_size=0; fix the upstream filter");
    }

    // Token stats already in payload; per-batch seq_len from payload, not config.

    const auto& token_stats = payload.token_stats;
    const int long_seq_threshold = payload.max_seq_len;
    const auto clip_selection = GRIM::TNC::computeClipSelection(
        hp.grad_clip_norm, token_stats, 1.0f, long_seq_threshold);

    // CoreRunHP view — single source of truth for epochs/accumulation in this
    // function. Do not introduce direct ctx.config.hyperparameters reads for
    // CoreRunHP fields below this line.
    const auto core = validatedCoreRunHP(ctx);

    // Gradient zeroing: backward(accumulate=false) at micro_step=0 zeros grads;
    // backward(accumulate=true) at micro_step>0 accumulates.
    const int accum_steps = core.gradient_accumulation_steps;

    // beginBatch() must run EVERY batch to clear previous entries; otherwise
    // micro-batches 1+ inherit stale entries.
    GRIM::GradStats::beginBatch();

    PHASE2_DEBUG_STDERR("[DEBUG-PROCESS] After BATCH_INFO log, checking shouldLogAtomStats...\n");
    GRIM::Diagnostics::runAtomStatsDiagnostic(ctx, payload, batch_idx);

    PHASE2_DEBUG_STDERR("[DEBUG-PROCESS] After atom stats, entering boundary diagnostic...\n");
    GRIM::Diagnostics::runBoundaryDiagnostic(ctx, payload, batch_idx);

    PHASE2_DEBUG_STDERR("[DEBUG-PROCESS] After boundary diagnostic, entering forward pass...\n");
    static int forward_call_count = 0;
    ++forward_call_count;

    PHASE2_DEBUG_STDERR("[DEBUG-PROCESS] forward_call_count=%d, building target distribution...\n", forward_call_count);
    GRIM::Diagnostics::runTargetDistributionLog(ctx, payload, batch_idx);

    // Unified training step: forward → loss → backward.
    // Rule 20: micro_step MUST be in bounds before starting the step.
    if (ctx.optimizer.current_micro_step >= accum_steps) {
        fprintf(stderr, "\n[Phase2] FATAL: Training step attempted with micro_step=%d >= accum_steps=%d\n", 
                ctx.optimizer.current_micro_step, accum_steps);
        fprintf(stderr, "[Phase2] batch=%d global_step=%d\n", batch_idx + 1, ctx.global_step);
        fprintf(stderr, "[Phase2] This indicates current_micro_step was not reset after optimizer step.\n");
        std::abort();
    }

    // Issue #22: first micro-batch overwrites (accumulate=false), rest accumulate.
    const bool should_accumulate = ctx.optimizer.current_micro_step > 0;

    PHASE2_DEBUG_STDERR("[DEBUG-PROCESS] About to call autogradTrainingStep...\n");
    // First-batch CUDA checkpoint: surface any error before fwd/loss/bwd.
    if (batch_idx == 0) {
        cudaError_t e = cudaDeviceSynchronize();
        cudaError_t last = (e != cudaSuccess) ? e : cudaGetLastError();
        if (last != cudaSuccess) {
            ctx.logging.logger->log("[CUDA] first_batch BEFORE autogradTrainingStep: " + std::string(cudaGetErrorString(last)));
            cudaGetLastError();
        } else {
            ctx.logging.logger->log("[CUDA] first_batch BEFORE autogradTrainingStep: ok");
        }
    }
    // Issue #27: 1/M scaling at backward source (matches PyTorch). Without
    // this, accumulated gradients are off by a factor of M (sum-of-averages).
    const float grad_scale = 1.0f / static_cast<float>(accum_steps);
    // Rule 20 ownership taxonomy: AutogradStepScope is the SINGLE owner of
    // AutogradIntermediates::clear() for this batch. Do NOT add an explicit
    // clear() anywhere inside this scope.
    GRIM::Autograd::AutogradStepScope autograd_step_scope(ctx.model->getTrainingState());
    // Sync slice: upload the prebuilt host BatchPayload once and reuse the
    // returned BatchDeviceBindings inside autogradTrainingStep — payload itself
    // is host-only/immutable and never carries device pointers.
    const auto train_bindings = ctx.model->uploadBatchToDevice(payload);
    auto loss_result = GRIM::Autograd::autogradTrainingStep(
        *ctx.model,
        ctx.model->getTrainingState(),
        payload,
        train_bindings,
        should_accumulate,
        grad_scale,
        static_cast<uint64_t>(ctx.optimizer.optimizer_state.step)
    );
    result.loss = loss_result.loss_value;
    PHASE2_DEBUG_STDERR("[DEBUG-PROCESS] autogradTrainingStep returned, loss=%f success=%d\n", 
                        result.loss, static_cast<int>(loss_result.success));

    // First-batch CUDA checkpoint: a fault here means fwd/loss/bwd produced it.
    // Rule 20: crash with the exact error — don't defer to teardown.
    if (batch_idx == 0) {
        cudaError_t e = cudaDeviceSynchronize();
        cudaError_t last = (e != cudaSuccess) ? e : cudaGetLastError();
        if (last != cudaSuccess) {
            throw std::runtime_error(
                std::string("[CUDA] first_batch AFTER autogradTrainingStep: ") +
                cudaGetErrorString(last) +
                " (fault is in forward, loss, or backward)");
        }
        ctx.logging.logger->log("[CUDA] first_batch AFTER autogradTrainingStep: ok");
    }

    // Rule 20: NaN/Inf loss or backward error is a real bug; crash with the
    // propagated message instead of skipping the batch.
    if (!loss_result.success) {
        throw std::runtime_error(
            "[autogradTrainingStep] FAILED batch=" + std::to_string(batch_idx + 1) +
            ": " + loss_result.error_message);
    }
    
    // Update guess cache with predictions from this batch
    if (std::isfinite(result.loss) && ctx.config.hyperparameters.guess_aux_enabled) {
        GRIMTS::Training::updateGuessCacheFromBatch(
            ctx.model->getTrainingState(),
            payload,
            result.loss,
            epoch_idx,
            ctx.global_step,
            state.guess_cache);
    }
    
    // Log model predictions (what it predicts vs targets) - uses ForwardPass module for filtering
    // (extracted to Diagnostics/PredictionDistributionDiagnostic.cu)
    GRIM::Diagnostics::runPredictionDistributionAndLogitTrace(ctx, payload, result.loss, batch_idx);
    // NOTE: Loss variance computation removed (was causing 5-second GPU sync bottleneck).
    // Variance is now tracked on GPU by TelemetryLattice (σ_tilde, v_σ fields).
    // Use computeTelemetryFeedback() to access grad_norm variance for adaptive decisions.

    // ========================================================================
    // BATCH_LOSS equation log + LossStats summary line
    // (extracted to Diagnostics/LossStatsDiagnostic.cu)
    // ========================================================================
    GRIM::Diagnostics::runLossStatsDiagnostic(ctx, result, batch_idx);

    // ========================================================================
    // TRAINING SIGNAL: Logit Statistics (argmax distribution, confidence)
    // (extracted to Diagnostics/LogitScaleDiagnostic.cu)
    // ========================================================================
    GRIM::Diagnostics::runLogitScaleDiagnostic(ctx, payload, batch_idx);
    
    if (!std::isfinite(result.loss)) {
        throw std::runtime_error("Non-finite batch loss: " + std::to_string(result.loss));
    }

    // Publish this batch's loss to the central detector BEFORE any consumer
    // diagnostic reads it. Consumers (LossSpikeDiagnostic / DynamicLR / ...)
    // will be migrated to read state.loss_signals->latest() in subsequent
    // refactor tasks; today they still own their own detection.
    state.loss_signals->recordTrainStep(
        static_cast<int64_t>(ctx.global_step), result.loss);
    
    // ========================================================================
    // DIAGNOSTIC: Per-sequence breakdown when batch loss spikes far above
    // the captured baseline (initial) loss.
    // (extracted to Diagnostics/LossSpikeDiagnostic.cu)
    // ========================================================================
    GRIM::Diagnostics::runLossSpikeDiagnostic(ctx, payload, result.loss, batch_idx,
                                              *state.loss_signals);

    // ========================================================================
    // Adaptive loss baseline tracking + invalid-token validation
    // (extracted to Diagnostics/LossBaselineDiagnostic.cu)
    // Mutates: state.initial_loss, state.min_observed_loss, state.warmup_batches.
    // Throws on data corruption (Rule 20).
    // ========================================================================
    GRIM::Diagnostics::runLossBaselineAndTokenValidation(
        ctx, state, payload, result.loss, batch_idx);

    // ═══════════════════════════════════════════════════════════════════════════
    // POST-STEP: Backward already ran inside autogradTrainingStep().
    // Diagnostics below read from TrainingState (persists through backward).
    // ═══════════════════════════════════════════════════════════════════════════

    // Issue #142: special-token weight & gradient verification
    GRIM::Diagnostics::runSpecialTokenDiagnostic(ctx, payload, batch_idx);

    // Issue #23: sync grad-norm measurement EVERY batch for accurate diagnostics.
    // Norm here is for diagnostics/spike detection only — clipping later
    // (Issue #135) recomputes its own norm on accumulation-scaled gradients.
    // Mutates result.grad_rms / normalized_grad_rms; throws on NaN/Inf.
    const auto grad_norm_snap = GRIM::Diagnostics::runGradientNormDiagnostic(
        ctx, state, result, batch_idx);
    const float preclip_grad_rms = grad_norm_snap.preclip_grad_rms;
    const float enc_rms_pre = grad_norm_snap.enc_rms_pre;

    // Issue #135: gradient clipping is DEFERRED to post-accumulation (inside
    // should_step block). Clipping ONCE on the averaged gradients matches
    // PyTorch; the old per-micro-batch clipping crushed text gradients M×.
    const float effective_per_token_limit = clip_selection.per_token_limit;
    const bool clipping_enabled = (hp.grad_clip_norm > 0.0f);

    // LR: index by optimizer step (NOT global_step). global_step is per-micro-batch;
    // using it advances warmup/decay accum_steps times too fast.
    const int optimizer_step = static_cast<int>(ctx.optimizer.optimizer_state.step);
    if (!ctx.lr_schedule) {
        throw std::runtime_error("lr_schedule is not initialized at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    const auto lr_inputs = ::GRIM::HyperParameters::learningRateScheduleInputs(hp);
    result.learning_rate = Internal::getScheduledLearningRate(
        *ctx.lr_schedule, optimizer_step, lr_inputs.learning_rate,
        ctx.config.hyperparameters.stability_overrides_enabled);

    // Optimizer step
    const bool sync_diag = GRIM::Diagnostics::shouldSyncDiagnostics(ctx, batch_idx);
    GRIM::Diagnostics::WeightSample pre_sample{};
    if (sync_diag) {
        pre_sample = GRIM::Diagnostics::sampleWeightStats(ctx.model->getLmHeadLayer(), ctx.model->getTrainingState(), true);
    }

    if (sync_diag) {
        auto& training_state = ctx.model->getTrainingState();
        const auto flush_result = GRIM::GradStats::flushAndLog(
            training_state.stream_ctrl.getPrimaryStream(),
            ctx.global_step,
            ModuleId::BackwardPass);
        if (flush_result.has_explosion || flush_result.has_nan || flush_result.has_inf) {
            std::ostringstream oss;
            oss << "[GradStats] FATAL: gradient stats flagged "
                << (flush_result.has_explosion ? "explosion " : "")
                << (flush_result.has_nan ? "NaN " : "")
                << (flush_result.has_inf ? "Inf " : "")
                << "at batch=" << (batch_idx + 1)
                << " step=" << ctx.global_step;
            EmitModuleError(ModuleId::BackwardPass, oss.str(), ctx.global_step);
            throw std::runtime_error(oss.str());
        }
        
        // CRITICAL: sync stream BEFORE endOptimizerStep() zeros gradient buffers.
        // GradStats::flushAndLog() launches async D2H copies that read from grad
        // buffers; without this sync those copies see zeroed memory and produce
        // corrupted stats (e.g. negative max_abs, inverted min/max).
        cudaError_t sync_err = cudaStreamSynchronize(training_state.stream_ctrl.getPrimaryStream());
        if (sync_err != cudaSuccess) {
            std::ostringstream oss;
            oss << "[GradStats] FATAL: Failed to synchronize stream before optimizer step: "
                << cudaGetErrorString(sync_err);
            EmitModuleError(ModuleId::BackwardPass, oss.str(), ctx.global_step);
            throw std::runtime_error(oss.str());
        }
    }
    
    // Run optimizer step only when accumulation is complete.
    // Increment micro_step BEFORE checking (we already processed this batch's backward).
    ctx.optimizer.current_micro_step++;
    const bool should_step = (ctx.optimizer.current_micro_step >= accum_steps);

    if (should_step) {
        const int micro_step_for_log = ctx.optimizer.current_micro_step;
        const int accum_steps_for_log = accum_steps;

        // Issue #149: zero PAD/UNK gradients before optimizer step.
        // PAD (id=1) and UNK (id=0) are never valid targets but accumulate grads via:
        //   1. LM head bwd: label-smoothing redistributes tiny grad to all rows
        //   2. Embedding bwd: attention bwd leaks grad from valid positions to PAD
        //      input positions through K/V cross-attention (~76× BOS/EOS)
        // Zero them here so they don't inflate norms or waste optimizer capacity.
        {
            auto& zero_ts = ctx.model->getTrainingState();
            cudaStream_t zero_stream = zero_ts.stream_ctrl.getPrimaryStream();
            const auto& zero_cfg = ctx.model->getConfig();
            const size_t row_bytes = static_cast<size_t>(zero_cfg.d_model) * sizeof(float);
            
            constexpr int NON_TRAINABLE_TOKENS[] = {
                GRIM::Tokenizer::UNK_TOKEN_ID,  // 0
                GRIM::Tokenizer::PAD_TOKEN_ID   // 1
            };
            
            // Zero embedding gradients for non-trainable tokens
            float* emb_grads = ctx.model->getEmbeddingLayer()->tokenWeights().grad_data();
            if (emb_grads) {
                for (int tok : NON_TRAINABLE_TOKENS) {
                    cudaMemsetAsync(
                        emb_grads + static_cast<size_t>(tok) * zero_cfg.d_model,
                        0, row_bytes, zero_stream);
                }
            }
            
            // Zero LM head gradients for non-trainable tokens
            float* lm_grads = ctx.model->getLmHeadLayer()->weights().grad_data();
            if (lm_grads) {
                for (int tok : NON_TRAINABLE_TOKENS) {
                    cudaMemsetAsync(
                        lm_grads + static_cast<size_t>(tok) * zero_cfg.d_model,
                        0, row_bytes, zero_stream);
                }
            }
        }
        
        // Issue #135: per-component clipping on accumulated + 1/M-scaled gradients.
        //   1. emb_clip — LM_HEAD (+ EMBEDDING if untied)
        //   2. enc_clip — ATTENTION + FFN + RMSNORM + SCRATCHBLOCK + MTP +
        //                  REASONING_HEAD + EXECUTION_BLOCK
        // Norm measurement, bucket aggregation, and gradient scaling all happen
        // inside GradientCC against the registered ParameterGroup tensors.
        if (clipping_enabled) {
            auto& clip_ts = ctx.model->getTrainingState();
            cudaStream_t clip_stream = clip_ts.stream_ctrl.getPrimaryStream();
            auto& clip_groups = ctx.model->parameterGroups();

            if (!clip_ts.grad_norm_scratch) {
                throw std::runtime_error("[FATAL] grad_norm_scratch is NULL at clipping stage - "
                                         "diagnostic norm should have allocated it");
            }

            GRIM::GradClip::ClipConfig clip_cfg;
            clip_cfg.max_rms = effective_per_token_limit;
            clip_cfg.tie_embeddings = ctx.model->getConfig().tie_embeddings;

            const auto clip = GRIM::GradClip::clipGradientNorms(
                clip_groups.data(), clip_groups.size(),
                clip_ts.grad_norm_scratch, clip_cfg, clip_stream);

            result.grad_rms = clip.total_rms_post;
            result.normalized_grad_rms = clip.total_rms_post;
            result.gradient_clipped = clip.any_clipped();
        }
        
        // ════════════════════════════════════════════════════════════════════
        // RUNTIME tie_embeddings pointer verification (every batch)
        // (extracted to Diagnostics/TieVerifyDiagnostic.cu)
        // ════════════════════════════════════════════════════════════════════
        GRIM::Diagnostics::runTieVerifyDiagnostic(ctx, batch_idx);

        const int emb_freeze_step = ctx.config.hyperparameters.embedding_freeze_enabled
            ? ctx.config.hyperparameters.embedding_freeze_after_step : -1;

        if (emb_freeze_step > 0 && ctx.optimizer.optimizer_state.step == emb_freeze_step) {
            if (ctx.config.hyperparameters.architecture.tie_embeddings) {
                ctx.logging.logger->log("[EmbeddingFreeze] WARNING: tie_embeddings=true — "
                    "embedding and LM head share weights. Set tie_embeddings=false to freeze "
                    "embedding independently. Freeze has no effect on tied weights.");
            } else {
                ctx.logging.logger->log("[EmbeddingFreeze] Embedding weights FROZEN at step "
                    + std::to_string(emb_freeze_step) + " — no further embedding updates");
            }
        }

        // Optimizer dispatch — single source of truth: ctx.config.hyperparameters
        // (loaded from training.config.optimizer in ai_config.json). Rule 20:
        // kind already validated at config load ("adamw" | "radamw" only).
        const auto& opt_hp = ctx.config.hyperparameters;
        if (opt_hp.optimizer_kind == "radamw") {
            GRIM::launchRAdamWStep(ctx.model->parameterGroups(),
                                   result.learning_rate,
                                   opt_hp.weight_decay,
                                   ctx.optimizer.optimizer_state.step,
                                   opt_hp.optimizer_beta1,
                                   opt_hp.optimizer_beta2,
                                   opt_hp.optimizer_epsilon,
                                   ctx.model->getTrainingState().stream_ctrl.getPrimaryStream(),
                                   emb_freeze_step);
        } else {
            GRIM::launchAdamWStep(ctx.model->parameterGroups(),
                                  result.learning_rate,
                                  opt_hp.weight_decay,
                                  ctx.optimizer.optimizer_state.step,
                                  ctx.model->getTrainingState().stream_ctrl.getPrimaryStream(),
                                  emb_freeze_step);
        }

        // Rule 20: post-optimizer weight NaN spot check.
        GRIM::Diagnostics::checkPostOptimizerWeightsFinite(ctx, result, batch_idx);

        // Reset micro_step counter after optimizer step completes
        ctx.optimizer.current_micro_step = 0;

        // Tape flush at end of processBatch is the safety flush.
        GRIM::Diagnostics::runOptimizerMomentDiagnostic(
            ctx, batch_idx, micro_step_for_log, accum_steps_for_log, sync_diag);

        // Post-optimizer LM-head sample, GradTrace POST log, [UpdateMag],
        // and per-component Adam update_rms trace (Issue #150).
        GRIM::Diagnostics::runPostOptimizerWeightTrace(
            ctx, result, pre_sample, batch_idx, sync_diag);

        ctx.optimizer.optimizer_state.step++;
    }
    
    // Rule 20: an async CUDA error here means a kernel launch faulted earlier
    // in the step. Crash with the exact error rather than logging and dropping it.
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("[CUDA] async error after optimizer step: ") +
                cudaGetErrorString(err));
        }
    }
    
    result.sequences_processed = payload.batch_size;
    result.tokens_processed = clip_selection.stats.total_tokens;

    // Flush device logs on the diagnostic sync interval.
    if (sync_diag) {
        GRIM::Logging::FlushDeviceLogs();
    }

    // Update telemetry lattice — metric computation lives in TelemetryUpdate.cu.
    {
        GRIM::Telemetry::TelemetryBatchInput tel_input;
        tel_input.loss              = result.loss;
        tel_input.preclip_grad_rms  = preclip_grad_rms;
        tel_input.learning_rate     = result.learning_rate;
        tel_input.total_tokens      = payload.token_stats.total_tokens;
        tel_input.enc_rms_pre       = enc_rms_pre;
        tel_input.optimizer_step    = static_cast<int>(ctx.optimizer.optimizer_state.step);
        tel_input.should_step       = should_step;
        tel_input.total_loss_value  = loss_result.loss_value;
        tel_input.aux_loss          = loss_result.aux_loss;
        tel_input.max_seq_len       = payload.max_seq_len;
        tel_input.batch_idx         = batch_idx;
        tel_input.global_step       = ctx.global_step;
        tel_input.actual_vocab_size = ctx.config.actual_vocab_size;
        tel_input.d_model           = ctx.model->getConfig().d_model;

        GRIM::Telemetry::updateTelemetryObservations(ctx, tel_input, grad_norm_snap.metrics, &payload);
    }
    
    // First-batch CUDA checkpoint (runs even if telemetry disabled): last point before step++
    if (batch_idx == 0) {
        cudaError_t e = cudaDeviceSynchronize();
        cudaError_t last = (e != cudaSuccess) ? e : cudaGetLastError();
        if (last != cudaSuccess) {
            ctx.logging.logger->log("[CUDA] first_batch END processBatch: " + std::string(cudaGetErrorString(last)));
            cudaGetLastError();
        } else {
            ctx.logging.logger->log("[CUDA] first_batch END processBatch: ok");
        }
    }
    
    ctx.global_step++;
    state.last_grad_rms = result.grad_rms;

    // Rule 20 single-owner clear: AutogradStepScope at processBatch entry owns
    // the clear; the tape flush below does not touch autograd intermediates.
    if (ctx.logging.tape) {
        ctx.logging.tape->flush();
    }

    return result;
}

//======================================================//
//  Epoch Implementation
//======================================================//

EpochResult runEpoch(
    TrainingContext& ctx,
    TrainingLoopState& state,
    int epoch_idx) {
    
    EpochResult result;
    result.epoch = epoch_idx;
    
    const auto& hp = ctx.config.hyperparameters;
    // CoreRunHP view — Phase2's single read path for epochs / accumulation /
    // single-batch-overfit. Phase2 must NOT re-read these fields from `hp`.
    const auto core = validatedCoreRunHP(ctx);
    const int num_epochs = core.epochs;
    
    ctx.logging.logger->log("Epoch " + std::to_string(epoch_idx + 1) + "/" + std::to_string(num_epochs));
    
    // Reset guess cache at epoch start
    if (ctx.config.hyperparameters.guess_aux_enabled) {
        GRIMTS::Training::resetGuessCacheForEpoch(
            ctx.model->getTrainingState(), state.guess_cache);
    }
    
    PHASE2_DEBUG_STDERR("[DEBUG-EPOCH] After ResetGuessCache, indexing Phase1 schedule...\n");

    // Phase1 owns all batch packing (Startup/Batching/PlannedBatches.cu).
    // Phase2 NEVER shuffles ctx.data.train_views, never rebuilds
    // ctx.data.train_catalog, and never calls buildEpochBatches /
    // buildBatchPayload — diversity across epochs is expressed solely as a
    // permutation of fixed batch indices in ctx.epoch_batch_order.
    if (ctx.train_payloads.empty()) {
        throw std::runtime_error(
            "FATAL: ctx.train_payloads is empty at runEpoch — "
            "PlannedBatchesReady must run during Phase1");
    }
    if (epoch_idx < 0 || epoch_idx >= static_cast<int>(ctx.epoch_batch_order.size())) {
        throw std::runtime_error(
            "FATAL: epoch_idx " + std::to_string(epoch_idx) +
            " out of range for ctx.epoch_batch_order (size=" +
            std::to_string(ctx.epoch_batch_order.size()) + ")");
    }
    const auto& batch_order = ctx.epoch_batch_order[epoch_idx];
    if (batch_order.size() != ctx.train_payloads.size()) {
        throw std::runtime_error(
            "FATAL: epoch_batch_order[" + std::to_string(epoch_idx) +
            "].size()=" + std::to_string(batch_order.size()) +
            " != train_payloads.size()=" + std::to_string(ctx.train_payloads.size()));
    }

    const int total_batches = static_cast<int>(batch_order.size());
    int total_batches_to_run = total_batches;
    const bool single_batch_overfit = core.single_batch_overfit_enabled;
    if (single_batch_overfit) {
        if (core.single_batch_overfit_max_steps <= 0) {
            throw std::runtime_error("FATAL: single_batch_overfit_max_steps must be > 0 when single_batch_overfit_enabled=true (got " +
                                     std::to_string(core.single_batch_overfit_max_steps) + ")");
        }
        total_batches_to_run = core.single_batch_overfit_max_steps;
        ctx.logging.logger->log("[SingleBatch] Enabled: repeating batch 1 for " +
                                std::to_string(total_batches_to_run) + " steps.");
    }
    const auto epoch_start = std::chrono::steady_clock::now();
    float epoch_loss = 0.0f;

    // Accumulation steps come from the CoreRunHP view built at function entry.
    const int accum_steps = core.gradient_accumulation_steps;

    // Process batches: ctx.epoch_batch_order[epoch_idx] dictates which
    // Phase1-authored payload is active each step. The hard invariant from
    // the plan is:
    //     active_batch = ctx.train_payloads[ctx.epoch_batch_order[epoch][i]]
    for (int batch_idx = 0; batch_idx < total_batches_to_run; ++batch_idx) {
        const int active_idx = batch_order[single_batch_overfit ? 0 : batch_idx];
        const GRIM::Batching::BatchPayload& payload = ctx.train_payloads[active_idx];

        // Log progress periodically (from payload — single source of truth)
        if (batch_idx % 5 == 0) {
            std::ostringstream msg;
            msg << "[Batch " << (batch_idx + 1) << "/" << total_batches_to_run << "] "
                << "size=" << payload.seq_ids.size()
                << " len=" << payload.min_seq_len << "-" << payload.max_seq_len
                << " eff=" << static_cast<int>(payload.packing_efficiency * 100) << "%"
                << " accum_steps=" << accum_steps;
            ctx.logging.logger->log(msg.str());
        }

        BatchResult batch_result = processBatch(ctx, state, payload, batch_idx, epoch_idx);

        // Rule 20: surface any first-batch CUDA error here so the real fault
        // shows up rather than a teardown cudaFree failure.
        if (batch_idx == 0) {
            cudaError_t sync_err = cudaDeviceSynchronize();
            cudaError_t last_err = (sync_err != cudaSuccess) ? sync_err : cudaGetLastError();
            if (last_err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("[CUDA] ERROR after first batch: ") + cudaGetErrorString(last_err) +
                    " (sync=" + (sync_err != cudaSuccess ? "failed" : "ok") + ")");
            }
        }

        epoch_loss += batch_result.loss;
        result.batches_processed++;
        result.best_batch_loss = std::min(result.best_batch_loss, batch_result.loss);
        result.worst_batch_loss = std::max(result.worst_batch_loss, batch_result.loss);
        
        // Log at interval
        if (ctx.global_step % hp.log_interval == 0) {
            ctx.logging.logger->log("[Step " + std::to_string(ctx.global_step) + "] " +
                                    Internal::formatMetric("loss", batch_result.loss) + " " +
                                    Internal::formatMetric("lr", batch_result.learning_rate, 8));
            // Multi-Token-Prediction per-head telemetry + monitor
            GRIM::Diagnostics::runMtpDiagnostic(ctx, batch_result);
            if (ctx.config.hyperparameters.guess_aux_enabled) {
                GRIMTS::Training::logGuessCacheTelemetry(state.guess_cache, ctx.global_step);
            }
        }

        logDiagnosticSample(ctx, state);
        
        // Status update
        float current_avg_loss = epoch_loss / result.batches_processed;
        float train_perplexity = (std::isfinite(current_avg_loss) && current_avg_loss < 50.0f)
            ? std::exp(current_avg_loss)
            : std::numeric_limits<float>::infinity();
        
        ctx.logging.status_writer->writeStatus(
            GRIMText::Control::TrainingState_Training,
            epoch_idx + 1, num_epochs,
            batch_idx + 1, total_batches_to_run,
            batch_result.loss, current_avg_loss,
            train_perplexity, 0.0f, 0.0f, 0.0f,
            "Training epoch " + std::to_string(epoch_idx + 1) + " batch " + std::to_string(batch_idx + 1));
        

    }
    
    if (result.batches_processed <= 0) {
        throw std::runtime_error("FATAL: epoch completed with 0 processed batches");
    }
    result.avg_loss = epoch_loss / result.batches_processed;
    result.duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - epoch_start);
    
    ctx.logging.logger->log("[Epoch " + std::to_string(epoch_idx + 1) + "] " +
                            Internal::formatMetric("avg_loss", result.avg_loss));
    
    // Log telemetry vectors (multi-scale summary)
    GRIM::Telemetry::logTelemetrySummary(ctx);
    
    // Run validation
    result.validation = runValidation(ctx);

    // Publish validation loss to central detector before any consumer
    // (evaluateAutoStop / SoftRestart) reads it.
    if (std::isfinite(result.validation.loss)) {
        state.loss_signals->recordValidation(epoch_idx, result.validation.loss);
    }
    
    // Auto-stop checks (plateau + high-loss). See Internal::evaluateAutoStop.
    Internal::evaluateAutoStop(ctx, state, result, epoch_idx);
    
    // Checkpoint
    if (result.validation.is_best) {
        Internal::maybeSaveCheckpoint(ctx, result.validation.loss, epoch_idx);
    }
    
    // Update status — GPU memory query at epoch granularity (cudaMemGetInfo
    // is implicitly synchronizing on some drivers).
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    float gpu_used_mb = static_cast<float>((total_mem - free_mem)) / (1024.0f*1024.0f);
    float gpu_total_mb = static_cast<float>(total_mem) / (1024.0f*1024.0f);
    
    ctx.logging.status_writer->writeStatus(
        GRIMText::Control::TrainingState_Training,
        epoch_idx + 1, num_epochs,
        total_batches, total_batches,
        result.validation.loss, result.avg_loss,
        result.validation.perplexity, 0.0f,
        gpu_used_mb, gpu_total_mb,
        "Validation complete - epoch " + std::to_string(epoch_idx + 1));
    
    return result;
}

//======================================================//
//  Phase2 Main Entry Point
//======================================================//

bool executePhase2(TrainingContext& ctx) {
    const auto& hp = ctx.config.hyperparameters;
    
    // Initialize loop state
    TrainingLoopState state;

    // Construct the central loss-signal detector. Wire validation auto-stop
    // policy from hp now so evaluateAutoStop subscribes to validation_high.
    // Other thresholds stay defaulted; task 7 plumbs the full ai_config.json
    // "loss_signals" block through HyperParameters.
    {
        GRIM::Loss::LossSignalConfig sig_cfg{};
        sig_cfg.validation_high_threshold = hp.auto_stop_high_loss_threshold;
        sig_cfg.validation_high_patience  = hp.auto_stop_high_loss_patience;
        state.loss_signals = std::make_unique<GRIM::Loss::LossSignalBus>(sig_cfg);
    }
    
    // Initialize guess cache (Rule 22: pass TrainingState for buffer allocation)
    auto guess_cache_scope = GRIMTS::Training::initGuessCache(
        ctx.model->getTrainingState(),
        ctx.config.hyperparameters.guess_aux_enabled,
        ctx.config.hyperparameters.single_stream_mode,
        ctx.global_step,
        state.guess_cache);
    
    PHASE2_DEBUG_STDERR("[DEBUG] About to initialize training log...");

    // NOTE: per-field hyperparameter logging (warmup_fraction, learning_rate,
    // cosine_decay_*, soft_restart_*, auto_stop_*, embedding_freeze_*) is the
    // exclusive responsibility of dumpAllHyperparameters() — invoked from
    // Phase1_Startup. Do NOT re-log individual hp fields here; that creates a
    // second source of truth that drifts every time a field is added.
    EmitModuleInfo(ModuleId::Training, "Starting training...", ctx.global_step);

    // CoreRunHP view for executePhase2's top-level epoch loop. Subroutines
    // (runEpoch / processBatch) build their own views — keeping the read path
    // local makes each function self-contained and independently auditable.
    const auto core = validatedCoreRunHP(ctx);
    const int num_epochs = core.epochs;
    EmitModuleInfo(ModuleId::Training,
        std::string("Total epochs to run: ") + std::to_string(num_epochs), ctx.global_step);
    
    // Reconstruct adam_cumulative_disp = Σlr(0..optimizer_step-1) for resume
    // correctness. Iterate optimizer steps (NOT global_step / batch counter).
    const int resumed_optimizer_step = static_cast<int>(ctx.optimizer.optimizer_state.step);
    if (resumed_optimizer_step > 0 && ctx.telemetry.adam_cumulative_disp == 0.0f) {
        float reconstructed = 0.0f;
        for (int t = 0; t < resumed_optimizer_step; ++t) {
            reconstructed += ctx.lr_schedule->lr(t);
        }
        ctx.telemetry.adam_cumulative_disp = reconstructed;
        EmitModuleInfo(ModuleId::Training,
            "[AdamCausation] Reconstructed cumulative_displacement=" + std::to_string(reconstructed) +
            " from " + std::to_string(resumed_optimizer_step) + " optimizer steps", ctx.global_step);
    }
    
    try {
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            EpochResult epoch_result = runEpoch(ctx, state, epoch);
            ctx.epochs_completed = epoch + 1;
            
            if (epoch_result.auto_stop_triggered) {
                EmitModuleInfo(ModuleId::Training,
                    std::string("Auto-stop engaged after epoch ") + std::to_string(epoch + 1) +
                    " (" + epoch_result.auto_stop_reason + "). Skipping remaining epochs.", ctx.global_step);
                break;
            }
            if (epoch + 1 < num_epochs) {
                EmitModuleInfo(ModuleId::Training,
                    "[Epoch boundary] Epoch " + std::to_string(epoch + 1) + "/" + std::to_string(num_epochs) +
                    " complete. Transitioning to epoch " + std::to_string(epoch + 2) + ".", ctx.global_step);
            } else {
                EmitModuleInfo(ModuleId::Training,
                    "[Epoch boundary] All " + std::to_string(num_epochs) + " epochs complete.", ctx.global_step);
            }
        }
    } catch (const std::exception& e) {
        EmitModuleError(ModuleId::Training, 
            std::string("TRAINING ERROR: ") + e.what(), ctx.global_step);
        
        ctx.logging.status_writer->writeStatus(
            GRIMText::Control::TrainingState_Error,
            0, num_epochs, 0, 0,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            "Training error", std::string(e.what()));
        
        throw;
    }
    return true;
}

} // namespace GRIMText::Training
