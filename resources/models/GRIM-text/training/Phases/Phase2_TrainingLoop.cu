//======================================================//
//  Phase2_TrainingLoop.cu
//  Core training computation and logic
//======================================================//
//
//  IMPLEMENTATION
//  ==============
//  This file implements all training computation:
//  1. Epoch iteration with dynamic batching
//  2. Batch construction with content weighting
//  3. Forward/backward passes with gradient accumulation
//  4. Gradient clipping (token-normalized + adaptive)
//  5. Optimizer steps with dynamic LR
//  6. Validation and checkpointing
//  7. Auto-stop and soft restart logic
//
//  Author: Austin Wadkins
//  Date: December 2025
//  Version: 1.0.0
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
#include "../../Shared/Batching/EpochBatching.hpp"  // GRIM::Batching::buildEpochBatches
#include "../Autograd/AutogradTraining.hpp"  // autogradTrainingStep: unified forward+loss+backward
#include "../../Shared/Optimizers/AdamW/AdamW_Kernal_GPU.hpp"  // launchAdamWStep, resetAdamWMoments, scaleAdamWMoments
#include "../../Shared/Optimizers/RAdamW/RAdamW_Kernal_GPU.hpp"  // launchRAdamWStep — selectable via training.config.optimizer.kind
#include "../../../../../control/ai_config_paths.hpp"  // For resolveGrimRoot()
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
//  String Utilities
//======================================================//

// Anon-namespace gate helpers (readEnvInt, readEnvString,
// isPhase2DebugEnabled, PHASE2_DEBUG_STDERR/FLUSH_STDERR,
// shouldSyncDiagnostics, shouldLogLogitTrace, shouldLogAtomStats)
// and the AtomStats / MomentSample / sampleOptimizerMomentStats helpers
// have all been moved to Diagnostics/DiagnosticGates.{hpp,cu},
// Diagnostics/AtomStatsDiagnostic.{hpp,cu}, and
// Diagnostics/OptimizerMomentDiagnostic.{hpp,cu}.
// They are reachable here via #include "../Diagnostics/DiagnosticGates.hpp"
// at the top of this file.

// GuessCacheScope, GuessCacheBatchBuffers implementations moved to
// Layers/GRIMTS/GuessCacheTraining.cu (namespace GRIMTS::Training)

//======================================================//
//  GPU-Native Telemetry Control
//======================================================//
// All decision logic runs on GPU via TelemetryControl::evaluate()
// See: TelemetryControl_GPU.{cu,hpp} for kernel implementation
// Only 48-byte ControlDecision struct synced to CPU

//======================================================//
//  Internal Helper Implementations
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

std::string formatGradientComponents(const GRIM::GradNorm::GradMetrics& gm, bool tied) {
    using GM = GRIM::GradNorm::GradMetrics;
    std::ostringstream comp_msg;
    comp_msg << "[GradTrace] COMPONENTS(rms):";
    
    // When tie_embeddings=true: tied buffer registered as LM_HEAD type (embedding_sum_sq=0)
    // When tie_embeddings=false: separate EMBEDDING and LM_HEAD groups
    // Use precision=6 so small per-parameter RMS values (e.g. attn ~0.00004) don't
    // display as 0.0000 with the default precision=4 (Issue #150)
    constexpr int kComponentPrecision = 10;

    if (tied) {
        comp_msg << " emb_lm_tied=" << formatScalar(GM::rms(gm.lm_head_sum_sq, gm.lm_head_count), kComponentPrecision);
    } else {
        comp_msg << " emb=" << formatScalar(GM::rms(gm.embedding_sum_sq, gm.embedding_count), kComponentPrecision)
                 << " lm=" << formatScalar(GM::rms(gm.lm_head_sum_sq, gm.lm_head_count), kComponentPrecision);
    }
    
    comp_msg << " attn=" << formatScalar(GM::rms(gm.attention_sum_sq, gm.attention_count), kComponentPrecision)
             << " ffn=" << formatScalar(GM::rms(gm.ffn_sum_sq, gm.ffn_count), kComponentPrecision)
             << " rmsnorm=" << formatScalar(GM::rms(gm.rmsnorm_sum_sq, gm.rmsnorm_count), kComponentPrecision);

    comp_msg << " tied=" << (tied ? "yes" : "no");
    
    // Include ScratchBlock if enabled
    if (gm.scratchblock_sum_sq > 0.0f) {
        comp_msg << " sb=" << formatScalar(GM::rms(gm.scratchblock_sum_sq, gm.scratchblock_count), kComponentPrecision);
    }
    
    return comp_msg.str();
}

/// Query the LR schedule at a given step, respecting stability overrides.
/// Delegates to GRIM::LR::LRSchedule for the actual curve computation.
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

// Emits the standard "[Batching] ..." log block summarizing a freshly-built
// BatchSchedule. Kept out of buildEpochBatches() so the orchestration step
// only contains the construction call.
// NOTE: buildEpochBatches() and logBatchSchedule() now live in
// Shared/Batching/EpochBatching.{hpp,cu} (namespace GRIM::Batching) so the
// per-epoch batching policy is co-located with the rest of the batch
// construction logic. Phase2 calls GRIM::Batching::buildEpochBatches(...)
// directly via the include above.

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

    // High-loss detection (validation policy) — delegated to the central
    // LossSignalBus. The bus owns the consecutive-validation-high counter and
    // sets `validation_high = true` on the epoch the patience trips. We just
    // honor it. The hp.auto_stop_high_loss_patience guard is kept so a value of
    // 0 disables the policy outright (consistent with the plateau guard above).
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
    
    // NOTE: model->save() handles its own sync internally before reading weights
    // No need for device sync here - let save() manage sync granularity
    
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
//  Validation Implementation
//======================================================//

ValidationResult runValidation(TrainingContext& ctx) {
    ValidationResult result;
    
    ctx.logging.logger->log("Running validation...");

    // ═══════════════════════════════════════════════════════════════════════════
    // PRE-VALIDATION SAFETY: Sync GPU and check for deferred CUDA errors.
    // After 24+ hours of training, deferred errors can accumulate. If we don't
    // drain them here, they manifest as SEH exceptions inside the validation
    // loop — bypassing C++ catch blocks and silently killing the process.
    // ═══════════════════════════════════════════════════════════════════════════
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

    // ═══════════════════════════════════════════════════════════════════════════
    // PRE-VALIDATION CHECKPOINT: Save model state BEFORE validation.
    // Validation runs hundreds of forward passes that can crash via SEH.
    // Without this, a validation crash loses the entire epoch of training.
    // ═══════════════════════════════════════════════════════════════════════════
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

    // Match GPU state to "start of training step" so validation has same memory as training.
    // Rule 20: pre-validation reset is one boundary; per-batch RAII handles the rest.
    {
        GRIM::Autograd::AutogradStepScope pre_val_scope(ctx.model->getTrainingState());
    }
    flushDeferredCleanup();
    cudaDeviceSynchronize();
    (void)cudaGetLastError();

    // Log available GPU memory before validation for diagnostics
    {
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        ctx.logging.logger->log("[Val] GPU memory: " +
            std::to_string(free_mem / (1024*1024)) + " MB free / " +
            std::to_string(total_mem / (1024*1024)) + " MB total");
    }

    // Use the same batch limits as training so validation uses the same memory path.
    const auto& model_cfg = ctx.model->getConfig();
    const int batch_size = std::max(1, ctx.config.hyperparameters.batch_size);
    const uint32_t config_token_budget = static_cast<uint32_t>(batch_size) *
        static_cast<uint32_t>(std::max(1, model_cfg.max_seq_len));

    ctx.logging.logger->log("[Val] Token budget: " + std::to_string(config_token_budget) +
        " (same as training: batch_size=" + std::to_string(batch_size) + " x max_seq_len=" + std::to_string(model_cfg.max_seq_len) + ")");

    GRIM::Batching::BatchOptions val_opts;
    val_opts.max_tokens_per_batch = config_token_budget;
    val_opts.max_batch_size = static_cast<uint32_t>(batch_size);
    val_opts.bucket_step = 256;
    
    auto val_schedule = GRIM::Batching::buildBatches(ctx.data.val_catalog, val_opts);
    const int total_val_batches = static_cast<int>(val_schedule.batches.size());
    ctx.logging.logger->log("Created " + std::to_string(total_val_batches) + " validation batches");
    
    float val_loss = 0.0f;
    int val_sequences_processed = 0;
    int val_batches_failed = 0;
    
    const auto& val_model_cfg = ctx.model->getConfig();
    const size_t val_max_cached_batch = static_cast<size_t>(std::max(1, val_model_cfg.max_cached_batch));
    const size_t val_max_cached_seq = static_cast<size_t>(std::max(1, std::min(val_model_cfg.max_seq_len, val_model_cfg.max_cached_seq_len)));
    
    const auto val_start_time = std::chrono::steady_clock::now();
    
    for (int val_idx = 0; val_idx < total_val_batches; ++val_idx) {
        const auto& val_batch = val_schedule.batches[val_idx];
        
        // Progress logging every 50 batches
        if (val_idx % 50 == 0) {
            ctx.logging.logger->log("[Val] Processing batch " + std::to_string(val_idx + 1) + 
                "/" + std::to_string(total_val_batches) +
                " (processed=" + std::to_string(val_sequences_processed) +
                " failed=" + std::to_string(val_batches_failed) + ")");
        }
        
        try {
            const auto val_token_layout = ctx.tokenizer.tokenLayout();
            auto val_payload = GRIM::Batching::buildBatchPayload(
                val_batch, ctx.data.val_views, ctx.config.actual_vocab_size,
                val_token_layout,
                val_max_cached_batch, val_max_cached_seq,
                val_model_cfg.execution_block_num_slots,
                val_model_cfg.execution_block_num_ops,
                val_model_cfg.execution_block_num_steps,
                0);  // mtp_k=0 for validation — no MTP shifting needed
            if (val_payload.batch_size == 0) continue;
            
            // Rule 20 single-owner clear: AutogradStepScope covers this validation
            // batch and unwinds on both normal exit and exception (catch handler
            // below MUST NOT call clear() again).
            float batch_val_loss = 0.0f;
            {
                GRIM::Autograd::AutogradStepScope val_step_scope(ctx.model->getTrainingState());
                batch_val_loss = ctx.model->computeLossBatch(val_payload, /*is_training=*/false);
            }
            flushDeferredCleanup();

            // Check for deferred CUDA errors after each batch
            cudaError_t batch_err = cudaGetLastError();
            if (batch_err != cudaSuccess) {
                ctx.logging.logger->log("[Val] CUDA error after batch " + std::to_string(val_idx + 1) + 
                    ": " + std::string(cudaGetErrorString(batch_err)));
                val_batches_failed++;
                // Sync and clear to attempt recovery for remaining batches
                cudaDeviceSynchronize();
                cudaGetLastError();
                continue;
            }
            
            // Validate the loss value before accumulating
            if (!std::isfinite(batch_val_loss)) {
                ctx.logging.logger->log("[Val] WARNING: Non-finite loss at batch " + 
                    std::to_string(val_idx + 1) + " (loss=" + std::to_string(batch_val_loss) + "), skipping");
                val_batches_failed++;
                continue;
            }
            
            val_loss += batch_val_loss * val_payload.batch_size;
            val_sequences_processed += val_payload.batch_size;
            
        } catch (const std::exception& e) {
            ctx.logging.logger->log("[Val] Exception at batch " + std::to_string(val_idx + 1) + 
                "/" + std::to_string(total_val_batches) + ": " + std::string(e.what()));
            val_batches_failed++;
            
            // Rule 20 single-owner clear: AutogradStepScope already destructed when
            // the try-block scope unwound the exception. Do NOT add a clear() here.
            
            // Sync and clear CUDA state for recovery — also flushes deferred GPU frees
            cudaDeviceSynchronize();
            flushDeferredCleanup();
            cudaGetLastError();
            
            // If too many batches fail (>10%), abort validation with partial results
            if (val_batches_failed > total_val_batches / 10 && val_batches_failed >= 5) {
                ctx.logging.logger->log("[Val] ABORTING: Too many failed batches (" + 
                    std::to_string(val_batches_failed) + "/" + std::to_string(val_idx + 1) + 
                    "). Using partial results.");
                break;
            }
            continue;
        }
    }
    
    auto val_duration = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - val_start_time);
    
    if (val_sequences_processed > 0) {
        result.loss = val_loss / val_sequences_processed;
    } else {
        ctx.logging.logger->log("[Val] ERROR: No validation sequences processed successfully!");
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
                            " failed=" + std::to_string(val_batches_failed) +
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
    
    // Issue #142b: Eagerly detect any deferred CUDA errors from prior operations
    // (e.g., sample generation, previous batch backward). Without this, the error
    // manifests deep inside encoderForward as an SEH exception → silent exit.
    {
        cudaError_t pre_err = cudaGetLastError();
        if (pre_err != cudaSuccess) {
            std::string err_msg = "[processBatch] CUDA error BEFORE batch " +
                std::to_string(batch_idx + 1) + ": " +
                std::string(cudaGetErrorString(pre_err)) +
                " (code=" + std::to_string(static_cast<int>(pre_err)) + ")";
            ctx.logging.logger->log(err_msg);
            fprintf(stderr, "%s\n", err_msg.c_str());
            // Attempt recovery: sync and clear
            cudaDeviceSynchronize();
            cudaGetLastError();
        }
    }
    
    const auto& hp = ctx.config.hyperparameters;
    // logit_trace_enabled gating now lives inside
    // GRIM::Diagnostics::runPredictionDistributionAndLogitTrace.
    
    // ========================================================================
    // RULE 20: Step Counter Clarity
    // Three step counters exist:
    // 1. batch_number = batch_idx + 1 (increases every batch: 1,2,3,...,N)
    // 2. ctx.global_step = training token counter (increments with every batch)
    // 3. ctx.optimizer.optimizer_state.step = actual optimizer updates (every accum_steps)
    //
    // CONVENTION: Log ONLY the relevant counter:
    // - During FORWARD/BACKWARD: use batch_number (most relevant to user)
    // - During OPTIMIZER step: use optimizer_step (shows actual weight updates)
    // - Remove global_step from logs (creates confusion with near-duplicate batch_number)
    // ========================================================================

    // Payload is built in runEpoch from scheduler (BatchAssignment); single source of truth here
    if (payload.batch_size == 0) {
        result.skipped = true;
        result.skip_reason = "filtered";
        return result;
    }

    // First batch diagnostics - check weight initialization
    if (batch_idx == 0 && ctx.global_step == 0) {
        ctx.logging.logger->log("[GradTrace] FIRST_BATCH: Checking initial model state...");
        auto model_stats = ctx.model->getModelStats();
        ctx.logging.logger->log("[GradTrace] FIRST_BATCH: total_params=" + std::to_string(model_stats.total_params) +
                                " embedding_params=" + std::to_string(model_stats.embedding_params) +
                                " encoder_params=" + std::to_string(model_stats.encoder_params) +
                                " lm_head_params=" + std::to_string(model_stats.lm_head_params));
    }
    
    // Token stats already computed in payload — single source of truth
    const auto& token_stats = payload.token_stats;
    // Per-batch seq_len from BatchPayload (single source of truth), not config
    const int long_seq_threshold = payload.max_seq_len;
    const auto clip_selection = GRIM::TNC::computeClipSelection(
        hp.grad_clip_norm, token_stats, 1.0f, long_seq_threshold);
    
    // Start gradient accumulation window only when at micro_step 0
    // (i.e., first micro-batch after an optimizer step or at very start)
    // Gradient zeroing is handled by backward() method based on accumulate parameter.
    // When micro_step=0, backward() is called with accumulate=false → zeros gradients.
    // When micro_step>0, backward() is called with accumulate=true → accumulates.
    const int accum_steps = std::max(1, ctx.config.hyperparameters.gradient_accumulation_steps);
    
    // BUG FIX: beginBatch() must be called EVERY batch to clear previous entries,
    // not just at accumulation window start. Otherwise micro-batches 1+ have stale entries.
    GRIM::GradStats::beginBatch();
    
    // DIAGNOSTIC: Disabled for performance (was causing 4 device syncs per batch)
    // static int diag_batch_count = 0;
    // ++diag_batch_count;
    // if (diag_batch_count <= 3) {
    //     const auto& ts = ctx.model->getTrainingState();
    //     ctx.logging.logger->log("[GradDiag] AFTER_ZERO: " + sampleEmbeddingGrads(ts));
    // }
    
    // Log batch info for diagnostics
    {
        std::ostringstream batch_info;
        batch_info << "[GradTrace] BATCH_INFO batch=" << (batch_idx + 1)
                   << " seqs=[";
        for (size_t i = 0; i < payload.seq_ids.size(); ++i) {
            batch_info << payload.seq_ids[i];
            if (i + 1 < payload.seq_ids.size()) batch_info << ",";
        }
        batch_info << "] lens=[";
        for (int i = 0; i < payload.batch_size; ++i) {
            batch_info << payload.seq_lengths[i];
            if (i + 1 < payload.batch_size) batch_info << ",";
        }
        batch_info << "]";
        ctx.logging.logger->log(batch_info.str());
    }

    PHASE2_DEBUG_STDERR("[DEBUG-PROCESS] After BATCH_INFO log, checking shouldLogAtomStats...\n");
    // ========================================================================
    // DIAGNOSTIC: Atom-token statistics for the batch
    // (extracted to Diagnostics/AtomStatsDiagnostic.cu)
    // ========================================================================
    GRIM::Diagnostics::runAtomStatsDiagnostic(ctx, payload, batch_idx);
    
    PHASE2_DEBUG_STDERR("[DEBUG-PROCESS] After atom stats, entering boundary diagnostic...\n");
    // ========================================================================
    // DIAGNOSTIC: Boundary crossing check (simplified for FlashAttention v2)
    // (extracted to Diagnostics/BoundaryDiagnostic.cu)
    // ========================================================================
    GRIM::Diagnostics::runBoundaryDiagnostic(ctx, payload, batch_idx);
    
    PHASE2_DEBUG_STDERR("[DEBUG-PROCESS] After boundary diagnostic, entering forward pass...\n");
    // Forward pass
    static int forward_call_count = 0;
    ++forward_call_count;
    
    PHASE2_DEBUG_STDERR("[DEBUG-PROCESS] forward_call_count=%d, building target distribution...\n", forward_call_count);
    // Log target and prediction distributions (uses ForwardPass module for filtering)
    GRIM::Diagnostics::runTargetDistributionLog(ctx, payload, batch_idx);
    
    // ═══════════════════════════════════════════════════════════════════════════
    // UNIFIED TRAINING STEP: forward → loss → backward via autogradTrainingStep
    // Replaces the old computeLossBatch() + backward() two-call pattern.
    // ═══════════════════════════════════════════════════════════════════════════
    
    // Rule 20: micro_step MUST be within bounds before starting the step
    if (ctx.optimizer.current_micro_step >= accum_steps) {
        fprintf(stderr, "\n[Phase2] FATAL: Training step attempted with micro_step=%d >= accum_steps=%d\n", 
                ctx.optimizer.current_micro_step, accum_steps);
        fprintf(stderr, "[Phase2] batch=%d global_step=%d\n", batch_idx + 1, ctx.global_step);
        fprintf(stderr, "[Phase2] This indicates current_micro_step was not reset after optimizer step.\n");
        std::abort();
    }
    
    // BUG FIX Issue #22: First micro-batch overwrites (accumulate=false), subsequent accumulate
    const bool should_accumulate = ctx.optimizer.current_micro_step > 0;
    
    PHASE2_DEBUG_STDERR("[DEBUG-PROCESS] About to call autogradTrainingStep...\n");
    // First-batch CUDA checkpoint: surface any error before forward/loss/backward
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
    // STABILITY FIX (Issue #27/Math Audit): Apply 1/M scaling at the source (backward pass) to match PyTorch.
    // This ensures that gradients averaged per micro-batch are further averaged across the accumulation window,
    // preventing the "Sum of Averages" discrepancy that causes gradient explosion by a factor of M.
    const float grad_scale = 1.0f / static_cast<float>(accum_steps);
    // FIX: Pass optimizer step, not global_step (batch counter).
    // ctx.step is used downstream by MTP alpha warmup ramp and dropout seeds.
    // With gradient_accumulation_steps > 1, global_step advances accum_steps
    // times faster than actual weight updates, completing warmup ramps too early.
    // Rule 20 ownership taxonomy: AutogradStepScope is the SINGLE owner of
    // AutogradIntermediates::clear() for this batch. It covers autogradTrainingStep
    // + GuessCache update + post-step diagnostics + tape logging. Do NOT add an
    // explicit clear() anywhere inside this scope.
    GRIM::Autograd::AutogradStepScope autograd_step_scope(ctx.model->getTrainingState());
    auto loss_result = GRIM::Autograd::autogradTrainingStep(
        *ctx.model,
        ctx.model->getTrainingState(),
        payload,
        should_accumulate,
        grad_scale,
        static_cast<uint64_t>(ctx.optimizer.optimizer_state.step)
    );
    result.loss = loss_result.loss_value;
    PHASE2_DEBUG_STDERR("[DEBUG-PROCESS] autogradTrainingStep returned, loss=%f success=%d\n", 
                        result.loss, static_cast<int>(loss_result.success));
    
    // First-batch CUDA checkpoint: fault is in forward/loss/backward if you see error here
    if (batch_idx == 0) {
        cudaError_t e = cudaDeviceSynchronize();
        cudaError_t last = (e != cudaSuccess) ? e : cudaGetLastError();
        if (last != cudaSuccess) {
            ctx.logging.logger->log("[CUDA] first_batch AFTER autogradTrainingStep: " + std::string(cudaGetErrorString(last)) +
                " (fault is in forward, loss, or backward)");
            cudaGetLastError();
        } else {
            ctx.logging.logger->log("[CUDA] first_batch AFTER autogradTrainingStep: ok");
        }
    }
    
    // Handle training step failure (NaN/Inf loss or backward error)
    if (!loss_result.success) {
        ctx.logging.logger->log("[autogradTrainingStep] FAILED batch=" + std::to_string(batch_idx + 1) +
                                " error: " + loss_result.error_message);
        // zeroGrad removed: next autogradTrainingStep with accumulate=false zeros gradients
        ctx.optimizer.current_micro_step = 0;
        result.skipped = true;
        result.skip_reason = "autograd_step_failed";
        ctx.global_step++;
        return result;
    }
    
    // Log gradient accumulation status
    ctx.logging.logger->log("[GradAccum] batch=" + std::to_string(batch_idx + 1) +
                            " micro_step=" + std::to_string(ctx.optimizer.current_micro_step) +
                            "/" + std::to_string(accum_steps) +
                            " accumulate=" + (should_accumulate ? "true" : "false"));

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
    
    if (forward_call_count <= 3) {
        ctx.logging.logger->log("[GradTrace] POST-autogradTrainingStep call #" + std::to_string(forward_call_count) + 
                                " returned=" + std::to_string(result.loss));
        
    }
    
    // NOTE: Loss variance computation removed (was causing 5-second GPU sync bottleneck).
    // Variance is now tracked on GPU by TelemetryLattice (σ_tilde, v_σ fields).
    // Use computeTelemetryFeedback() to access grad_norm variance for adaptive decisions.
    
    ctx.logging.logger->log("[GradTrace] POST-FORWARD loss=" + Internal::formatScalar(result.loss, 4));

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
    
    ctx.logging.logger->log("[GradTrace] POST-BACKWARD batch=" + std::to_string(batch_idx + 1) + 
                            " loss=" + Internal::formatScalar(result.loss) + 
                            " valid_tokens=" + std::to_string(payload.valid_tokens));
    
    // NOTE: Gradient component logging happens later after measureGradientNorms()
    // via formatGradientComponents(). Premature logging here would use undefined variables.

    // ========================================================================
    // DIAGNOSTIC: Issue #142 - Special Token Weight & Gradient Verification
    // (extracted to Diagnostics/SpecialTokenDiagnostic.cu)
    // ========================================================================
    GRIM::Diagnostics::runSpecialTokenDiagnostic(ctx, payload, batch_idx);
    
    // NOTE: Window closes automatically via beginOptimizerStep() → endOptimizerStep()
    // State flow: ACCUMULATING → READY_FOR_STEP → IDLE
    
    // DISABLED for performance: Diagnostic causes device sync
    // if (diag_batch_count <= 3) {
    //     const auto& ts = ctx.model->getTrainingState();
    //     ctx.logging.logger->log("[GradDiag] AFTER_BACKWARD: " + sampleEmbeddingGrads(ts, ts.stream_ctrl.getPrimaryStream()));
    // }
    
    // FIX Issue #23: Sync gradient norms EVERY batch for accurate diagnostics.
    // Previously: only synced every 10th batch (batch_idx % 10 == 0), used stale cached values.
    // Evidence: Batches 611-620 all showed grad_norm=2.5877 despite actual norms varying 2.1-5.3.
    // Impact: Diagnostics showed wrong values, spike detection used stale data.
    // NOTE: Performance cost is ~1ms per batch (acceptable for correctness).
    // NOTE: After Issue #135, clipping happens LATER (in should_step block) and recomputes
    //       its own norm on scaled gradients. This norm is for diagnostics/spike detection only.

    // ========================================================================
    // Synced grad-norm measurement + [EMB_GRAD_EQUATION] diagnostic
    // (extracted to Diagnostics/GradientNormDiagnostic.cu)
    // Mutates: result.grad_rms, result.normalized_grad_rms.
    // Throws on NaN/Inf gradients (Rule 20).
    // ========================================================================
    const auto grad_norm_snap = GRIM::Diagnostics::runGradientNormDiagnostic(
        ctx, state, result, batch_idx);
    const float preclip_grad_rms = grad_norm_snap.preclip_grad_rms;
    const float enc_rms_pre = grad_norm_snap.enc_rms_pre;

    // === TIMING GUARD: Track operations between POST-BACKWARD and PRE-OPTIMIZER ===
    auto pre_optimizer_start = std::chrono::steady_clock::now();
    
    // Gradient spike handling removed (Rule 20: spikes indicate real bugs, not something to silently skip)
    
    // Telemetry control interventions removed (Rule 20: monitoring-only, crash on real bugs)
    
    // ========================================================================
    // Issue #135: Gradient clipping DEFERRED to after accumulation scaling
    //
    // OLD BUG: Clipping ran EVERY micro-batch, crushing text gradients 3x
    //   (once per micro-batch × 3 accum steps). With accum_steps=3, text
    //   gradients that were already tiny (~0.04) got clipped 3 times.
    //
    // FIX: Clipping now runs ONCE, inside should_step block, AFTER
    //   accum_scale(1/accum_steps). This means gradients are:
    //   1. Accumulated across all micro-batches (summed)
    //   2. Scaled by 1/accum_steps (averaged)
    //   3. Norm recomputed on the averaged gradients
    //   4. Clipped once against the limit
    //   5. Fed to optimizer
    // ========================================================================
    auto clipping_start = std::chrono::steady_clock::now();
    
    const float effective_per_token_limit = clip_selection.per_token_limit;
    const bool clipping_enabled = (hp.grad_clip_norm > 0.0f);
    
    // No clipping here — deferred to post-accumulation inside should_step
    auto clipping_elapsed_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - clipping_start).count();
    
    // Learning rate computation (accum_steps already computed above)
    // FIX: Use optimizer step count, NOT global_step (batch counter).
    // global_step increments every micro-batch; optimizer_state.step increments
    // only on actual weight updates. With gradient_accumulation_steps > 1,
    // using global_step makes warmup/decay advance accum_steps times too fast.
    auto lr_start = std::chrono::steady_clock::now();
    const int optimizer_step = static_cast<int>(ctx.optimizer.optimizer_state.step);
    if (!ctx.lr_schedule) {
        throw std::runtime_error("lr_schedule is not initialized at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    const float scheduled_lr = Internal::getScheduledLearningRate(
        *ctx.lr_schedule, optimizer_step, hp.learning_rate,
        ctx.config.stability.enabled);
    
    result.learning_rate = scheduled_lr;
    
    auto lr_elapsed_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - lr_start).count();
    
    // Optimizer step
    const bool sync_diag = GRIM::Diagnostics::shouldSyncDiagnostics(ctx, batch_idx);
    auto optimizer_step_start = std::chrono::steady_clock::now();
    float sample_elapsed_ms = 0.0f;
    GRIM::Diagnostics::WeightSample pre_sample{};
    std::string pre_weights = "lm_head_weights=skipped";
    if (sync_diag) {
        auto sample_start = std::chrono::steady_clock::now();
        pre_sample = GRIM::Diagnostics::sampleWeightStats(ctx.model->getLmHeadLayer(), ctx.model->getTrainingState(), true);
        if (pre_sample.valid) {
            pre_weights = GRIM::Diagnostics::formatWeightSample(pre_sample);
        }
        sample_elapsed_ms = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - sample_start).count();
    }
    
    // Log total time for pre-optimizer setup (spike handling, telemetry, clipping, LR)
    auto pre_optimizer_elapsed_ms = std::chrono::duration<float, std::milli>(
        optimizer_step_start - pre_optimizer_start).count();
    if (pre_optimizer_elapsed_ms > 1000.0f) {  // Log if > 1 second
        ctx.logging.logger->log("[PERF] Pre-optimizer setup took " + Internal::formatScalar(pre_optimizer_elapsed_ms, 2) + 
                                "ms (clipping=" + Internal::formatScalar(clipping_elapsed_ms, 2) + 
                                "ms, lr=" + Internal::formatScalar(lr_elapsed_ms, 2) + 
                                "ms, sample=" + Internal::formatScalar(sample_elapsed_ms, 2) + "ms)");
    }
    
    ctx.logging.logger->log("[GradTrace] PRE-OPTIMIZER batch=" + std::to_string(batch_idx + 1) +
                            " lr=" + Internal::formatScalar(result.learning_rate, 8) +
                            " grad_rms=" + Internal::formatScalar(result.grad_rms) +
                            " step=" + std::to_string(ctx.optimizer.optimizer_state.step) +
                            " " + pre_weights);
    
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
        
        // CRITICAL FIX: Synchronize stream BEFORE endOptimizerStep() zeros gradient buffers!
        // GradStats::flushAndLog() launches async D2H copies that read from gradient buffers.
        // If endOptimizerStep() zeros those buffers before the copies complete, GradStats
        // will read zeros instead of actual gradient values, causing corrupted stats.
        // This was the root cause of: max_abs=-1.53835e+13 (negative!), min/max inverted,
        // and unexplained zeros in fields that should contain computed values.
        cudaError_t sync_err = cudaStreamSynchronize(training_state.stream_ctrl.getPrimaryStream());
        if (sync_err != cudaSuccess) {
            std::ostringstream oss;
            oss << "[GradStats] FATAL: Failed to synchronize stream before optimizer step: "
                << cudaGetErrorString(sync_err);
            EmitModuleError(ModuleId::BackwardPass, oss.str(), ctx.global_step);
            throw std::runtime_error(oss.str());
        }
    }
    
    // Only run optimizer step when accumulation is complete
    // This enables gradient accumulation across multiple micro-batches
    // Increment micro_step BEFORE checking, since we already processed this batch's backward
    ctx.optimizer.current_micro_step++;
    const bool should_step = (ctx.optimizer.current_micro_step >= accum_steps);
    
    if (should_step) {
        const int micro_step_for_log = ctx.optimizer.current_micro_step;
        const int accum_steps_for_log = accum_steps;
        
        // ========================================================================
        // Issue #149: Zero PAD/UNK gradients before optimizer step
        //
        // PAD (id=1) and UNK (id=0) are never valid targets, yet they accumulate
        // non-zero gradients through two paths:
        //   1. LM head backward: label smoothing redistributes tiny gradient to ALL
        //      vocab rows, including PAD/UNK (via softmax Jacobian)
        //   2. Embedding backward: attention backward leaks gradient from valid
        //      positions to PAD input positions through K/V cross-attention,
        //      then scatter-adds to PAD's embedding row
        // Path 2 is dominant (~76x larger than BOS/EOS) because attention has
        // cross-position interactions even for masked-target positions.
        //
        // Zeroing these BEFORE norm computation ensures PAD/UNK don't inflate
        // gradient norms or waste optimizer capacity.
        // ========================================================================
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
        
        // ========================================================================
        // Issue #135: POST-ACCUMULATION gradient clipping
        //
        // Clipping runs ONCE on the fully accumulated + scaled gradients.
        // Recompute grad norm since accum_scale changed magnitudes.
        // Per-component clipping (Issue #139):
        //   1. emb_clip  — LM_HEAD (+ EMBEDDING if untied)
        //   2. enc_clip  — ATTENTION + FFN + RMSNORM + SCRATCHBLOCK +
        //                  MTP + REASONING_HEAD + EXECUTION_BLOCK
        //
        // Clipping operates through the ParameterGroup tensor registry
        // via GradClip::clipGradientNorms() — norm measurement, bucket
        // aggregation, and gradient scaling all happen inside GradientCC
        // against the registered tensors.
        // ========================================================================
        if (clipping_enabled) {
            clipping_start = std::chrono::steady_clock::now();
            
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
            
            ctx.logging.logger->log("[PostAccumClip] batch=" + std::to_string(batch_idx + 1) +
                                    " post_accum_rms=" + Internal::formatScalar(clip.total_rms_pre, 6) +
                                    " emb_rms=" + Internal::formatScalar(clip.emb_rms, 6) +
                                    " enc_rms=" + Internal::formatScalar(clip.enc_rms, 6) +
                                    " emb_clipped=" + (clip.emb_clipped ? "YES" : "NO") +
                                    " enc_clipped=" + (clip.enc_clipped ? "YES" : "NO") +
                                    " post_clip_total=" + Internal::formatScalar(clip.total_rms_post, 6));
            
            clipping_elapsed_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - clipping_start).count();
        }
        
        // ════════════════════════════════════════════════════════════════════
        // RUNTIME tie_embeddings pointer verification (every batch)
        // (extracted to Diagnostics/TieVerifyDiagnostic.cu)
        // ════════════════════════════════════════════════════════════════════
        GRIM::Diagnostics::runTieVerifyDiagnostic(ctx, batch_idx);

        const int emb_freeze_step = ctx.config.hyperparameters.embedding_freeze_enabled
            ? ctx.config.hyperparameters.embedding_freeze_after_step : -1;

        if (emb_freeze_step > 0 && ctx.optimizer.optimizer_state.step == emb_freeze_step) {
            if (ctx.config.architecture.tie_embeddings) {
                ctx.logging.logger->log("[EmbeddingFreeze] WARNING: tie_embeddings=true — "
                    "embedding and LM head share weights. Set tie_embeddings=false to freeze "
                    "embedding independently. Freeze has no effect on tied weights.");
            } else {
                ctx.logging.logger->log("[EmbeddingFreeze] Embedding weights FROZEN at step "
                    + std::to_string(emb_freeze_step) + " — no further embedding updates");
            }
        }

        // Optimizer dispatch — single source of truth: ctx.config.hyperparameters
        // (loaded from training.config.optimizer in ai_config.json). Kernel hyperparams
        // are passed by signature so kernels read no globals. Rule 20: kind already
        // validated at config-load time ("adamw" | "radamw" only).
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
        
        // RULE 20: Post-optimizer weight NaN spot check
        // (extracted to Diagnostics/OptimizerStepGuards.cu)
        GRIM::Diagnostics::checkPostOptimizerWeightsFinite(ctx, result, batch_idx);

        // Reset micro_step counter after optimizer step completes
        ctx.optimizer.current_micro_step = 0;

        // Periodic tape flush (every 10 optimizer steps) — sinks handle I/O
        // (tape.flush() already called at end of processBatch; this is a mid-batch safety flush)

        // ====================================================================
        // DIAGNOSTIC: Adam optimizer m/v moment-state telemetry
        // (extracted to Diagnostics/OptimizerMomentDiagnostic.cu)
        // ====================================================================
        GRIM::Diagnostics::runOptimizerMomentDiagnostic(
            ctx, batch_idx, micro_step_for_log, accum_steps_for_log, sync_diag);
        
        GRIM::Diagnostics::WeightSample post_sample{};
        std::string post_weights = "lm_head_weights=skipped";
        if (sync_diag) {
            post_sample = GRIM::Diagnostics::sampleWeightStats(ctx.model->getLmHeadLayer(), ctx.model->getTrainingState(), true);
            if (post_sample.valid) {
                post_weights = GRIM::Diagnostics::formatWeightSample(post_sample);
            }
        }
        ctx.logging.logger->log("[GradTrace] POST-OPTIMIZER batch=" + std::to_string(batch_idx + 1) + 
                                " lr=" + Internal::formatScalar(result.learning_rate, 8) +
                                " step=" + std::to_string(ctx.optimizer.optimizer_state.step) +
                                " t=" + std::to_string(ctx.optimizer.optimizer_state.step) +
                                " " + post_weights);
        
        if (pre_sample.valid && post_sample.valid) {
            const float update_rms = GRIM::Diagnostics::computeUpdateRms(pre_sample, post_sample);
            const std::string update_msg = "[UpdateMag] batch=" + std::to_string(batch_idx + 1) +
                                           " update_rms=" + Internal::formatScalar(update_rms, 8) +
                                           " param_rms=" + Internal::formatScalar(pre_sample.rms, 8);
            ctx.logging.logger->log(update_msg);
            EmitModuleInfo(ModuleId::Optimizer, update_msg, ctx.global_step);
        }
        
        // Per-component Adam update_rms diagnostic (Issue #150)
        // Answers: "Does Adam normalize the gradient gap across component types?"
        // Only on diagnostic-sync batches to avoid blocking the pipeline.
        if (sync_diag) {
            const auto update_trace = GRIM::Diagnostics::computePerComponentUpdateTrace(
                ctx.model->parameterGroups(),
                result.learning_rate,
                ctx.optimizer.optimizer_state.step + 1,  // 1-based iteration count (matches AdamW bias correction)
                ctx.model->getTrainingState().stream_ctrl.getPrimaryStream()
            );
            if (update_trace.valid) {
                const std::string trace_str = GRIM::Diagnostics::formatUpdateTrace(
                    update_trace, batch_idx + 1, ctx.model->getConfig().tie_embeddings);
                ctx.logging.logger->log(trace_str);
            }
        }
        
        ctx.optimizer.optimizer_state.step++;
    } else {
        ctx.logging.logger->log("[GradTrace] ACCUMULATING batch=" + std::to_string(batch_idx + 1) + 
                                " micro_step=" + std::to_string(ctx.optimizer.current_micro_step) +
                                " of " + std::to_string(accum_steps) +
                                " (skipping optimizer step)");
    }
    
    // Check for CUDA errors (non-blocking)
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            ctx.logging.logger->log("[CUDA ERROR] Async check: " + std::string(cudaGetErrorString(err)));
        }
    }
    
    result.sequences_processed = payload.batch_size;
    result.tokens_processed = clip_selection.stats.total_tokens;
    
    // Flush device logs on the diagnostic sync interval.
    if (sync_diag) {
        GRIM::Logging::FlushDeviceLogs();
    }
    
    // Record layer log for post-run analysis (matches old train_gpu.cu behavior)
    GRIM::Logging::RecordLayerLogHost(
        GRIM::LayerType::kEncoding,         // aggregate marker
        -1,                                 // no specific layer (-1 = global)
        static_cast<std::uint64_t>(ctx.global_step),
        result.grad_rms,                   // primary: gradient RMS
        result.normalized_grad_rms,        // secondary: per-token RMS
        "grad_rms",
        "post_backward");
    
    // Update telemetry lattice — all metric computation delegated to TelemetryUpdate.cu
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

        GRIM::Telemetry::updateTelemetryObservations(ctx, tel_input, gm, &payload);
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

    // UpdateProbe consumer deleted (Rule 26): hasUpdateProbe() never returned
    // true because update_probe_ready_ was never set anywhere. Entire probe
    // subsystem removed; PostLoss cached_logits trace at line ~1362 remains.

    // Rule 20 single-owner clear: handled by AutogradStepScope at processBatch entry.
    // Tape flush below does not read autograd intermediates.
    
    // Flush tape: sort by phase, dispatch to all sinks
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
    const int num_epochs = hp.epochs;
    
    ctx.logging.logger->log("Epoch " + std::to_string(epoch_idx + 1) + "/" + std::to_string(num_epochs));
    
    // Reset guess cache at epoch start
    if (ctx.config.hyperparameters.guess_aux_enabled) {
        GRIMTS::Training::resetGuessCacheForEpoch(
            ctx.model->getTrainingState(), state.guess_cache);
    }
    
    PHASE2_DEBUG_STDERR("[DEBUG-EPOCH] After ResetGuessCache, checking shuffle...\n");
    
    // Shuffle train catalog if enabled
    const bool shuffle_this_epoch = hp.shuffle_train_enabled &&
        (hp.shuffle_train_epochs == 0 || epoch_idx < hp.shuffle_train_epochs);
    PHASE2_DEBUG_STDERR("[DEBUG-EPOCH] shuffle_this_epoch=%d hp.shuffle_train_enabled=%d\n",
                        shuffle_this_epoch ? 1 : 0, hp.shuffle_train_enabled ? 1 : 0);
    
    if (shuffle_this_epoch) {
        PHASE2_DEBUG_STDERR("[DEBUG-EPOCH] Entering shuffle block\n");
        ctx.logging.logger->log("[Shuffle] Randomizing train catalog for epoch " + std::to_string(epoch_idx + 1));
        PHASE2_DEBUG_STDERR("[DEBUG-EPOCH] train_views.size()=%zu\n", ctx.data.train_views.size());
        
        std::vector<size_t> perm(ctx.data.train_views.size());
        for (size_t i = 0; i < perm.size(); ++i) perm[i] = i;
        std::shuffle(perm.begin(), perm.end(), ctx.rng.data_rng);
        
        std::vector<TrainingSequence*> shuffled;
        shuffled.reserve(ctx.data.train_views.size());
        GRIM::DynaSeq::Catalog shuffled_catalog;
        
        for (size_t new_idx = 0; new_idx < perm.size(); ++new_idx) {
            auto* seq = ctx.data.train_views[perm[new_idx]];
            shuffled.push_back(seq);
            const uint32_t len = static_cast<uint32_t>(seq->token_ids.size());
            shuffled_catalog.add(len, len, 0, 0, 0);
        }
        
        ctx.data.train_views.swap(shuffled);
        ctx.data.train_catalog = std::move(shuffled_catalog);
        PHASE2_DEBUG_STDERR("[DEBUG-EPOCH] Shuffle complete\n");
    }
    
    PHASE2_DEBUG_STDERR("[DEBUG-EPOCH] About to build epoch batches...\n");
    // Build batches for this epoch (per-epoch batching policy lives in
    // Shared/Batching/EpochBatching.cu).
    auto log_batching = [&](const std::string& msg) { ctx.logging.logger->log(msg); };
    auto schedule = GRIM::Batching::buildEpochBatches(
        ctx.data.train_catalog,
        hp.batch_size,
        static_cast<uint32_t>(ctx.model->getConfig().max_seq_len),
        ctx.global_step,
        epoch_idx,
        ctx.rng.data_seed,
        log_batching);
    
    const int total_batches = static_cast<int>(schedule.batches.size());
    if (ctx.estimated_total_steps == 0 && total_batches > 0) {
        // estimated_total_steps counts OPTIMIZER STEPS (not micro-batches).
        // LR schedule, warmup, and cosine decay all index by optimizer step.
        const int accum = std::max(1, hp.gradient_accumulation_steps);
        ctx.estimated_total_steps = (num_epochs * total_batches) / accum;
        
        // Derive warmup_steps from warmup_fraction now that total_steps is known.
        GRIM::Config::deriveWarmupSteps(ctx.config.hyperparameters, ctx.estimated_total_steps);
        
        // Construct the deterministic LR schedule now that total_steps is known.
        // Cosine decay spans all epochs; warm restarts are disabled.
        GRIM::LR::LRScheduleConfig lr_cfg;
        lr_cfg.base_lr = hp.learning_rate;
        lr_cfg.cosine_decay_min_lr = hp.cosine_decay_min_lr;
        lr_cfg.warmup_steps = ctx.config.hyperparameters.warmup_steps;
        lr_cfg.total_steps = ctx.estimated_total_steps;
        lr_cfg.steps_per_epoch = total_batches / accum;
        lr_cfg.cosine_decay_enabled = hp.cosine_decay_enabled;
        lr_cfg.warm_restarts = hp.cosine_warm_restarts;
        ctx.lr_schedule.emplace(lr_cfg);
        
        ctx.logging.logger->log("[LRSchedule] warmup_fraction=" + std::to_string(hp.warmup_fraction)
            + " -> warmup_steps=" + std::to_string(ctx.config.hyperparameters.warmup_steps)
            + " / total_steps=" + std::to_string(ctx.estimated_total_steps));
    }
    int total_batches_to_run = total_batches;
    const bool single_batch_overfit = hp.single_batch_overfit_enabled;
    if (single_batch_overfit) {
        if (total_batches == 0) {
            ctx.logging.logger->log("[SingleBatch] WARNING: No batches available to repeat.");
            total_batches_to_run = 0;
        } else {
            total_batches_to_run = std::max(1, hp.single_batch_overfit_max_steps);
            ctx.logging.logger->log("[SingleBatch] Enabled: repeating batch 1 for " +
                                    std::to_string(total_batches_to_run) + " steps.");
        }
    }
    const auto epoch_start = std::chrono::steady_clock::now();
    float epoch_loss = 0.0f;
    
    // Get accumulation steps from hyperparameters
    const int accum_steps = std::max(1, ctx.config.hyperparameters.gradient_accumulation_steps);
    
    // Process batches: scheduler (BatchAssignment) dictates order; we build BatchPayload and act on it
    const auto& model_cfg = ctx.model->getConfig();
    const auto token_layout = ctx.tokenizer.tokenLayout();
    const size_t max_cached_batch = static_cast<size_t>(std::max(1, model_cfg.max_cached_batch));
    const size_t max_cached_seq = static_cast<size_t>(std::max(1, std::min(model_cfg.max_seq_len, model_cfg.max_cached_seq_len)));

    for (int batch_idx = 0; batch_idx < total_batches_to_run; ++batch_idx) {
        const auto& assignment = schedule.batches[single_batch_overfit ? 0 : batch_idx];
        GRIM::Batching::BatchPayload payload = GRIM::Batching::buildBatchPayload(
            assignment, ctx.data.train_views, ctx.config.actual_vocab_size,
            token_layout, max_cached_batch, max_cached_seq,
            model_cfg.execution_block_num_slots,
            model_cfg.execution_block_num_ops,
            model_cfg.execution_block_num_steps,
            model_cfg.mtp_enabled ? model_cfg.mtp_k : 0);

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
        
        // After first batch: sync and surface any CUDA error so we see the real fault
        // (otherwise it only appears later as cudaFree failures during teardown)
        if (batch_idx == 0 && !batch_result.skipped) {
            cudaError_t sync_err = cudaDeviceSynchronize();
            cudaError_t last_err = (sync_err != cudaSuccess) ? sync_err : cudaGetLastError();
            if (last_err != cudaSuccess) {
                ctx.logging.logger->log("[CUDA] ERROR after first batch: " + std::string(cudaGetErrorString(last_err)) +
                    " (sync=" + (sync_err != cudaSuccess ? "failed" : "ok") + "). "
                    "Fix this to avoid cudaFree 'illegal memory access' during teardown.");
                cudaGetLastError(); // clear so subsequent code can run if desired
            }
        }
        
        if (batch_result.skipped) {
            result.batches_skipped++;
            ctx.logging.logger->log("[Batch " + std::to_string(batch_idx + 1) + 
                                    "] skipped (" + batch_result.skip_reason + ")");
            continue;
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
            // ================================================================
            // DIAGNOSTIC: Multi-Token-Prediction per-head telemetry + monitor
            // (extracted to Diagnostics/MtpDiagnostic.cu)
            // ================================================================
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
    
    result.avg_loss = epoch_loss / std::max(result.batches_processed, 1);
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
    
    // Update status - GPU memory query at epoch end only (not per-batch)
    // cudaMemGetInfo is implicitly synchronizing on some drivers, but acceptable
    // at epoch granularity (once per ~100-1000 batches)
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

    // Construct the central loss-signal detector. Wire the validation policy
    // (auto-stop high-loss) from hyperparameters NOW so evaluateAutoStop can
    // subscribe to signals.validation_high directly. Other thresholds (smoothed
    // sigma, baseline multiplier, etc.) stay defaulted; task 7 will plumb the
    // full ai_config.json "loss_signals" block through HyperParameters.
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
        ctx.config.cuda_exec.single_stream_mode,
        ctx.global_step,
        state.guess_cache);
    
    PHASE2_DEBUG_STDERR("[DEBUG] About to initialize training log...");

    // Log configuration
    EmitModuleInfo(ModuleId::Training, "Starting training...", ctx.global_step);
    EmitModuleInfo(ModuleId::Training, std::string("  Warmup fraction: ") + std::to_string(hp.warmup_fraction) + " (steps derived after batch count known)", ctx.global_step);
    EmitModuleInfo(ModuleId::Training, std::string("  Target learning rate: ") + std::to_string(hp.learning_rate), ctx.global_step);
    if (hp.cosine_decay_enabled) {
        EmitModuleInfo(ModuleId::Training,
            std::string("  Cosine decay: enabled, min_lr=") + std::to_string(hp.cosine_decay_min_lr), ctx.global_step);
    }
    EmitModuleInfo(ModuleId::Training, 
        std::string("  Soft restart: ") + (hp.soft_restart_enabled ? "enabled" : "disabled"), ctx.global_step);
    EmitModuleInfo(ModuleId::Training, 
        std::string("  Auto-stop: ") + (hp.auto_stop_enabled ? "enabled" : "disabled"), ctx.global_step);
    if (hp.embedding_freeze_enabled) {
        EmitModuleInfo(ModuleId::Training,
            std::string("  Embedding freeze: after step ") + std::to_string(hp.embedding_freeze_after_step)
            + (ctx.config.architecture.tie_embeddings ? " (WARNING: tie_embeddings=true, set to false for freeze to take effect)" : ""),
            ctx.global_step);
    }
    
    const int num_epochs = std::max(1, hp.epochs);
    EmitModuleInfo(ModuleId::Training,
        std::string("Total epochs to run: ") + std::to_string(num_epochs), ctx.global_step);
    
    // Reconstruct adam_cumulative_disp = Σlr(0..optimizer_step-1) for checkpoint resume correctness.
    // On fresh start this is a no-op. On resume, it reconstructs the exact cumulative
    // displacement by iterating over actual optimizer steps (not micro-batches).
    // FIX: Used to loop over ctx.global_step (batch counter), overcounting by
    // gradient_accumulation_steps. Now loops over optimizer_state.step.
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
