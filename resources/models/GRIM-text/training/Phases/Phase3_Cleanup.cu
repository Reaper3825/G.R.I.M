//======================================================//
//  Phase3_Cleanup.cu
//  Final cleanup, saves, and status reporting
//======================================================//
//
//  IMPLEMENTATION
//  ==============
//  This file implements cleanup operations:
//  1. Final model checkpoint save
//  2. Status file updates
//  3. Resource cleanup (GPU/CPU)
//  4. Training summary generation
//
//  Author: Austin Wadkins
//  Date: December 2025
//  Version: 1.0.0
//======================================================//

#include "Phase3_Cleanup.hpp"
#include "Phase2_TrainingLoop.hpp"
#include "../OptimizerCheckpoint.hpp"
#include "../LoRACheckpointLifecycle.hpp"

#include "../../Common/ParameterCheckpoint.hpp"
#include "../../Shared/LogRecorder/LogRecorder.hpp"
#include "../../Shared/LogRecorder/BatchLogTape.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../Shared/Telemetry/TelemetryUpdate.hpp"
#include "../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../../../../../control/training_control_generated.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <filesystem>
#include <limits>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

using GRIM::Logging::ModuleId;
using GRIM::Logging::EmitModuleInfo;
using GRIM::Logging::EmitModuleWarning;
using GRIM::Logging::EmitModuleError;

namespace GRIMText::Training {
namespace fs = std::filesystem;

//======================================================//
//  Internal Helpers
//======================================================//

namespace Internal {

std::string formatDuration(double seconds) {
    if (seconds < 60.0) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << seconds << "s";
        return oss.str();
    }
    
    int total_secs = static_cast<int>(seconds);
    int hours = total_secs / 3600;
    int minutes = (total_secs % 3600) / 60;
    int secs = total_secs % 60;
    
    std::ostringstream oss;
    if (hours > 0) {
        oss << hours << "h " << minutes << "m " << secs << "s";
    } else if (minutes > 0) {
        oss << minutes << "m " << secs << "s";
    } else {
        oss << secs << "s";
    }
    return oss.str();
}

std::pair<float, float> getGPUMemoryStats() {
    size_t free_mem = 0, total_mem = 0;
#ifdef USE_CUDA
    cudaMemGetInfo(&free_mem, &total_mem);
#endif
    float used_mb = static_cast<float>(total_mem - free_mem) / (1024.0f * 1024.0f);
    float total_mb = static_cast<float>(total_mem) / (1024.0f * 1024.0f);
    return {used_mb, total_mb};
}

void cleanupTemporaryFiles(const ::GRIM::Config::AiConfigSnapshot& config) {
    const auto paths_hp = ::GRIM::HyperParameters::pathsHP(config);
    // Remove any temporary checkpoint files
    try {
        for (const auto& entry : fs::directory_iterator(paths_hp.checkpoint_dir)) {
            if (entry.path().extension() == ".tmp") {
                fs::remove(entry.path());
            }
        }
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("cleanupTemporaryFiles failed: ") + e.what());
    }
}

TrainingSummary computeTrainingSummary(const TrainingContext& ctx) {
    TrainingSummary summary;
    
    summary.end_time = std::chrono::steady_clock::now();
    summary.start_time = ctx.start_time;
    summary.total_duration_seconds = std::chrono::duration<double>(
        summary.end_time - summary.start_time).count();
    
    summary.total_steps = ctx.global_step;
    summary.epochs_completed = ctx.epochs_completed;
    summary.total_batches_processed = ctx.global_step;
    summary.best_val_loss = ctx.best_val_loss;
    summary.best_perplexity = (std::isfinite(ctx.best_val_loss) && ctx.best_val_loss < 50.0f)
        ? std::exp(ctx.best_val_loss)
        : std::numeric_limits<float>::infinity();
    
    summary.auto_stopped = ctx.auto_stop_triggered;
    summary.auto_stop_reason = ctx.auto_stop_reason;
    summary.auto_stop_epoch = ctx.auto_stop_epoch;
    
    // Peak GPU memory: prefer the high-water mark tracked across the run (Phase2
    // samples it each batch). getGPUMemoryStats() here is a cleanup-time snapshot
    // taken AFTER most buffers were freed, so it understates the true peak — only
    // fall back to it if nothing was sampled (e.g. a run that never trained).
    if (ctx.peak_gpu_used_bytes > 0) {
        summary.peak_gpu_memory_mb = static_cast<float>(
            static_cast<double>(ctx.peak_gpu_used_bytes) / (1024.0 * 1024.0));
    } else {
        auto [gpu_used, gpu_total] = getGPUMemoryStats();
        (void)gpu_total;
        summary.peak_gpu_memory_mb = gpu_used;
    }

    if (summary.total_steps > 0 && summary.total_duration_seconds > 0) {
        summary.avg_batch_time_ms = static_cast<float>(
            (summary.total_duration_seconds * 1000.0) / summary.total_steps);
    }
    
    return summary;
}

} // namespace Internal

namespace {

GRIM::Checkpoint::LatestCurriculumCompletionRecord
latestCurriculumCompletionForSave(
    const TrainingContext& ctx,
    int epochs_completed_this_run)
{
    if (!ctx.current_curriculum_metadata) {
        throw std::runtime_error(
            "Checkpoint save has no current curriculum metadata");
    }
    if (epochs_completed_this_run < 0) {
        throw std::runtime_error(
            "Checkpoint save received a negative completed-epoch count");
    }
    const std::uint64_t run_epochs =
        static_cast<std::uint64_t>(epochs_completed_this_run);
    if (run_epochs >
        std::numeric_limits<std::uint64_t>::max() -
            ctx.curriculum_epochs_completed_at_start) {
        throw std::runtime_error(
            "Checkpoint curriculum completed-epoch count overflow");
    }

    GRIM::Checkpoint::LatestCurriculumCompletionRecord latest;
    latest.curriculum = *ctx.current_curriculum_metadata;
    latest.epochs_completed =
        ctx.curriculum_epochs_completed_at_start + run_epochs;
    return latest;
}

} // namespace

//======================================================//
//  Final Model Save
//======================================================//

std::string saveFinalModel(TrainingContext& ctx, const std::string& suffix) {
    EmitModuleInfo(ModuleId::Checkpoint, "Saving final model...", ctx.global_step);
    const auto paths_hp = ::GRIM::HyperParameters::pathsHP(ctx.config);
    
    std::string final_path = paths_hp.checkpoint_dir + "/checkpoint" + suffix + ".grimckpt";
    
#ifdef USE_CUDA
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        EmitModuleWarning(ModuleId::Checkpoint, 
            std::string("CUDA sync warning: ") + cudaGetErrorString(sync_err), ctx.global_step);
    }
#endif
    
    try {
        const auto latest_curriculum_completion =
            latestCurriculumCompletionForSave(ctx, ctx.epochs_completed);
        bool save_result = GRIM::Checkpoint::saveParameterCheckpoint(
            ctx.config,
            ctx.parameter_registry,
            ctx.requireTrainingState("saveFinalModel").stream_ctrl.getPrimaryStream(),
            final_path,
            latest_curriculum_completion);
        if (save_result) {
            EmitModuleInfo(ModuleId::Checkpoint, 
                "✓ Final model saved: " + final_path, ctx.global_step);
            
            if (fs::exists(final_path)) {
                auto file_size = fs::file_size(final_path);
                EmitModuleInfo(ModuleId::Checkpoint, 
                    "File size: " + std::to_string(file_size / (1024*1024)) + " MB", ctx.global_step);
            }
            // Save optimizer sidecar (.opt) alongside final checkpoint
            try {
                std::string opt_path = optimizerSidecarPath(final_path);
                saveOptimizerState(ctx, opt_path);
            } catch (const std::exception& e) {
                EmitModuleWarning(ModuleId::Checkpoint,
                    std::string("Optimizer state save failed: ") + e.what(), ctx.global_step);
            }
            saveLoRATrainingCheckpointAtBoundary(
                ctx, "final", ctx.epochs_completed);
            return final_path;
        } else {
            EmitModuleError(ModuleId::Checkpoint, "Save returned false", ctx.global_step);
        }
    } catch (const std::exception& e) {
        EmitModuleError(ModuleId::Checkpoint, 
            std::string("Exception during save: ") + e.what(), ctx.global_step);
        if (ctx.lora_checkpoint.active) {
            throw;
        }
    }
    
    return "";
}

//======================================================//
//  Epoch Outcome Finalization
//======================================================//

namespace {

void evaluateAutoStop(
    TrainingContext& ctx,
    TrainingLoopState& state,
    EpochResult& result,
    int epoch_idx)
{
    const auto auto_stop_hp =
        ::GRIM::HyperParameters::autoStopHP(ctx.config);

    if (!auto_stop_hp.enabled || ctx.auto_stop_triggered) {
        return;
    }

    const float prev_best = ctx.best_val_loss;
    bool significant_improvement = true;
    if (std::isfinite(prev_best)) {
        significant_improvement =
            (prev_best - result.validation.loss) >
            auto_stop_hp.plateau_min_delta;
    }

    auto trip = [&](const char* reason) {
        ctx.auto_stop_triggered = true;
        ctx.auto_stop_reason = reason;
        ctx.auto_stop_epoch = epoch_idx + 1;
        ctx.auto_stop_metric = result.validation.loss;
        result.auto_stop_triggered = true;
        result.auto_stop_reason = reason;
    };

    if (auto_stop_hp.plateau_patience > 0) {
        if (significant_improvement) {
            state.plateau_epochs_without_improvement = 0;
        } else {
            state.plateau_epochs_without_improvement++;
            if (state.plateau_epochs_without_improvement >=
                auto_stop_hp.plateau_patience) {
                trip("plateau");
            }
        }
    }

    if (!ctx.auto_stop_triggered &&
        auto_stop_hp.high_loss_patience > 0) {
        if (state.loss_signals->latest().validation_high) {
            trip("high_loss");
        }
    }
}

bool saveBestCheckpoint(
    TrainingContext& ctx,
    float val_loss,
    int epoch)
{
    if (val_loss >= ctx.best_val_loss) {
        return false;
    }

    ctx.best_val_loss = val_loss;
    ctx.logging.logger->log("✓ New best! Saving checkpoint...");

    const auto paths_hp = ::GRIM::HyperParameters::pathsHP(ctx.config);
    std::string checkpoint_path = paths_hp.checkpoint_dir +
                                  "/checkpoint_epoch_" + std::to_string(epoch + 1) + ".grimckpt";
    try {
        const auto latest_curriculum_completion =
            latestCurriculumCompletionForSave(ctx, epoch + 1);
        bool save_result = GRIM::Checkpoint::saveParameterCheckpoint(
            ctx.config,
            ctx.parameter_registry,
            ctx.requireTrainingState("saveBestCheckpoint").stream_ctrl.getPrimaryStream(),
            checkpoint_path,
            latest_curriculum_completion);
        if (save_result) {
            ctx.logging.logger->log("  ✓ Checkpoint saved: " + checkpoint_path);
            if (fs::exists(checkpoint_path)) {
                auto file_size = fs::file_size(checkpoint_path);
                ctx.logging.logger->log("  File size: " + std::to_string(file_size / (1024 * 1024)) + " MB");
            }

            try {
                std::string opt_path = optimizerSidecarPath(checkpoint_path);
                saveOptimizerState(ctx, opt_path);
            } catch (const std::exception& e) {
                ctx.logging.logger->log(std::string("  ⚠ Optimizer state save failed: ") + e.what());
            }
            saveLoRATrainingCheckpointAtBoundary(
                ctx, "best_epoch_" + std::to_string(epoch + 1), epoch + 1);
            return true;
        }
        ctx.logging.logger->log("  ✗ Save returned false");
    } catch (const std::exception& e) {
        ctx.logging.logger->log(std::string("  ✗ Exception: ") + e.what());
        if (ctx.lora_checkpoint.active) {
            throw;
        }
    }

    return false;
}

} // namespace

void finalizeEpochOutcome(
    TrainingContext& ctx,
    TrainingLoopState& state,
    EpochResult& result,
    int epoch_idx,
    float epoch_loss,
    std::chrono::steady_clock::time_point epoch_start)
{
    result.avg_loss = epoch_loss / result.batches_processed;
    result.duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - epoch_start);

    ctx.logging.logger->log("[Epoch " + std::to_string(epoch_idx + 1) + "] " +
                            Internal::formatMetric("avg_loss", result.avg_loss));

    // Peak GPU memory high-water mark across the run so far, sampled each batch
    // in Phase2 to reflect params + grads + activations + optimizer state.
    if (ctx.peak_gpu_used_bytes > 0) {
        const double mib = static_cast<double>(ctx.peak_gpu_used_bytes) / (1024.0 * 1024.0);
        const double gib = mib / 1024.0;
        std::ostringstream peak_line;
        peak_line << "[Epoch " << (epoch_idx + 1) << "] peak_gpu_used = "
                  << ctx.peak_gpu_used_bytes << " B ("
                  << std::fixed << std::setprecision(2) << mib << " MiB, "
                  << std::setprecision(3) << gib << " GiB";
        if (ctx.gpu_total_bytes > 0) {
            const double total_gib =
                static_cast<double>(ctx.gpu_total_bytes) / (1024.0 * 1024.0 * 1024.0);
            const double pct = 100.0 * static_cast<double>(ctx.peak_gpu_used_bytes) /
                               static_cast<double>(ctx.gpu_total_bytes);
            peak_line << ", " << std::setprecision(1) << pct << "% of "
                      << std::setprecision(2) << total_gib << " GiB total";
        }
        peak_line << ")";
        ctx.logging.logger->log(peak_line.str());
    }

    GRIM::Telemetry::logTelemetrySummary(ctx);

    if (std::isfinite(result.validation.loss)) {
        state.loss_signals->recordValidation(epoch_idx, result.validation.loss);
    }

    evaluateAutoStop(ctx, state, result, epoch_idx);

    result.validation.is_best =
        saveBestCheckpoint(ctx, result.validation.loss, epoch_idx);
}

void writeTrainingProgressStatus(
    TrainingContext& ctx,
    int epoch_idx,
    int num_epochs,
    int batch_idx,
    int total_batches,
    float batch_loss,
    float epoch_loss,
    int batches_processed)
{
    const float current_avg_loss = epoch_loss / batches_processed;
    const float train_perplexity =
        (std::isfinite(current_avg_loss) && current_avg_loss < 50.0f)
            ? std::exp(current_avg_loss)
            : std::numeric_limits<float>::infinity();

    ctx.logging.status_writer->writeStatus(
        GRIMText::Control::TrainingState_Training,
        epoch_idx + 1, num_epochs,
        batch_idx + 1, total_batches,
        batch_loss, current_avg_loss,
        train_perplexity, 0.0f, 0.0f, 0.0f,
        "Training epoch " + std::to_string(epoch_idx + 1) +
            " batch " + std::to_string(batch_idx + 1));
}

void writeTrainingErrorStatus(
    TrainingContext& ctx,
    int num_epochs,
    const std::string& error)
{
    ctx.logging.status_writer->writeStatus(
        GRIMText::Control::TrainingState_Error,
        0, num_epochs, 0, 0,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        "Training error", error);
}

//======================================================//
//  Training Summary Logging
//======================================================//

void logTrainingSummary(TrainingContext& ctx, const TrainingSummary& summary) {
    EmitModuleInfo(ModuleId::Training, "╔══════════════════════════════════════╗", ctx.global_step);
    EmitModuleInfo(ModuleId::Training, "║        TRAINING SUMMARY              ║", ctx.global_step);
    EmitModuleInfo(ModuleId::Training, "╠══════════════════════════════════════╣", ctx.global_step);
    
    // Duration
    EmitModuleInfo(ModuleId::Training, 
        "║  Duration: " + Internal::formatDuration(summary.total_duration_seconds), ctx.global_step);
    
    // Steps
    EmitModuleInfo(ModuleId::Training, 
        "║  Total steps: " + std::to_string(summary.total_steps), ctx.global_step);
    
    // Best metrics
    std::ostringstream loss_line;
    loss_line << "║  Best val loss: " << std::fixed << std::setprecision(4) 
              << summary.best_val_loss;
    EmitModuleInfo(ModuleId::Training, loss_line.str(), ctx.global_step);
    
    std::ostringstream ppl_line;
    ppl_line << "║  Best perplexity: " << std::fixed << std::setprecision(2)
             << summary.best_perplexity;
    EmitModuleInfo(ModuleId::Training, ppl_line.str(), ctx.global_step);
    
    // Auto-stop info
    if (summary.auto_stopped) {
        EmitModuleInfo(ModuleId::Training, 
            "║  Auto-stop: " + summary.auto_stop_reason + 
            " (epoch " + std::to_string(summary.auto_stop_epoch) + ")", ctx.global_step);
    }
    
    // GPU memory
    std::ostringstream gpu_line;
    gpu_line << "║  Peak GPU memory: " << std::fixed << std::setprecision(0)
             << summary.peak_gpu_memory_mb << " MB";
    EmitModuleInfo(ModuleId::Training, gpu_line.str(), ctx.global_step);
    
    // Throughput
    if (summary.avg_batch_time_ms > 0) {
        std::ostringstream time_line;
        time_line << "║  Avg batch time: " << std::fixed << std::setprecision(1)
                  << summary.avg_batch_time_ms << " ms";
        EmitModuleInfo(ModuleId::Training, time_line.str(), ctx.global_step);
    }
    
    EmitModuleInfo(ModuleId::Training, "╚══════════════════════════════════════╝", ctx.global_step);
}

//======================================================//
//  Final Status Writing
//======================================================//

void writeFinalStatus(
    TrainingContext& ctx,
    const TrainingSummary& summary,
    bool success) {
    
    GRIMText::Control::TrainingState state = success
        ? GRIMText::Control::TrainingState_Completed
        : GRIMText::Control::TrainingState_Error;
    
    std::string message = success
        ? "Training completed successfully"
        : "Training ended with errors";
    
    if (summary.auto_stopped) {
        message = "Training auto-stopped: " + summary.auto_stop_reason;
        // Use Completed state for auto-stop (no separate AutoStopped enum)
        state = GRIMText::Control::TrainingState_Completed;
    }
    
    auto [gpu_used, gpu_total] = Internal::getGPUMemoryStats();
    
    ctx.logging.status_writer->writeStatus(
        state,
        summary.epochs_completed, summary.epochs_completed,
        summary.total_batches_processed, summary.total_batches_processed,
        summary.final_val_loss, summary.final_loss,
        summary.final_perplexity, summary.best_perplexity,
        gpu_used, gpu_total,
        message);
    
    EmitModuleInfo(ModuleId::Training, "Final status written to training_status.fb", ctx.global_step);
    }

//======================================================//
//  Resource Release
//======================================================//

void releaseResources(TrainingContext& ctx) {
    EmitModuleInfo(ModuleId::Training, "Releasing resources...", ctx.global_step);
    
    // Clear training data
    ctx.data.train_seqs.clear();
    ctx.data.val_seqs.clear();
    ctx.data.train_views.clear();
    ctx.data.val_views.clear();
    ctx.data.train_seq_lengths.clear();
    ctx.data.val_seq_lengths.clear();
    EmitModuleInfo(ModuleId::Training, "✓ Training data released", ctx.global_step);
    
    // Note: Gradient accumulation is tracked by OptimizerContext's accumulation-slot owner.
    // No separate grad_controller to release
    EmitModuleInfo(ModuleId::Training, "✓ Gradient state released", ctx.global_step);
    
#ifdef USE_CUDA
    // Final CUDA synchronization
    cudaDeviceSynchronize();
    
    // Release module-static autograd resources (cleanup stream + cuBLAS handle)
    shutdownAutogradResources();
    
    EmitModuleInfo(ModuleId::Training, "✓ CUDA synchronized", ctx.global_step);
#endif
}

//======================================================//
//  Phase3 Main Entry Point
//======================================================//

CleanupResult executePhase3(
    TrainingContext& ctx,
    const CleanupConfig& cleanup_config) {
    
    EmitModuleInfo(ModuleId::Training, "========================================", ctx.global_step);
    EmitModuleInfo(ModuleId::Training, "  Phase 3: Cleanup and Finalization", ctx.global_step);
    EmitModuleInfo(ModuleId::Training, "========================================", ctx.global_step);
    
    CleanupResult result;
    
    try {
        // Compute summary
        result.summary = Internal::computeTrainingSummary(ctx);
        
        // Save final model
        if (cleanup_config.save_final_model) {
            result.final_model_path = saveFinalModel(ctx, cleanup_config.final_model_suffix);
            result.final_model_saved = !result.final_model_path.empty();
        }
        
        // Generate and log summary
        if (cleanup_config.generate_summary_report) {
            logTrainingSummary(ctx, result.summary);
        }
        
        // Write final status
        if (cleanup_config.write_final_status) {
            writeFinalStatus(ctx, result.summary, true);
            result.status_written = true;
        }
        
        // Cleanup temporary files
        if (cleanup_config.cleanup_temporary_files) {
            Internal::cleanupTemporaryFiles(ctx.config);
            EmitModuleInfo(ModuleId::Training, "✓ Temporary files cleaned", ctx.global_step);
        }
        
        // Release resources
        releaseResources(ctx);

        // Flush telemetry CSV before shutdown
        if (ctx.telemetry.csv_logger) {
            ctx.telemetry.csv_logger->flush();
        }

        // Flush all pending device logs to disk before exit
        EmitModuleInfo(ModuleId::Training, "Flushing device logs...", ctx.global_step);
        GRIM::Logging::FlushDeviceLogs();
        GRIM::Logging::ShutdownLogRecorder();
        
        // Final summary entry to tape
        if (ctx.logging.tape) {
            GRIM::Logging::LogEntry entry{};
            entry.level = GRIM::Logging::LogLevel::Info;
            entry.group = GRIM::Logging::LogGroup::System;
            entry.phase = GRIM::Logging::LogPhase::LIFECYCLE;
            entry.layer_idx = -1;
            entry.global_step = static_cast<int>(ctx.global_step);
            entry.batch_idx = -1;
            entry.setTag("TRAINING_COMPLETE");
            std::ostringstream eq;
            eq << "final_loss=avg(batch_losses) batches_completed=" << ctx.global_step;
            entry.setMessage("%s", eq.str().c_str());
            ctx.logging.tape->emitImmediate(entry);
            
            // Final flush and shutdown
            ctx.logging.tape->flush();
            ctx.logging.tape->flushSinks();
            GRIM::Logging::setGlobalTape(nullptr);
        }
        
        EmitModuleInfo(ModuleId::Training, "✓ Logs flushed", ctx.global_step);
        
        result.success = true;
        
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
        EmitModuleError(ModuleId::Training, 
            std::string("CLEANUP ERROR: ") + e.what(), ctx.global_step);
        
        // Still try to write error status
        ctx.logging.status_writer->writeStatus(
            GRIMText::Control::TrainingState_Error,
            0, 0, 0, 0,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            "Cleanup error", e.what());
    }
    
    EmitModuleInfo(ModuleId::Training, "========================================", ctx.global_step);
    EmitModuleInfo(ModuleId::Training, "  Phase 3 Complete", ctx.global_step);
    EmitModuleInfo(ModuleId::Training, "========================================", ctx.global_step);
    
    return result;
}

} // namespace GRIMText::Training
