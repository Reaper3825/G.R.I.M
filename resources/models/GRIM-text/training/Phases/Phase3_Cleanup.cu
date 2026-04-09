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
#include "../OptimizerCheckpoint.hpp"

#include "../../Shared/LogRecorder/LogRecorder.hpp"
#include "../../Shared/EquationLogging/EquationLogging.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

using GRIM::Logging::ModuleId;
using GRIM::Logging::EmitModuleInfo;
using GRIM::Logging::EmitModuleWarning;
using GRIM::Logging::EmitModuleError;

namespace GRIMText::Training {

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
    float used_mb = static_cast<float>((total_mem - free_mem) / (1024 * 1024));
    float total_mb = static_cast<float>(total_mem / (1024 * 1024));
    return {used_mb, total_mb};
}

void cleanupTemporaryFiles(const PathConfig& paths) {
    // Remove any temporary checkpoint files
    try {
        for (const auto& entry : fs::directory_iterator(paths.checkpoint_dir)) {
            if (entry.path().extension() == ".tmp") {
                fs::remove(entry.path());
            }
        }
    } catch (const std::exception&) {
        // Ignore cleanup errors
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
    
    auto [gpu_used, gpu_total] = getGPUMemoryStats();
    (void)gpu_total;
    summary.peak_gpu_memory_mb = gpu_used;
    
    if (summary.total_steps > 0 && summary.total_duration_seconds > 0) {
        summary.avg_batch_time_ms = static_cast<float>(
            (summary.total_duration_seconds * 1000.0) / summary.total_steps);
    }
    
    return summary;
}

} // namespace Internal

//======================================================//
//  Final Model Save
//======================================================//

std::string saveFinalModel(TrainingContext& ctx, const std::string& suffix) {
    EmitModuleInfo(ModuleId::Checkpoint, "Saving final model...", ctx.global_step);
    
    std::string final_path = ctx.config.paths.checkpoint_dir + "/checkpoint" + suffix + ".bin";
    
#ifdef USE_CUDA
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        EmitModuleWarning(ModuleId::Checkpoint, 
            std::string("CUDA sync warning: ") + cudaGetErrorString(sync_err), ctx.global_step);
    }
#endif
    
    try {
        bool save_result = ctx.model->save(final_path);
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
            return final_path;
        } else {
            EmitModuleError(ModuleId::Checkpoint, "Save returned false", ctx.global_step);
        }
    } catch (const std::exception& e) {
        EmitModuleError(ModuleId::Checkpoint, 
            std::string("Exception during save: ") + e.what(), ctx.global_step);
    }
    
    return "";
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

#ifdef USE_CUDA
    // Clear autograd intermediates before destroying model so all grad_fns and layer
    // intermediates are released in a controlled order (avoids relying only on destructor order).
    if (ctx.model) {
        ctx.model->getTrainingState().autograd_intermediates.clear();
        EmitModuleInfo(ModuleId::Training, "✓ Autograd intermediates cleared", ctx.global_step);
    }
#endif

    // Release model (TrainingState destructor frees all Tensor buffers, PBM, TeacherLogits, etc.)
    if (ctx.model) {
        ctx.model.reset();
        EmitModuleInfo(ModuleId::Training, "✓ Model released", ctx.global_step);
    }
    
    // Clear training data
    ctx.data.train_seqs.clear();
    ctx.data.val_seqs.clear();
    ctx.data.train_views.clear();
    ctx.data.val_views.clear();
    ctx.data.train_catalog = GRIM::DynaSeq::Catalog{};
    ctx.data.val_catalog = GRIM::DynaSeq::Catalog{};
    EmitModuleInfo(ModuleId::Training, "✓ Training data released", ctx.global_step);
    
    // Note: Gradient accumulation now tracked via current_micro_step counter in OptimizerContext
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
            Internal::cleanupTemporaryFiles(ctx.config.paths);
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
        
        // Final summary entry to equation log
        {
            std::ostringstream eq;
            eq << "[TRAINING_COMPLETE] final_loss = avg(batch_losses)\n";
            eq << "  batches_completed=" << ctx.global_step << "\n";
            eq << "  training_completed_successfully\n";
            EQ_LOG("TRAINING_COMPLETE", eq.str(),
                   static_cast<int>(ctx.global_step), -1, static_cast<int>(ctx.global_step),
                   GRIM::EquationPhase::LOSS_COMPUTATION);
        }
        
        GRIM::getEquationLogger().shutdown();
        EmitModuleInfo(ModuleId::Training, "✓ Device logs flushed", ctx.global_step);
        EmitModuleInfo(ModuleId::Training, "✓ Equation logs flushed", ctx.global_step);
        
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
