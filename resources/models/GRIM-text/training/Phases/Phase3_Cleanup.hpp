//======================================================//
//  Phase3_Cleanup.hpp
//  Final cleanup, saves, and status reporting
//======================================================//
//
//  HEADER
//  ======
//  This file defines cleanup operations performed
//  after training completes:
//  1. Final model save
//  2. Status file updates
//  3. Resource cleanup
//  4. Final metrics reporting
//
//  Author: Austin Wadkins
//  Date: December 2025
//  Version: 1.0.0
//======================================================//

#pragma once

#include "Phase1_Startup.hpp"

#include <string>
#include <chrono>
#include <optional>

namespace GRIMText::Training {

//======================================================//
//  Cleanup Configuration
//======================================================//

struct CleanupConfig {
    bool save_final_model = true;
    bool save_optimizer_state = true;
    bool write_final_status = true;
    bool cleanup_temporary_files = false;
    bool generate_summary_report = true;
    std::string final_model_suffix = "_final";
};

//======================================================//
//  Training Summary
//======================================================//

struct TrainingSummary {
    // Timing
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;
    double total_duration_seconds = 0.0;
    
    // Training stats
    int epochs_completed = 0;
    int total_batches_processed = 0;
    int total_steps = 0;
    
    // Loss metrics
    float initial_loss = 0.0f;
    float final_loss = 0.0f;
    float best_loss = std::numeric_limits<float>::infinity();
    int best_loss_epoch = -1;
    
    // Validation metrics
    float final_val_loss = 0.0f;
    float best_val_loss = std::numeric_limits<float>::infinity();
    float final_perplexity = 0.0f;
    float best_perplexity = std::numeric_limits<float>::infinity();
    int best_val_epoch = -1;
    
    // Auto-stop info
    bool auto_stopped = false;
    std::string auto_stop_reason;
    int auto_stop_epoch = -1;
    
    // Resource usage
    float peak_gpu_memory_mb = 0.0f;
    float avg_batch_time_ms = 0.0f;
    float throughput_tokens_per_sec = 0.0f;
};

//======================================================//
//  Cleanup Result
//======================================================//

struct CleanupResult {
    bool success = false;
    bool final_model_saved = false;
    bool optimizer_state_saved = false;
    bool status_written = false;
    std::string final_model_path;
    std::string error_message;
    
    TrainingSummary summary;
};

//======================================================//
//  Public API
//======================================================//

/**
 * Execute Phase 3: Cleanup and finalization
 * 
 * @param ctx The training context from Phase 1/2
 * @param cleanup_config Optional cleanup configuration
 * @return CleanupResult with summary and status
 */
CleanupResult executePhase3(
    TrainingContext& ctx,
    const CleanupConfig& cleanup_config = CleanupConfig{});

/**
 * Save the final model checkpoint
 * 
 * @param ctx Training context
 * @param suffix Suffix for the checkpoint filename
 * @return Path to saved checkpoint, or empty on failure
 */
std::string saveFinalModel(
    TrainingContext& ctx,
    const std::string& suffix = "_final");

/**
 * Generate and log training summary
 * 
 * @param ctx Training context
 * @param summary Pre-computed summary (if available)
 */
void logTrainingSummary(
    TrainingContext& ctx,
    const TrainingSummary& summary);

/**
 * Write final status to FlatBuffer file
 * 
 * @param ctx Training context
 * @param summary Training summary
 * @param success Whether training succeeded
 */
void writeFinalStatus(
    TrainingContext& ctx,
    const TrainingSummary& summary,
    bool success);

/**
 * Release all GPU and CPU resources
 * 
 * @param ctx Training context to cleanup
 */
void releaseResources(TrainingContext& ctx);

//======================================================//
//  Internal Helpers
//======================================================//

namespace Internal {

/**
 * Compute final training metrics summary
 */
TrainingSummary computeTrainingSummary(const TrainingContext& ctx);

/**
 * Format duration as human-readable string
 */
std::string formatDuration(double seconds);

/**
 * Get GPU memory statistics
 */
std::pair<float, float> getGPUMemoryStats();

/**
 * Clean up temporary files if requested
 */
void cleanupTemporaryFiles(const PathConfig& paths);

} // namespace Internal

} // namespace GRIMText::Training
