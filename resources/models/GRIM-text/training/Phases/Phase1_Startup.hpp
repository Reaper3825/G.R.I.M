#pragma once
//======================================================//
//  Phase1_Startup.hpp
//  Configuration loading, model initialization, data loading
//======================================================//
//
//  PURPOSE
//  =======
//  This phase handles all startup operations before training:
//  - Configuration loading (ai_config.json)
//  - Tokenizer initialization
//  - Model initialization
//  - Data loading and preprocessing
//  - Optimizer state initialization
//  - Gradient accumulation controller setup
//
//  DESIGN PRINCIPLES
//  =================
//  1. FAIL FAST: All validation happens here, not during training
//  2. SINGLE RESPONSIBILITY: Only startup operations
//  3. RETURN CONTRACT: Returns a fully initialized TrainingContext
//  4. EXCEPTION SAFETY: All errors throw, no partial initialization
//
//  Author: Austin Wadkins
//  Date: December 2025
//  Version: 1.0.0
//======================================================//

#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <random>
#include <filesystem>
#include <chrono>

// Core includes
// NOTE: ai_config_paths.hpp must NOT be included directly. It is pulled in
// transitively (and in the correct order) by HyperParameters_GPU.hpp below.
#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"
#include "../../GRIM/grim_language_model_cuda.hpp"
#include "../../Shared/UnigramByte/UniByte.hpp"
#include "../../Layers/Encoding/Encoding_GPU.hpp"
#include "../../Shared/DataLoader/DataLoader.hpp"
#include "../../Shared/Batching/Batching_GPU.hpp"
#include "../../Shared/Dynamic_LR/DynamicLR.hpp"
#include "../../Shared/Dynamic_LR/LRSchedule.hpp"
#include "../../Shared/SoftRestart/SoftRestart.hpp"
#include "../../Shared/Loss/LossContext/LossContext.hpp"
#include "../../Shared/Telemetry/TelemetryLattice_GPU.hpp"
#include "../../Shared/Telemetry/TelemetryControl_GPU.hpp"
#include "../../Shared/Telemetry/TelemetryCsvLogger.hpp"
#include "../../Shared/LogRecorder/BatchLogTape.hpp"
#include "../../Shared/LogRecorder/Sinks/TextLogSink.hpp"
#include "../../Shared/LogRecorder/Sinks/CsvEquationSink.hpp"
#include "../../Shared/LogRecorder/Sinks/StderrSink.hpp"
#include "../training_logger.hpp"
#include "../training_status_writer.hpp"
#include "../metrics_collector.hpp"
#include "../training_data_loader.hpp"

// Extracted startup subsystems
#include "Startup/Rng.hpp"
#include "Startup/Logging.hpp"
#include "Startup/Capacity/RunCapacity.hpp"

namespace GRIMText::Training {

// Use types from training_data_loader.hpp (global namespace)
using ::TrainingSequence;
using ::GRMTDataLoader;

//======================================================//
//  Startup Configuration Structures
//======================================================//

// PathConfig + StartupConfig are owned by Shared/HyperParameters/HyperParameters_GPU.hpp
// (single source of truth for ai_config.json loading + validation +
// derivation). The aliases below give phase-internal code the short
// names it has always used; do NOT redefine these types here.
using PathConfig    = ::GRIM::HyperParameters::PathConfig;
using StartupConfig = ::GRIM::HyperParameters::StartupConfig;

//======================================================//
//  Data Structures for Training Loop
//======================================================//

/**
 * @brief Sequence data for training
 */
struct SequenceData {
    std::vector<TrainingSequence> train_seqs;
    std::vector<TrainingSequence> val_seqs;
    std::vector<TrainingSequence*> train_views;
    std::vector<TrainingSequence*> val_views;
    GRIM::DynaSeq::Catalog train_catalog;
    GRIM::DynaSeq::Catalog val_catalog;
    uint32_t vocab_size = 0;  // Vocab size from training data file
};

/**
 * @brief Optimizer and controller state
 */
struct OptimizerContext {
    GRIM::OptimizerState optimizer_state;
    GRIM::DynamicLR::DynamicLRController dynamic_lr_controller;
    GRIM::SoftRestart::SoftRestartController soft_restart_controller;
    int current_micro_step = 0;  // Tracks position within accumulation window [0, accum_steps)
};

/**
 * @brief Logging and status tracking context
 */
struct LoggingContext {
    std::unique_ptr<TrainingLogger> logger;
    std::unique_ptr<MetricsCollector> metrics_collector;
    std::unique_ptr<StatusFileWriter> status_writer;
    std::string session_id;
    std::string raw_log_path;
    
    // ---- Unified batch-tape logging system ----
    std::unique_ptr<GRIM::Logging::BatchLogTape> tape;
    std::unique_ptr<GRIM::Logging::TextLogSink> text_sink;
    std::unique_ptr<GRIM::Logging::CsvEquationSink> equation_sink;
    std::unique_ptr<GRIM::Logging::StderrSink> stderr_sink;
};

/**
 * @brief Telemetry lattice context for multi-scale monitoring
 * Pattern B: TelemetryLattice is self-managing (RAII via unique_ptr).
 * 
 * last_obs[55] holds the most recent raw observation for ALL metric streams (0-54 inclusive).
 * Streams 0-4 are updated every batch; streams 5-8 (rho) are updated at
 * diagnostic intervals. Streams 9-13 (Adam warmup causation) are updated
 * every batch. Streams 14-20 (exec block health). Streams 21-26 (EB/SB injection diagnostics).
 * Streams 27-30 (PBM positional bias diagnostics). Streams 31-34 (rho raw decomposition).
 * Streams 35-37 (RMSNorm learned gamma tracking). Stream 38 (RHO_RAW_RMS_SPREAD).
 * Streams 39-44 (h↔W alignment / LM-head leak channel — updated at LOGIT_SCALE_EQUATION cadence).
 * Streams 45-46 (unigram-frequency-direction collapse detector — LOGIT_SCALE_EQUATION cadence).
 * Stream 47 (LM_HEAD_W_RMS_RMS — LOGIT_SCALE_EQUATION cadence).
 * Streams 48-54 (INIT_*: tie/ownership/optimizer-group invariants — set once
 *   by Phases/Startup/InitFacts.cu after model construction; constant for run).
 * lattice->update() always receives the full array.
 */
struct TelemetryContext {
    std::unique_ptr<GRIM::Telemetry::TelemetryLattice> lattice;
    GRIM::Telemetry::LatticeConfig config;
    GRIM::Telemetry::TelemetryControlConfig control_config;
    std::unique_ptr<GRIM::Telemetry::TelemetryControl> controller;
    std::unique_ptr<GRIM::Telemetry::TelemetryCsvLogger> csv_logger;
    float last_obs[55] = {};  // All metric streams (0-54 inclusive) — rho slots persist between diagnostic intervals; INIT_* slots (48-54) are constant for run
    float adam_cumulative_disp = 0.0f;  // Running sum of lr(t) for Adam disruption tracking
    bool enabled = true;

    TelemetryContext() = default;
    ~TelemetryContext() = default;
    TelemetryContext(const TelemetryContext&) = delete;
    TelemetryContext& operator=(const TelemetryContext&) = delete;
    TelemetryContext(TelemetryContext&&) noexcept = default;
    TelemetryContext& operator=(TelemetryContext&&) noexcept = default;
};

/**
 * @brief Complete training context returned by Phase1
 * 
 * This is the "contract" between phases - everything needed to run training
 * 
 * NOTE: Move-only type (contains std::unique_ptr)
 */
struct TrainingContext {
    // Configuration
    StartupConfig config;
    // Capacity stem (single author after HP policy)
    RunCapacity run_capacity;
    
    // Model and tokenizer
    std::unique_ptr<GRIM::LanguageModel> model;
    GRIM::Tokenizer::UniByte tokenizer;
    
    // Data
    SequenceData data;
    
    // Optimizer and controllers
    OptimizerContext optimizer;
    
    // Logging
    LoggingContext logging;
    
    // Telemetry (multi-scale monitoring)
    TelemetryContext telemetry;
    
    // RNG (hierarchical seeding for reproducibility)
    RNGContext rng;
    
    // Derived schedule
    GRIM::HyperParameters::DerivedScheduleInfo derived_schedule;
    
    // Timing
    std::chrono::steady_clock::time_point start_time;
    
    // State tracking
    int global_step = 0;
    /** Estimated total steps (epochs * batches per epoch), set in runEpoch; used for cosine LR decay. */
    int estimated_total_steps = 0;
    
    /** Deterministic LR schedule — exposed curve queryable at any step.
     *  Constructed in Phase2 once estimated_total_steps is known. */
    std::optional<GRIM::LR::LRSchedule> lr_schedule;
    
    
    float best_val_loss = std::numeric_limits<float>::infinity();
    /** Number of epochs actually completed (set by Phase 2; used by Phase 3 for summary). */
    int epochs_completed = 0;
    
    /** Path to the checkpoint that was loaded at startup (empty if fresh start). */
    std::string loaded_checkpoint_path;
    
    // Auto-stop state
    bool auto_stop_triggered = false;
    std::string auto_stop_reason;
    int auto_stop_epoch = 0;
    float auto_stop_metric = 0.0f;
    
    // Move-only semantics (std::unique_ptr cannot be copied)
    // Use compiler-generated move to avoid breaking GPU resource pointers
    TrainingContext() = default;
    TrainingContext(const TrainingContext&) = delete;
    TrainingContext& operator=(const TrainingContext&) = delete;
    TrainingContext(TrainingContext&&) noexcept = default;
    TrainingContext& operator=(TrainingContext&&) noexcept = default;
    
    // Validation check
    bool is_valid() const {
        return model != nullptr && 
               logging.logger != nullptr &&
               !data.train_views.empty();
    }
};

//======================================================//
//  Phase1 Entry Point
//======================================================//

/**
 * @brief Execute Phase 1 - Startup and initialization
 * 
 * @param argc Command line argument count
 * @param argv Command line arguments
 * @return unique_ptr<TrainingContext> Fully initialized context for Phase 2 (heap-allocated to avoid move)
 * @throws std::runtime_error on any initialization failure
 */
std::unique_ptr<TrainingContext> executePhase1(int argc, char** argv);

//======================================================//
//  Internal Helper Functions (exposed for testing)
//======================================================//

namespace Internal {

/**
 * @brief Load configuration from ai_config.json
 */
StartupConfig loadConfiguration(int argc, char** argv);

/**
 * @brief Validate all paths exist and are accessible
 */
void validatePaths(const PathConfig& paths);

/**
 * @brief Initialize logging infrastructure (declared in Startup/Logging.hpp)
 */

/**
 * @brief Load and initialize tokenizer
 */
GRIM::Tokenizer::UniByte initializeTokenizer(
    const std::string& vocab_path,
    const GRIM::Config::TokenizerConfig& tok_config,
    const GRIM::Config::TrainingHyperparameters& hyperparameters,
    TrainingLogger& logger);

/**
 * @brief Load and preprocess training data
 * 
 * @param add_bos_token If true, adds BOS token to start of sequences (from config)
 * @param add_eos_token If true, adds EOS token to end of sequences (from config)
 */
SequenceData loadTrainingData(
    const std::string& data_path,
    int max_seq_len,
    int min_seq_valid_tokens,
    int sliding_window_stride,
    bool add_bos_token,
    bool add_eos_token,
    const GRIM::Tokenizer::UniByte& tokenizer,
    TrainingLogger& logger);

/**
 * @brief Initialize the model with configuration
 * @param[out] loaded_checkpoint_path Set to the path of the checkpoint that
 *             actually loaded successfully (empty if fresh start).
 */
std::unique_ptr<GRIM::LanguageModel> initializeModel(
    const StartupConfig& config,
    const RunCapacity& run_capacity,
    uint32_t vocab_size,
    uint64_t xavier_seed,
    TrainingLogger& logger,
    std::string& loaded_checkpoint_path);

/**
 * @brief Initialize optimizer state AND restore the sidecar that pairs with
 *        the .bin checkpoint that loaded the model weights.
 *
 * Writes ctx.optimizer in-place. When ctx.loaded_checkpoint_path is non-empty,
 * pairs it with optimizerSidecarPath() and calls loadOptimizerState(ctx, ...);
 * a missing sidecar is logged and treated as a fresh-optimizer start. A failed
 * sidecar load is also logged and tolerated — fresh optimizer state, weights
 * keep what initializeModel loaded.
 *
 * Why this lives here (Rule 20: single source of truth): the sidecar path
 * MUST be derived from the EXACT checkpoint that loaded weights. Splitting
 * the work across the call site let an independent rescan of the checkpoint
 * dir creep in and pick a different epoch's .opt.
 */
void initializeOptimizer(TrainingContext& ctx);

// initializeRNG declaration moved to Startup/Rng.hpp
} // namespace Internal

} // namespace GRIMText::Training
