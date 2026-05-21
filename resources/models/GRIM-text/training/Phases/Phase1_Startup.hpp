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
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include <random>
#include <filesystem>
#include <chrono>
#include <stdexcept>

// Core includes
// NOTE: ai_config_paths.hpp must NOT be included directly. It is pulled in
// transitively (and in the correct order) by HyperParameters_GPU.hpp below.
#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"
#include "../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../../GRIM/grim_language_model_cuda.hpp"
#include "../../Shared/Optimizers/OptimizerState_GPU.hpp"
#include "../../Shared/Optimizers/OptimizerStep.hpp"
#include "../../Shared/UnigramByte/UniByte.hpp"
#include "../../Layers/Encoding/Encoding_GPU.hpp"
#include "../../Shared/DataLoader/DataLoader.hpp"
#include "../../Shared/Batching/Batching_GPU.hpp"
#include "../../Shared/Batching/BatchPayload.hpp"
#include "../../Shared/Dynamic_LR/LRSchedule.hpp"
#include "../../Shared/SoftRestart/SoftRestart.hpp"
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
#include "Startup/Capacity/MemorySnapshot.hpp"
#include "Startup/Data/DataInfo.hpp"
#include "Startup/Model/ModelAllocationState.hpp"
#include "Startup/Epoch/EpochPlan.hpp"
#include "Startup/Payload/PayloadBuildInputs.hpp"
#include "Startup/Resume/ResumeState.hpp"

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
    std::vector<std::uint32_t> train_seq_lengths;
    std::vector<std::uint32_t> val_seq_lengths;
    uint32_t vocab_size = 0;  // Vocab size from training data file
};

/**
 * @brief Optimizer and controller state
 */
struct OptimizerContext {
    GRIM::OptimizerStep optimizer_step;
    GRIM::OptimizerState optimizer_state;
    GRIM::SoftRestart::SoftRestartController soft_restart_controller;

    int accumulationSlot() const { return accumulation_slot_; }

    bool shouldAccumulateGradients() const {
        return accumulation_slot_ > 0;
    }

    void validateBeforeAccumulationSlot(int accumulation_window_slots) const {
        if (accumulation_window_slots <= 0) {
            throw std::runtime_error("FATAL: accumulation_window_slots must be > 0");
        }
        if (accumulation_slot_ < 0 || accumulation_slot_ >= accumulation_window_slots) {
            throw std::runtime_error("FATAL: accumulation slot cursor out of range before autograd pass");
        }
    }

    bool completeAccumulationSlot(int accumulation_window_slots) {
        validateBeforeAccumulationSlot(accumulation_window_slots);
        accumulation_slot_++;
        return accumulation_slot_ >= accumulation_window_slots;
    }

    void completeOptimizerStepAfterFullAccumulationWindow(int accumulation_window_slots) {
        if (accumulation_slot_ != accumulation_window_slots) {
            throw std::runtime_error(
                "FATAL: optimizer step requested before accumulation window completed (completed=" +
                std::to_string(accumulation_slot_) + " required=" +
                std::to_string(accumulation_window_slots) + ")");
        }
        accumulation_slot_ = 0;
        optimizer_step.step++;
    }

    void resetAccumulationWindow() {
        accumulation_slot_ = 0;
    }

    void restoreAccumulationSlotFromCheckpoint(int accumulation_slot) {
        if (accumulation_slot < 0) {
            throw std::runtime_error("FATAL: checkpoint accumulation_slot is negative");
        }
        accumulation_slot_ = accumulation_slot;
    }

private:
    // Single authoritative in-window cursor. One accumulation slot is exactly
    // one BatchPayload upload → forward → loss → backward pass. There is no
    // separate secondary counter/lifecycle.
    int accumulation_slot_ = 0;
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
 * last_obs[58] holds the most recent raw observation for ALL metric streams (0-57 inclusive).
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
 * Streams 55-57 (rho signed/centered/mean-vector diagnostics — RHO_BUILDUP_EQUATION cadence).
 * lattice->update() always receives the full array.
 */
struct TelemetryContext {
    std::unique_ptr<GRIM::Telemetry::TelemetryLattice> lattice;
    GRIM::Telemetry::LatticeConfig config;
    GRIM::Telemetry::TelemetryControlConfig control_config;
    std::unique_ptr<GRIM::Telemetry::TelemetryControl> controller;
    std::unique_ptr<GRIM::Telemetry::TelemetryCsvLogger> csv_logger;
    float last_obs[58] = {};  // All metric streams (0-57 inclusive) — rho slots persist between diagnostic intervals; INIT_* slots (48-54) are constant for run
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
    // Phase1-authored static model config. This is the single model-config set
    // handed to Phase2; training code must not rebuild or route alternate static
    // model wrappers around it.
    GRIM::HyperParameters::LanguageModelConfig model_config;
    // Capacity stem (single author after HP policy)
    RunCapacity run_capacity;
    // Memory snapshot (evidence only; never authors capacity)
    MemorySnapshot memory_snapshot;
    // Post-allocation validation evidence (fails loud on mismatch)
    ModelAllocationState model_allocation;
    // Resume metadata (populated after optimizer sidecar restore attempt)
    ResumeState resume_state;
    // Data summary/reference artifact (SequenceData remains storage owner)
    DataInfo data_info;
    // Startup-owned epoch plan facts (LR schedule config, total steps, warmup)
    EpochPlan epoch_plan;
    // Phase1-authored snapshot of static inputs to GRIM::Batching::buildBatchPayload.
    // Contract-checks the model ↔ run_capacity cache agreement once at startup so
    // Phase2's per-batch payload builder never re-reads or re-validates them.
    PayloadBuildInputs payload_build_inputs;

    //==================================================//
    // Phase1-owned PLANNED BATCHES (PrecomputeBatchPayloads.plan.md)
    //
    // Phase1 builds the train and val schedules ONCE and materializes every
    // BatchPayload into the vectors below. Phase2 NEVER calls buildBatches /
    // buildEpochBatches / buildBatchPayload — it only:
    //   1. Indexes train_payloads via epoch_batch_order[epoch][batch_i] for
    //      training (the per-epoch "shuffle" is a permutation over fixed batch
    //      indices, never over sequence membership).
    //   2. Iterates val_payloads in order for validation.
    //
    // BatchPayload is host-only and immutable after the builder returns
    // (BatchDeviceBindings is the parallel device-pointer surface produced by
    // LanguageModel::uploadBatchToDevice at the per-step sync boundary).
    //==================================================//
    GRIM::Batching::BatchSchedule fixed_train_schedule;
    std::vector<GRIM::Batching::BatchPayload> train_payloads;
    GRIM::Batching::BatchSchedule fixed_val_schedule;
    std::vector<GRIM::Batching::BatchPayload> val_payloads;
    /** Per-epoch executable train batch-index order. Outer size = number of
     *  epochs to train; inner vector is the exact step order Phase2 reads.
     *  Normal mode is a permutation; single-batch diagnostic mode repeats
     *  index 0 for the authored step count. */
    std::vector<std::vector<int>> epoch_batch_order;

    // Model and tokenizer
    std::unique_ptr<GRIM::LanguageModel> model;
    std::unique_ptr<GRIM::Tokenizer::UniByte> tokenizer;
    
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
    /** Estimated total steps (epochs * batches per epoch), set during Phase1 EpochPlanReady. */
    int estimated_total_steps = 0;
    
    /** Deterministic LR schedule — exposed curve queryable at any step.
     *  Constructed in Phase1 after PlannedBatchesReady authors the train payload count. */
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

enum class Phase1Outcome : int {
    ready_for_training = 0,
    tokenizer_only_complete = 1,
};

struct Phase1Result {
    Phase1Outcome outcome = Phase1Outcome::ready_for_training;
    std::unique_ptr<TrainingContext> context;
};

/**
 * @brief Execute Phase 1 - Startup and initialization
 * 
 * @param argc Command line argument count
 * @param argv Command line arguments
 * @return Phase1Result containing either a fully initialized context for Phase 2
 *         or an explicit tokenizer-only completion outcome.
 * @throws std::runtime_error on any initialization failure
 */
Phase1Result executePhase1(int argc, char** argv);

} // namespace GRIMText::Training
