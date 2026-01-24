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
#include <string>
#include <vector>
#include <random>
#include <filesystem>
#include <chrono>

// Core includes
#include "../../../../control/ai_config_paths.hpp"
#include "../../GRIM/grim_language_model_cuda.hpp"
#include "../../Shared/UnigramByte/UniByte.hpp"
#include "../../Layers/Encoding/Encoding_GPU.hpp"
#include "../../Shared/DataLoader/DataLoader.hpp"
#include "../../Shared/Batching/Batching_GPU.hpp"
#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"
#include "../../Shared/Dynamic_LR/DynamicLR.hpp"
#include "../../Shared/SoftRestart/SoftRestart.hpp"
#include "../../Shared/GradAccumulationController/GradAccumulationController_Integration.hpp"
#include "../../Shared/RareTokens/RareTokens_GPU.hpp"
#include "../../Shared/Loss/LossContext/LossContext.hpp"
#include "../../Shared/Telemetry/TelemetryLattice_GPU.hpp"
#include "../../Shared/Telemetry/TelemetryControl_GPU.hpp"
#include "../training_logger.hpp"
#include "../training_status_writer.hpp"
#include "../metrics_collector.hpp"
#include "../training_data_loader.hpp"

// Forward declaration for setGradCheckLogPath (defined in LanguageModel_Training.cu)
namespace GRIM { void setGradCheckLogPath(const std::string& path); }

namespace fs = std::filesystem;

namespace GRIMText::Training {

// Use types from training_data_loader.hpp (global namespace)
using ::TrainingSequence;
using ::GRMTDataLoader;

//======================================================//
//  Startup Configuration Structures
//======================================================//

/**
 * @brief Configuration paths loaded from ai_config.json
 */
struct PathConfig {
    std::string data_path;
    std::string vocab_path;
    std::string output_model_path;
    std::string checkpoint_dir;
    std::string log_dir;
    std::string status_path;
    fs::path config_path;
    
    bool validate() const;
};

/**
 * @brief CUDA execution mode flags for debugging/profiling
 */
struct CUDAExecutionConfig {
    bool single_stream_mode = false;
    bool disable_async_frees = false;
    bool synchronize_after_kernels = false;
};

/**
 * @brief Prediction comparison logging configuration
 */
struct PredictionComparisonConfig {
    bool enabled = false;
    int interval = 100;
    int top_k = 5;
    int max_positions = 10;
    std::string log_path;
};

/**
 * @brief Stability override configuration
 */
struct StabilityOverrides {
    bool enabled = false;
    int batch_size = 0;
    int max_seq_len = 0;
    int max_tokens_per_batch = 0;
    float clip_abs = 0.0f;
    float clip_norm = 0.0f;
    float lr_min = 0.0f;
};

/**
 * @brief Scratch block configuration
 */
struct ScratchBlockConfig {
    bool enabled = false;
    size_t max_tokens_per_block = 0;
    size_t num_blocks = 0;
    bool write_combined = false;
};

/**
 * @brief All startup configuration combined
 */
struct StartupConfig {
    PathConfig paths;
    GRIM::Config::TrainingHyperparameters hyperparameters;
    GRIM::LossContext::LossOptions loss_options;
    GRIM::HyperParameters::ModelArchitecture architecture;
    CUDAExecutionConfig cuda_exec;
    PredictionComparisonConfig pred_comparison;
    StabilityOverrides stability;
    ScratchBlockConfig scratch;
    
    // Derived values
    int max_seq_len = 512;
    int sliding_window_stride = 256;
    uint32_t actual_vocab_size = 0;
    
    // Flags
    bool save_test_mode = false;
    bool force_rebuild_vocab = false;
    bool clear_merged_cache = false;
};

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
    std::vector<float> sequence_rarity;
    std::vector<float> val_sequence_rarity;
};

/**
 * @brief Optimizer and controller state
 */
struct OptimizerContext {
    GRIM::OptimizerState optimizer_state;
    GRIM::DynamicLR::DynamicLRController dynamic_lr_controller;
    GRIM::SoftRestart::SoftRestartController soft_restart_controller;
    std::unique_ptr<GRIM::ModelGradAccumulationController> grad_controller;
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
    
    // Module log sinks (must persist throughout training)
    std::unique_ptr<GRIM::Logging::ModuleLogSink> backward_sink;
    std::unique_ptr<GRIM::Logging::ModuleLogSink> stream_controller_sink;
    std::unique_ptr<GRIM::Logging::ModuleLogSink> checkpoint_sink;
    std::unique_ptr<GRIM::Logging::ModuleLogSink> activations_sink;
    std::unique_ptr<GRIM::Logging::ModuleLogSink> guess_cache_sink;
};

/**
 * @brief Telemetry lattice context for multi-scale monitoring
 */
struct TelemetryContext {
    GRIM::Telemetry::TelemetryLattice* lattice = nullptr;
    GRIM::Telemetry::LatticeConfig config;
    GRIM::Telemetry::TelemetryControlConfig control_config;
    std::unique_ptr<GRIM::Telemetry::TelemetryControl> controller;
    bool enabled = true;
    
    ~TelemetryContext() {
        if (lattice) {
            GRIM::Telemetry::freeTelemetryLattice(&lattice);
        }
    }
    
    // Move-only semantics (std::unique_ptr cannot be copied)
    TelemetryContext() = default;
    TelemetryContext(const TelemetryContext&) = delete;
    TelemetryContext& operator=(const TelemetryContext&) = delete;
    TelemetryContext(TelemetryContext&& other) noexcept
        : lattice(other.lattice)
        , config(std::move(other.config))
        , control_config(std::move(other.control_config))
        , controller(std::move(other.controller))
        , enabled(other.enabled)
    {
        // Transfer ownership - critical to prevent double-free
        other.lattice = nullptr;
    }
    TelemetryContext& operator=(TelemetryContext&& other) noexcept {
        if (this != &other) {
            // Clean up our current lattice
            if (lattice) {
                GRIM::Telemetry::freeTelemetryLattice(&lattice);
            }
            
            // Transfer from other
            lattice = other.lattice;
            config = std::move(other.config);
            control_config = std::move(other.control_config);
            controller = std::move(other.controller);
            enabled = other.enabled;
            
            // Nullify source
            other.lattice = nullptr;
        }
        return *this;
    }
};

/**
 * @brief Production-grade RNG context with reproducibility support
 * 
 * Hierarchical seeding strategy (like PyTorch/JAX):
 * - base_seed: Master seed from config (user-specified or time-based)
 * - data_seed = base_seed + 0: CPU RNG for data shuffling
 * - init_seed = base_seed + 1000: Weight initialization (Xavier)
 * - cuda_seed = base_seed + 2000: GPU dropout/sampling operations
 * 
 * This ensures:
 * 1. Different RNG streams avoid statistical correlations
 * 2. Exact reproducibility when same seed is provided
 * 3. CUDA operations (dropout, attention dropout) use controlled RNG
 * 4. Checkpoints can save/restore RNG state for perfect resume
 */
struct RNGContext {
    // Seeds
    uint64_t base_seed = 0;       // Master seed (0 = use timestamp)
    uint64_t data_seed = 0;       // Data shuffling seed
    uint64_t init_seed = 0;       // Weight initialization seed
    uint64_t cuda_seed = 0;       // CUDA RNG seed
    
    // CPU RNG for data operations
    std::mt19937_64 data_rng;
    
    // CUDA RNG for GPU dropout/sampling (nullptr if not initialized)
    void* cuda_rng_generator = nullptr;  // curandGenerator_t (void* to avoid curand.h here)
    
    // State tracking
    bool deterministic = false;   // Whether using fixed seed
    bool cuda_rng_initialized = false;
    
    // Rule of Five: Move-only semantics (CUDA generator cannot be copied)
    RNGContext() = default;
    ~RNGContext();
    
    // Delete copy operations (CUDA generator is unique resource)
    RNGContext(const RNGContext&) = delete;
    RNGContext& operator=(const RNGContext&) = delete;
    
    // Move operations transfer ownership
    RNGContext(RNGContext&& other) noexcept;
    RNGContext& operator=(RNGContext&& other) noexcept;
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
    float best_val_loss = std::numeric_limits<float>::infinity();
    
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
 * @brief Initialize logging infrastructure
 */
LoggingContext initializeLogging(const PathConfig& paths);

/**
 * @brief Load and initialize tokenizer
 */
GRIM::Tokenizer::UniByte initializeTokenizer(
    const std::string& vocab_path,
    const GRIM::Config::TokenizerConfig& tok_config,
    TrainingLogger& logger);

/**
 * @brief Load and preprocess training data
 */
SequenceData loadTrainingData(
    const std::string& data_path,
    int max_seq_len,
    int sliding_window_stride,
    const GRIM::Tokenizer::UniByte& tokenizer,
    TrainingLogger& logger);

/**
 * @brief Initialize the model with configuration
 */
std::unique_ptr<GRIM::LanguageModel> initializeModel(
    const StartupConfig& config,
    uint32_t vocab_size,
    uint64_t xavier_seed,
    TrainingLogger& logger);

/**
 * @brief Initialize optimizer and gradient controller
 */
OptimizerContext initializeOptimizer(
    GRIM::LanguageModel& model,
    const StartupConfig& config,
    TrainingLogger& logger);

/**
 * @brief Compute rare token scores for prioritized batching
 */
std::vector<float> computeRareTokenScores(
    const std::vector<TrainingSequence*>& train_views,
    uint32_t vocab_size,
    size_t train_count,
    TrainingLogger& logger);
/**
 * @brief Initialize production-grade RNG system with hierarchical seeding
 * 
 * Implements PyTorch/JAX-style reproducibility:
 * - Reads seed from ai_config.json (training.config.seed)
 * - seed = -1 or missing: random seed from system clock
 * - seed >= 0: deterministic mode with that seed
 * - Creates separate RNG streams for data/init/cuda
 * - Initializes cuRAND for GPU dropout operations
 * - Logs seed values for reproducibility
 */
RNGContext initializeRNG(
    const StartupConfig& config,
    TrainingLogger& logger);
} // namespace Internal

} // namespace GRIMText::Training
