#pragma once
//======================================================//
//  Phase2_TrainingLoop.hpp
//  Core training computation and logic
//======================================================//
//
//  PURPOSE
//  =======
//  This phase handles all training computation:
//  - Epoch iteration
//  - Batch construction and processing
//  - Forward/backward passes
//  - Gradient clipping and optimizer steps
//  - Validation and checkpointing
//  - Auto-stop detection
//
//  DESIGN PRINCIPLES
//  =================
//  1. SINGLE EPOCH FUNCTION: One entry point per epoch
//  2. CLEAR CONTRACTS: Input/output well-defined
//  3. RECOVERABLE: Can resume from any epoch
//  4. OBSERVABLE: All metrics exposed via events/callbacks
//
//  Author: Austin Wadkins
//  Date: December 2025
//  Version: 1.0.0
//======================================================//

#include "Phase1_Startup.hpp"
#include "../TrainingEvents.hpp"
#include "../../Shared/TNC/Token-normalized_clipping.hpp"
#include "../../Layers/GRIMTS/GRIM-TS.hpp"

#include <functional>
#include <unordered_map>
#include <unordered_set>

namespace GRIMText::Training {

class GuessCacheBatchBuffers;

//======================================================//
//  Training Constants
//======================================================//

constexpr float kGuessRewardMomentum = 0.85f;
constexpr std::size_t kDefaultGuessCacheCapacity = 16384;
constexpr int kDefaultMaxTokensPerBatch = 8192;
constexpr int kWarmupTokenCap = 4096;
constexpr int kWarmupTokenSteps = 2048;
constexpr int kCurriculumEpochs = 1;
constexpr float kTokenAwareGradReference = 1024.0f;
constexpr float kBoilerplateBaseWeight = 0.7f;
constexpr float kJunkBaseWeight = 0.5f;
constexpr int kGradSpikeCooldownSteps = 2;
constexpr float kGradSpikeLrFraction = 0.10f;

//======================================================//
//  Batch Processing Structures
//======================================================//

/**
 * @brief Result of processing a single batch
 */
struct BatchResult {
    int batch_idx = 0;
    float loss = 0.0f;
    float grad_rms = 0.0f;
    float normalized_grad_rms = 0.0f;
    float learning_rate = 0.0f;
    int sequences_processed = 0;
    int tokens_processed = 0;
    bool skipped = false;
    bool gradient_clipped = false;
    std::string skip_reason;
};

/**
 * @brief Result of running validation
 */
struct ValidationResult {
    float loss = 0.0f;
    float perplexity = 0.0f;
    int sequences_processed = 0;
    bool is_best = false;
};

/**
 * @brief Result of running one epoch
 */
struct EpochResult {
    int epoch = 0;
    float avg_loss = 0.0f;
    float best_batch_loss = std::numeric_limits<float>::infinity();
    float worst_batch_loss = 0.0f;
    int batches_processed = 0;
    int batches_skipped = 0;
    ValidationResult validation;
    bool auto_stop_triggered = false;
    std::string auto_stop_reason;
    std::chrono::milliseconds duration{0};
};

//======================================================//
//  Tracking State (persists across epochs)
//======================================================//

/**
 * @brief State that persists across batches and epochs
 */
struct TrainingLoopState {
    // Spike tracking
    std::unordered_map<uint32_t, int> spike_counts;
    std::unordered_set<uint32_t> quarantined_seqs;
    int grad_spike_cooldown = 0;
    
    // Loss baseline tracking (for adaptive spike detection)
    float initial_loss = 0.0f;
    float min_observed_loss = std::numeric_limits<float>::infinity();
    int warmup_batches = 0;
    
    // Micro-validation state
    std::vector<GRIM::Batching::BatchAssignment> micro_validation_batches;
    std::size_t micro_validation_cursor = 0;
    
    // GuessCache state
    bool guess_cache_ready = false;
    bool guess_cache_faulted = false;
    std::unique_ptr<GuessCacheBatchBuffers> guess_cache_buffers;
    
    // Shuffle tracking
    bool shuffle_window_exhausted_notified = false;
    
    // Auto-stop tracking
    int plateau_epochs_without_improvement = 0;
    int high_loss_epochs = 0;
    
    // Prediction comparison counter
    int prediction_comparison_good_batch_counter = 0;
    
    // Last gradient RMS for growth detection
    float last_grad_rms = 0.0f;

    // Last optimizer step that emitted a sample
    int last_sample_step = -1;

    ~TrainingLoopState();
};

//======================================================//
//  Phase2 Entry Points
//======================================================//

/**
 * @brief Execute Phase 2 - Run all training epochs
 * 
 * @param ctx Training context from Phase 1
 * @return bool True if training completed successfully
 */
bool executePhase2(TrainingContext& ctx);

/**
 * @brief Run a single training epoch
 * 
 * @param ctx Training context
 * @param state Loop state that persists across epochs
 * @param epoch_idx Zero-based epoch index
 * @return EpochResult Results from this epoch
 */
EpochResult runEpoch(
    TrainingContext& ctx,
    TrainingLoopState& state,
    int epoch_idx);

/**
 * @brief Process a single batch
 * 
 * @param ctx Training context
 * @param state Loop state
 * @param batch Dynamic batch assignment
 * @param batch_idx Batch index within epoch
 * @param total_batches Total batches in epoch
 * @return BatchResult Result from this batch
 */
BatchResult processBatch(
    TrainingContext& ctx,
    TrainingLoopState& state,
    const GRIM::Batching::BatchAssignment& batch,
    int batch_idx,
    int total_batches,
    int epoch_idx);

/**
 * @brief Run validation after an epoch
 * 
 * @param ctx Training context
 * @return ValidationResult Validation metrics
 */
ValidationResult runValidation(TrainingContext& ctx);

//======================================================//
//  Internal Helper Functions
//======================================================//

namespace Internal {

/**
 * @brief Get scheduled learning rate with warmup, constant after warmup
 */
float getScheduledLearningRate(
    int step,
    float base_lr,
    int warmup_steps,
    bool stability_overrides_enabled);

/**
 * @brief Check if a batch should be skipped due to quarantine
 */
bool isBatchQuarantined(
    const std::vector<uint32_t>& seq_ids,
    const std::unordered_set<uint32_t>& quarantined_seqs);

/**
 * @brief Format scalar for logging
 */
std::string formatScalar(float value, int precision = 4);

/**
 * @brief Format metric for logging
 */
std::string formatMetric(std::string_view name, float value, int precision = 4);

/**
 * @brief Build batches for an epoch with appropriate settings
 */
GRIM::Batching::BatchSchedule buildEpochBatches(
    const GRIM::DynaSeq::Catalog& catalog,
    int batch_size,
    int global_step,
    int epoch,
    int max_tokens_override,
    TrainingLogger& logger);

/**
 * @brief Run micro-validation if conditions are met
 */
void maybeRunMicroValidation(
    TrainingContext& ctx,
    TrainingLoopState& state,
    float latest_batch_loss,
    float current_lr);

/**
 * @brief Handle gradient spike detection and quarantine
 */
bool handleGradientSpike(
    TrainingContext& ctx,
    TrainingLoopState& state,
    const GRIM::Batching::BatchAssignment& batch,
    float preclip_grad_rms,
    float preclip_norm_grad,
    float batch_loss,
    const GRIM::TNC::ClipSelection& clip_selection,
    int batch_idx);

/**
 * @brief Save checkpoint if this is the best validation loss
 */
bool maybeSaveCheckpoint(
    TrainingContext& ctx,
    float val_loss,
    int epoch);

} // namespace Internal

//======================================================//
//  GuessCache RAII Scope (Rule 22 Compliant)
//======================================================//

class GuessCacheScope {
public:
    // Rule 22: TrainingState owns all GPU memory
    // GuessCacheScope just manages initialization/shutdown lifecycle
    explicit GuessCacheScope(::GRIM::TrainingState& training_state, 
                             std::size_t capacity, 
                             bool enable_async = true);
    ~GuessCacheScope();
    
    GuessCacheScope(const GuessCacheScope&) = delete;
    GuessCacheScope& operator=(const GuessCacheScope&) = delete;
    
    bool active() const { return active_; }

private:
    ::GRIM::TrainingState& training_state_;
    bool active_ = false;
    bool buffers_allocated_ = false;
};

class GuessCacheBatchBuffers {
public:
    GuessCacheBatchBuffers() = default;
    ~GuessCacheBatchBuffers();
    
    cudaError_t ensure(std::size_t capacity);
    
    GRIMTS::GuessMetadata* metadata() const { return device_metadata_; }
    float* rewards() const { return device_rewards_; }
    GRIMTS::GuessRewardStats* stats() const { return device_stats_; }

private:
    void release();
    
    GRIMTS::GuessMetadata* device_metadata_ = nullptr;
    float* device_rewards_ = nullptr;
    GRIMTS::GuessRewardStats* device_stats_ = nullptr;
    std::size_t capacity_ = 0;
};

class MicroValidationScope {
public:
    explicit MicroValidationScope(int step);
    ~MicroValidationScope();
    
    MicroValidationScope(const MicroValidationScope&) = delete;
    MicroValidationScope& operator=(const MicroValidationScope&) = delete;
    
    void complete(const GRIMTS::MicroValidationPulse& pulse);

private:
    int step_ = 0;
    bool completed_ = false;
};

} // namespace GRIMText::Training
