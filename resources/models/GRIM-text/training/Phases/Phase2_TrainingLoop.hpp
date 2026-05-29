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
//  - Validation measurement
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
#include "../../Shared/Batching/BatchPayload.hpp"
#include "../../Shared/Loss/LossSignals/LossSignals.hpp"
#include "../../Shared/MTP/MTPDiagnostics.hpp"

#include <functional>
#include <memory>

namespace GRIMText::Training {

//======================================================//
//  Batch Processing Structures
//======================================================//

/**
 * @brief Immutable per-microbatch autograd timing facts authored by runEpoch.
 *
 * processBatch consumes this plan but does not read or mutate OptimizerContext.
 * Accumulation-window normalization is owned later by the optimizer window,
 * not by autograd.
 */
struct BatchAutogradPlan {
    bool should_accumulate = false;
    uint64_t batch_idx = 0;
};

/**
 * @brief Result of processing a single batch
 */
struct BatchResult {
    int batch_idx = 0;
    float loss = 0.0f;
    float aux_loss = 0.0f;
    float grad_rms = 0.0f;
    bool grad_rms_valid = false;
    float learning_rate = 0.0f;
    int sequences_processed = 0;
    int tokens_processed = 0;
    bool gradient_clipped = false;
    GRIM::MTP::MTPDiagnostics mtp_diagnostics;
    // NOTE: `skipped` / `skip_reason` were removed. Per Rule 20 a batch either
    // completes a real step or the trainer crashes — there is no third "silent
    // skip" outcome for callers to handle.
};

/**
 * @brief Epoch metric historically serialized as validation.
 *
 * There is no second validation/eval autograd loop. This metric is derived
 * from the training batches already executed in the epoch.
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
    // batches_skipped removed — batches no longer skip silently (Rule 20).
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
    struct DiagnosticsState {
        // Embedding-spike diagnostic history. Updated only by
        // runGradientNormClipDiagnostic() when it emits the gated trace.
        float prev_emb_rms = 0.0f;
        bool has_prev_emb_rms = false;
    };

    // Loss baseline tracking
    float initial_loss = 0.0f;
    float min_observed_loss = std::numeric_limits<float>::infinity();
    int warmup_batches = 0;

    // Shuffle tracking
    bool shuffle_window_exhausted_notified = false;
    
    // Auto-stop tracking. Plateau is local (best-so-far improvement gap);
    // high-loss patience is owned by GRIM::Loss::LossSignalBus.
    int plateau_epochs_without_improvement = 0;
    
    // Prediction comparison counter
    int prediction_comparison_good_batch_counter = 0;
    
    // Last optimizer step that emitted a sample
    int last_sample_step = -1;

    // Diagnostic-owned persistent state. Phase2 carries it across steps but
    // does not interpret or mutate individual diagnostic fields directly.
    DiagnosticsState diagnostics;

    // High-loss patience detector. Train-loss spike/EWMA measurement is owned
    // by TelemetryLattice. Constructed in executePhase2; never null after
    // Phase2 enters the loop.
    std::unique_ptr<GRIM::Loss::LossSignalBus> loss_signals;

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
 * @param num_epochs Static epoch count from grouped startup hyperparameters
 * @param accum_steps Startup-authored gradient accumulation window size
 * @return EpochResult Results from this epoch
 */
EpochResult runEpoch(
    TrainingContext& ctx,
    TrainingLoopState& state,
    int epoch_idx,
    int num_epochs,
    int accum_steps);

/**
 * @brief Process a single batch
 *
 * @param ctx Training context
 * @param state Loop state
 * @param payload Batch payload (single source of truth; built from BatchAssignment in runEpoch)
 * @param batch_idx Batch index within epoch
 * @param epoch_idx Epoch index
 * @param plan Epoch-authored autograd accumulation/step facts
 * @return BatchResult Result from this batch
 */
BatchResult processBatch(
    TrainingContext& ctx,
    TrainingLoopState& state,
    GRIM::Batching::BatchPayload& payload,
    int batch_idx,
    int epoch_idx,
    const BatchAutogradPlan& plan);

//======================================================//
//  Internal Helper Functions
//======================================================//

namespace Internal {

/**
 * @brief Get scheduled learning rate: linear warmup, then constant or cosine decay.
 * @param total_steps Estimated total training steps (epochs * batches per epoch). 0 disables cosine decay.
 * @param cosine_decay_min_lr Minimum LR at end of schedule when cosine_decay_enabled.
 */
float getScheduledLearningRate(
    int step,
    float base_lr,
    int warmup_steps,
    int total_steps,
    float cosine_decay_min_lr,
    bool cosine_decay_enabled,
    bool stability_overrides_enabled);

/**
 * @brief Format scalar for logging
 */
std::string formatScalar(float value, int precision = 4);

/**
 * @brief Format metric for logging
 */
std::string formatMetric(std::string_view name, float value, int precision = 4);

} // namespace Internal

} // namespace GRIMText::Training
