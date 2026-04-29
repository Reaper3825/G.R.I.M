#pragma once
//======================================================//
//  LossSignals.hpp — single source of truth for
//  loss-based control signals consumed by:
//    - Diagnostics/LossSpikeDiagnostic   (BaselineSpike, per-batch log)
//    - Shared/Dynamic_LR                 (SmoothedSpike, per-step LR cut)
//    - Shared/SoftRestart                (ValidationDelta, momentum reset)
//    - training/Phases evaluateAutoStop  (ValidationHigh, halt)
//
//  RATIONALE:
//    Each subsystem previously kept its own threshold, its own state
//    (initial_loss, smoothed_loss, last_val_loss, …), and its own
//    "is this a spike?" computation. They disagreed on the definition
//    of high loss. LossSignalBus owns ALL detection state and emits
//    the canonical signals; subsystems own ONLY the action they take
//    when a signal fires.
//
//  Rule 20 / Category 2:
//    LossSignalBus is durable persistent training state. It is owned by
//    TrainingState. Per-step / per-epoch detection state lives ONLY here.
//======================================================//

#include <cstdint>
#include <limits>

namespace GRIM::Loss {

//------------------------------------------------------//
// CONFIG — all loss-spike thresholds in one place.
// Mirror these into ai_config.json under "loss_signals".
//------------------------------------------------------//
struct LossSignalConfig {
    // BaselineSpike (per-batch): loss > initial_loss * baseline_spike_multiplier.
    //   Used by LossSpikeDiagnostic for the per-sequence breakdown log line.
    float baseline_spike_multiplier = 1.5f;

    // SmoothedSpike (per-step, statistical): smoothed_loss > mean + sigma*std,
    //   with a minimum-samples guard so the band isn't garbage at startup.
    //   Published for consumers that need loss-spike state.
    float smoothed_ema_alpha   = 0.10f;
    float smoothed_sigma       = 3.0f;
    float smoothed_floor       = 0.0f;
    int   smoothed_min_samples = 32;

    // StepDeltaSpike (per-step): loss - prev_step_loss > step_delta_threshold.
    //   Optional fast-twitch signal. Currently unused by core path; kept so
    //   subsystems can opt in without rolling their own delta tracking.
    float step_delta_threshold = 0.5f;

    // ValidationHigh (per-epoch policy): val_loss >= validation_high_threshold
    //   for validation_high_patience consecutive epochs → halt.
    //   Used by evaluateAutoStop.
    float validation_high_threshold = 10.0f;
    int   validation_high_patience  = 3;

    // ValidationDelta (per-epoch): (val_loss - last_val_loss) > validation_delta_threshold.
    //   Used by SoftRestart to reset optimizer momentum.
    float validation_delta_threshold = 0.2f;
};

//------------------------------------------------------//
// CANONICAL SIGNALS — consumers READ these, never recompute.
// All booleans reflect the LATEST record* call.
//------------------------------------------------------//
struct LossSignals {
    // Per-step (set by recordTrainStep)
    bool  baseline_spike    = false;
    bool  smoothed_spike    = false;
    bool  step_delta_spike  = false;

    // Per-epoch (set by recordValidation)
    bool  validation_high         = false; // patience tripped this epoch
    bool  validation_delta_spike  = false;

    // Diagnostic values (always populated for the latest record* call)
    float current_loss          = 0.0f;
    float baseline_loss         = 0.0f;   // initial_loss snapshot
    float smoothed_loss         = 0.0f;
    float smoothed_threshold    = 0.0f;   // mean + sigma*std (or 0 if not ready)
    float prev_step_loss        = 0.0f;
    float last_validation_loss  = 0.0f;
    int   consecutive_validation_high = 0;
};

//------------------------------------------------------//
// LossSignalBus — owns all loss-detection state.
// Construction is cheap; not GPU-resident.
// Thread model: called from the training-loop thread only.
//------------------------------------------------------//
class LossSignalBus {
public:
    explicit LossSignalBus(const LossSignalConfig& cfg);

    // Per-batch train-loss event. Returns the canonical signals
    // computed from the running state AFTER this sample is folded in.
    // Rule 20: caller MUST pass a finite loss. Non-finite loss is the
    // autograd path's responsibility to crash on, not this module's
    // to silently ignore.
    const LossSignals& recordTrainStep(int64_t global_step, float loss);

    // Per-epoch validation-loss event.
    const LossSignals& recordValidation(int epoch, float val_loss);

    // Read the most recent signals without mutating state.
    const LossSignals& latest() const { return latest_; }
    const LossSignalConfig& config() const { return cfg_; }

    // For diagnostics / checkpoint-resume bookkeeping.
    int64_t samples() const { return samples_; }
    float   initialLoss() const { return initial_loss_; }
    float   minObservedLoss() const { return min_observed_loss_; }

private:
    void updateSmoothedStats(float loss);

    LossSignalConfig cfg_;
    LossSignals      latest_{};

    // Detection state (Category 2: durable across steps).
    float   initial_loss_       = 0.0f;
    bool    initial_captured_   = false;
    float   min_observed_loss_  = std::numeric_limits<float>::infinity();
    float   smoothed_loss_      = 0.0f;
    float   smoothed_var_       = 0.0f;   // EWMA of squared deviation
    int64_t samples_            = 0;
    float   prev_step_loss_     = std::numeric_limits<float>::quiet_NaN();

    int     consecutive_validation_high_ = 0;
    float   last_validation_loss_        = std::numeric_limits<float>::quiet_NaN();
};

} // namespace GRIM::Loss
