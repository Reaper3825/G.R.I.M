#pragma once
//======================================================//
//  LossSignals.hpp — validation-loss high-loss patience for auto-stop.
//
//  RATIONALE:
//    Train-loss spike / EWMA detection is owned by TelemetryLattice.
//    This bus owns only validation-epoch high-loss patience state for the
//    auto-stop policy. It does not compute spike or delta signals.
//
//  Rule 20 / Category 2:
//    LossSignalBus is durable persistent training state. It is owned by
//    TrainingLoopState. Validation policy state lives ONLY here.
//======================================================//

namespace GRIM::Loss {

//------------------------------------------------------//
// CONFIG — validation-loss policy thresholds.
//------------------------------------------------------//
struct LossSignalConfig {
    // ValidationHigh (per-epoch policy): val_loss >= validation_high_threshold
    //   for validation_high_patience consecutive epochs → halt.
    //   Used by evaluateAutoStop.
    float validation_high_threshold = 10.0f;
    int   validation_high_patience  = 3;

};

//------------------------------------------------------//
// CANONICAL VALIDATION SIGNALS — consumers READ these, never recompute.
// All fields reflect the latest recordValidation() call.
//------------------------------------------------------//
struct LossSignals {
    bool  validation_high         = false; // patience tripped this epoch

    float last_validation_loss  = 0.0f;
    int   consecutive_validation_high = 0;
};

//------------------------------------------------------//
// LossSignalBus — owns validation-loss policy state.
// Construction is cheap; not GPU-resident.
// Thread model: called from the training-loop thread only.
//------------------------------------------------------//
class LossSignalBus {
public:
    explicit LossSignalBus(const LossSignalConfig& cfg);

    // Per-epoch validation-loss event.
    const LossSignals& recordValidation(int epoch, float val_loss);

    // Read the most recent signals without mutating state.
    const LossSignals& latest() const { return latest_; }
    const LossSignalConfig& config() const { return cfg_; }

private:
    LossSignalConfig cfg_;
    LossSignals      latest_{};

    int     consecutive_validation_high_ = 0;
};

} // namespace GRIM::Loss
