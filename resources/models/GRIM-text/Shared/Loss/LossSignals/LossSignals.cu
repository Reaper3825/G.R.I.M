//======================================================//
//  LossSignals.cu — central loss-spike detector.
//  See LossSignals.hpp for design / Rule 20 notes.
//======================================================//

#include "LossSignals.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace GRIM::Loss {

namespace {
inline bool isFinite(float v) { return std::isfinite(static_cast<double>(v)); }
}

LossSignalBus::LossSignalBus(const LossSignalConfig& cfg) : cfg_(cfg) {
    // Rule 20: validate construction-time invariants loudly.
    if (!(cfg_.smoothed_ema_alpha > 0.0f && cfg_.smoothed_ema_alpha <= 1.0f)) {
        throw std::runtime_error(
            "LossSignalBus: smoothed_ema_alpha must be in (0,1] (got " +
            std::to_string(cfg_.smoothed_ema_alpha) + ") at " +
            std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    if (cfg_.smoothed_min_samples < 1) {
        throw std::runtime_error(
            "LossSignalBus: smoothed_min_samples must be >= 1 (got " +
            std::to_string(cfg_.smoothed_min_samples) + ")");
    }
    if (cfg_.validation_high_patience < 0) {
        throw std::runtime_error(
            "LossSignalBus: validation_high_patience must be >= 0 (got " +
            std::to_string(cfg_.validation_high_patience) + ")");
    }
}

void LossSignalBus::updateSmoothedStats(float loss) {
    const float a = cfg_.smoothed_ema_alpha;
    if (samples_ == 1) {
        // Seed EWMA on the first sample so we don't bias toward zero.
        smoothed_loss_ = loss;
        smoothed_var_  = 0.0f;
        return;
    }
    const float prev_mean = smoothed_loss_;
    smoothed_loss_ = (1.0f - a) * smoothed_loss_ + a * loss;
    const float dev = loss - prev_mean;
    // EWMA of squared deviation (Welford-like, biased but stable).
    smoothed_var_ = (1.0f - a) * smoothed_var_ + a * dev * dev;
}

const LossSignals& LossSignalBus::recordTrainStep(int64_t global_step, float loss) {
    (void)global_step; // reserved for future use (warmup gating, etc.)

    // Rule 20: non-finite loss MUST crash upstream (autograd). If it slipped
    // through, surface it loudly here rather than poisoning baseline state.
    if (!isFinite(loss)) {
        throw std::runtime_error(
            "LossSignalBus::recordTrainStep got non-finite loss (" +
            std::to_string(loss) + ") at " +
            std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    // Reset the per-step booleans; per-epoch booleans persist until next
    // recordValidation call so consumers can read them after the fact.
    latest_.baseline_spike   = false;
    latest_.smoothed_spike   = false;
    latest_.step_delta_spike = false;
    latest_.current_loss     = loss;

    // 1) Capture baseline on the very first sample.
    if (!initial_captured_) {
        initial_loss_       = loss;
        initial_captured_   = true;
        min_observed_loss_  = loss;
    } else {
        if (loss < min_observed_loss_) min_observed_loss_ = loss;
    }
    latest_.baseline_loss = initial_loss_;

    // 2) BaselineSpike: loss > initial * multiplier (skip on the seed sample).
    if (samples_ >= 1 && initial_loss_ > 0.0f) {
        latest_.baseline_spike =
            (loss > initial_loss_ * cfg_.baseline_spike_multiplier);
    }

    // 3) StepDeltaSpike.
    if (isFinite(prev_step_loss_)) {
        latest_.step_delta_spike =
            ((loss - prev_step_loss_) > cfg_.step_delta_threshold);
    }
    latest_.prev_step_loss = prev_step_loss_;

    // 4) Update EWMA mean/var, then evaluate SmoothedSpike.
    samples_++;
    updateSmoothedStats(loss);
    latest_.smoothed_loss      = smoothed_loss_;
    latest_.smoothed_threshold = 0.0f;
    if (samples_ >= cfg_.smoothed_min_samples) {
        const float std_dev = std::sqrt(std::max(smoothed_var_, 0.0f));
        const float thresh =
            std::max(cfg_.smoothed_floor,
                     smoothed_loss_ + cfg_.smoothed_sigma * std_dev);
        latest_.smoothed_threshold = thresh;
        latest_.smoothed_spike     = (loss > thresh);
    }

    prev_step_loss_ = loss;
    return latest_;
}

const LossSignals& LossSignalBus::recordValidation(int epoch, float val_loss) {
    (void)epoch;

    if (!isFinite(val_loss)) {
        throw std::runtime_error(
            "LossSignalBus::recordValidation got non-finite val_loss (" +
            std::to_string(val_loss) + ") at " +
            std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    // Reset per-epoch booleans.
    latest_.validation_high        = false;
    latest_.validation_delta_spike = false;

    // ValidationDelta vs previous validation sample.
    if (isFinite(last_validation_loss_)) {
        latest_.validation_delta_spike =
            ((val_loss - last_validation_loss_) > cfg_.validation_delta_threshold);
    }

    // ValidationHigh patience counter.
    if (val_loss >= cfg_.validation_high_threshold) {
        consecutive_validation_high_++;
    } else {
        consecutive_validation_high_ = 0;
    }
    if (cfg_.validation_high_patience > 0 &&
        consecutive_validation_high_ >= cfg_.validation_high_patience) {
        latest_.validation_high = true;
    }

    last_validation_loss_              = val_loss;
    latest_.last_validation_loss       = val_loss;
    latest_.consecutive_validation_high = consecutive_validation_high_;
    return latest_;
}

} // namespace GRIM::Loss
