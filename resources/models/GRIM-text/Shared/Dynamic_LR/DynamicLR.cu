//======================================================//
//  DynamicLR.cu
//  Adaptive learning-rate controller implementation
//======================================================//

#include "DynamicLR.hpp"
#include "../HyperParameters/HyperParameters_GPU.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <sstream>
#include <vector>

namespace GRIM {
namespace DynamicLR {

namespace {
constexpr float kEpsilon = HyperParameters::EPSILON_DYNAMIC_LR;
inline bool isFinite(float value) {
    return std::isfinite(static_cast<double>(value));
}

// Logging infrastructure
std::mutex g_dynlr_log_mutex;
std::vector<Logging::LogCallback> g_dynlr_log_callbacks;
}

DynamicLRController::DynamicLRController(const DynamicLRConfig& config)
    : config_(config) {
    reset();
}

void DynamicLRController::setEnabled(bool enabled) {
    enabled_ = enabled;
    if (enabled_ && current_lr_ <= 0.0f) {
        current_lr_ = std::clamp(config_.base_learning_rate,
                                 config_.min_learning_rate,
                                 config_.max_learning_rate);
    }
}

void DynamicLRController::reset() {
    current_lr_ = std::clamp(config_.base_learning_rate,
                             config_.min_learning_rate,
                             config_.max_learning_rate);
    smoothed_grad_norm_ = 0.0f;
    smoothed_loss_ = 0.0f;
    prev_smoothed_loss_ = std::numeric_limits<float>::quiet_NaN();
    diagnostics_ = {};
    diagnostics_.applied_learning_rate = current_lr_;
    diagnostics_.proposed_learning_rate = current_lr_;
    diagnostics_.reason = DynamicLRDiagnostics::AdjustmentReason::None;
    diagnostics_.adjustment_applied = false;
    diagnostics_.cooldown_remaining = 0;
    diagnostics_.stable_band_steps = 0;
    diagnostics_.at_floor_steps = 0;
    diagnostics_.neutral_band_lower = config_.lower_grad_norm;
    diagnostics_.neutral_band_upper = config_.upper_grad_norm;
    diagnostics_.scaled_band_lower = diagnostics_.neutral_band_lower;
    diagnostics_.scaled_band_upper = diagnostics_.neutral_band_upper;
    diagnostics_.safety_band_lower = config_.band_floor;
    diagnostics_.safety_band_upper = config_.band_ceiling;
    diagnostics_.momentum_score = 0.0f;
    lr_override_.reset();
    step_ = 0;
    cooldown_ = 0;
    stable_band_steps_ = 0;
    at_floor_steps_ = 0;
    current_smoothing_ = std::clamp(config_.smoothing, 0.0f, 0.99f);
    current_lower_band_ = config_.lower_grad_norm;
    current_upper_band_ = config_.upper_grad_norm;
    neutral_lower_band_ = current_lower_band_;
    neutral_upper_band_ = current_upper_band_;
    scaled_lower_band_ = current_lower_band_;
    scaled_upper_band_ = current_upper_band_;
    safety_lower_band_ = config_.band_floor;
    safety_upper_band_ = config_.band_ceiling;
    baseline_capture_remaining_ = std::max(config_.baseline_capture_steps, 0);
    baseline_ready_ = (baseline_capture_remaining_ == 0);
    baseline_midpoint_ = 0.0f;
    baseline_half_span_ = 0.0f;
    momentum_counter_ = 0;
    safety_counter_ = 0;
    momentum_score_ = 0.0f;
    current_loss_spike_threshold_ = 0.0f;
    adaptive_cooldown_steps_ = std::max(config_.cooldown_steps, 0);
    runtime_min_lr_ = config_.min_learning_rate;
    runtime_max_lr_ = config_.max_learning_rate;
    grad_stats_.reset();
    loss_stats_.reset();
}

void DynamicLRController::setBaseLearningRate(float lr) {
    if (lr > config_.max_learning_rate) {
        config_.max_learning_rate = lr;
    }
    const float effective_min = std::min(runtime_min_lr_, runtime_max_lr_);
    const float effective_max = std::max(runtime_min_lr_, runtime_max_lr_);
    config_.base_learning_rate = std::clamp(lr, effective_min, effective_max);
    if (!lr_override_) {
        if (step_ <= 0) {
            current_lr_ = std::clamp(config_.base_learning_rate,
                                     effective_min,
                                     effective_max);
        } else {
            current_lr_ = std::clamp(current_lr_,
                                     effective_min,
                                     effective_max);
        }
    }
}

void DynamicLRController::setManualOverride(std::optional<float> lr_override) {
    lr_override_ = lr_override;
    if (lr_override_) {
        float proposed = std::clamp(*lr_override_,
                                    runtime_min_lr_,
                                    runtime_max_lr_);
        current_lr_ = proposed;
        diagnostics_.reason = DynamicLRDiagnostics::AdjustmentReason::ManualOverride;
        diagnostics_.applied_learning_rate = proposed;
        diagnostics_.proposed_learning_rate = proposed;
        diagnostics_.adjustment_applied = true;
    }
}

void DynamicLRController::setRuntimeLimits(float min_lr, float max_lr) {
    if (min_lr <= 0.0f || max_lr <= 0.0f) {
        runtime_min_lr_ = config_.min_learning_rate;
        runtime_max_lr_ = config_.max_learning_rate;
        return;
    }
    if (min_lr > max_lr) {
        std::swap(min_lr, max_lr);
    }
    runtime_min_lr_ = std::max(min_lr, 1e-8f);
    runtime_max_lr_ = std::max(max_lr, runtime_min_lr_ * 1.05f);
}

float DynamicLRController::update(float grad_norm, float loss, float scheduled_lr_ceiling) {
    diagnostics_ = {};
    diagnostics_.applied_learning_rate = current_lr_;
    diagnostics_.proposed_learning_rate = current_lr_;
    diagnostics_.smoothed_grad_norm = smoothed_grad_norm_;
    diagnostics_.smoothed_loss = smoothed_loss_;
    diagnostics_.adjustment_applied = false;
    diagnostics_.cooldown_remaining = cooldown_;
    diagnostics_.reason = DynamicLRDiagnostics::AdjustmentReason::None;
    diagnostics_.stable_band_steps = stable_band_steps_;
    diagnostics_.at_floor_steps = at_floor_steps_;

    // If scheduled ceiling provided, temporarily cap runtime max to respect decay schedule
    const float saved_max = runtime_max_lr_;
    if (scheduled_lr_ceiling > 0.0f) {
        runtime_max_lr_ = std::min(runtime_max_lr_, scheduled_lr_ceiling);
    }

    ++step_;

    if (!enabled_) {
        diagnostics_.reason = DynamicLRDiagnostics::AdjustmentReason::Disabled;
        diagnostics_.applied_learning_rate = current_lr_;
        diagnostics_.proposed_learning_rate = current_lr_;
        runtime_max_lr_ = saved_max;  // Restore before return
        return current_lr_;
    }

    if (lr_override_) {
        float proposed = std::clamp(*lr_override_,
                                    config_.min_learning_rate,
                                    config_.max_learning_rate);
        clampAndCommit(proposed, DynamicLRDiagnostics::AdjustmentReason::ManualOverride);
        diagnostics_.smoothed_grad_norm = smoothed_grad_norm_;
        diagnostics_.smoothed_loss = smoothed_loss_;
        runtime_max_lr_ = saved_max;  // Restore before return
        return current_lr_;
    }

    if (!isFinite(grad_norm) || !isFinite(loss)) {
        clampAndCommit(runtime_min_lr_,
                       DynamicLRDiagnostics::AdjustmentReason::InvalidSample);
        smoothed_grad_norm_ = config_.upper_grad_norm;
        smoothed_loss_ = std::max(loss, 0.0f);
        cooldown_ = activeCooldownSteps();
        diagnostics_.cooldown_remaining = cooldown_;
        runtime_max_lr_ = saved_max;  // Restore before return
        return current_lr_;
    }

    applySmoothing(grad_norm, loss);
    refreshAdaptiveParameters();
    updateBandScaling();

    diagnostics_.smoothed_grad_norm = smoothed_grad_norm_;
    diagnostics_.smoothed_loss = smoothed_loss_;
    diagnostics_.cooldown_remaining = cooldown_;
    diagnostics_.stable_band_steps = stable_band_steps_;
    diagnostics_.at_floor_steps = at_floor_steps_;

    // Track band / floor state
    if (inNeutralBand()) {
        stable_band_steps_++;
    } else {
        stable_band_steps_ = 0;
    }
    if (atLearningRateFloor()) {
        at_floor_steps_++;
    } else {
        at_floor_steps_ = 0;
    }
    diagnostics_.stable_band_steps = stable_band_steps_;
    diagnostics_.at_floor_steps = at_floor_steps_;

    if (step_ <= config_.warmup_steps) {
        applyWarmupPhase();
        prev_smoothed_loss_ = smoothed_loss_;
        runtime_max_lr_ = saved_max;  // Restore before return
        return current_lr_;
    }

    if (cooldown_ > 0) {
        --cooldown_;
        diagnostics_.cooldown_remaining = cooldown_;
        prev_smoothed_loss_ = smoothed_loss_;
        runtime_max_lr_ = saved_max;  // Restore before return
        return current_lr_;
    }

    // Attempt recovery if stuck at floor inside neutral band
    maybeApplyFloorRecovery();
    diagnostics_.stable_band_steps = stable_band_steps_;
    diagnostics_.at_floor_steps = at_floor_steps_;
    if (diagnostics_.reason == DynamicLRDiagnostics::AdjustmentReason::FloorRecovery) {
        prev_smoothed_loss_ = smoothed_loss_;
        runtime_max_lr_ = saved_max;  // Restore before return
        return current_lr_;
    }

    applyDynamicAdjustment();
    prev_smoothed_loss_ = smoothed_loss_;
    runtime_max_lr_ = saved_max;  // Restore before return
    return current_lr_;
}

void DynamicLRController::applySmoothing(float grad_norm, float loss) {
    const float smoothing = std::clamp(current_smoothing_, 0.0f, 0.99f);
    if (step_ <= 1 || !isFinite(smoothed_grad_norm_)) {
        smoothed_grad_norm_ = grad_norm;
    } else {
        smoothed_grad_norm_ = smoothing * smoothed_grad_norm_ + (1.0f - smoothing) * grad_norm;
    }

    if (step_ <= 1 || !isFinite(smoothed_loss_)) {
        smoothed_loss_ = loss;
    } else {
        smoothed_loss_ = smoothing * smoothed_loss_ + (1.0f - smoothing) * loss;
    }
}

void DynamicLRController::applyWarmupPhase() {
    if (config_.warmup_steps <= 0) {
        return;
    }

    float progress = static_cast<float>(step_) /
                     static_cast<float>(std::max(config_.warmup_steps, 1));
    progress = std::clamp(progress, 0.0f, 1.0f);
    float proposed = config_.base_learning_rate * progress;
    clampAndCommit(proposed, DynamicLRDiagnostics::AdjustmentReason::Warmup);
}

void DynamicLRController::refreshAdaptiveParameters() {
    grad_stats_.push(static_cast<double>(smoothed_grad_norm_));
    loss_stats_.push(static_cast<double>(smoothed_loss_));

    const float grad_mean = grad_stats_.meanf();
    const float grad_std = std::max(grad_stats_.stddevf(), 1e-6f);

    float lower_band = config_.lower_grad_norm;
    float upper_band = config_.upper_grad_norm;
    if (config_.auto_band && grad_stats_.samples() >= config_.band_min_samples) {
        const float sigma = std::max(config_.band_sigma, 0.1f);
        lower_band = grad_mean - sigma * grad_std;
        upper_band = grad_mean + sigma * grad_std;
        lower_band = std::max(lower_band, config_.band_floor);
        upper_band = std::min(upper_band, config_.band_ceiling);
        if ((upper_band - lower_band) < config_.band_min_span) {
            const float midpoint = grad_mean;
            const float half_span = config_.band_min_span * 0.5f;
            lower_band = midpoint - half_span;
            upper_band = midpoint + half_span;
        }
    }

    current_lower_band_ = std::max(lower_band, 0.0f);
    const float min_span = std::max(config_.band_min_span, 0.5f);
    current_upper_band_ = std::max(current_lower_band_ + min_span, upper_band);
    current_upper_band_ = std::min(current_upper_band_, config_.band_ceiling);

    diagnostics_.grad_mean = grad_mean;
    diagnostics_.grad_stddev = grad_std;
    neutral_lower_band_ = current_lower_band_;
    neutral_upper_band_ = current_upper_band_;
    diagnostics_.neutral_band_lower = neutral_lower_band_;
    diagnostics_.neutral_band_upper = neutral_upper_band_;

    if (config_.adaptive_smoothing) {
        const float variance = grad_stats_.variancef();
        const float reference = std::max(config_.variance_reference, 1e-3f);
        const float normalized = std::clamp(variance / reference, 0.0f, 1.0f);
        current_smoothing_ = config_.smoothing_min +
            (config_.smoothing_max - config_.smoothing_min) * normalized;
    } else {
        current_smoothing_ = std::clamp(config_.smoothing, 0.0f, 0.99f);
    }

    if (config_.adaptive_cooldown) {
        const float variance = grad_stats_.variancef();
        const float reference = std::max(config_.variance_reference, 1e-3f);
        const float normalized = std::clamp(variance / reference, 0.0f, 1.0f);
        const float cooldown_span = static_cast<float>(config_.cooldown_max - config_.cooldown_min);
        adaptive_cooldown_steps_ = static_cast<int>(std::round(
            static_cast<float>(config_.cooldown_min) + cooldown_span * normalized));
        adaptive_cooldown_steps_ = std::max(adaptive_cooldown_steps_, 1);
    } else {
        adaptive_cooldown_steps_ = std::max(config_.cooldown_steps, 0);
    }

    if (config_.adaptive_loss && loss_stats_.samples() >= config_.loss_min_samples) {
        const float loss_mean = loss_stats_.meanf();
        const float loss_std = std::max(loss_stats_.stddevf(), 1e-6f);
        current_loss_spike_threshold_ = std::max(config_.loss_floor,
                                                loss_mean + config_.loss_sigma * loss_std);
    } else {
        current_loss_spike_threshold_ = 0.0f;
    }
    diagnostics_.loss_spike_threshold = current_loss_spike_threshold_;
}

void DynamicLRController::applyDynamicAdjustment() {
    using AdjustmentReason = DynamicLRDiagnostics::AdjustmentReason;

    float proposed = current_lr_;
    AdjustmentReason reason = AdjustmentReason::None;
    bool changed = false;

    if (smoothed_grad_norm_ > safety_upper_band_) {
        proposed = current_lr_ * config_.decrease_factor;
        reason = AdjustmentReason::AggressiveGradients;
        changed = true;
    } else if (smoothed_grad_norm_ < safety_lower_band_) {
        proposed = current_lr_ * config_.increase_factor;
        reason = AdjustmentReason::VanishingGradients;
        changed = true;
    } else if (smoothed_grad_norm_ > current_upper_band_) {
        proposed = current_lr_ * config_.decrease_factor;
        reason = AdjustmentReason::AggressiveGradients;
        changed = true;
    } else if (smoothed_grad_norm_ < current_lower_band_) {
        proposed = current_lr_ * config_.increase_factor;
        reason = AdjustmentReason::VanishingGradients;
        changed = true;
    }

    const bool adaptive_loss_ready = config_.adaptive_loss && current_loss_spike_threshold_ > 0.0f;
    if (!changed && adaptive_loss_ready) {
        if (smoothed_loss_ > current_loss_spike_threshold_) {
            proposed = current_lr_ * config_.decrease_factor;
            reason = AdjustmentReason::LossSpike;
            changed = true;
        }
    } else if (!changed && isFinite(prev_smoothed_loss_) && prev_smoothed_loss_ > kEpsilon) {
        float loss_ratio = smoothed_loss_ / prev_smoothed_loss_;
        if (loss_ratio > config_.max_loss_jump) {
            proposed = current_lr_ * config_.decrease_factor;
            reason = AdjustmentReason::LossSpike;
            changed = true;
        }
    }

    if (!changed) {
        diagnostics_.reason = AdjustmentReason::None;
        diagnostics_.proposed_learning_rate = current_lr_;
        diagnostics_.applied_learning_rate = current_lr_;
        diagnostics_.adjustment_applied = false;
        return;
    }

    clampAndCommit(proposed, reason);
    cooldown_ = activeCooldownSteps();
    diagnostics_.cooldown_remaining = cooldown_;
}

void DynamicLRController::clampAndCommit(float proposed_lr,
                                         DynamicLRDiagnostics::AdjustmentReason reason) {
    float clamped = proposed_lr;

    const bool enforce_ratio_limits =
        reason == DynamicLRDiagnostics::AdjustmentReason::AggressiveGradients ||
        reason == DynamicLRDiagnostics::AdjustmentReason::VanishingGradients ||
        reason == DynamicLRDiagnostics::AdjustmentReason::LossSpike ||
        reason == DynamicLRDiagnostics::AdjustmentReason::FloorRecovery;

    if (enforce_ratio_limits && current_lr_ > 0.0f) {
        const float max_step_up = std::max(config_.max_step_up_ratio, 1.0f);
        const float max_step_down = std::clamp(config_.max_step_down_ratio, 0.0f, 1.0f);
        const float max_allowed = current_lr_ * max_step_up;
        const float min_allowed = current_lr_ * max_step_down;
        if (clamped > max_allowed) {
            clamped = max_allowed;
        }
        if (clamped < min_allowed) {
            clamped = min_allowed;
        }
    }

    clamped = std::clamp(clamped,
                         runtime_min_lr_,
                         runtime_max_lr_);

    diagnostics_.proposed_learning_rate = proposed_lr;
    diagnostics_.applied_learning_rate = clamped;
    diagnostics_.reason = reason;
    diagnostics_.adjustment_applied = std::fabs(clamped - current_lr_) > kEpsilon;
    current_lr_ = clamped;
}

bool DynamicLRController::inNeutralBand() const {
    return enabled_ && smoothed_grad_norm_ >= current_lower_band_ &&
           smoothed_grad_norm_ <= current_upper_band_;
}

bool DynamicLRController::atLearningRateFloor() const {
    return current_lr_ <= runtime_min_lr_ + 1e-12f;
}

void DynamicLRController::initializeSafetyBands(float midpoint, float half_span) {
    const float scale = std::max(config_.safety_scale, 1.0f);
    const float span = std::max(half_span * scale, half_span);
    safety_lower_band_ = std::max(midpoint - span, config_.band_floor);
    safety_upper_band_ = std::min(midpoint + span, config_.band_ceiling);
}

void DynamicLRController::updateBandScaling() {
    diagnostics_.scaled_band_lower = scaled_lower_band_;
    diagnostics_.scaled_band_upper = scaled_upper_band_;
    diagnostics_.safety_band_lower = safety_lower_band_;
    diagnostics_.safety_band_upper = safety_upper_band_;
    diagnostics_.momentum_score = momentum_score_;

    if (!baseline_ready_) {
        if (step_ > config_.warmup_steps && baseline_capture_remaining_ > 0) {
            baseline_capture_remaining_--;
            if (baseline_capture_remaining_ == 0) {
                baseline_ready_ = true;
                baseline_midpoint_ = 0.5f * (neutral_lower_band_ + neutral_upper_band_);
                baseline_half_span_ = std::max((neutral_upper_band_ - neutral_lower_band_) * 0.5f,
                                               config_.band_min_span * 0.5f);
                initializeSafetyBands(baseline_midpoint_, baseline_half_span_);
            }
        }

        scaled_lower_band_ = neutral_lower_band_;
        scaled_upper_band_ = neutral_upper_band_;
        current_lower_band_ = scaled_lower_band_;
        current_upper_band_ = scaled_upper_band_;
        diagnostics_.scaled_band_lower = scaled_lower_band_;
        diagnostics_.scaled_band_upper = scaled_upper_band_;
        diagnostics_.safety_band_lower = safety_lower_band_;
        diagnostics_.safety_band_upper = safety_upper_band_;
        diagnostics_.momentum_score = momentum_score_;
        return;
    }

    const float drift = std::clamp(config_.baseline_drift, 0.0f, 1.0f);
    if (drift > 0.0f) {
        const float observed_mid = 0.5f * (neutral_lower_band_ + neutral_upper_band_);
        const float observed_half = std::max((neutral_upper_band_ - neutral_lower_band_) * 0.5f,
                                             config_.band_min_span * 0.5f);
        baseline_midpoint_ = baseline_midpoint_ * (1.0f - drift) + observed_mid * drift;
        baseline_half_span_ = baseline_half_span_ * (1.0f - drift) + observed_half * drift;
    } else if (baseline_half_span_ <= 0.0f) {
        baseline_midpoint_ = 0.5f * (neutral_lower_band_ + neutral_upper_band_);
        baseline_half_span_ = std::max((neutral_upper_band_ - neutral_lower_band_) * 0.5f,
                                       config_.band_min_span * 0.5f);
    }

    const int momentum_interval = std::max(config_.momentum_interval, 1);
    momentum_counter_ = (momentum_counter_ + 1) % momentum_interval;
    if (momentum_counter_ == 0) {
        const float span = std::max(baseline_half_span_, 0.25f);
        const float grad_score = (baseline_midpoint_ - smoothed_grad_norm_) / span;
        float loss_score = 0.0f;
        if (std::isfinite(prev_smoothed_loss_)) {
            const float loss_den = std::max(std::fabs(prev_smoothed_loss_), 1.0f);
            if (loss_den > kEpsilon) {
                loss_score = (prev_smoothed_loss_ - smoothed_loss_) / loss_den;
            }
        }
        float combined = 0.6f * grad_score + 0.4f * loss_score;
        combined = std::clamp(combined, -3.0f, 3.0f);
        const float decay = std::clamp(config_.momentum_decay, 0.0f, 0.999f);
        momentum_score_ = decay * momentum_score_ + (1.0f - decay) * combined;
        momentum_score_ = std::clamp(momentum_score_, -1.5f, 1.5f);

        const int safety_interval = std::max(config_.safety_interval, 1);
        safety_counter_ = (safety_counter_ + 1) % safety_interval;
        if (safety_counter_ == 0) {
            const float adjust = momentum_score_ * config_.safety_gain * span;
            safety_upper_band_ = std::min(safety_upper_band_ + adjust, config_.band_ceiling);
            safety_lower_band_ = std::max(safety_lower_band_ - adjust, config_.band_floor);
            if (safety_lower_band_ >= safety_upper_band_) {
                const float mid = baseline_midpoint_;
                safety_lower_band_ = mid - span * config_.safety_scale;
                safety_upper_band_ = mid + span * config_.safety_scale;
                safety_lower_band_ = std::max(safety_lower_band_, config_.band_floor);
                safety_upper_band_ = std::min(safety_upper_band_, config_.band_ceiling);
            }
        }
    }

    const float span = std::max(baseline_half_span_, 0.25f);
    const float upper_scale = 1.0f + std::max(0.0f, momentum_score_) * config_.momentum_gain;
    const float lower_scale = 1.0f + std::max(0.0f, -momentum_score_) * config_.momentum_gain;
    float desired_upper = baseline_midpoint_ + span * upper_scale;
    float desired_lower = baseline_midpoint_ - span * lower_scale;
    desired_upper = std::min(desired_upper, safety_upper_band_);
    desired_lower = std::max(desired_lower, safety_lower_band_);

    scaled_lower_band_ = desired_lower;
    scaled_upper_band_ = desired_upper;
    current_lower_band_ = scaled_lower_band_;
    current_upper_band_ = scaled_upper_band_;
    diagnostics_.scaled_band_lower = scaled_lower_band_;
    diagnostics_.scaled_band_upper = scaled_upper_band_;
    diagnostics_.safety_band_lower = safety_lower_band_;
    diagnostics_.safety_band_upper = safety_upper_band_;
    diagnostics_.momentum_score = momentum_score_;
}

void DynamicLRController::maybeApplyFloorRecovery() {
    if (!enabled_) return;
    if (!atLearningRateFloor()) return;
    // Require several consecutive stable steps to avoid oscillation
    const int kMinStableBandSteps = 3;
    const int kMinFloorSteps = 6;
    if (stable_band_steps_ < kMinStableBandSteps || at_floor_steps_ < kMinFloorSteps) {
        return;
    }
    float proposed = current_lr_ * config_.increase_factor;
    clampAndCommit(proposed, DynamicLRDiagnostics::AdjustmentReason::FloorRecovery);
    // Reset stable band counter so we don't immediately stack recoveries
    stable_band_steps_ = 0;
    cooldown_ = std::max(activeCooldownSteps() / 2, 1); // short cooldown after recovery
    diagnostics_.cooldown_remaining = cooldown_;
    diagnostics_.stable_band_steps = stable_band_steps_;
    diagnostics_.at_floor_steps = at_floor_steps_;
}

int DynamicLRController::activeCooldownSteps() const {
    return std::max(adaptive_cooldown_steps_, 0);
}

void DynamicLRController::RunningStats::push(double value) {
    ++count;
    const double delta = value - mean;
    mean += delta / static_cast<double>(count);
    const double delta2 = value - mean;
    m2 += delta * delta2;
}

float DynamicLRController::RunningStats::meanf() const {
    if (count <= 0) {
        return 0.0f;
    }
    return static_cast<float>(mean);
}

float DynamicLRController::RunningStats::variancef() const {
    if (count <= 1) {
        return 0.0f;
    }
    return static_cast<float>(m2 / static_cast<double>(count - 1));
}

float DynamicLRController::RunningStats::stddevf() const {
    return std::sqrt(std::max(variancef(), 0.0f));
}

void DynamicLRController::RunningStats::reset() {
    mean = 0.0;
    m2 = 0.0;
    count = 0;
}

} // namespace DynamicLR

//======================================================//
//  DynamicLR Logging Implementation
//======================================================//

namespace DynamicLR::Logging {

namespace {

const char* ReasonToString(DynamicLRDiagnostics::AdjustmentReason reason) {
    using R = DynamicLRDiagnostics::AdjustmentReason;
    switch (reason) {
        case R::None: return "none";
        case R::Warmup: return "warmup";
        case R::AggressiveGradients: return "high_grad";
        case R::VanishingGradients: return "low_grad";
        case R::LossSpike: return "loss_spike";
        case R::ManualOverride: return "manual";
        case R::Disabled: return "disabled";
        case R::InvalidSample: return "invalid_sample";
        case R::FloorRecovery: return "floor_recovery";
        default: return "unknown";
    }
}

std::string FormatFloat(float value, int precision = 6) {
    std::ostringstream oss;
    oss.precision(precision);
    oss << std::fixed << value;
    return oss.str();
}

} // namespace

void RegisterLogCallback(LogCallback callback) {
    if (!callback) return;
    std::lock_guard<std::mutex> lock(g_dynlr_log_mutex);
    g_dynlr_log_callbacks.push_back(std::move(callback));
}

void ClearLogCallbacks() {
    std::lock_guard<std::mutex> lock(g_dynlr_log_mutex);
    g_dynlr_log_callbacks.clear();
}

void EmitLog(LogLevel level, std::string_view message) {
    std::lock_guard<std::mutex> lock(g_dynlr_log_mutex);
    for (const auto& cb : g_dynlr_log_callbacks) {
        if (cb) {
            cb(level, message);
        }
    }
}

void LogAdjustment(float base_lr, float proposed_lr, float applied_lr,
                   DynamicLRDiagnostics::AdjustmentReason reason,
                   float grad_norm, float loss) {
    std::ostringstream msg;
    msg << "[DynamicLR] base=" << FormatFloat(base_lr, 8)
        << " proposed=" << FormatFloat(proposed_lr, 8)
        << " applied=" << FormatFloat(applied_lr, 8)
        << " reason=" << ReasonToString(reason)
        << " grad_norm=" << FormatFloat(grad_norm, 4)
        << " loss=" << FormatFloat(loss, 4);
    EmitLog(LogLevel::Info, msg.str());
}

void LogFloorRecovery(float lr, int at_floor_steps) {
    std::ostringstream msg;
    msg << "[DynamicLR] floor_recovery lr=" << FormatFloat(lr, 8)
        << " at_floor_steps=" << at_floor_steps;
    EmitLog(LogLevel::Info, msg.str());
}

void LogBandUpdate(float lower, float upper, float safety_lower, float safety_upper) {
    std::ostringstream msg;
    msg << "[DynamicLR] band_update lower=" << FormatFloat(lower, 4)
        << " upper=" << FormatFloat(upper, 4)
        << " safety=[" << FormatFloat(safety_lower, 4)
        << ", " << FormatFloat(safety_upper, 4) << "]";
    EmitLog(LogLevel::Info, msg.str());
}

} // namespace DynamicLR::Logging

} // namespace GRIM

