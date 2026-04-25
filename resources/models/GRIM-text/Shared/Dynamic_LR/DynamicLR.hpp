//======================================================//
//  DynamicLR.hpp
//  Adaptive learning-rate controller (host side)
//======================================================//

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

namespace GRIM::Loss { struct LossSignals; }

namespace GRIM {
namespace DynamicLR {

struct DynamicLRConfig {
    float base_learning_rate = 3.0e-5f;
    float min_learning_rate = 1.0e-6f;
    float max_learning_rate = 3.0e-4f;
    float increase_factor = 1.05f;
    float decrease_factor = 0.5f;
    float target_grad_norm = 8.0f;
    float upper_grad_norm = 12.0f;
    float lower_grad_norm = 4.0f;
    // NOTE: loss-spike detection (max_loss_jump / adaptive_loss / loss_sigma /
    // loss_min_samples / loss_floor) used to live here. It now lives in
    // GRIM::Loss::LossSignalConfig (Shared/Loss/LossSignals/). DynamicLR is a
    // pure subscriber to LossSignals::smoothed_spike for LR cuts.
    float smoothing = 0.2f;
    int warmup_steps = 25;
    int cooldown_steps = 5;
    float max_step_up_ratio = 1.12f;   // cap per-step LR increase (comfort band)
    float max_step_down_ratio = 0.72f; // cap per-step LR decrease
    bool auto_band = true;
    float band_sigma = 1.5f;
    float band_floor = 3.0f;
    float band_ceiling = 250.0f;
    int band_min_samples = 12;
    float band_min_span = 1.0f;
    bool adaptive_smoothing = true;
    float smoothing_min = 0.1f;
    float smoothing_max = 0.6f;
    float variance_reference = 25.0f;
    bool adaptive_cooldown = true;
    int cooldown_min = 1;
    int cooldown_max = 8;
    bool guard_logging = true;
    int guard_floor_steps = 200;
    float guard_grad_multiplier = 1.6f;
    int guard_loss_patience = 12;
    float guard_loss_multiplier = 1.05f;
    int baseline_capture_steps = 32;
    float baseline_drift = 0.05f;
    int momentum_interval = 12;
    float momentum_gain = 0.35f;
    float momentum_decay = 0.7f;
    int safety_interval = 4;
    float safety_gain = 0.1f;
    float safety_scale = 2.4f;
};

struct DynamicLRDiagnostics {
    enum class AdjustmentReason {
        None,
        Warmup,
        AggressiveGradients,
        VanishingGradients,
        LossSpike,
        ManualOverride,
        Disabled,
        InvalidSample,
        FloorRecovery
    };

    float proposed_learning_rate = 0.0f;
    float applied_learning_rate = 0.0f;
    float smoothed_grad_norm = 0.0f;
    float smoothed_loss = 0.0f;
    bool adjustment_applied = false;
    AdjustmentReason reason = AdjustmentReason::None;
    int cooldown_remaining = 0;
    int stable_band_steps = 0;      // consecutive steps inside neutral grad band
    int at_floor_steps = 0;         // consecutive steps stuck at min LR
    float neutral_band_lower = 0.0f;
    float neutral_band_upper = 0.0f;
    float scaled_band_lower = 0.0f;
    float scaled_band_upper = 0.0f;
    float safety_band_lower = 0.0f;
    float safety_band_upper = 0.0f;
    float grad_mean = 0.0f;
    float grad_stddev = 0.0f;
    float loss_spike_threshold = 0.0f;
    float momentum_score = 0.0f;
};

class DynamicLRController {
public:
    explicit DynamicLRController(const DynamicLRConfig& config = {});

    void setEnabled(bool enabled);
    bool enabled() const { return enabled_; }

    void reset();
    void setBaseLearningRate(float lr);

    float update(float grad_rms,
                 float loss,
                 const GRIM::Loss::LossSignals& signals,
                 float scheduled_lr_ceiling = -1.0f);
    float currentLearningRate() const { return current_lr_; }
    void setRuntimeLimits(float min_lr, float max_lr);

    void setManualOverride(std::optional<float> lr_override);
    std::optional<float> manualOverride() const { return lr_override_; }

    const DynamicLRDiagnostics& diagnostics() const { return diagnostics_; }

private:
    DynamicLRConfig config_;
    bool enabled_ = false;
    float current_lr_ = 0.0f;
    float smoothed_grad_norm_ = 0.0f;
    float smoothed_loss_ = 0.0f;  // diagnostics-only EMA; not used for spike detection
    // Previous smoothed loss is kept ONLY to compute the loss-trend component
    // of momentum_score_; spike detection lives in GRIM::Loss::LossSignalBus.
    float prev_smoothed_loss_ = std::numeric_limits<float>::quiet_NaN();
    int step_ = 0;
    int cooldown_ = 0;
    float current_smoothing_ = 0.2f;
    float current_lower_band_ = 0.0f;
    float current_upper_band_ = 0.0f;
    int adaptive_cooldown_steps_ = 5;
    float runtime_min_lr_ = 0.0f;
    float runtime_max_lr_ = 0.0f;
    std::optional<float> lr_override_;
    DynamicLRDiagnostics diagnostics_{};
    int stable_band_steps_ = 0;
    int at_floor_steps_ = 0;
    float neutral_lower_band_ = 0.0f;
    float neutral_upper_band_ = 0.0f;
    float scaled_lower_band_ = 0.0f;
    float scaled_upper_band_ = 0.0f;
    float safety_lower_band_ = 0.0f;
    float safety_upper_band_ = 0.0f;
    int baseline_capture_remaining_ = 0;
    bool baseline_ready_ = false;
    float baseline_midpoint_ = 0.0f;
    float baseline_half_span_ = 0.0f;
    int momentum_counter_ = 0;
    int safety_counter_ = 0;
    float momentum_score_ = 0.0f;
    struct RunningStats {
        double mean = 0.0;
        double m2 = 0.0;
        std::int64_t count = 0;

        void push(double value);
        float meanf() const;
        float variancef() const;
        float stddevf() const;
        void reset();
        std::int64_t samples() const { return count; }
    };
    RunningStats grad_stats_{};
    // loss_stats_ removed — LossSignalBus owns the EWMA mean/var used for
    // SmoothedSpike. DynamicLR no longer recomputes its own loss statistics.

    void applySmoothing(float grad_rms, float loss);
    void applyWarmupPhase();
    void applyDynamicAdjustment(const GRIM::Loss::LossSignals& signals);
    void clampAndCommit(float proposed_lr, DynamicLRDiagnostics::AdjustmentReason reason);
    bool inNeutralBand() const;
    bool atLearningRateFloor() const;
    void maybeApplyFloorRecovery();
    void refreshAdaptiveParameters();
    void updateBandScaling();
    void initializeSafetyBands(float midpoint, float half_span);
    int activeCooldownSteps() const;
};

} // namespace DynamicLR

//======================================================//
//  DynamicLR Logging Integration
//======================================================//

namespace DynamicLR::Logging {

enum class LogLevel { Info, Warning, Error };

using LogCallback = std::function<void(LogLevel level, std::string_view message)>;

void RegisterLogCallback(LogCallback callback);
void ClearLogCallbacks();
void EmitLog(LogLevel level, std::string_view message);

inline void LogInfo(std::string_view message) { EmitLog(LogLevel::Info, message); }
inline void LogWarning(std::string_view message) { EmitLog(LogLevel::Warning, message); }
inline void LogError(std::string_view message) { EmitLog(LogLevel::Error, message); }

// Structured logging helpers
void LogAdjustment(float base_lr, float proposed_lr, float applied_lr,
                   DynamicLRDiagnostics::AdjustmentReason reason,
                   float grad_rms, float loss);
void LogFloorRecovery(float lr, int at_floor_steps);
void LogBandUpdate(float lower, float upper, float safety_lower, float safety_upper);

} // namespace DynamicLR::Logging

} // namespace GRIM

