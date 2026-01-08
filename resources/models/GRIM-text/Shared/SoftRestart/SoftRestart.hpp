//======================================================//
//  SoftRestart.hpp
//  Validation-spike detection and optimizer reset helpers
//======================================================//

#pragma once

#include <cstdint>
#include <functional>
#include <limits>
#include <string_view>

namespace GRIM {

struct OptimizerState;
class LanguageModel;

namespace SoftRestart {

struct SoftRestartConfig {
    float loss_increase_threshold = 0.2f;    // Minimum delta to consider a spike
    int max_step_window = 50;                // Maximum steps between evaluations
    int cooldown_steps = 200;                // Minimum steps between soft restarts
};

struct SoftRestartState {
    float last_val_loss = std::numeric_limits<float>::quiet_NaN();
    int64_t last_val_step = -1;
    int64_t last_restart_step = -1;
};

class SoftRestartController {
public:
    explicit SoftRestartController(const SoftRestartConfig& config = {});

    // Returns true when the latest validation loss warrants a soft restart.
    bool shouldTrigger(float val_loss, int64_t global_step);

    // Record that a restart was executed.
    void markRestart(int64_t global_step);

    const SoftRestartState& state() const { return state_; }
    SoftRestartConfig& config() { return config_; }
    const SoftRestartConfig& config() const { return config_; }

private:
    SoftRestartConfig config_;
    SoftRestartState state_;
};

// Zero-out the optimizer momentum/variance buffers without touching weights.
void zeroOptimizerMoments(LanguageModel* model, OptimizerState* optimizer);
void scaleOptimizerMoments(LanguageModel* model, float scale);

//======================================================//
//  SoftRestart Logging Integration
//======================================================//

namespace Logging {

enum class LogLevel { Info, Warning, Error };

using LogCallback = std::function<void(LogLevel level, std::string_view message)>;

void RegisterLogCallback(LogCallback callback);
void ClearLogCallbacks();
void EmitLog(LogLevel level, std::string_view message);

inline void LogInfo(std::string_view message) { EmitLog(LogLevel::Info, message); }
inline void LogWarning(std::string_view message) { EmitLog(LogLevel::Warning, message); }
inline void LogError(std::string_view message) { EmitLog(LogLevel::Error, message); }

// Structured logging helpers
void LogSpikeDetected(float loss_delta, int steps_since_last, bool momentum_reset);
void LogRestartTriggered(int64_t global_step, float val_loss);
void LogCooldownActive(int64_t steps_remaining);

} // namespace Logging

} // namespace SoftRestart
} // namespace GRIM
