//======================================================//
//  SoftRestart.hpp
//  Optimizer momentum reset helpers
//======================================================//

#pragma once

#include <cstdint>
#include <functional>
#include <string_view>

#include "../../GRIM/grim_language_model_cuda.hpp"
#include "../Optimizers/OptimizerStep.hpp"

namespace GRIM {

namespace SoftRestart {

struct SoftRestartConfig {
    int cooldown_steps = 200;                // Minimum steps between soft restarts
};

struct SoftRestartState {
    int64_t last_restart_step = -1;
};

class SoftRestartController {
public:
    explicit SoftRestartController(const SoftRestartConfig& config = {});

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
void zeroOptimizerMoments(LanguageModel* model, OptimizerStep* optimizer);
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
void LogRestartTriggered(int64_t global_step, float val_loss);
void LogCooldownActive(int64_t steps_remaining);

} // namespace Logging

} // namespace SoftRestart
} // namespace GRIM
