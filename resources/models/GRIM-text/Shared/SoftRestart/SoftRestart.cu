//======================================================//
//  SoftRestart.cu
//  Validation-spike detection and optimizer reset helpers
//======================================================//


#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <mutex>
#include <sstream>
#include <vector>
#include "SoftRestart.hpp"
#include "../../GRIM/grim_language_model_cuda.hpp"

namespace GRIM {
namespace SoftRestart {

namespace {
inline bool isFinite(float value) {
    return std::isfinite(static_cast<double>(value));
}

// Logging infrastructure
std::mutex g_sr_log_mutex;
std::vector<Logging::LogCallback> g_sr_log_callbacks;
}

SoftRestartController::SoftRestartController(const SoftRestartConfig& config)
    : config_(config) {}

bool SoftRestartController::shouldTrigger(float val_loss, int64_t global_step) {
    if (!isFinite(val_loss) || global_step < 0) {
        return false;
    }

    bool trigger = false;
    if (isFinite(state_.last_val_loss) && state_.last_val_step >= 0) {
        const float loss_delta = val_loss - state_.last_val_loss;
        const int64_t step_delta = global_step - state_.last_val_step;
        const int64_t since_restart = (state_.last_restart_step < 0)
            ? std::numeric_limits<int64_t>::max()
            : global_step - state_.last_restart_step;

        if (loss_delta > config_.loss_increase_threshold &&
            step_delta > 0 && step_delta <= config_.max_step_window &&
            since_restart >= config_.cooldown_steps) {
            trigger = true;
        }
    }

    state_.last_val_loss = val_loss;
    state_.last_val_step = global_step;
    return trigger;
}

void SoftRestartController::markRestart(int64_t global_step) {
    state_.last_restart_step = global_step;
}

void zeroOptimizerMoments(LanguageModel* model, OptimizerState* optimizer) {
    if (!model) {
        std::cerr << "[SoftRestart] Warning: model pointer is null, cannot zero moments." << std::endl;
        return;
    }

    // Reset all parameter-group moment buffers (m_state, v_state) managed by the model
    model->resetOptimizerMoments();

    // OptimizerState only tracks step count - no moment buffers to clear
    // ParameterGroup owns m_state/v_state, which are zeroed by resetOptimizerMoments()
    if (optimizer) {
        optimizer->step = 0;  // Reset bias correction step counter
    }
}

void scaleOptimizerMoments(LanguageModel* model, float scale) {
    if (!model) {
        std::cerr << "[SoftRestart] Warning: model pointer is null, cannot scale moments." << std::endl;
        return;
    }

    // Scale all parameter-group moment buffers (m_state, v_state)
    model->scaleOptimizerMoments(scale);
    
    // Don't reset step counter - we're just damping momentum, not restarting
}

//======================================================//
//  SoftRestart Logging Implementation
//======================================================//

namespace Logging {

namespace {

std::string FormatFloat(float value, int precision = 4) {
    std::ostringstream oss;
    oss.precision(precision);
    oss << std::fixed << value;
    return oss.str();
}

} // namespace

void RegisterLogCallback(LogCallback callback) {
    if (!callback) return;
    std::lock_guard<std::mutex> lock(g_sr_log_mutex);
    g_sr_log_callbacks.push_back(std::move(callback));
}

void ClearLogCallbacks() {
    std::lock_guard<std::mutex> lock(g_sr_log_mutex);
    g_sr_log_callbacks.clear();
}

void EmitLog(LogLevel level, std::string_view message) {
    std::lock_guard<std::mutex> lock(g_sr_log_mutex);
    for (const auto& cb : g_sr_log_callbacks) {
        if (cb) {
            cb(level, message);
        }
    }
}

void LogSpikeDetected(float loss_delta, int steps_since_last, bool momentum_reset) {
    std::ostringstream msg;
    msg << "[SoftRestart] spike loss_delta=" << FormatFloat(loss_delta)
        << " steps_since_last=" << steps_since_last
        << " momentum_reset=" << (momentum_reset ? "1" : "0");
    EmitLog(LogLevel::Warning, msg.str());
}

void LogRestartTriggered(int64_t global_step, float val_loss) {
    std::ostringstream msg;
    msg << "[SoftRestart] triggered step=" << global_step
        << " val_loss=" << FormatFloat(val_loss);
    EmitLog(LogLevel::Info, msg.str());
}

void LogCooldownActive(int64_t steps_remaining) {
    std::ostringstream msg;
    msg << "[SoftRestart] cooldown_active steps_remaining=" << steps_remaining;
    EmitLog(LogLevel::Info, msg.str());
}

} // namespace Logging

} // namespace SoftRestart
} // namespace GRIM
