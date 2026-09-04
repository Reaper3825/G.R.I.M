//======================================================//
//  SoftRestart.cu
//  Optimizer momentum reset helpers
//======================================================//


#include <cuda_runtime.h>
#include <iostream>
#include <mutex>
#include <sstream>
#include <vector>
#include "SoftRestart.hpp"
#include "../Optimizers/AdamW/AdamW_Kernal_GPU.hpp"
namespace GRIM {
namespace SoftRestart {

namespace {
// Logging infrastructure
std::mutex g_sr_log_mutex;
std::vector<Logging::LogCallback> g_sr_log_callbacks;
}

SoftRestartController::SoftRestartController(const SoftRestartConfig& config)
    : config_(config) {}

void SoftRestartController::markRestart(int64_t global_step) {
    state_.last_restart_step = global_step;
}

void SoftRestartController::restoreState(const SoftRestartState& state) {
    if (state.last_restart_step < -1) {
        throw std::runtime_error("SoftRestartController::restoreState: last_restart_step is invalid");
    }
    state_ = state;
}

void zeroOptimizerMoments(std::vector<ParameterGroup>& parameter_groups,
                          TrainingState& training_state,
                          GRIM::OptimizerStep* optimizer) {
    if (parameter_groups.empty()) {
        throw std::runtime_error("[SoftRestart] parameter_groups is empty, cannot zero moments");
    }

    // Reset all parameter-group moment buffers (m_state, v_state) via free function
    GRIM::resetAdamWMoments(parameter_groups,
                            training_state.stream_ctrl.getPrimaryStream());

    // OptimizerStep only tracks step count - moment buffers live in OptimizerState.
    // ParameterGroup borrows m_state/v_state, which are zeroed by resetAdamWMoments().
    if (optimizer) {
        optimizer->step = 0;  // Reset bias correction step counter
    }
}

void scaleOptimizerMoments(std::vector<ParameterGroup>& parameter_groups,
                           TrainingState& training_state,
                           float scale) {
    if (parameter_groups.empty()) {
        throw std::runtime_error("[SoftRestart] parameter_groups is empty, cannot scale moments");
    }

    // Scale all parameter-group moment buffers (m_state, v_state) via free function
    GRIM::scaleAdamWMoments(parameter_groups, scale,
                            training_state.stream_ctrl.getPrimaryStream());
    
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
