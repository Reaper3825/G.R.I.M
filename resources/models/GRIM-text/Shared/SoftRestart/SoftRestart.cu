//======================================================//
//  SoftRestart.cu
//  Validation-spike detection and optimizer reset helpers
//======================================================//


#include <cuda_runtime.h>
#include <iostream>
#include <mutex>
#include <sstream>
#include <vector>
#include "SoftRestart.hpp"
#include "../Loss/LossSignals/LossSignals.hpp"
#include "../../GRIM/grim_language_model_cuda.hpp"
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

bool SoftRestartController::shouldTrigger(const GRIM::Loss::LossSignals& signals,
                                          int64_t global_step) {
    if (global_step < 0) {
        return false;
    }
    if (!signals.validation_delta_spike) {
        return false;
    }
    const int64_t since_restart = (state_.last_restart_step < 0)
        ? std::numeric_limits<int64_t>::max()
        : global_step - state_.last_restart_step;
    return since_restart >= config_.cooldown_steps;
}

void SoftRestartController::markRestart(int64_t global_step) {
    state_.last_restart_step = global_step;
}

void zeroOptimizerMoments(LanguageModel* model, GRIM::OptimizerStep* optimizer) {
    if (!model) {
        std::cerr << "[SoftRestart] Warning: model pointer is null, cannot zero moments." << std::endl;
        return;
    }

    // Reset all parameter-group moment buffers (m_state, v_state) via free function
    GRIM::resetAdamWMoments(model->parameterGroups(),
                            model->getTrainingState().stream_ctrl.getPrimaryStream());

    // OptimizerStep only tracks step count - moment buffers live in OptimizerState.
    // ParameterGroup borrows m_state/v_state, which are zeroed by resetAdamWMoments().
    if (optimizer) {
        optimizer->step = 0;  // Reset bias correction step counter
    }
}

void scaleOptimizerMoments(LanguageModel* model, float scale) {
    if (!model) {
        std::cerr << "[SoftRestart] Warning: model pointer is null, cannot scale moments." << std::endl;
        return;
    }

    // Scale all parameter-group moment buffers (m_state, v_state) via free function
    GRIM::scaleAdamWMoments(model->parameterGroups(), scale,
                            model->getTrainingState().stream_ctrl.getPrimaryStream());
    
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
