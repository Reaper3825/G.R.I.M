#pragma once

#include <sstream>
#include <string>

#include "../../Shared/LogRecorder/LogRecorder.hpp"

#ifdef USE_CUDA
#include "../../Shared/TrainingState/TrainingState_GPU.hpp"
#endif

namespace GRIM::ForwardOps {

#ifdef USE_CUDA
inline bool ShouldLogForwardInfo() {
    return GRIM::Logging::GetModuleLogLevel(
               GRIM::Logging::ModuleIdToString(GRIM::Logging::ModuleId::ForwardPass)) <=
           GRIM::Logging::ModuleLogLevel::Info;
}

inline bool HasGradController(const TrainingState& ts) {
    return ts.grad_ctrl.controller().bufferCount() > 0;
}

inline std::string BuildForwardPrefix(const TrainingState& ts, const char* entry) {
    std::ostringstream oss;
    oss << "[" << entry << "]";
    if (HasGradController(ts)) {
        const auto& ctrl = ts.grad_ctrl.controller();
        oss << " grad=" << ctrl.stateString()
            << " micro=" << ctrl.currentMicroStep()
            << "/" << ctrl.accumSteps();
    }
    return oss.str();
}

inline void LogUnexpectedGradState(const TrainingState& ts, const char* entry) {
    if (!HasGradController(ts)) {
        return;
    }

    const auto& ctrl = ts.grad_ctrl.controller();
    const auto state = ctrl.state();
    if (state == GradControllerState::READY_FOR_STEP || state == GradControllerState::STEPPING) {
        std::ostringstream oss;
        oss << "[" << entry << "] forward invoked while grad_state=" << ctrl.stateString()
            << " micro=" << ctrl.currentMicroStep()
            << "/" << ctrl.accumSteps();
        GRIM::Logging::EmitModuleWarning(GRIM::Logging::ModuleId::ForwardPass, oss.str());
    }
}
#endif  // USE_CUDA

}  // namespace GRIM::ForwardOps

#define FWD_INFO(msg) \
    do { \
        if (GRIM::ForwardOps::ShouldLogForwardInfo()) { \
            std::ostringstream _oss; \
            _oss << msg; \
            GRIM::Logging::EmitModuleInfo(GRIM::Logging::ModuleId::ForwardPass, _oss.str()); \
        } \
    } while (0)

#define FWD_WARN(msg) \
    do { \
        std::ostringstream _oss; \
        _oss << msg; \
        GRIM::Logging::EmitModuleWarning(GRIM::Logging::ModuleId::ForwardPass, _oss.str()); \
    } while (0)

#define FWD_ERROR(msg) \
    do { \
        std::ostringstream _oss; \
        _oss << msg; \
        GRIM::Logging::EmitModuleError(GRIM::Logging::ModuleId::ForwardPass, _oss.str()); \
    } while (0)
