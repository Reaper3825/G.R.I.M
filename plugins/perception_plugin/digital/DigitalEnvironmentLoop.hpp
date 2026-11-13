#pragma once

#include <chrono>
#include <cstdint>
#include <string>

#include "DigitalCaptureTypes.hpp"

namespace GRIM { namespace Perception { namespace Digital {

struct DigitalEnvironmentConfig {
    DigitalCaptureRequest request{};
    std::chrono::milliseconds capture_interval{1000};
};

struct DigitalEnvironmentStatus {
    bool running = false;
    std::uint64_t capture_attempts = 0;
    std::uint64_t successful_captures = 0;
    std::uint64_t failed_captures = 0;
    std::uint64_t last_frame_counter = 0;
    DigitalCaptureStatus last_status = DigitalCaptureStatus::SourceUnavailable;
    std::string last_error;
    std::string backend;
    double last_capture_duration_ms = 0.0;
};

void StartDigitalEnvironment(const DigitalEnvironmentConfig& config = {});
void UpdateDigitalEnvironmentConfig(const DigitalEnvironmentConfig& config);
void RequestDigitalCaptureNow();
void ShutdownDigitalEnvironment();
bool IsDigitalEnvironmentRunning();
DigitalEnvironmentConfig GetDigitalEnvironmentConfigSnapshot();
DigitalEnvironmentStatus GetDigitalEnvironmentStatusSnapshot();

}}} // namespace GRIM::Perception::Digital
