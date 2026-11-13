#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace GRIM { namespace Perception { namespace Digital {

struct DigitalPerceptionPrimitivesConfig {
    bool ocr_enabled = true;
    bool automation_enabled = true;
    std::size_t max_automation_elements = 384;
};

struct DigitalPerceptionPrimitivesStatus {
    bool running = false;
    std::uint64_t processed_frames = 0;
    std::uint64_t last_source_frame_counter = 0;
    std::string ocr_provider;
    std::string automation_provider;
    std::string last_error;
    double last_ocr_duration_ms = 0.0;
    double last_automation_duration_ms = 0.0;
};

void StartDigitalPerceptionPrimitives(
    const DigitalPerceptionPrimitivesConfig& config = {});
void UpdateDigitalPerceptionPrimitivesConfig(
    const DigitalPerceptionPrimitivesConfig& config);
void ShutdownDigitalPerceptionPrimitives();
bool IsDigitalPerceptionPrimitivesRunning();
DigitalPerceptionPrimitivesConfig GetDigitalPerceptionPrimitivesConfigSnapshot();
DigitalPerceptionPrimitivesStatus GetDigitalPerceptionPrimitivesStatusSnapshot();

}}} // namespace GRIM::Perception::Digital
