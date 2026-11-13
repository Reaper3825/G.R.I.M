#pragma once

#include "PhysicalHandGestureResult.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace GRIM { namespace Perception { namespace Physical {

struct PhysicalHandGestureConfig {
    bool        enabled = true;
    std::string model_path;
    int         max_hands = 2;
    float       min_hand_detection_confidence = 0.5f;
    float       min_hand_presence_confidence  = 0.5f;
    float       min_tracking_confidence       = 0.5f;
    float       min_gesture_confidence        = 0.5f;
    double      target_fps                    = 30.0;
    double      minimum_adaptive_fps          = 12.0;
    bool        adaptive_fps                  = true;
};

// RGB pixels are owned by the caller and remain valid for the duration of the
// synchronous Process call. The worker performs this call off the main thread.
struct PhysicalHandGestureFrame {
    const uint8_t* rgb_data = nullptr;
    int            width = 0;
    int            height = 0;
    int            byte_count = 0;
    uint64_t       source_frame_counter = 0;
    uint64_t       source_capture_steady_ns = 0;
};

class IPhysicalHandGestureBackend {
public:
    virtual ~IPhysicalHandGestureBackend() = default;

    virtual bool Initialize(const PhysicalHandGestureConfig& config,
                            std::string& error) = 0;
    virtual bool Process(const PhysicalHandGestureFrame& frame,
                         std::vector<PhysicalHandObservation>& observations,
                         double& inference_ms,
                         std::string& error) = 0;
    virtual void Shutdown() noexcept = 0;

    virtual const char* BackendName() const noexcept = 0;
    virtual const char* BackendVersion() const noexcept = 0;
    virtual bool IsCompiled() const noexcept = 0;
    virtual bool TelemetryDisabled() const noexcept = 0;
};

std::unique_ptr<IPhysicalHandGestureBackend>
CreatePhysicalMediaPipeHandGestureBackend();

}}} // namespace GRIM::Perception::Physical
