#pragma once

#include <string>

namespace GRIM { namespace Perception { namespace Physical {

struct PhysicalCameraControlRange {
    bool   available = false;
    double minimum = 0.0;
    double maximum = 0.0;
    double step = 0.0;
    double default_value = 0.0;
};

struct PhysicalCameraControlProfile {
    PhysicalCameraControlRange exposure{};
    PhysicalCameraControlRange gain{};
    bool   manual_auto_exposure_value_available = false;
    double manual_auto_exposure_value = 0.0;
    std::string provider;
    std::string reason;

    bool SupportsMotionExposure() const {
        return exposure.available && gain.available
            && manual_auto_exposure_value_available;
    }
};

// Read-only native capability discovery. This never changes a camera setting.
PhysicalCameraControlProfile DiscoverPhysicalCameraControlProfile(
    int device_index,
    int requested_opencv_backend);

}}} // namespace GRIM::Perception::Physical
