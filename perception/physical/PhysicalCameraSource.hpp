#pragma once

#include <cstdint>
#include <string>

namespace GRIM { namespace Perception { namespace Physical {

// Origin of a discovered camera candidate.
enum class PhysicalCameraOrigin : uint8_t {
    LocalNic    = 0, // a NIC bound to this machine; informational only
    HubDevice   = 1, // a paired device in the local DeviceRegistry advertising "camera"
    LocalDevice = 2  // a camera directly attached to this host (USB/builtin), opened by index
};

// Why a candidate is/isn't connectable. NEVER falls back silently — UI surfaces this.
enum class PhysicalCameraCandidateStatus : uint8_t {
    Ready              = 0, // host known, URL constructable, attempt allowed
    NoNetworkAddress   = 1, // hub device has no IP recorded yet
    Disabled           = 2  // explicitly excluded (loopback, link-local, etc.)
};

// One row in the camera directory.
//
// `url_template` is what we hand to cv::VideoCapture. It is REQUIRED to be
// non-empty when status == Ready. UI may let the user override it before
// opening the stream.
struct PhysicalCameraSource {
    PhysicalCameraOrigin          origin           = PhysicalCameraOrigin::LocalNic;
    PhysicalCameraCandidateStatus status           = PhysicalCameraCandidateStatus::Disabled;
    std::string                   label;            // human readable, shown in UI
    std::string                   host;             // bare IPv4/IPv6, no scheme
    std::string                   url_template;     // full RTSP/HTTP URL to attempt
    std::string                   device_id;        // populated only when origin == HubDevice
    std::string                   status_reason;    // human readable reason when status != Ready
};

// Helper — build a default RTSP URL for a host. Centralised so all call sites
// agree on the convention. Throws if host is empty (Rule 20).
std::string MakeDefaultRtspUrlForHost(const std::string& host);

}}} // namespace GRIM::Perception::Physical
