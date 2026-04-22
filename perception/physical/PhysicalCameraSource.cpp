#include "PhysicalCameraSource.hpp"

#include <stdexcept>

namespace GRIM { namespace Perception { namespace Physical {

std::string MakeDefaultRtspUrlForHost(const std::string& host) {
    if (host.empty()) {
        throw std::runtime_error(
            "MakeDefaultRtspUrlForHost: host is empty — caller MUST provide an address");
    }
    // Default RTSP port; user can override the full URL in the UI before opening.
    return "rtsp://" + host + ":554/";
}

}}} // namespace
