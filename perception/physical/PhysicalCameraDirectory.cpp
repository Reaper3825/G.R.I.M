#include "PhysicalCameraDirectory.hpp"
#include "PhysicalEnvironmentLogTag.hpp"
#include "PhysicalNicScan.hpp"

#include "control/devices/server/device_comm_server.hpp"
#include "control/devices/registry/device_snapshot.hpp"
#include "logger.hpp"

#include <opencv2/videoio.hpp>

#include <algorithm>
#include <stdexcept>

namespace GRIM { namespace Perception { namespace Physical {

std::vector<PhysicalCameraSource> EnumerateLocalCameraCandidates() {
    std::vector<PhysicalCameraSource> out;
    auto nics = ScanLocalNicAddresses(); // throws on OS error — fail loud

    for (const auto& nic : nics) {
        PhysicalCameraSource src;
        src.origin = PhysicalCameraOrigin::LocalNic;
        src.host   = nic.ipv4;
        src.label  = nic.interface_name + " (" + nic.ipv4 + ")";

        // A local NIC IP is THIS machine's own address — it is NOT a camera.
        // Dialing rtsp://<own-ip>:554/ fails because nothing is listening.
        // Surface these entries for informational/diagnostic use only so the
        // user knows which networks are reachable, but never auto-fill a URL
        // that cannot work. Rule 20: fail loud, never fabricate.
        src.status = PhysicalCameraCandidateStatus::Disabled;
        if (nic.is_loopback) {
            src.status_reason = "loopback interface — cannot host an IP camera";
        } else if (nic.is_link_local) {
            src.status_reason = "link-local 169.254.x.x — no DHCP assigned address";
        } else {
            src.status_reason =
                "this is your machine's own NIC IP — enter the REMOTE camera's "
                "RTSP/HTTP URL in the URL field manually";
        }
        // url_template intentionally left empty — Connect must use user-typed URL.
        out.push_back(std::move(src));
    }
    LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
              "EnumerateLocalCameraCandidates: produced " + std::to_string(out.size())
              + " entries");
    return out;
}

std::vector<PhysicalCameraSource>
EnumerateHubCameraCandidates(const ::GRIM::DeviceCommServer* server) {
    std::vector<PhysicalCameraSource> out;
    if (!server) {
        // Legitimate state: hub server not started. Honest empty list.
        return out;
    }

    const auto snapshots = server->listDeviceSnapshots();
    for (const auto& snap : snapshots) {
        const bool advertises_camera =
            std::find(snap.allowed_actions.begin(), snap.allowed_actions.end(),
                      std::string("camera")) != snap.allowed_actions.end();
        if (!advertises_camera) continue;

        PhysicalCameraSource src;
        src.origin    = PhysicalCameraOrigin::HubDevice;
        src.label     = snap.device_name + " [" + snap.device_id + "]";
        src.device_id = snap.device_id;

        // DeviceSnapshot does not currently carry an IP. Until the device
        // protocol is extended, fail loud so the UI shows exactly why we
        // can't connect — never silently fabricate an address.
        src.status        = PhysicalCameraCandidateStatus::NoNetworkAddress;
        src.status_reason =
            "hub device advertises camera but no IP recorded in DeviceSnapshot";
        out.push_back(std::move(src));
    }

    LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
              "EnumerateHubCameraCandidates: produced " + std::to_string(out.size())
              + " entries (server=" + (server ? "yes" : "no") + ")");
    return out;
}

std::vector<PhysicalCameraSource>
RefreshPhysicalCameraDirectory(const ::GRIM::DeviceCommServer* server) {
    auto devices = EnumerateLocalDeviceCameras();
    auto local   = EnumerateLocalCameraCandidates();
    auto hub     = EnumerateHubCameraCandidates(server);
    devices.insert(devices.end(),
                   std::make_move_iterator(local.begin()),
                   std::make_move_iterator(local.end()));
    devices.insert(devices.end(),
                   std::make_move_iterator(hub.begin()),
                   std::make_move_iterator(hub.end()));
    return devices;
}

std::vector<PhysicalCameraSource> EnumerateLocalDeviceCameras(int max_probe) {
    std::vector<PhysicalCameraSource> out;
    if (max_probe <= 0) {
        throw std::runtime_error(
            "EnumerateLocalDeviceCameras: max_probe must be > 0, got "
            + std::to_string(max_probe));
    }
    // On macOS CAP_ANY often picks an unsuitable backend for the FaceTime
    // camera. Try AVFoundation first, then fall back. On Linux CAP_V4L2 is
    // the native path; on Windows CAP_DSHOW or CAP_MSMF. We try the platform-
    // native backend first, then CAP_ANY.
    const std::vector<int> backends_to_try = {
#if defined(__APPLE__)
        cv::CAP_AVFOUNDATION,
#elif defined(_WIN32)
        cv::CAP_DSHOW,
        cv::CAP_MSMF,
#else
        cv::CAP_V4L2,
#endif
        cv::CAP_ANY
    };

    for (int i = 0; i < max_probe; ++i) {
        cv::VideoCapture cap;
        bool opened      = false;
        int  used_backend = -1;
        std::string last_throw;

        for (int backend : backends_to_try) {
            try {
                if (cap.open(i, backend)) {
                    opened       = true;
                    used_backend = backend;
                    break;
                }
            } catch (const std::exception& e) {
                last_throw = e.what();
            }
            cap.release();
        }

        if (opened) {
            PhysicalCameraSource src;
            src.origin       = PhysicalCameraOrigin::LocalDevice;
            src.status       = PhysicalCameraCandidateStatus::Ready;
            src.label        = "Local device " + std::to_string(i)
                             + " (backend=" + std::to_string(used_backend) + ")";
            src.url_template = "device:" + std::to_string(i);
            out.push_back(std::move(src));
            cap.release();
        } else if (i == 0) {
            PhysicalCameraSource src;
            src.origin        = PhysicalCameraOrigin::LocalDevice;
            src.label         = "Local device 0 (not present)";
            src.status        = PhysicalCameraCandidateStatus::Disabled;
            src.status_reason =
                last_throw.empty()
                ? std::string("cv::VideoCapture::open(0) returned false on every backend "
                              "(CAP_ANY + platform-native) — no attached camera, or macOS "
                              "camera permission (TCC) has not been granted to this binary yet. "
                              "Grant camera access in System Settings → Privacy & Security → Camera.")
                : std::string("cv::VideoCapture::open(0) threw on every backend; last what(): ")
                      + last_throw;
            out.push_back(std::move(src));
            break;
        } else {
            break;
        }
    }
    LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
              "EnumerateLocalDeviceCameras: produced " + std::to_string(out.size())
              + " entries (probed up to index " + std::to_string(max_probe - 1) + ")");
    return out;
}

}}} // namespace
