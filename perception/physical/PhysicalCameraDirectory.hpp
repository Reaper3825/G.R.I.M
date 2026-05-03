#pragma once

#include "PhysicalCameraSource.hpp"

#include <string>
#include <vector>

namespace GRIM { class DeviceCommServer; }

namespace GRIM { namespace Perception { namespace Physical {

// Build the candidate list from this machine's NICs.
//
// One entry per non-loopback IPv4. Loopback / link-local interfaces are
// included with status=Disabled and a status_reason explaining why, so the
// UI can show the full picture instead of pretending they don't exist.
std::vector<PhysicalCameraSource> EnumerateLocalCameraCandidates();

// Probe cameras directly attached to this host (USB webcams, builtin
// FaceTime/iSight camera, virtual cameras, etc.) via OpenCV backends.
// Probes every device index in 0..max_probe-1, emitting Ready entries with
// backend-pinned url_template values ("device:<N>?backend=<CAP_ID>") for
// indices that open successfully. Gaps are tolerated because Windows camera
// index assignment is not guaranteed to be contiguous across virtual + USB
// devices.
//
// This uses a short-lived cv::VideoCapture open/close on each index — there
// is no cheaper "list attached cameras" API that works across macOS/Linux/
// Windows without adding AVFoundation/MediaFoundation/v4l2 directly.
std::vector<PhysicalCameraSource> EnumerateLocalDeviceCameras(int max_probe = 16);

// Build the candidate list from devices currently registered with the local
// hub (DeviceCommServer). Only includes devices whose `allowed_actions`
// contains the literal string "camera".
//
// `server` MAY be nullptr (e.g. when the hub server hasn't started). In that
// case this returns an empty list — it does NOT throw, because "no hub yet"
// is a legitimate runtime state at startup.
//
// If a hub camera device has no recorded IP, its entry is emitted with
// status=NoNetworkAddress and a clear status_reason. We never silently skip.
std::vector<PhysicalCameraSource>
EnumerateHubCameraCandidates(const ::GRIM::DeviceCommServer* server);

// Convenience: local + hub, concatenated.
std::vector<PhysicalCameraSource>
RefreshPhysicalCameraDirectory(const ::GRIM::DeviceCommServer* server);

}}} // namespace GRIM::Perception::Physical
