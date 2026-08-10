#pragma once

#include "PhysicalCameraSource.hpp"
#include "PhysicalFrameConditioner.hpp"
#include "PhysicalStereoCapture.hpp"

#include <string>
#include <vector>

namespace GRIM { class DeviceCommServer; }

namespace GRIM { namespace Perception { namespace Physical {

// ─────────────────────────────────────────────────────────────────────────────
//  PhysicalEnvironmentLoop — THE single integration point with the main loop.
//
//  The mainloop calls TickPhysicalEnvironment() exactly once per frame.
//  On the first call (per process lifetime) the subsystem lazy-initialises:
//    - records the optional DeviceCommServer pointer for hub enumeration
//    - builds the initial camera directory
//    - leaves the active stream Idle (UI must explicitly choose a source)
//
//  Every subsequent call:
//    - drains the active PhysicalCameraStream into the PhysicalFrameBus
//    - notices Failed streams and surfaces them via GetLastEnvironmentError()
//
//  No other file in the codebase should touch PhysicalCameraStream directly
//  during the main loop. UI mutation (choose source, connect, disconnect)
//  goes through the Request* functions below; they are thread-safe.
// ─────────────────────────────────────────────────────────────────────────────

// Optional one-time configuration. If `server` is null, hub enumeration
// returns an empty list (legitimate at early startup).
void RegisterPhysicalEnvironmentDeviceServer(const ::GRIM::DeviceCommServer* server);

// Mainloop entry point. Cheap; safe to call every frame. Lazy-inits.
void TickPhysicalEnvironment();

// Stop worker threads, release the active stream, clear the frame bus.
void ShutdownPhysicalEnvironment();

bool IsPhysicalEnvironmentRunning();

// ── Read-only state for UI ──────────────────────────────────────────────────

std::vector<PhysicalCameraSource> GetPhysicalCameraDirectorySnapshot();
std::string                       GetActivePhysicalCameraUrl();
std::string                       GetActivePhysicalCameraLabel();
std::string                       GetLastEnvironmentError();
bool                              IsPhysicalCameraStreamActive();
bool                              IsPhysicalCameraStreamFailed();
double                            GetActiveStreamFps();
uint64_t                          GetActiveStreamFrameCounter();
PhysicalStereoCaptureStatus       GetPhysicalStereoCaptureStatusSnapshot();
bool                              IsPhysicalStereoCaptureActive();

PhysicalSignalConditioningConfig  GetPhysicalSignalConditioningConfigSnapshot();
PhysicalSignalConditioningStatus  GetPhysicalSignalConditioningStatusSnapshot();

// ── UI-facing requests ──────────────────────────────────────────────────────

// Re-runs local + hub enumeration. Throws on OS failure (Rule 20: NIC scan
// failure is real).
void RequestPhysicalCameraDirectoryRefresh();

// Open the given URL as the active source. Closes any prior stream first.
// Empty URL throws.
void RequestOpenPhysicalCameraSource(const std::string& url,
                                     const std::string& display_label);

// Close the active stream and clear the frame bus.
void RequestClosePhysicalCameraSource();

// Opens an explicit two-camera capture mode. Any active single-camera stream
// is closed first so a local device is never opened by both modes at once.
void RequestOpenPhysicalStereoCameraPair(const PhysicalStereoCaptureConfig& config);
void RequestClosePhysicalStereoCameraPair();

// Replaces the full config atomically. Throws on invalid config.
void RequestConfigurePhysicalSignalConditioning(const PhysicalSignalConditioningConfig& config);

// Restores default low-level signal conditioning settings.
void RequestResetPhysicalSignalConditioningDefaults();

}}} // namespace GRIM::Perception::Physical
