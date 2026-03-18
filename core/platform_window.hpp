#pragma once

// =============================================================================
// Platform window/display abstraction for BGFX init and main loop
// Same UI flow on all platforms; only these APIs are OS-specific.
// =============================================================================

namespace PlatformWindow {

// Create a minimal window for BGFX init. Returns native handle (HWND on Windows,
// NSWindow* on macOS). Caller passes to WindowManager::initGlobalBGFX(handle).
void* createBGFXInitWindow();

void destroyBGFXInitWindow(void* handle);

void setWindowVisible(void* handle, bool visible);

// Virtual screen bounds (all displays). Used for overlay size.
void getVirtualScreenRect(int& x, int& y, int& width, int& height);

// Process one frame of events.
// mouseWheelDeltaOut: accumulated scroll delta for this frame.
// quitRequested: set to true if OS quit signal received.
bool pumpEvents(float& mouseWheelDeltaOut, bool& quitRequested);

// Create an overlay window spanning the given rect.
// Returns native handle (HWND on Windows, NSWindow* on macOS).
void* createOverlayWindow(int x, int y, int width, int height);

} // namespace PlatformWindow
