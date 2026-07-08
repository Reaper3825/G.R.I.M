#pragma once

#include <string>
#include <functional>
#include <cstdint>

// =============================================================================
// Platform window/display abstraction for BGFX init and main loop
// Same UI flow on all platforms; only these APIs are OS-specific.
// =============================================================================

namespace PlatformWindow {

enum class ViewportInputEventType {
    MouseMove,
    MouseDown,
    MouseUp,
    MouseWheel,
    KeyDown,
    KeyUp,
    FocusLost
};

enum class ViewportMouseButton {
    None,
    Left,
    Right,
    Middle,
    X1,
    X2
};

struct ViewportInputModifiers {
    bool ctrl = false;
    bool shift = false;
    bool alt = false;
};

struct ViewportInputEvent {
    ViewportInputEventType type = ViewportInputEventType::MouseMove;
    ViewportMouseButton button = ViewportMouseButton::None;
    int x = 0;
    int y = 0;
    int wheelDelta = 0;
    int keyCode = 0;
    bool repeat = false;
    ViewportInputModifiers modifiers{};
};

using ViewportInputCallback = std::function<void(const ViewportInputEvent&)>;

struct OverlayBlurStyle {
    bool enabled = true;
    float opacity = 0.99f;
    int intensity = 2;
    unsigned int generation = 0;
};

// Optional callback for injecting typed text (used on macOS; Windows uses WM_CHAR).
// Set before pumpEvents runs. Signature: void(const std::string&).
void setTextInputCallback(std::function<void(const std::string&)> callback);

// Create a minimal window for BGFX init. Returns native handle (HWND on Windows,
// NSWindow* on macOS). Caller passes to WindowManager::initGlobalBGFX(handle).
void* createBGFXInitWindow();

void destroyBGFXInitWindow(void* handle);

void setWindowVisible(void* handle, bool visible);

// Native viewport windows are positioned from overlay-window local pixels.
// They sit below CPU overlay cutouts and can later receive bgfx native-window framebuffers.
void* createViewportWindow(void* overlayWindowHandle, const char* debugName);
void destroyViewportWindow(void* viewportWindowHandle);
void setViewportWindowVisible(void* viewportWindowHandle, bool visible);
void setViewportInputCallback(void* viewportWindowHandle, ViewportInputCallback callback);
void setViewportWindowBounds(void* viewportWindowHandle,
                             void* overlayWindowHandle,
                             int x,
                             int y,
                             int width,
                             int height);

// Virtual screen bounds (all displays). Used for overlay size.
void getVirtualScreenRect(int& x, int& y, int& width, int& height);

// Process one frame of events.
// mouseWheelDeltaOut: accumulated scroll delta for this frame.
// quitRequested: set to true if OS quit signal received.
bool pumpEvents(float& mouseWheelDeltaOut, bool& quitRequested);

// Create an overlay window spanning the given rect.
// Returns native handle (HWND on Windows, NSWindow* on macOS).
void* createOverlayWindow(int x, int y, int width, int height);

// macOS: Constrain NSVisualEffectView blur so it only appears behind UI panels.
// For other platforms this can be a no-op.
// panelRects is an array of [x, y, w, h, cornerRadius] tuples (5 floats each).
void setOverlayBlurMask(void* overlayWindowHandle,
                         const float* panelRects,
                         int panelCount);

// Height of the macOS menu bar (or 0 on other platforms).
// Used to offset maximized panels below the system chrome.
float getMenuBarHeight();

// Update blur style at runtime: toggle, opacity (0..1), and intensity (layer count).
void setOverlayBlurStyle(void* overlayWindowHandle,
                          bool enabled,
                          float opacity,
                          int intensity);

// Read the current runtime blur style for the overlay window.
OverlayBlurStyle getOverlayBlurStyle(void* overlayWindowHandle);

// Toggle click-through on the overlay window.
// When clickThrough is true, all mouse events pass to windows behind the overlay.
// When false, the overlay captures mouse events normally.
void setOverlayClickThrough(void* overlayWindowHandle, bool clickThrough);

// Capture pixels from the real desktop behind the overlay window.
// Input coordinates are in overlay-window local space (origin = top-left of overlay window).
// Output pixels are packed ARGB: (a<<24)|(r<<16)|(g<<8)|b.
bool captureDesktopBehindOverlay(void* overlayWindowHandle,
                                 int x,
                                 int y,
                                 int width,
                                 int height,
                                 uint32_t* outPixelsARGB);

} // namespace PlatformWindow
