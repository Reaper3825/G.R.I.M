#pragma once
#include "core/grim_platform.h"

#ifdef _WIN32
#include <bgfx/bgfx.h>
#include <string>

// ===========================================================
// Popup window interface (Win32 layered window)
// ===========================================================

HWND createOverlayWindow(int width, int height);
void queueWindowAlphaReadback(int width, int height);
void applyWindowAlphaIfReady(HWND hwnd, int width, int height, uint32_t frameIdx);
void applyAnimationToWindow(HWND hwnd, int width, int height, float scale, float alpha, float voiceIntensity);

// Present a 3D-rendered BGRA frame (straight alpha) to the layered window.
void presentPopup3DFrame(HWND hwnd, const uint8_t* bgraData, int width, int height);

#else
#include <cstdint>

// ===========================================================
// Popup window interface (macOS — NSWindow + CALayer)
// ===========================================================

// Create a borderless floating NSWindow for popup rendering.
// Returns native handle (NSWindow* as void*).
void* createPopupWindow(int width, int height);

void showPopupWindow(void* handle);
void hidePopupWindow(void* handle);
bool isPopupWindowValid(void* handle);

// Present a 3D-rendered BGRA frame (straight alpha) to the popup CALayer.
void presentPopup3DFrame(void* handle, const uint8_t* bgraData, int width, int height);

// Hit-test: is the screen point inside the popup window?
bool isPointInPopupWindow(void* handle, int screenX, int screenY);

#endif
