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

#endif
