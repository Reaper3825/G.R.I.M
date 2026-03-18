#pragma once
#include "core/grim_platform.h"
#include <bgfx/bgfx.h>
#include <string>

// ===========================================================
// Popup window interface
// ===========================================================

// Creates a layered overlay window (debug visible mode)
HWND createOverlayWindow(int width, int height);

// Queues an alpha readback from a texture (for per-pixel transparency)
void queueWindowAlphaReadback(int width, int height);

// Applies alpha map to layered window if ready
void applyWindowAlphaIfReady(HWND hwnd, int width, int height, uint32_t frameIdx);

// Apply animation state to window (scale, alpha, voice-reactive emissive)
void applyAnimationToWindow(HWND hwnd, int width, int height, float scale, float alpha, float voiceIntensity);
