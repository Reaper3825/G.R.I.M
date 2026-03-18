#pragma once
#include "core/grim_platform.h"

#ifdef _WIN32
#include "popup_anim.hpp"
#define WM_GRIM_SHOW_POPUP (WM_APP + 1)

// ===========================================================
// GRIM Popup UI Control Header
// ===========================================================

void runPopupUI(int width, int height);
void showPopup();
void hidePopup();
void notifyPopupActivity();

PopupAnimState getPopupAnimState();
float getPopupAlpha();
float getPopupScale();
float getPopupPulse();
bool isPopupVisible();

#else
// Stubs for non-Windows platforms (popup not yet implemented)
inline void runPopupUI(int, int) {}
inline void showPopup() {}
inline void hidePopup() {}
inline void notifyPopupActivity() {}
inline bool isPopupVisible() { return false; }
inline float getPopupAlpha() { return 0.0f; }
inline float getPopupScale() { return 1.0f; }
inline float getPopupPulse() { return 0.0f; }
#endif

namespace bx {
    void mtxSRT(
        float* result,
        float sx, float sy, float sz,
        float rx, float ry, float rz,
        float tx, float ty, float tz
    );
}
