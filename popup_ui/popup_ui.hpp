#pragma once
#include <windows.h>
#include "popup_anim.hpp"
#define WM_GRIM_SHOW_POPUP (WM_APP + 1)

// ===========================================================
// GRIM Popup UI Control Header
// ===========================================================
//
// Exports the top-level popup UI control loop and helper functions
// to show/hide/notify overlay activity.
//
// ===========================================================

// Main popup UI loop. Launches and manages bgfx rendering + alpha.
void runPopupUI(int width, int height);

// Control visibility of the popup window.
void showPopup();
void hidePopup();

// Notify the popup of user or system activity.
// Resets idle timers and triggers display.
void notifyPopupActivity();

// Get animation state (for rendering)
PopupAnimState getPopupAnimState();
float getPopupAlpha();
float getPopupScale();
float getPopupPulse();
bool isPopupVisible();

namespace bx {
    void mtxSRT(
        float* result,
        float sx, float sy, float sz,
        float rx, float ry, float rz,
        float tx, float ty, float tz
    );
}
