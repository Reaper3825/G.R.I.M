#pragma once
#include <windows.h>

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
namespace bx {
    void mtxSRT(
        float* result,
        float sx, float sy, float sz,
        float rx, float ry, float rz,
        float tx, float ty, float tz
    );
}
