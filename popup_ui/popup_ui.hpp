#pragma once

// ===========================================================
// Popup UI control (public API)
// ===========================================================

// Launch the popup UI loop in its own thread
void runPopupUI(int width, int height);

// Show the popup immediately
void showPopup();

// Hide the popup immediately
void hidePopup();

// Notify that GRIM activity occurred (voice speaking, command, etc.)
// - Ensures the popup is visible
// - Resets idle timer so it stays open a few seconds after activity
void notifyPopupActivity();
