#pragma once
#include <windows.h>

// ===========================================================
// Popup window creation + management
// ===========================================================

// Custom Win32 window procedure
LRESULT CALLBACK OverlayProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

// Create the overlay window (centered on chosen monitor)
HWND createOverlayWindow(int width, int height);
