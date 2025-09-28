#include "popup_window.hpp"
#include "logger.hpp"
#include "system_detect.hpp"

#include <string>

// ===========================================================
// Win32 overlay window procedure
// ===========================================================
LRESULT CALLBACK OverlayProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
        default:
            return DefWindowProc(hwnd, msg, wParam, lParam);
    }
}

// ===========================================================
// Create overlay window (centered on chosen monitor)
// ===========================================================
// Create a 128x128 overlay and position it in the bottom-right of the
// primary monitor (with a small margin). The width/height params are kept
// for compatibility but are ignored.
HWND createOverlayWindow(int /*width*/, int /*height*/) {
    WNDCLASSW wc{};
    wc.lpfnWndProc   = OverlayProc;
    wc.hInstance     = GetModuleHandle(nullptr);
    wc.lpszClassName = L"SFML3_Overlay";

    ATOM reg = RegisterClassW(&wc);
    if (!reg) {
        DWORD err = GetLastError();
        if (err == ERROR_CLASS_ALREADY_EXISTS) {
            LOG_DEBUG("PopupWindow", "RegisterClassW: class already exists, reusing it");
        } else {
            LOG_ERROR("PopupWindow", "RegisterClassW failed, code=" + std::to_string(err));
            return nullptr;
        }
    }

    // Default position (will be overridden to bottom-right)
    int posX = 100;
    int posY = 100;

    // Enforce our desired overlay size
    constexpr int overlayW = 128;
    constexpr int overlayH = 128;

    SystemInfo sys = detectSystem();
    if (!sys.monitors.empty()) {
        // Prefer primary monitor
        const MonitorInfo* target = nullptr;
        for (const auto& m : sys.monitors) {
            if (m.isPrimary) {
                target = &m;
                break;
            }
        }
        if (!target) target = &sys.monitors.front();

    // Position bottom-right with a 16px margin
    const int margin = 16;
    posX = target->x + (target->width  - overlayW)  - margin;
    posY = target->y + (target->height - overlayH) - margin;

        LOG_DEBUG("PopupWindow",
                  "Centering on monitor (" + std::to_string(target->x) + "," +
                  std::to_string(target->y) + ") size=" +
                  std::to_string(target->width) + "x" +
                  std::to_string(target->height));
    } else {
        LOG_DEBUG("PopupWindow", "No monitor info, fallback to (100,100)");
    }

    LOG_DEBUG("PopupWindow", "Creating window at posX=" + std::to_string(posX) +
                             " posY=" + std::to_string(posY) +
                             " size=" + std::to_string(overlayW) + "x" + std::to_string(overlayH));

    HWND hwnd = CreateWindowExW(
        WS_EX_LAYERED | WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
        wc.lpszClassName,
        L"GRIM PopupUI",
        WS_POPUP,
        posX, posY, overlayW, overlayH,
        nullptr, nullptr, wc.hInstance, nullptr
    );

    if (!hwnd) {
        DWORD err = GetLastError();
        LOG_ERROR("PopupWindow", "CreateWindowExW failed, code=" + std::to_string(err));
        return nullptr;
    }

    // Log the actual window rect so we can verify the placement.
    RECT wr{};
    if (GetWindowRect(hwnd, &wr)) {
        LOG_DEBUG("PopupWindow", "HWND created (PID=" + std::to_string(GetCurrentProcessId()) + ") rect=(" +
                  std::to_string(wr.left) + "," + std::to_string(wr.top) + ")-(" +
                  std::to_string(wr.right) + "," + std::to_string(wr.bottom) + ")");
    } else {
        LOG_DEBUG("PopupWindow", "HWND created (PID=" + std::to_string(GetCurrentProcessId()) + ") (GetWindowRect failed)");
    }
    return hwnd;
}
