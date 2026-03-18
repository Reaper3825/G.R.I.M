// Windows implementation of platform window/display API
#ifdef _WIN32

#include "platform_window.hpp"
#include "grim_platform.h"
#include <windows.h>

namespace PlatformWindow {

void* createBGFXInitWindow() {
    HWND hwnd = CreateWindowExW(
        0, L"STATIC", L"TempBGFXWindow",
        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
        1, 1, nullptr, nullptr, GetModuleHandle(nullptr), nullptr);
    return hwnd;
}

void destroyBGFXInitWindow(void* handle) {
    if (handle && IsWindow(static_cast<HWND>(handle)))
        DestroyWindow(static_cast<HWND>(handle));
}

void setWindowVisible(void* handle, bool visible) {
    if (handle && IsWindow(static_cast<HWND>(handle)))
        ShowWindow(static_cast<HWND>(handle), visible ? SW_SHOW : SW_HIDE);
}

void getVirtualScreenRect(int& x, int& y, int& width, int& height) {
    x = GetSystemMetrics(SM_XVIRTUALSCREEN);
    y = GetSystemMetrics(SM_YVIRTUALSCREEN);
    width = GetSystemMetrics(SM_CXVIRTUALSCREEN);
    height = GetSystemMetrics(SM_CYVIRTUALSCREEN);
}

bool pumpEvents(float& mouseWheelDeltaOut, bool& quitRequested) {
    mouseWheelDeltaOut = 0.0f;
    MSG msg{};
    while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
        if (msg.message == WM_MOUSEWHEEL) {
            mouseWheelDeltaOut = static_cast<float>(GET_WHEEL_DELTA_WPARAM(msg.wParam));
        }
        if (msg.message == WM_QUIT) {
            quitRequested = true;
        }
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    return true;
}

void* createOverlayWindow(int x, int y, int width, int height) {
    (void)x; (void)y; (void)width; (void)height;
    return nullptr;
}

} // namespace PlatformWindow

#endif
