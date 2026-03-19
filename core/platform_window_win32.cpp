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

void setOverlayBlurMask(void* overlayWindowHandle,
                         const float* panelRects,
                         int panelCount,
                         float cornerRadius)
{
    (void)overlayWindowHandle;
    (void)panelRects;
    (void)panelCount;
    (void)cornerRadius;
}

bool captureDesktopBehindOverlay(void* overlayWindowHandle,
                                 int x,
                                 int y,
                                 int width,
                                 int height,
                                 uint32_t* outPixelsARGB)
{
    if (!overlayWindowHandle || !outPixelsARGB || width <= 0 || height <= 0)
        return false;

    HWND hwnd = static_cast<HWND>(overlayWindowHandle);

    RECT wr{};
    if (!GetWindowRect(hwnd, &wr))
        return false;

    const int screenX = wr.left + x;
    const int screenY = wr.top + y;

    HDC hdcScreen = GetDC(nullptr);
    if (!hdcScreen) return false;

    HDC hdcMem = CreateCompatibleDC(hdcScreen);
    if (!hdcMem) {
        ReleaseDC(nullptr, hdcScreen);
        return false;
    }

    BITMAPINFO bmi{};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = width;
    bmi.bmiHeader.biHeight = -height; // top-down
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;

    void* dibBits = nullptr;
    HBITMAP hbm = CreateDIBSection(hdcScreen, &bmi, DIB_RGB_COLORS, &dibBits, nullptr, 0);
    if (!hbm || !dibBits) {
        if (hbm) DeleteObject(hbm);
        DeleteDC(hdcMem);
        ReleaseDC(nullptr, hdcScreen);
        return false;
    }

    HGDIOBJ old = SelectObject(hdcMem, hbm);

    // Capture desktop pixels. Note: this may include our previous frame for overlapping areas.
    BitBlt(hdcMem, 0, 0, width, height, hdcScreen, screenX, screenY, SRCCOPY);

    // Convert BGRA (little-endian DIB) to packed ARGB.
    const uint8_t* src = static_cast<const uint8_t*>(dibBits);
    for (int yy = 0; yy < height; ++yy) {
        const uint8_t* row = src + (size_t)yy * (size_t)width * 4;
        for (int xx = 0; xx < width; ++xx) {
            const uint8_t b = row[xx * 4 + 0];
            const uint8_t g = row[xx * 4 + 1];
            const uint8_t r = row[xx * 4 + 2];
            outPixelsARGB[yy * width + xx] = 0xFF000000u | (uint32_t(r) << 16) | (uint32_t(g) << 8) | uint32_t(b);
        }
    }

    SelectObject(hdcMem, old);
    DeleteObject(hbm);
    DeleteDC(hdcMem);
    ReleaseDC(nullptr, hdcScreen);
    return true;
}

} // namespace PlatformWindow

#endif
