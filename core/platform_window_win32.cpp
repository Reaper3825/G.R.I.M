// Windows implementation of platform window/display API
#ifdef _WIN32

#include "platform_window.hpp"
#include "grim_platform.h"
#include <algorithm>
#include <atomic>
#include <windows.h>

namespace {
std::atomic<bool> s_overlayBlurEnabled{true};
std::atomic<float> s_overlayBlurOpacity{0.99f};
std::atomic<int> s_overlayBlurIntensity{2};
std::atomic<unsigned int> s_overlayBlurGeneration{0};
}

namespace PlatformWindow {

void setTextInputCallback(std::function<void(const std::string&)>) {
    // Windows uses WM_CHAR in OverlayWndProc; no callback needed
}

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
                         int panelCount)
{
    (void)overlayWindowHandle;
    (void)panelRects;
    (void)panelCount;
}

float getMenuBarHeight()
{
    return 0.0f;
}

void setOverlayBlurStyle(void* overlayWindowHandle,
                          bool enabled,
                          float opacity,
                          int intensity)
{
    (void)overlayWindowHandle;
    s_overlayBlurEnabled.store(enabled, std::memory_order_release);
    s_overlayBlurOpacity.store(std::clamp(opacity, 0.0f, 1.0f), std::memory_order_release);
    s_overlayBlurIntensity.store(std::clamp(intensity, 0, 5), std::memory_order_release);
    s_overlayBlurGeneration.fetch_add(1, std::memory_order_acq_rel);
}

OverlayBlurStyle getOverlayBlurStyle(void* overlayWindowHandle)
{
    (void)overlayWindowHandle;

    OverlayBlurStyle style;
    style.enabled = s_overlayBlurEnabled.load(std::memory_order_acquire);
    style.opacity = s_overlayBlurOpacity.load(std::memory_order_acquire);
    style.intensity = s_overlayBlurIntensity.load(std::memory_order_acquire);
    style.generation = s_overlayBlurGeneration.load(std::memory_order_acquire);
    return style;
}

void setOverlayClickThrough(void* overlayWindowHandle, bool clickThrough) {
    (void)overlayWindowHandle;
    (void)clickThrough;
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

    // Reuse cached GDI resources when the capture dimensions haven't changed.
    // Creating/destroying HDC+HBITMAP per call per panel per frame is extremely
    // expensive on Windows (~0.5ms each).
    static thread_local HDC s_hdcScreen = nullptr;
    static thread_local HDC s_hdcMem = nullptr;
    static thread_local HBITMAP s_hbm = nullptr;
    static thread_local HGDIOBJ s_oldObj = nullptr;
    static thread_local void* s_dibBits = nullptr;
    static thread_local int s_cachedW = 0;
    static thread_local int s_cachedH = 0;

    if (!s_hdcScreen) {
        s_hdcScreen = GetDC(nullptr);
        if (!s_hdcScreen) return false;
    }

    // Recreate memory DC and DIB section only when size changes
    if (!s_hdcMem || s_cachedW != width || s_cachedH != height) {
        if (s_oldObj && s_hdcMem) SelectObject(s_hdcMem, s_oldObj);
        if (s_hbm) DeleteObject(s_hbm);
        if (s_hdcMem) DeleteDC(s_hdcMem);

        s_hdcMem = CreateCompatibleDC(s_hdcScreen);
        if (!s_hdcMem) return false;

        BITMAPINFO bmi{};
        bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bmi.bmiHeader.biWidth = width;
        bmi.bmiHeader.biHeight = -height; // top-down
        bmi.bmiHeader.biPlanes = 1;
        bmi.bmiHeader.biBitCount = 32;
        bmi.bmiHeader.biCompression = BI_RGB;

        s_hbm = CreateDIBSection(s_hdcScreen, &bmi, DIB_RGB_COLORS, &s_dibBits, nullptr, 0);
        if (!s_hbm || !s_dibBits) {
            if (s_hbm) { DeleteObject(s_hbm); s_hbm = nullptr; }
            DeleteDC(s_hdcMem); s_hdcMem = nullptr;
            s_dibBits = nullptr;
            return false;
        }
        s_oldObj = SelectObject(s_hdcMem, s_hbm);
        s_cachedW = width;
        s_cachedH = height;
    }

    // Capture desktop pixels.
    BitBlt(s_hdcMem, 0, 0, width, height, s_hdcScreen, screenX, screenY, SRCCOPY);

    // Convert BGRA (little-endian DIB) to packed ARGB.
    const uint8_t* src = static_cast<const uint8_t*>(s_dibBits);
    for (int yy = 0; yy < height; ++yy) {
        const uint8_t* row = src + (size_t)yy * (size_t)width * 4;
        for (int xx = 0; xx < width; ++xx) {
            const uint8_t b = row[xx * 4 + 0];
            const uint8_t g = row[xx * 4 + 1];
            const uint8_t r = row[xx * 4 + 2];
            outPixelsARGB[yy * width + xx] = 0xFF000000u | (uint32_t(r) << 16) | (uint32_t(g) << 8) | uint32_t(b);
        }
    }

    return true;
}

} // namespace PlatformWindow

#endif
