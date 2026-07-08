// Windows implementation of platform window/display API
#ifdef _WIN32

#include "platform_window.hpp"
#include "grim_platform.h"
#include <algorithm>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <windows.h>
#include <windowsx.h>

namespace {
std::atomic<bool> s_overlayBlurEnabled{true};
std::atomic<float> s_overlayBlurOpacity{0.99f};
std::atomic<int> s_overlayBlurIntensity{2};
std::atomic<unsigned int> s_overlayBlurGeneration{0};
std::mutex s_viewportInputMutex;
std::unordered_map<HWND, PlatformWindow::ViewportInputCallback> s_viewportInputCallbacks;

PlatformWindow::ViewportInputModifiers currentViewportModifiers()
{
    PlatformWindow::ViewportInputModifiers modifiers;
    modifiers.ctrl = (GetKeyState(VK_CONTROL) & 0x8000) != 0;
    modifiers.shift = (GetKeyState(VK_SHIFT) & 0x8000) != 0;
    modifiers.alt = (GetKeyState(VK_MENU) & 0x8000) != 0;
    return modifiers;
}

PlatformWindow::ViewportInputCallback viewportInputCallback(HWND hwnd)
{
    std::lock_guard lock(s_viewportInputMutex);
    auto it = s_viewportInputCallbacks.find(hwnd);
    if (it == s_viewportInputCallbacks.end())
        return {};
    return it->second;
}

bool dispatchViewportInput(HWND hwnd, const PlatformWindow::ViewportInputEvent& event)
{
    PlatformWindow::ViewportInputCallback callback = viewportInputCallback(hwnd);
    if (!callback)
        return false;

    callback(event);
    return true;
}

void dispatchViewportMouse(HWND hwnd,
                           PlatformWindow::ViewportInputEventType type,
                           PlatformWindow::ViewportMouseButton button,
                           LPARAM lParam)
{
    PlatformWindow::ViewportInputEvent event;
    event.type = type;
    event.button = button;
    event.x = GET_X_LPARAM(lParam);
    event.y = GET_Y_LPARAM(lParam);
    event.modifiers = currentViewportModifiers();
    dispatchViewportInput(hwnd, event);
}

void dispatchViewportKey(HWND hwnd, PlatformWindow::ViewportInputEventType type, WPARAM wParam, LPARAM lParam)
{
    PlatformWindow::ViewportInputEvent event;
    event.type = type;
    event.keyCode = static_cast<int>(wParam);
    event.repeat = type == PlatformWindow::ViewportInputEventType::KeyDown && ((lParam & (1L << 30)) != 0);
    event.modifiers = currentViewportModifiers();
    dispatchViewportInput(hwnd, event);
}

bool anyViewportMouseButtonDown()
{
    return (GetKeyState(VK_LBUTTON) & 0x8000) != 0
        || (GetKeyState(VK_RBUTTON) & 0x8000) != 0
        || (GetKeyState(VK_MBUTTON) & 0x8000) != 0
        || (GetKeyState(VK_XBUTTON1) & 0x8000) != 0
        || (GetKeyState(VK_XBUTTON2) & 0x8000) != 0;
}

LRESULT CALLBACK ViewportWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg) {
    case WM_ERASEBKGND:
        return 1;
    case WM_CLOSE:
        ShowWindow(hwnd, SW_HIDE);
        return 0;
    case WM_LBUTTONDOWN:
        SetFocus(hwnd);
        SetCapture(hwnd);
        dispatchViewportMouse(hwnd, PlatformWindow::ViewportInputEventType::MouseDown, PlatformWindow::ViewportMouseButton::Left, lParam);
        return 0;
    case WM_RBUTTONDOWN:
        SetFocus(hwnd);
        SetCapture(hwnd);
        dispatchViewportMouse(hwnd, PlatformWindow::ViewportInputEventType::MouseDown, PlatformWindow::ViewportMouseButton::Right, lParam);
        return 0;
    case WM_MBUTTONDOWN:
        SetFocus(hwnd);
        SetCapture(hwnd);
        dispatchViewportMouse(hwnd, PlatformWindow::ViewportInputEventType::MouseDown, PlatformWindow::ViewportMouseButton::Middle, lParam);
        return 0;
    case WM_XBUTTONDOWN:
        SetFocus(hwnd);
        SetCapture(hwnd);
        dispatchViewportMouse(hwnd,
                              PlatformWindow::ViewportInputEventType::MouseDown,
                              HIWORD(wParam) == XBUTTON1 ? PlatformWindow::ViewportMouseButton::X1 : PlatformWindow::ViewportMouseButton::X2,
                              lParam);
        return TRUE;
    case WM_LBUTTONUP:
        dispatchViewportMouse(hwnd, PlatformWindow::ViewportInputEventType::MouseUp, PlatformWindow::ViewportMouseButton::Left, lParam);
        if (!anyViewportMouseButtonDown() && GetCapture() == hwnd)
            ReleaseCapture();
        return 0;
    case WM_RBUTTONUP:
        dispatchViewportMouse(hwnd, PlatformWindow::ViewportInputEventType::MouseUp, PlatformWindow::ViewportMouseButton::Right, lParam);
        if (!anyViewportMouseButtonDown() && GetCapture() == hwnd)
            ReleaseCapture();
        return 0;
    case WM_MBUTTONUP:
        dispatchViewportMouse(hwnd, PlatformWindow::ViewportInputEventType::MouseUp, PlatformWindow::ViewportMouseButton::Middle, lParam);
        if (!anyViewportMouseButtonDown() && GetCapture() == hwnd)
            ReleaseCapture();
        return 0;
    case WM_XBUTTONUP:
        dispatchViewportMouse(hwnd,
                              PlatformWindow::ViewportInputEventType::MouseUp,
                              HIWORD(wParam) == XBUTTON1 ? PlatformWindow::ViewportMouseButton::X1 : PlatformWindow::ViewportMouseButton::X2,
                              lParam);
        if (!anyViewportMouseButtonDown() && GetCapture() == hwnd)
            ReleaseCapture();
        return TRUE;
    case WM_MOUSEMOVE:
        dispatchViewportMouse(hwnd, PlatformWindow::ViewportInputEventType::MouseMove, PlatformWindow::ViewportMouseButton::None, lParam);
        return 0;
    case WM_MOUSEWHEEL: {
        POINT point{GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)};
        ScreenToClient(hwnd, &point);
        PlatformWindow::ViewportInputEvent event;
        event.type = PlatformWindow::ViewportInputEventType::MouseWheel;
        event.x = point.x;
        event.y = point.y;
        event.wheelDelta = GET_WHEEL_DELTA_WPARAM(wParam);
        event.modifiers = currentViewportModifiers();
        dispatchViewportInput(hwnd, event);
        return 0;
    }
    case WM_KEYDOWN:
    case WM_SYSKEYDOWN:
        dispatchViewportKey(hwnd, PlatformWindow::ViewportInputEventType::KeyDown, wParam, lParam);
        return 0;
    case WM_KEYUP:
    case WM_SYSKEYUP:
        dispatchViewportKey(hwnd, PlatformWindow::ViewportInputEventType::KeyUp, wParam, lParam);
        return 0;
    case WM_KILLFOCUS: {
        PlatformWindow::ViewportInputEvent event;
        event.type = PlatformWindow::ViewportInputEventType::FocusLost;
        event.modifiers = currentViewportModifiers();
        dispatchViewportInput(hwnd, event);
        if (GetCapture() == hwnd)
            ReleaseCapture();
        return 0;
    }
    case WM_DESTROY:
        {
            std::lock_guard lock(s_viewportInputMutex);
            s_viewportInputCallbacks.erase(hwnd);
        }
        return 0;
    default:
        return DefWindowProcW(hwnd, msg, wParam, lParam);
    }
}

bool ensureViewportClassRegistered()
{
    HINSTANCE hInstance = GetModuleHandleW(nullptr);
    WNDCLASSEXW wc{};
    wc.cbSize = sizeof(WNDCLASSEXW);
    wc.lpfnWndProc = ViewportWndProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = L"GRIMViewportWindowClass";
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = reinterpret_cast<HBRUSH>(GetStockObject(BLACK_BRUSH));

    if (RegisterClassExW(&wc))
        return true;

    return GetLastError() == ERROR_CLASS_ALREADY_EXISTS;
}
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

void* createViewportWindow(void* overlayWindowHandle, const char* debugName)
{
    (void)overlayWindowHandle;

    if (!ensureViewportClassRegistered())
        return nullptr;

    std::wstring title = L"GRIM Viewport";
    if (debugName && debugName[0] != '\0') {
        std::string name(debugName);
        title.assign(name.begin(), name.end());
    }

    HWND hwnd = CreateWindowExW(
        WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
        L"GRIMViewportWindowClass",
        title.c_str(),
        WS_POPUP | WS_CLIPSIBLINGS | WS_CLIPCHILDREN,
        0, 0, 1, 1,
        nullptr,
        nullptr,
        GetModuleHandleW(nullptr),
        nullptr
    );

    if (!hwnd)
        return nullptr;

    ShowWindow(hwnd, SW_HIDE);
    return hwnd;
}

void destroyViewportWindow(void* viewportWindowHandle)
{
    HWND hwnd = static_cast<HWND>(viewportWindowHandle);
    if (hwnd && IsWindow(hwnd)) {
        setViewportInputCallback(viewportWindowHandle, {});
        DestroyWindow(hwnd);
    }
}

void setViewportWindowVisible(void* viewportWindowHandle, bool visible)
{
    HWND hwnd = static_cast<HWND>(viewportWindowHandle);
    if (!hwnd || !IsWindow(hwnd))
        return;

    ShowWindow(hwnd, visible ? SW_SHOWNOACTIVATE : SW_HIDE);
}

void setViewportInputCallback(void* viewportWindowHandle, ViewportInputCallback callback)
{
    HWND hwnd = static_cast<HWND>(viewportWindowHandle);
    if (!hwnd || !IsWindow(hwnd))
        return;

    std::lock_guard lock(s_viewportInputMutex);
    if (callback) {
        s_viewportInputCallbacks[hwnd] = std::move(callback);
    } else {
        s_viewportInputCallbacks.erase(hwnd);
    }
}

void setViewportWindowBounds(void* viewportWindowHandle,
                             void* overlayWindowHandle,
                             int x,
                             int y,
                             int width,
                             int height)
{
    HWND hwnd = static_cast<HWND>(viewportWindowHandle);
    HWND overlay = static_cast<HWND>(overlayWindowHandle);
    if (!hwnd || !IsWindow(hwnd))
        return;
    if (width <= 0 || height <= 0) {
        ShowWindow(hwnd, SW_HIDE);
        return;
    }

    int screenX = x;
    int screenY = y;
    if (overlay && IsWindow(overlay)) {
        RECT overlayRect{};
        if (GetWindowRect(overlay, &overlayRect)) {
            screenX = overlayRect.left + x;
            screenY = overlayRect.top + y;
        }
    }

    SetWindowPos(hwnd,
                 HWND_TOPMOST,
                 screenX,
                 screenY,
                 width,
                 height,
                 SWP_NOACTIVATE | SWP_SHOWWINDOW);

    if (overlay && IsWindow(overlay)) {
        SetWindowPos(overlay,
                     HWND_TOPMOST,
                     0,
                     0,
                     0,
                     0,
                     SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW);
    }
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
