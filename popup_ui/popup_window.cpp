#include "core/grim_platform.h"
#ifdef _WIN32

#include "pch.hpp"
#include "popup_window.hpp"
#include "ui/ui_root.hpp"
#include "core/ui_sync.hpp"
#include <algorithm>
#include <sstream>
#define WM_GRIM_SHOW_POPUP (WM_APP + 1)

// Enable DWM blur-behind so popup translucency blurs the real desktop backdrop.
static void enableDwmBlurBehind(HWND hwnd)
{
    if (!hwnd)
        return;

    struct ACCENT_POLICY
    {
        int AccentState;
        int AccentFlags;
        int GradientColor;
        int AnimationId;
    };

    struct WINDOWCOMPOSITIONATTRIBDATA
    {
        int Attrib;
        PVOID pvData;
        SIZE_T cbData;
    };

    enum
    {
        WCA_ACCENT_POLICY = 19,
        ACCENT_ENABLE_BLURBEHIND = 3
    };

    HMODULE user32 = GetModuleHandleW(L"user32.dll");
    if (!user32)
        return;

    using SetWindowCompositionAttributeFn = BOOL(WINAPI*)(HWND, WINDOWCOMPOSITIONATTRIBDATA*);
    auto fn = reinterpret_cast<SetWindowCompositionAttributeFn>(
        GetProcAddress(user32, "SetWindowCompositionAttribute"));
    if (!fn)
        return;

    ACCENT_POLICY policy{};
    policy.AccentState = ACCENT_ENABLE_BLURBEHIND;
    policy.AccentFlags = 0;
    policy.GradientColor = 0;
    policy.AnimationId = 0;

    WINDOWCOMPOSITIONATTRIBDATA data{};
    data.Attrib = WCA_ACCENT_POLICY;
    data.pvData = &policy;
    data.cbData = sizeof(policy);

    fn(hwnd, &data);
}

// ===========================================================

// ===========================================================
// Custom Window Procedure
// ===========================================================
static LRESULT CALLBACK PopupWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    case WM_CLOSE:
        ShowWindow(hwnd, SW_HIDE);
        LOG_DEBUG("PopupWindow", "WM_CLOSE intercepted — hiding popup instead of closing");
        return 0;

    case WM_DESTROY:
        LOG_DEBUG("PopupWindow", "WM_DESTROY received — ignoring PostQuitMessage to keep GRIM alive");
        return 0;

    case WM_GRIM_SHOW_POPUP:
        LOG_DEBUG("PopupWindow", "Received WM_GRIM_SHOW_POPUP — showing popup (thread-safe)");
        ShowWindow(hwnd, SW_SHOW);
        return 0;

    // =====================================================
    // 🟢 Handle popup click (show or launch console)
    // =====================================================
    case WM_LBUTTONDOWN:
    {
        LOG_DEBUG("PopupWindow", "Popup clicked — showing GRIM console");
        auto consolePanel = UIRoot::get().getPanel("Console");
        if (consolePanel) {
            consolePanel->setVisible(true);
            LOG_DEBUG("PopupWindow", "Console panel shown");
        }
        return 0;
    }

    // Optional: Log window adjustments
    case WM_SIZE:
    case WM_MOVE:
    case WM_PAINT:
    case WM_DISPLAYCHANGE:
        break;

    default:
        return DefWindowProcW(hwnd, msg, wParam, lParam);
    }

    return DefWindowProcW(hwnd, msg, wParam, lParam);
}



    // ===========================================================
    // Window creation (Debug Visible Mode)
    // ===========================================================
    HWND createOverlayWindow(int width, int height)
    {
        // ======================================================
    // Wait for BGFX/UI lock before creating popup overlay
    // ======================================================
    auto start = std::chrono::steady_clock::now();

    // ======================================================
    // Non-blocking popup creation (no BGFX lock contention)
    // ======================================================
    LOG_DEBUG("PopupWindow", "Creating popup overlay (non-blocking mode)");
    HINSTANCE hInstance = GetModuleHandleW(nullptr);

    WNDCLASSW wc{};
    wc.lpfnWndProc   = PopupWndProc;  // Custom procedure prevents unwanted quit
    wc.hInstance     = hInstance;
    wc.lpszClassName = L"GRIMPopupClass";
    wc.hCursor       = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = nullptr;       // No background for layered window

    if (!RegisterClassW(&wc) && GetLastError() != ERROR_CLASS_ALREADY_EXISTS)
    {
        LOG_ERROR("PopupWindow", "RegisterClassW failed: " + std::to_string(GetLastError()));
        return nullptr;
    }
    LOG_TRACE("PW", "CreateWindowExW");
    HWND hwnd = CreateWindowExW(
        
        WS_EX_LAYERED | WS_EX_TOPMOST | WS_EX_NOACTIVATE,  // Remove WS_EX_TRANSPARENT
        L"GRIMPopupClass",
        L"GRIM Debug Popup",
        WS_POPUP,  // Remove WS_VISIBLE - we'll show it later
        100, 100, width, height,
        nullptr, nullptr, hInstance, nullptr
    );

    if (!hwnd)
    {
        LOG_ERROR("PopupWindow", "CreateWindowExW failed: " + std::to_string(GetLastError()));
        return nullptr;
    }

    UpdateWindow(hwnd);
    ShowWindow(hwnd, SW_SHOWDEFAULT);

    LOG_DEBUG("PopupWindow", "HWND created successfully (" +
        std::to_string(width) + "x" + std::to_string(height) + ")");

    return hwnd;
}


// ===========================================================
// Present 3D-rendered frame via UpdateLayeredWindow
// Input: straight-alpha BGRA8 pixels from offscreen readback
// ===========================================================
void presentPopup3DFrame(HWND hwnd, const uint8_t* bgraData, int width, int height)
{
    if (!hwnd || !IsWindow(hwnd) || !bgraData || width <= 0 || height <= 0)
        return;

    HDC hdcScreen = GetDC(nullptr);
    HDC hdcMem = CreateCompatibleDC(hdcScreen);

    BITMAPINFO bmi{};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = width;
    bmi.bmiHeader.biHeight = -height; // top-down
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;

    void* bits = nullptr;
    HBITMAP hBmp = CreateDIBSection(hdcMem, &bmi, DIB_RGB_COLORS, &bits, nullptr, 0);
    if (!hBmp)
    {
        DeleteDC(hdcMem);
        ReleaseDC(nullptr, hdcScreen);
        return;
    }

    // Readback data is already premultiplied from the GPU blend
    // (separate RGB/Alpha blend functions). UpdateLayeredWindow expects
    // premultiplied BGRA, which is exactly what we have — just copy.
    uint8_t* dst = static_cast<uint8_t*>(bits);
    std::memcpy(dst, bgraData, static_cast<size_t>(width) * height * 4);

    HBITMAP oldBmp = (HBITMAP)SelectObject(hdcMem, hBmp);

    POINT winPos{ 100, 100 };
    SIZE wndSize{ width, height };
    POINT srcPos{ 0, 0 };

    BLENDFUNCTION blend{};
    blend.BlendOp = AC_SRC_OVER;
    blend.SourceConstantAlpha = 255;
    blend.AlphaFormat = AC_SRC_ALPHA;

    UpdateLayeredWindow(hwnd, hdcScreen, &winPos, &wndSize, hdcMem,
                        &srcPos, 0, &blend, ULW_ALPHA);

    SelectObject(hdcMem, oldBmp);
    DeleteObject(hBmp);
    DeleteDC(hdcMem);
    ReleaseDC(nullptr, hdcScreen);
}

#endif // _WIN32
