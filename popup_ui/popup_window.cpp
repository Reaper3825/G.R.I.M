#include "pch.hpp"
#include "popup_window.hpp"
#include "ui/console_ui.hpp"
#include <windows.h>
#include <stb/stb_image.h>
#include "core/ui_sync.hpp"
#include <algorithm>
#include <sstream>
#define WM_GRIM_SHOW_POPUP (WM_APP + 1)


// ===========================================================
// Globals
// ===========================================================
std::atomic<bool> g_alphaReady{ false };
std::vector<uint8_t> g_alphaPixels;
std::mutex g_alphaMutex;

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

    case WM_SHOWWINDOW:
        LOG_DEBUG("PopupWindow", "WM_SHOWWINDOW received - wParam: " + std::to_string(wParam) +
                  ", lParam: " + std::to_string(lParam));
        break;

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
        GRIMConsole::showConsole();
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
        
        WS_EX_LAYERED | WS_EX_TOPMOST,  // Layered for transparency
        L"GRIMPopupClass",
        L"GRIM Debug Popup",
        WS_POPUP | WS_VISIBLE,
        50, 50, width, height,          // Initial position and size
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
// Load alpha directly from Oreo RGBA map
// ===========================================================
void queueWindowAlphaReadback(int width, int height)
{
    int diffuseW = 0, diffuseH = 0, diffuseC = 0;
    unsigned char* diffuseData = stbi_load("D:/G.R.I.M/resources/shaders/g_sprite_Diffuse.png", &diffuseW, &diffuseH, &diffuseC, 4);
    if (!diffuseData)
    {
        LOG_ERROR("PopupWindow", "Failed to load diffuse texture g_sprite_Diffuse.png");
        return;
    }

    int oreoW = 0, oreoH = 0, oreoC = 0;
    unsigned char* oreoData = stbi_load("D:/G.R.I.M/resources/shaders/g_sprite_oreo.png", &oreoW, &oreoH, &oreoC, 4);
    if (!oreoData)
    {
        LOG_ERROR("PopupWindow", "Failed to load oreo alpha texture g_sprite_oreo.png");
        stbi_image_free(diffuseData);
        return;
    }

    LOG_DEBUG("PopupWindow", "Loaded diffuse " + std::to_string(diffuseW) + "x" + std::to_string(diffuseH) + " and oreo " + std::to_string(oreoW) + "x" + std::to_string(oreoH));

    std::vector<uint8_t> combinedData(width * height * 4);

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int srcX = (x * diffuseW) / width;
            int srcY = (y * diffuseH) / height;
            srcX = std::clamp(srcX, 0, diffuseW - 1);
            srcY = std::clamp(srcY, 0, diffuseH - 1);
            int diffuseIdx = (srcY * diffuseW + srcX) * 4;

            int oreoSrcX = (x * oreoW) / width;
            int oreoSrcY = (y * oreoH) / height;
            oreoSrcX = std::clamp(oreoSrcX, 0, oreoW - 1);
            oreoSrcY = std::clamp(oreoSrcY, 0, oreoH - 1);
            int oreoIdx = (oreoSrcY * oreoW + oreoSrcX) * 4;

            int dstIdx = (y * width + x) * 4;
            combinedData[dstIdx + 0] = diffuseData[diffuseIdx + 0]; // R
            combinedData[dstIdx + 1] = diffuseData[diffuseIdx + 1]; // G
            combinedData[dstIdx + 2] = diffuseData[diffuseIdx + 2]; // B
            combinedData[dstIdx + 3] = oreoData[oreoIdx + 3];       // A from oreo
        }
    }

    stbi_image_free(diffuseData);
    stbi_image_free(oreoData);

    {
        std::lock_guard<std::mutex> lock(g_alphaMutex);
        g_alphaPixels = std::move(combinedData);
        g_alphaReady = true;
    }

    uint64_t alphaSum = 0;
    uint8_t minAlpha = 255;
    uint8_t maxAlpha = 0;
    size_t pixelCount = g_alphaPixels.size() / 4;
    for (size_t i = 0; i < pixelCount; ++i)
    {
        uint8_t alpha = g_alphaPixels[i * 4 + 3];
        alphaSum += alpha;
        minAlpha = (std::min)(minAlpha, alpha);
        maxAlpha = (std::max)(maxAlpha, alpha);
    }
    uint64_t avgAlpha = pixelCount > 0 ? alphaSum / pixelCount : 0;
    LOG_DEBUG("PopupWindow", "Combined diffuse+oreo stats: avg=" + std::to_string(avgAlpha) +
                             ", min=" + std::to_string(minAlpha) +
                             ", max=" + std::to_string(maxAlpha));
}

// ===========================================================
// Apply alpha to window (per-pixel transparency)
// ===========================================================
void applyWindowAlphaIfReady(HWND hwnd, int width, int height, uint32_t frameIdx)
{
    LOG_TRACE("PW", "applyWindowAlphaIfReady (non-blocking)");

    // ------------------------------------------------------
    // Skip if alpha data isn't ready yet
    // ------------------------------------------------------
    if (!g_alphaReady.load())
    {
        LOG_DEBUG("PopupWindow", "No alpha data ready yet, will use when available");
        return;
    }

    // ------------------------------------------------------
    // Copy current alpha pixels safely
    // ------------------------------------------------------
    std::vector<uint8_t> pixelsCopy;
    {
        std::lock_guard<std::mutex> lock(g_alphaMutex);
        pixelsCopy = g_alphaPixels;
        g_alphaReady = false;
    }

    if (pixelsCopy.empty())
    {
        LOG_DEBUG("PopupWindow", "Alpha buffer empty — skipping update");
        return;
    }

    // ------------------------------------------------------
    // Create device contexts for layered window drawing
    // ------------------------------------------------------
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
        LOG_ERROR("PopupWindow", "CreateDIBSection failed: " + std::to_string(GetLastError()));
        DeleteDC(hdcMem);
        ReleaseDC(nullptr, hdcScreen);
        return;
    }

    // ------------------------------------------------------
    // Copy pixels (RGBA → BGRA)
    // ------------------------------------------------------
    uint8_t* dst = static_cast<uint8_t*>(bits);
    for (int i = 0; i < width * height; ++i)
    {
        dst[i * 4 + 0] = pixelsCopy[i * 4 + 2]; // B
        dst[i * 4 + 1] = pixelsCopy[i * 4 + 1]; // G
        dst[i * 4 + 2] = pixelsCopy[i * 4 + 0]; // R
        dst[i * 4 + 3] = pixelsCopy[i * 4 + 3]; // A
    }

    HBITMAP oldBmp = (HBITMAP)SelectObject(hdcMem, hBmp);

    SIZE wndSize{ width, height };
    POINT srcPos{ 0, 0 };
    POINT winPos{ 50, 50 }; // popup screen position

    BLENDFUNCTION blend{};
    blend.BlendOp = AC_SRC_OVER;
    blend.SourceConstantAlpha = 255;
    blend.AlphaFormat = AC_SRC_ALPHA;

    // ------------------------------------------------------
    // Apply layered window transparency (no lock)
    // ------------------------------------------------------
    BOOL result = UpdateLayeredWindow(
        hwnd, hdcScreen, &winPos, &wndSize, hdcMem,
        &srcPos, 0, &blend, ULW_ALPHA);

    if (!result)
    {
        LOG_ERROR("PopupWindow", "UpdateLayeredWindow failed: " + std::to_string(GetLastError()));
    }
    else
    {
        LOG_DEBUG("PopupWindow", "Applied alpha to window successfully (frame " + std::to_string(frameIdx) + ")");
    }

    // ------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------
    SelectObject(hdcMem, oldBmp);
    DeleteObject(hBmp);
    DeleteDC(hdcMem);
    ReleaseDC(nullptr, hdcScreen);
}
