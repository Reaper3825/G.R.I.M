#include "popup_window.hpp"
#include "logger.hpp"
#include <windows.h>
#include <vector>
#include <mutex>
#include <atomic>
#include <string>
#include <algorithm>
#include <stb_image.h>

// ===========================================================
// Globals
// ===========================================================
std::atomic<bool> g_alphaReady{ false };
std::vector<uint8_t> g_alphaPixels;
std::mutex g_alphaMutex;

// ===========================================================
// Window creation (Debug Visible Mode)
// ===========================================================
HWND createOverlayWindow(int width, int height)
{
    LOG_DEBUG("PopupWindow", "Creating debug overlay window...");

    HINSTANCE hInstance = GetModuleHandleW(nullptr);

    WNDCLASSW wc{};
    wc.lpfnWndProc = DefWindowProcW;
    wc.hInstance = hInstance;
    wc.lpszClassName = L"GRIMPopupClass";
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = nullptr;  // No background for layered window

    if (!RegisterClassW(&wc) && GetLastError() != ERROR_CLASS_ALREADY_EXISTS)
    {
        LOG_ERROR("PopupWindow", "RegisterClassW failed: " + std::to_string(GetLastError()));
        return nullptr;
    }

    HWND hwnd = CreateWindowExW(
        WS_EX_LAYERED | WS_EX_TOPMOST,  // Restore layered for transparency
        L"GRIMPopupClass",
        L"GRIM Debug Popup",
        WS_POPUP | WS_VISIBLE,
        50, 50, width, height,  // More visible position
        nullptr, nullptr, hInstance, nullptr);

    if (!hwnd)
    {
        LOG_ERROR("PopupWindow", "CreateWindowExW failed: " + std::to_string(GetLastError()));
        return nullptr;
    }

    // Note: For per-pixel alpha, we don't call SetLayeredWindowAttributes
    // The alpha will be set via UpdateLayeredWindow with ULW_ALPHA
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
    int texW = 0, texH = 0, texC = 0;
    unsigned char* data = stbi_load("D:/G.R.I.M/resources/shaders/g_sprite_oreo.png", &texW, &texH, &texC, 4);
    if (!data)
    {
        LOG_ERROR("PopupWindow", "Failed to load fallback alpha image g_sprite_oreo.png");
        return;
    }

    LOG_DEBUG("PopupWindow", "Loaded alpha map directly from disk " + std::to_string(texW) + "x" + std::to_string(texH));

    // Resize to window size if necessary
    std::vector<uint8_t> resizedData;
    if (texW != width || texH != height)
    {
        // Simple nearest-neighbor resize
        resizedData.resize(width * height * 4);
        float scaleX = (float)texW / width;
        float scaleY = (float)texH / height;

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                int srcX = (int)(x * scaleX);
                int srcY = (int)(y * scaleY);
                if (srcX >= texW) srcX = texW - 1;
                if (srcY >= texH) srcY = texH - 1;

                int srcIdx = (srcY * texW + srcX) * 4;
                int dstIdx = (y * width + x) * 4;

                resizedData[dstIdx + 0] = data[srcIdx + 0]; // R
                resizedData[dstIdx + 1] = data[srcIdx + 1]; // G
                resizedData[dstIdx + 2] = data[srcIdx + 2]; // B
                resizedData[dstIdx + 3] = data[srcIdx + 3]; // A
            }
        }

        LOG_DEBUG("PopupWindow", "Resized alpha map from " + std::to_string(texW) + "x" + std::to_string(texH) + " to " + std::to_string(width) + "x" + std::to_string(height));
    }
    else
    {
        resizedData.assign(data, data + (texW * texH * 4));
    }

    stbi_image_free(data);

    std::lock_guard<std::mutex> lock(g_alphaMutex);
    g_alphaPixels = std::move(resizedData);
    g_alphaReady = true;

    // Debug: Check alpha stats immediately
    uint64_t alphaSum = 0;
    uint8_t minAlpha = 255;
    uint8_t maxAlpha = 0;
    size_t pixelCount = g_alphaPixels.size() / 4;
    for (size_t i = 0; i < pixelCount; ++i) {
        uint8_t alpha = g_alphaPixels[i * 4 + 3];
        alphaSum += alpha;
        if (alpha < minAlpha) minAlpha = alpha;
        if (alpha > maxAlpha) maxAlpha = alpha;
    }
    uint64_t avgAlpha = pixelCount > 0 ? alphaSum / pixelCount : 0;
    LOG_DEBUG("PopupWindow", "Loaded alpha stats: avg=" + std::to_string(avgAlpha) + ", min=" + std::to_string(minAlpha) + ", max=" + std::to_string(maxAlpha));
}

// ===========================================================
// Apply alpha to window (per-pixel transparency)
// ===========================================================
void applyWindowAlphaIfReady(HWND hwnd, int width, int height, uint32_t frameIdx)
{
    LOG_DEBUG("PopupWindow", "applyWindowAlphaIfReady called at frame " + std::to_string(frameIdx));

    if (!g_alphaReady.load())
    {
        LOG_DEBUG("PopupWindow", "No alpha data ready yet, will use when available");
        return;
    }

    std::vector<uint8_t> pixelsCopy;
    {
        std::lock_guard<std::mutex> lock(g_alphaMutex);
        pixelsCopy = g_alphaPixels;
        g_alphaReady = false;
    }

    if (pixelsCopy.empty())
        return;

    uint64_t alphaSum = 0;
    uint8_t minAlpha = 255;
    uint8_t maxAlpha = 0;
    for (size_t i = 0; i < pixelsCopy.size() / 4; ++i) {
        uint8_t alpha = pixelsCopy[i * 4 + 3];
        alphaSum += alpha;
        if (alpha < minAlpha) minAlpha = alpha;
        if (alpha > maxAlpha) maxAlpha = alpha;
    }
    uint64_t avgAlpha = alphaSum / (pixelsCopy.size() / 4);
    LOG_DEBUG("PopupWindow", "Alpha stats: avg=" + std::to_string(avgAlpha) + ", min=" + std::to_string(minAlpha) + ", max=" + std::to_string(maxAlpha));

    HDC hdcScreen = GetDC(nullptr);
    HDC hdcMem = CreateCompatibleDC(hdcScreen);

    BITMAPINFO bmi{};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = width;
    bmi.bmiHeader.biHeight = -height;
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

    // Copy RGBA → BGRA for Windows
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
    POINT winPos{ 50, 50 };  // Match window creation position

    BLENDFUNCTION blend{};
    blend.BlendOp = AC_SRC_OVER;
    blend.SourceConstantAlpha = 255;
    blend.AlphaFormat = AC_SRC_ALPHA;

    BOOL result = UpdateLayeredWindow(
        hwnd, hdcScreen, &winPos, &wndSize, hdcMem,
        &srcPos, 0, &blend, ULW_ALPHA);

    if (!result)
        LOG_ERROR("PopupWindow", "UpdateLayeredWindow failed: " + std::to_string(GetLastError()));
    else
        LOG_DEBUG("PopupWindow", "Applied alpha to window successfully (frame " + std::to_string(frameIdx) + ")");

    SelectObject(hdcMem, oldBmp);
    DeleteObject(hBmp);
    DeleteDC(hdcMem);
    ReleaseDC(nullptr, hdcScreen);
}
