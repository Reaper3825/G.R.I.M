#include "core/grim_platform.h"
#ifdef _WIN32

#include "pch.hpp"
#include "popup_window.hpp"
#include "ui/ui_root.hpp"
#include <stb/stb_image.h>
#include "core/ui_sync.hpp"
#include <algorithm>
#include <sstream>
#define WM_GRIM_SHOW_POPUP (WM_APP + 1)


// ===========================================================
// Globals
// ===========================================================
std::atomic<bool> g_alphaReady{ false };
std::vector<uint8_t> g_popupPixels;
std::vector<uint8_t> g_emissiveMask;
std::vector<uint8_t> g_occlusionMask;
std::vector<uint8_t> g_shadowMask;
std::vector<uint8_t> g_roughnessMask;
std::mutex g_alphaMutex;
static int g_popupWidth = 0;
static int g_popupHeight = 0;

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
// Shadow helpers
// ===========================================================
static void boxBlurPass(const std::vector<float>& in, std::vector<float>& out,
                        int width, int height, int radius, bool horizontal)
{
    if (radius <= 0)
    {
        out = in;
        return;
    }

    out.assign(width * height, 0.0f);
    int kernel = radius * 2 + 1;

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            float sum = 0.0f;
            int samples = 0;

            for (int k = -radius; k <= radius; ++k)
            {
                int sampleX = horizontal ? x + k : x;
                int sampleY = horizontal ? y : y + k;

                if (sampleX < 0 || sampleX >= width || sampleY < 0 || sampleY >= height)
                    continue;

                sum += in[sampleY * width + sampleX];
                ++samples;
            }

            out[y * width + x] = (samples > 0) ? (sum / samples) : 0.0f;
        }
    }
}

static std::vector<uint8_t> buildShadowMask(const std::vector<uint8_t>& rgbaPixels,
                                            const std::vector<uint8_t>& occlusionMask,
                                            int width, int height)
{
    if (rgbaPixels.empty() || width == 0 || height == 0)
        return {};

    std::vector<float> mask(width * height, 0.0f);
    constexpr float alphaCutoff = 0.92f;      // drop anything below 92% opacity
    constexpr float occlusionCutoff = 0.2f;   // ignore very low occlusion contributions

    for (int i = 0; i < width * height; ++i)
    {
        float alpha = rgbaPixels[i * 4 + 3] / 255.0f;
        float occlusion = occlusionMask.empty() ? 1.0f : (occlusionMask[i] / 255.0f);

        if (alpha < alphaCutoff || occlusion < occlusionCutoff)
        {
            mask[i] = 0.0f;
            continue;
        }

        // Remap so surviving pixels still produce a full-strength shadow edge.
        float alphaRemapped = (alpha - alphaCutoff) / (1.0f - alphaCutoff);
        float occlusionRemapped = (occlusion - occlusionCutoff) / (1.0f - occlusionCutoff);
        alphaRemapped = std::clamp(alphaRemapped, 0.0f, 1.0f);
        occlusionRemapped = std::clamp(occlusionRemapped, 0.0f, 1.0f);

        mask[i] = alphaRemapped * (0.4f + 0.6f * occlusionRemapped);
    }

    std::vector<float> temp;
    temp.reserve(width * height);

    // Multiple passes for smoother blur
    for (int pass = 0; pass < 2; ++pass)
    {
        boxBlurPass(mask, temp, width, height, 6, true);
        boxBlurPass(temp, mask, width, height, 6, false);
    }

    std::vector<uint8_t> shadow(width * height);
    for (int i = 0; i < width * height; ++i)
    {
        float value = std::clamp(mask[i], 0.0f, 1.0f);
        shadow[i] = static_cast<uint8_t>(value * 255.0f);
    }
    return shadow;
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
    // NOTE: Do NOT call enableDwmBlurBehind() on the popup.
    // DWM blur fills the ENTIRE popup window with blurred desktop,
    // producing a blurred square instead of showing the sprite with
    // per-pixel alpha transparency via UpdateLayeredWindow(ULW_ALPHA).
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

    LOG_DEBUG("PopupWindow", "Loaded diffuse " + std::to_string(diffuseW) + "x" + std::to_string(diffuseH) +
                             " and oreo " + std::to_string(oreoW) + "x" + std::to_string(oreoH));

    std::vector<uint8_t> combinedData(width * height * 4);
    std::vector<uint8_t> emissiveData(width * height, 0);
    std::vector<uint8_t> occlusionData(width * height, 0);
    std::vector<uint8_t> roughnessData(width * height, 0);

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

            uint8_t opacity = oreoData[oreoIdx + 3];
            combinedData[dstIdx + 3] = opacity;

            int pixelIndex = (y * width + x);
            occlusionData[pixelIndex] = oreoData[oreoIdx + 0];
            roughnessData[pixelIndex] = oreoData[oreoIdx + 1];
            emissiveData[pixelIndex] = oreoData[oreoIdx + 2];
        }
    }

    stbi_image_free(diffuseData);
    stbi_image_free(oreoData);

    auto shadowData = buildShadowMask(combinedData, occlusionData, width, height);

    {
        std::lock_guard<std::mutex> lock(g_alphaMutex);
        g_popupPixels = std::move(combinedData);
        g_emissiveMask = std::move(emissiveData);
        g_occlusionMask = std::move(occlusionData);
        g_roughnessMask = std::move(roughnessData);
        g_shadowMask = std::move(shadowData);
        g_popupWidth = width;
        g_popupHeight = height;
        g_alphaReady = true;
    }

    uint64_t alphaSum = 0;
    uint8_t minAlpha = 255;
    uint8_t maxAlpha = 0;
    size_t pixelCount = g_popupPixels.size() / 4;
    for (size_t i = 0; i < pixelCount; ++i)
    {
        uint8_t alpha = g_popupPixels[i * 4 + 3];
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
// Apply alpha to window (per-pixel transparency) - INITIAL SETUP ONLY
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
    // Copy current alpha pixels safely (DON'T clear g_alphaReady!)
    // ------------------------------------------------------
    std::vector<uint8_t> pixelsCopy;
    {
        std::lock_guard<std::mutex> lock(g_alphaMutex);
        pixelsCopy = g_popupPixels;
        // DON'T SET g_alphaReady = false; - Animation needs continuous access!
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
        uint8_t a = pixelsCopy[i * 4 + 3];
        dst[i * 4 + 0] = static_cast<uint8_t>((pixelsCopy[i * 4 + 2] * a) / 255); // B
        dst[i * 4 + 1] = static_cast<uint8_t>((pixelsCopy[i * 4 + 1] * a) / 255); // G
        dst[i * 4 + 2] = static_cast<uint8_t>((pixelsCopy[i * 4 + 0] * a) / 255); // R
        dst[i * 4 + 3] = a; // A
    }

    HBITMAP oldBmp = (HBITMAP)SelectObject(hdcMem, hBmp);

    SIZE wndSize{ width, height };
    POINT srcPos{ 0, 0 };
    POINT winPos{ 100, 100 }; // popup screen position

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
        LOG_DEBUG("PopupWindow", "Applied initial alpha to window successfully (frame " + std::to_string(frameIdx) + ")");
    }

    // ------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------
    SelectObject(hdcMem, oldBmp);
    DeleteObject(hBmp);
    DeleteDC(hdcMem);
    ReleaseDC(nullptr, hdcScreen);
}

// ===========================================================
// Apply animation to window with SMOOTH BILINEAR INTERPOLATION
// ===========================================================
void applyAnimationToWindow(HWND hwnd, int width, int height, float scale, float alpha, float voiceIntensity)
{
    if (!hwnd || !IsWindow(hwnd))
        return;
    voiceIntensity = std::clamp(voiceIntensity, 0.0f, 1.0f);

    // Get current cached data
    std::vector<uint8_t> pixelsCopy;
    std::vector<uint8_t> emissiveCopy;
    std::vector<uint8_t> occlusionCopy;
    std::vector<uint8_t> shadowCopy;
    std::vector<uint8_t> roughnessCopy;
    int baseWidth = width;
    int baseHeight = height;
    {
        std::lock_guard<std::mutex> lock(g_alphaMutex);
        if (g_popupPixels.empty())
        {
            static int emptyWarningCount = 0;
            if (++emptyWarningCount <= 5) {
                LOG_ERROR("PopupWindow", "applyAnimationToWindow: g_popupPixels is EMPTY! Animation cannot render.");
            }
            return;
        }
        pixelsCopy = g_popupPixels;
        emissiveCopy = g_emissiveMask;
        occlusionCopy = g_occlusionMask;
        shadowCopy = g_shadowMask;
        roughnessCopy = g_roughnessMask;
        if (g_popupWidth > 0) baseWidth = g_popupWidth;
        if (g_popupHeight > 0) baseHeight = g_popupHeight;
    }

    if (pixelsCopy.empty())
        return;

    // Calculate scaled dimensions
    int scaledWidth = static_cast<int>(std::round(width * scale));
    int scaledHeight = static_cast<int>(std::round(height * scale));
    
    scaledWidth = std::clamp(scaledWidth, 50, 512);
    scaledHeight = std::clamp(scaledHeight, 50, 512);

    // ------------------------------------------------------
    // Use GDI HALFTONE mode for fast hardware-accelerated scaling
    // ------------------------------------------------------
    HDC hdcScreen = GetDC(nullptr);
    HDC hdcSrc = CreateCompatibleDC(hdcScreen);
    HDC hdcDst = CreateCompatibleDC(hdcScreen);

    // Create source bitmap
    BITMAPINFO bmiSrc{};
    bmiSrc.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmiSrc.bmiHeader.biWidth = width;
    bmiSrc.bmiHeader.biHeight = -height;
    bmiSrc.bmiHeader.biPlanes = 1;
    bmiSrc.bmiHeader.biBitCount = 32;
    bmiSrc.bmiHeader.biCompression = BI_RGB;

    void* bitsSrc = nullptr;
    HBITMAP hBmpSrc = CreateDIBSection(hdcSrc, &bmiSrc, DIB_RGB_COLORS, &bitsSrc, nullptr, 0);
    if (!hBmpSrc)
    {
        DeleteDC(hdcSrc);
        DeleteDC(hdcDst);
        ReleaseDC(nullptr, hdcScreen);
        LOG_ERROR("PopupWindow", "Failed to create source bitmap");
        return;
    }

    // Prepare source buffer with shadow + emissive tinting
    uint8_t* src = static_cast<uint8_t*>(bitsSrc);
    std::fill(src, src + (width * height * 4), 0);

    // Drop shadow (draw first so sprite overwrites overlap)
    if (!shadowCopy.empty())
    {
        const int shadowOffsetX = 8;
        const int shadowOffsetY = 10;
        const float shadowAlphaScale = std::clamp(alpha * 0.65f, 0.0f, 1.0f);

        for (int y = 0; y < baseHeight; ++y)
        {
            for (int x = 0; x < baseWidth; ++x)
            {
                int idx = y * baseWidth + x;
                uint8_t mask = shadowCopy[idx];
                if (mask == 0)
                    continue;

                int dstX = x + shadowOffsetX;
                int dstY = y + shadowOffsetY;
                if (dstX < 0 || dstX >= width || dstY < 0 || dstY >= height)
                    continue;

                uint8_t finalAlpha = static_cast<uint8_t>(mask * shadowAlphaScale);
                size_t dstIdx = (dstY * width + dstX) * 4;

                if (finalAlpha > src[dstIdx + 3])
                {
                    src[dstIdx + 0] = static_cast<uint8_t>((15 * finalAlpha) / 255);
                    src[dstIdx + 1] = static_cast<uint8_t>((18 * finalAlpha) / 255);
                    src[dstIdx + 2] = static_cast<uint8_t>((22 * finalAlpha) / 255);
                    src[dstIdx + 3] = finalAlpha;
                }
            }
        }
    }

    auto sampleMask = [](const std::vector<uint8_t>& mask, int idx) -> float {
        if (mask.empty()) return 0.0f;
        return mask[idx] / 255.0f;
    };

    for (int y = 0; y < baseHeight; ++y)
    {
        for (int x = 0; x < baseWidth; ++x)
        {
            if (x >= width || y >= height)
                continue;

            size_t srcIdx = (y * baseWidth + x) * 4;
            float occlusion = sampleMask(occlusionCopy, y * baseWidth + x);
            float shading = occlusion;

            float roughness = sampleMask(roughnessCopy, y * baseWidth + x);
            float smoothness = 1.0f - roughness;
            float specularScale = 1.0f + smoothness * 0.2f; // smoother = brighter highlight
            float specularAdd = smoothness * 25.0f;

            float emissiveMask = sampleMask(emissiveCopy, y * baseWidth + x);
            float emissiveBoost = 1.0f + voiceIntensity * emissiveMask;
            float emissiveAdd = voiceIntensity * emissiveMask * 180.0f;

            auto toneMap = [&](float value) -> uint8_t {
                float boosted = value * shading * emissiveBoost * specularScale + emissiveAdd + specularAdd;
                return static_cast<uint8_t>(std::clamp(boosted, 0.0f, 255.0f));
            };

            uint8_t originalAlpha = pixelsCopy[srcIdx + 3];
            uint8_t finalAlpha = static_cast<uint8_t>(originalAlpha * alpha);

            auto premul = [&](uint8_t channel) -> uint8_t {
                return static_cast<uint8_t>((toneMap(channel) * finalAlpha) / 255);
            };

            uint8_t finalR = premul(pixelsCopy[srcIdx + 0]);
            uint8_t finalG = premul(pixelsCopy[srcIdx + 1]);
            uint8_t finalB = premul(pixelsCopy[srcIdx + 2]);

            size_t dstIdx = (y * width + x) * 4;
            src[dstIdx + 0] = finalB;
            src[dstIdx + 1] = finalG;
            src[dstIdx + 2] = finalR;
            src[dstIdx + 3] = finalAlpha;
        }
    }

    // Create destination bitmap
    BITMAPINFO bmiDst{};
    bmiDst.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmiDst.bmiHeader.biWidth = scaledWidth;
    bmiDst.bmiHeader.biHeight = -scaledHeight;
    bmiDst.bmiHeader.biPlanes = 1;
    bmiDst.bmiHeader.biBitCount = 32;
    bmiDst.bmiHeader.biCompression = BI_RGB;

    void* bitsDst = nullptr;
    HBITMAP hBmpDst = CreateDIBSection(hdcDst, &bmiDst, DIB_RGB_COLORS, &bitsDst, nullptr, 0);
    if (!hBmpDst)
    {
        DeleteObject(hBmpSrc);
        DeleteDC(hdcSrc);
        DeleteDC(hdcDst);
        ReleaseDC(nullptr, hdcScreen);
        LOG_ERROR("PopupWindow", "Failed to create destination bitmap");
        return;
    }

    HBITMAP oldSrc = (HBITMAP)SelectObject(hdcSrc, hBmpSrc);
    HBITMAP oldDst = (HBITMAP)SelectObject(hdcDst, hBmpDst);

    // DON'T use StretchBlt - it breaks alpha on layered windows!
    // Instead, manually copy with simple nearest-neighbor scaling
    uint8_t* srcBits = static_cast<uint8_t*>(bitsSrc);
    uint8_t* dstBits = static_cast<uint8_t*>(bitsDst);
    
    for (int y = 0; y < scaledHeight; ++y)
    {
        for (int x = 0; x < scaledWidth; ++x)
        {
            // Nearest neighbor scaling
            int srcX = (x * width) / scaledWidth;
            int srcY = (y * height) / scaledHeight;
            srcX = std::clamp(srcX, 0, width - 1);
            srcY = std::clamp(srcY, 0, height - 1);
            
            int srcIdx = (srcY * width + srcX) * 4;
            int dstIdx = (y * scaledWidth + x) * 4;
            
            dstBits[dstIdx + 0] = srcBits[srcIdx + 0]; // B
            dstBits[dstIdx + 1] = srcBits[srcIdx + 1]; // G
            dstBits[dstIdx + 2] = srcBits[srcIdx + 2]; // R
            dstBits[dstIdx + 3] = srcBits[srcIdx + 3]; // A
        }
    }

    // ------------------------------------------------------
    // Update window
    // ------------------------------------------------------
    POINT winPos;
    winPos.x = 50 - (scaledWidth - width) / 2;
    winPos.y = 50 - (scaledHeight - height) / 2;

    SIZE wndSize{ scaledWidth, scaledHeight };
    POINT srcPos{ 0, 0 };

    BLENDFUNCTION blend{};
    blend.BlendOp = AC_SRC_OVER;
    blend.SourceConstantAlpha = 255;
    blend.AlphaFormat = AC_SRC_ALPHA;

    BOOL updateResult = UpdateLayeredWindow(hwnd, hdcScreen, &winPos, &wndSize, hdcDst,
                                           &srcPos, 0, &blend, ULW_ALPHA);

    if (!updateResult)
    {
        LOG_ERROR("PopupWindow", "UpdateLayeredWindow failed: " + std::to_string(GetLastError()));
    }

    // ------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------
    SelectObject(hdcSrc, oldSrc);
    SelectObject(hdcDst, oldDst);
    DeleteObject(hBmpSrc);
    DeleteObject(hBmpDst);
    DeleteDC(hdcSrc);
    DeleteDC(hdcDst);
    ReleaseDC(nullptr, hdcScreen);
}

#endif // _WIN32
