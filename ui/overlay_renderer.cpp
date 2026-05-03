#include "overlay_renderer.hpp"
#include "logger.hpp"
#include <algorithm>
#include <cstring>

// ---------------------------------------------------------------
// Init / Shutdown
// ---------------------------------------------------------------

#ifdef _WIN32
void OverlayRenderer::init(HWND hwnd, int width, int height)
{
    LOG_DEBUG("OverlayRenderer", "Initializing with thread safety");
    std::lock_guard<std::mutex> lock(m_renderMutex);

    m_hwnd = hwnd;
    m_width = width;
    m_height = height;

    m_hdcScreen = GetDC(nullptr);
    m_hdcMem = CreateCompatibleDC(m_hdcScreen);

    BITMAPINFO bmi{};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = width;
    bmi.bmiHeader.biHeight = -height;
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;

    m_bitmap = CreateDIBSection(m_hdcMem, &bmi, DIB_RGB_COLORS, &m_pixels, nullptr, 0);
    m_oldBitmap = (HBITMAP)SelectObject(m_hdcMem, m_bitmap);
    m_ownsPixels = false;
    m_backdropPixels = new uint32_t[width * height]();
    m_backdropDirty = true;

    // Windows does not use DWM blur for the transparent overlay; per-panel glass
    // is produced by the software desktop capture path in overlay_renderer_glass.cpp.
    m_usePlatformBlur = false;

    LOG_DEBUG("OverlayRenderer", "Initialized overlay renderer (" +
             std::to_string(width) + "x" + std::to_string(height) + ")");
}
#elif defined(__APPLE__)
void OverlayRenderer::init(HWND hwnd, int width, int height)
{
    LOG_DEBUG("OverlayRenderer", "Initializing (macOS software renderer)");
    std::lock_guard<std::mutex> lock(m_renderMutex);

    m_nativeWindow = hwnd;
    m_width = width;
    m_height = height;
    m_pixels = new uint32_t[width * height]();
    m_ownsPixels = true;
    m_backdropPixels = new uint32_t[width * height]();
    m_backdropDirty = true;

    // macOS uses NSVisualEffectView masking as the native panel backdrop owner.
    m_usePlatformBlur = true;

    LOG_DEBUG("OverlayRenderer", "Initialized overlay renderer (" +
             std::to_string(width) + "x" + std::to_string(height) + ")");
}
#endif

void OverlayRenderer::init(int width, int height, void* pixelBuffer)
{
    std::lock_guard<std::mutex> lock(m_renderMutex);
    m_width = width;
    m_height = height;

#if defined(__APPLE__)
    m_usePlatformBlur = true;
#else
    m_usePlatformBlur = false;
#endif

    if (pixelBuffer) {
        m_pixels = pixelBuffer;
        m_ownsPixels = false;
    } else {
        m_pixels = new uint32_t[width * height]();
        m_ownsPixels = true;
    }
    if (m_width > 0 && m_height > 0)
        m_backdropPixels = new uint32_t[m_width * m_height]();
    m_backdropDirty = true;
}

void OverlayRenderer::shutdown()
{
    std::lock_guard<std::mutex> lock(m_renderMutex);

#ifdef _WIN32
    if (m_oldBitmap) {
        SelectObject(m_hdcMem, m_oldBitmap);
        m_oldBitmap = nullptr;
    }
    if (m_bitmap) {
        DeleteObject(m_bitmap);
        m_bitmap = nullptr;
    }
    if (m_hdcMem) {
        DeleteDC(m_hdcMem);
        m_hdcMem = nullptr;
    }
    if (m_hdcScreen) {
        ReleaseDC(nullptr, m_hdcScreen);
        m_hdcScreen = nullptr;
    }
    m_hwnd = nullptr;
#elif defined(__APPLE__)
    m_nativeWindow = nullptr;
#endif

    if (m_backdropPixels) {
        delete[] m_backdropPixels;
        m_backdropPixels = nullptr;
    }
    if (m_ownsPixels) {
        delete[] static_cast<uint32_t*>(m_pixels);
    }
    m_pixels = nullptr;
    m_ownsPixels = false;

    m_fontFileData.clear();
    m_iconFontFileData.clear();
    m_fontAtlas.clear();
    m_fontLoaded = false;
    m_glassCache.clear();
}

// ---------------------------------------------------------------
// Clip stack
// ---------------------------------------------------------------

void OverlayRenderer::pushClipRect(const Vec2& pos, const Vec2& size)
{
    ClipRect incoming;
    incoming.x1 = std::max(0, (int)pos.x);
    incoming.y1 = std::max(0, (int)pos.y);
    incoming.x2 = std::min(m_width,  (int)(pos.x + size.x));
    incoming.y2 = std::min(m_height, (int)(pos.y + size.y));

    if (!m_clipStack.empty()) {
        const ClipRect& cur = m_clipStack.back();
        incoming.x1 = std::max(incoming.x1, cur.x1);
        incoming.y1 = std::max(incoming.y1, cur.y1);
        incoming.x2 = std::min(incoming.x2, cur.x2);
        incoming.y2 = std::min(incoming.y2, cur.y2);
    }

    m_clipStack.push_back(incoming);
}

void OverlayRenderer::popClipRect()
{
    if (!m_clipStack.empty())
        m_clipStack.pop_back();
}

ClipRect OverlayRenderer::activeClip() const
{
    if (!m_clipStack.empty())
        return m_clipStack.back();
    return { 0, 0, m_width, m_height };
}

// ---------------------------------------------------------------
// Dirty region tracking
// ---------------------------------------------------------------

void OverlayRenderer::expandDirtyRect(int x, int y, int w, int h)
{
    if (w <= 0 || h <= 0) return;
    std::lock_guard<std::mutex> lock(m_renderMutex);
    int x2 = x + w;
    int y2 = y + h;
    if (m_dirtyX1 >= m_dirtyX2) {
        m_dirtyX1 = x;
        m_dirtyY1 = y;
        m_dirtyX2 = x2;
        m_dirtyY2 = y2;
    } else {
        m_dirtyX1 = std::min(m_dirtyX1, x);
        m_dirtyY1 = std::min(m_dirtyY1, y);
        m_dirtyX2 = std::max(m_dirtyX2, x2);
        m_dirtyY2 = std::max(m_dirtyY2, y2);
    }
}

// ---------------------------------------------------------------
// Frame management
// ---------------------------------------------------------------

void OverlayRenderer::beginFrame()
{
    std::lock_guard<std::mutex> lock(m_renderMutex);

    if (!m_pixels)
        return;

    m_grainSeed ^= (m_grainSeed << 13);
    m_grainSeed ^= (m_grainSeed >> 17);
    m_grainSeed ^= (m_grainSeed << 5);
    ++m_frameIndex;

    m_clipStack.clear();

    uint32_t* pixels = static_cast<uint32_t*>(m_pixels);
    int cx1 = std::max(0, std::min(m_prevDirtyX1, m_width));
    int cy1 = std::max(0, std::min(m_prevDirtyY1, m_height));
    int cx2 = std::max(0, std::min(m_prevDirtyX2, m_width));
    int cy2 = std::max(0, std::min(m_prevDirtyY2, m_height));

    if (cx2 > cx1 && cy2 > cy1) {
        int clearW = cx2 - cx1;
        for (int y = cy1; y < cy2; ++y)
            std::memset(pixels + y * m_width + cx1, 0, clearW * sizeof(uint32_t));
    }

    m_dirtyX1 = m_dirtyX2 = m_dirtyY1 = m_dirtyY2 = 0;
}

void OverlayRenderer::endFrame()
{
    std::lock_guard<std::mutex> lock(m_renderMutex);

    m_prevDirtyX1 = m_dirtyX1;
    m_prevDirtyY1 = m_dirtyY1;
    m_prevDirtyX2 = m_dirtyX2;
    m_prevDirtyY2 = m_dirtyY2;

#ifdef _WIN32
    if (!m_hwnd || !m_hdcMem || !m_hdcScreen)
        return;

    RECT rect;
    GetWindowRect(m_hwnd, &rect);

    SIZE wndSize{ m_width, m_height };
    POINT srcPos{ 0, 0 };
    POINT wndPos{ rect.left, rect.top };

    BLENDFUNCTION blend{};
    blend.BlendOp = AC_SRC_OVER;
    blend.SourceConstantAlpha = 255;
    blend.AlphaFormat = AC_SRC_ALPHA;

    UpdateLayeredWindow(m_hwnd, m_hdcScreen, &wndPos, &wndSize, m_hdcMem,
                       &srcPos, 0, &blend, ULW_ALPHA);
#elif defined(__APPLE__)
    if (!m_nativeWindow || !m_pixels)
        return;

    extern void grimOverlayBlit(void* nsWindow, void* pixels, int width, int height);
    grimOverlayBlit(m_nativeWindow, m_pixels, m_width, m_height);
#endif
}
