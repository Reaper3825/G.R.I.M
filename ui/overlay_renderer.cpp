#include "overlay_renderer.hpp"
#include "logger.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>

#define STB_TRUETYPE_IMPLEMENTATION
#include <stb/stb_truetype.h>

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

    LOG_DEBUG("OverlayRenderer", "Initialized overlay renderer (" +
             std::to_string(width) + "x" + std::to_string(height) + ")");
}
#endif

void OverlayRenderer::init(int width, int height, void* pixelBuffer)
{
    std::lock_guard<std::mutex> lock(m_renderMutex);
    m_width = width;
    m_height = height;

    if (pixelBuffer) {
        m_pixels = pixelBuffer;
        m_ownsPixels = false;
    } else {
        m_pixels = new uint32_t[width * height]();
        m_ownsPixels = true;
    }
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
#endif

    if (m_ownsPixels) {
        delete[] static_cast<uint32_t*>(m_pixels);
    }
    m_pixels = nullptr;
    m_ownsPixels = false;

    m_fontFileData.clear();
    m_fontAtlas.clear();
    m_fontLoaded = false;
}

// ---------------------------------------------------------------
// Font loading via stb_truetype
// ---------------------------------------------------------------

void OverlayRenderer::setFont(const std::string& fontPath, int fontSize)
{
    std::lock_guard<std::mutex> lock(m_renderMutex);

    std::ifstream file(fontPath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LOG_ERROR("OverlayRenderer", "Failed to open font file: " + fontPath);
        return;
    }

    auto fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    m_fontFileData.resize(static_cast<size_t>(fileSize));
    file.read(reinterpret_cast<char*>(m_fontFileData.data()), fileSize);
    file.close();

    m_fontSize = static_cast<float>(fontSize);
    m_atlasWidth = 512;
    m_atlasHeight = 512;

    // Retry with larger atlas if needed
    for (int attempt = 0; attempt < 3; ++attempt) {
        m_fontAtlas.resize(m_atlasWidth * m_atlasHeight);

        stbtt_bakedchar bakedChars[kCharCount];
        int result = stbtt_BakeFontBitmap(
            m_fontFileData.data(), 0,
            m_fontSize,
            m_fontAtlas.data(), m_atlasWidth, m_atlasHeight,
            kFirstChar, kCharCount,
            bakedChars
        );

        if (result > 0) {
            for (int i = 0; i < kCharCount; ++i) {
                m_bakedChars[i].x0 = bakedChars[i].x0;
                m_bakedChars[i].y0 = bakedChars[i].y0;
                m_bakedChars[i].x1 = bakedChars[i].x1;
                m_bakedChars[i].y1 = bakedChars[i].y1;
                m_bakedChars[i].xoff = bakedChars[i].xoff;
                m_bakedChars[i].yoff = bakedChars[i].yoff;
                m_bakedChars[i].xadvance = bakedChars[i].xadvance;
            }
            m_fontLoaded = true;
            LOG_DEBUG("OverlayRenderer", "Font loaded: " + fontPath +
                     " (size " + std::to_string(fontSize) +
                     ", atlas " + std::to_string(m_atlasWidth) + "x" + std::to_string(m_atlasHeight) + ")");
            return;
        }

        m_atlasWidth *= 2;
        m_atlasHeight *= 2;
    }

    LOG_ERROR("OverlayRenderer", "Failed to bake font atlas for: " + fontPath);
    m_fontLoaded = false;
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
// Frame management
// ---------------------------------------------------------------

void OverlayRenderer::beginFrame()
{
    std::lock_guard<std::mutex> lock(m_renderMutex);

    if (!m_pixels)
        return;

    m_clipStack.clear();

    uint32_t* pixels = static_cast<uint32_t*>(m_pixels);
    int pixelCount = m_width * m_height;
    std::memset(pixels, 0, pixelCount * sizeof(uint32_t));
}

void OverlayRenderer::endFrame()
{
    std::lock_guard<std::mutex> lock(m_renderMutex);

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
#endif
}

// ---------------------------------------------------------------
// Drawing: filled rectangle
// ---------------------------------------------------------------

void OverlayRenderer::drawRect(const Vec2& pos, const Vec2& size, uint32_t color)
{
    std::lock_guard<std::mutex> lock(m_renderMutex);

    if (!m_pixels)
        return;

    uint8_t a = (color >> 24) & 0xFF;
    uint8_t r = (color >> 16) & 0xFF;
    uint8_t g = (color >> 8) & 0xFF;
    uint8_t b = color & 0xFF;

    r = (uint8_t)((r * a) / 255);
    g = (uint8_t)((g * a) / 255);
    b = (uint8_t)((b * a) / 255);

    ClipRect clip = activeClip();

    int x1 = std::max(clip.x1, (int)pos.x);
    int y1 = std::max(clip.y1, (int)pos.y);
    int x2 = std::min(clip.x2, (int)(pos.x + size.x));
    int y2 = std::min(clip.y2, (int)(pos.y + size.y));

    uint32_t* pixels = static_cast<uint32_t*>(m_pixels);

    for (int y = y1; y < y2; ++y) {
        for (int x = x1; x < x2; ++x) {
            int idx = y * m_width + x;
            pixels[idx] = (a << 24) | (r << 16) | (g << 8) | b;
        }
    }
}

// ---------------------------------------------------------------
// Drawing: text via stb_truetype baked atlas
// ---------------------------------------------------------------

void OverlayRenderer::drawText(const Vec2& pos, const std::string& text, uint32_t color)
{
    std::lock_guard<std::mutex> lock(m_renderMutex);

    if (!m_pixels || !m_fontLoaded || text.empty())
        return;

    ClipRect clip = activeClip();

    uint8_t a = (color >> 24) & 0xFF;
    if (a == 0) a = 255;
    uint8_t r = (color >> 16) & 0xFF;
    uint8_t g = (color >> 8) & 0xFF;
    uint8_t b = color & 0xFF;

    uint32_t* pixels = static_cast<uint32_t*>(m_pixels);
    float cursorX = pos.x;
    float cursorY = pos.y;

    for (char c : text) {
        if (c < kFirstChar || c >= kFirstChar + kCharCount) {
            if (c == '\n') {
                cursorX = pos.x;
                cursorY += m_fontSize;
                continue;
            }
            c = '?';
        }

        int ci = c - kFirstChar;
        const BakedChar& bc = m_bakedChars[ci];

        int glyphW = bc.x1 - bc.x0;
        int glyphH = bc.y1 - bc.y0;
        if (glyphW <= 0 || glyphH <= 0) {
            cursorX += bc.xadvance;
            continue;
        }

        int dstX0 = (int)std::floor(cursorX + bc.xoff);
        int dstY0 = (int)std::floor(cursorY + bc.yoff + m_fontSize);

        for (int gy = 0; gy < glyphH; ++gy) {
            int dy = dstY0 + gy;
            if (dy < clip.y1 || dy >= clip.y2) continue;

            for (int gx = 0; gx < glyphW; ++gx) {
                int dx = dstX0 + gx;
                if (dx < clip.x1 || dx >= clip.x2) continue;

                uint8_t coverage = m_fontAtlas[(bc.y0 + gy) * m_atlasWidth + (bc.x0 + gx)];
                if (coverage == 0) continue;

                uint8_t ca = (uint8_t)((a * coverage) / 255);
                uint8_t cr = (uint8_t)((r * ca) / 255);
                uint8_t cg = (uint8_t)((g * ca) / 255);
                uint8_t cb = (uint8_t)((b * ca) / 255);

                int idx = dy * m_width + dx;
                uint32_t dst = pixels[idx];
                uint8_t da = (dst >> 24) & 0xFF;
                uint8_t dr = (dst >> 16) & 0xFF;
                uint8_t dg = (dst >> 8) & 0xFF;
                uint8_t db = dst & 0xFF;

                uint8_t outA = ca + (uint8_t)((da * (255 - ca)) / 255);
                uint8_t outR = cr + (uint8_t)((dr * (255 - ca)) / 255);
                uint8_t outG = cg + (uint8_t)((dg * (255 - ca)) / 255);
                uint8_t outB = cb + (uint8_t)((db * (255 - ca)) / 255);

                pixels[idx] = (outA << 24) | (outR << 16) | (outG << 8) | outB;
            }
        }

        cursorX += bc.xadvance;
    }
}

// ---------------------------------------------------------------
// Drawing: line (Bresenham with thickness)
// ---------------------------------------------------------------

void OverlayRenderer::drawLine(const Vec2& start, const Vec2& end, uint32_t color, float thickness)
{
    std::lock_guard<std::mutex> lock(m_renderMutex);

    if (!m_pixels)
        return;

    uint8_t a = (color >> 24) & 0xFF;
    uint8_t r = (color >> 16) & 0xFF;
    uint8_t g = (color >> 8) & 0xFF;
    uint8_t b = color & 0xFF;

    r = (uint8_t)((r * a) / 255);
    g = (uint8_t)((g * a) / 255);
    b = (uint8_t)((b * a) / 255);

    uint32_t premultColor = (a << 24) | (r << 16) | (g << 8) | b;

    ClipRect clip = activeClip();

    float dx = end.x - start.x;
    float dy = end.y - start.y;
    float len = std::sqrt(dx * dx + dy * dy);

    if (len < 0.01f) {
        drawRect(start, {thickness, thickness}, color);
        return;
    }

    int segments = static_cast<int>(len * 2.0f) + 1;
    uint32_t* pixels = static_cast<uint32_t*>(m_pixels);

    for (int i = 0; i <= segments; ++i) {
        float t = static_cast<float>(i) / segments;
        float fx = start.x + dx * t;
        float fy = start.y + dy * t;
        int x = static_cast<int>(fx);
        int y = static_cast<int>(fy);

        int halfThick = static_cast<int>(thickness / 2.0f) + 1;
        float radiusSq = (thickness / 2.0f) * (thickness / 2.0f);

        for (int py = -halfThick; py <= halfThick; ++py) {
            for (int px = -halfThick; px <= halfThick; ++px) {
                float distSq = static_cast<float>(px * px + py * py);
                if (distSq <= radiusSq) {
                    int xx = x + px;
                    int yy = y + py;

                    if (xx >= clip.x1 && xx < clip.x2 && yy >= clip.y1 && yy < clip.y2) {
                        int idx = yy * m_width + xx;
                        pixels[idx] = premultColor;
                    }
                }
            }
        }
    }
}
