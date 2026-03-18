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

        if (result != 0) {
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
                     ", atlas " + std::to_string(m_atlasWidth) + "x" + std::to_string(m_atlasHeight) +
                     ", bake result " + std::to_string(result) + ")");
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
// Drawing: filled rounded rectangle
// ---------------------------------------------------------------

void OverlayRenderer::drawRoundedRect(const Vec2& pos, const Vec2& size, uint32_t color, float radius)
{
    std::lock_guard<std::mutex> lock(m_renderMutex);

    if (!m_pixels || size.x <= 0 || size.y <= 0)
        return;

    // Clamp radius so it doesn't exceed half the smaller dimension
    float maxR = std::min(size.x, size.y) * 0.5f;
    float r = std::min(radius, maxR);

    uint8_t a = (color >> 24) & 0xFF;
    uint8_t sr = (color >> 16) & 0xFF;
    uint8_t sg = (color >> 8) & 0xFF;
    uint8_t sb = color & 0xFF;

    ClipRect clip = activeClip();
    uint32_t* pixels = static_cast<uint32_t*>(m_pixels);

    int ix1 = (int)pos.x;
    int iy1 = (int)pos.y;
    int ix2 = (int)(pos.x + size.x);
    int iy2 = (int)(pos.y + size.y);

    int ri = (int)(r + 1.0f);  // expand check zone by 1 for AA fringe

    for (int y = std::max(clip.y1, iy1 - 1); y < std::min(clip.y2, iy2 + 1); ++y) {
        for (int x = std::max(clip.x1, ix1 - 1); x < std::min(clip.x2, ix2 + 1); ++x) {
            // Check if this pixel is in a corner region
            float dx = 0, dy = 0;

            if (x < ix1 + ri && y < iy1 + ri) {
                dx = (ix1 + r) - x - 0.5f;
                dy = (iy1 + r) - y - 0.5f;
            } else if (x >= ix2 - ri && y < iy1 + ri) {
                dx = x + 0.5f - (ix2 - r);
                dy = (iy1 + r) - y - 0.5f;
            } else if (x < ix1 + ri && y >= iy2 - ri) {
                dx = (ix1 + r) - x - 0.5f;
                dy = y + 0.5f - (iy2 - r);
            } else if (x >= ix2 - ri && y >= iy2 - ri) {
                dx = x + 0.5f - (ix2 - r);
                dy = y + 0.5f - (iy2 - r);
            }

            // Compute coverage for anti-aliasing
            float coverage = 1.0f;
            if (dx > 0 && dy > 0) {
                float dist = std::sqrt(dx * dx + dy * dy);
                if (dist > r + 0.5f)
                    continue;  // fully outside
                if (dist > r - 0.5f)
                    coverage = r + 0.5f - dist;  // partial coverage (0..1)
            }

            uint8_t pixelAlpha = (uint8_t)(a * coverage);
            if (pixelAlpha == 0) continue;

            // Premultiply source with coverage-adjusted alpha
            uint8_t pr = (uint8_t)((sr * pixelAlpha) / 255);
            uint8_t pg = (uint8_t)((sg * pixelAlpha) / 255);
            uint8_t pb = (uint8_t)((sb * pixelAlpha) / 255);

            int idx = y * m_width + x;
            
            // Alpha-blend (premultiplied over)
            if (pixelAlpha == 255) {
                pixels[idx] = (255 << 24) | (sr << 16) | (sg << 8) | sb;
            } else {
                uint32_t dst = pixels[idx];
                uint8_t da = (dst >> 24) & 0xFF;
                uint8_t dr = (dst >> 16) & 0xFF;
                uint8_t dg = (dst >> 8) & 0xFF;
                uint8_t db = dst & 0xFF;
                
                uint8_t invA = 255 - pixelAlpha;
                uint8_t outA = pixelAlpha + (uint8_t)((da * invA) / 255);
                uint8_t outR = pr + (uint8_t)((dr * invA) / 255);
                uint8_t outG = pg + (uint8_t)((dg * invA) / 255);
                uint8_t outB = pb + (uint8_t)((db * invA) / 255);
                
                pixels[idx] = (outA << 24) | (outR << 16) | (outG << 8) | outB;
            }
        }
    }
}

// ---------------------------------------------------------------
// Drawing: rounded border outline
// ---------------------------------------------------------------

void OverlayRenderer::drawRoundedBorder(const Vec2& pos, const Vec2& size, uint32_t color, float radius, float thickness)
{
    std::lock_guard<std::mutex> lock(m_renderMutex);

    if (!m_pixels || size.x <= 0 || size.y <= 0)
        return;

    float maxR = std::min(size.x, size.y) * 0.5f;
    float r = std::min(radius, maxR);

    uint8_t a = (color >> 24) & 0xFF;
    uint8_t sr = (color >> 16) & 0xFF;
    uint8_t sg = (color >> 8) & 0xFF;
    uint8_t sb = color & 0xFF;

    ClipRect clip = activeClip();
    uint32_t* pixels = static_cast<uint32_t*>(m_pixels);

    int ix1 = (int)pos.x;
    int iy1 = (int)pos.y;
    int ix2 = (int)(pos.x + size.x);
    int iy2 = (int)(pos.y + size.y);

    float th = std::max(1.0f, thickness);
    float halfTh = th * 0.5f;

    for (int y = std::max(clip.y1, iy1 - 1); y < std::min(clip.y2, iy2 + 1); ++y) {
        for (int x = std::max(clip.x1, ix1 - 1); x < std::min(clip.x2, ix2 + 1); ++x) {
            float px = x + 0.5f;
            float py = y + 0.5f;

            // Signed distance to rounded rect interior
            // For a rounded rect the SDF is: max(|p - center| - halfExtent, 0).length - r
            float cx = pos.x + size.x * 0.5f;
            float cy = pos.y + size.y * 0.5f;
            float hx = size.x * 0.5f - r;
            float hy = size.y * 0.5f - r;

            float qx = std::abs(px - cx) - hx;
            float qy = std::abs(py - cy) - hy;
            float outerDist = std::sqrt(std::max(qx, 0.0f) * std::max(qx, 0.0f) +
                                        std::max(qy, 0.0f) * std::max(qy, 0.0f)) 
                              + std::min(std::max(qx, qy), 0.0f) - r;

            // Distance to ring center line at the shape edge (outerDist == 0)
            float ringDist = std::abs(outerDist) - halfTh;
            
            // Anti-alias: smooth transition over 1px
            float coverage = std::max(0.0f, std::min(1.0f, 0.5f - ringDist));
            if (coverage <= 0.0f) continue;

            uint8_t pixelAlpha = (uint8_t)(a * coverage);
            if (pixelAlpha == 0) continue;

            uint8_t pr = (uint8_t)((sr * pixelAlpha) / 255);
            uint8_t pg = (uint8_t)((sg * pixelAlpha) / 255);
            uint8_t pb = (uint8_t)((sb * pixelAlpha) / 255);

            int idx = y * m_width + x;
            
            if (pixelAlpha == 255) {
                pixels[idx] = (255 << 24) | (sr << 16) | (sg << 8) | sb;
            } else {
                uint32_t dst = pixels[idx];
                uint8_t da = (dst >> 24) & 0xFF;
                uint8_t dr = (dst >> 16) & 0xFF;
                uint8_t dg = (dst >> 8) & 0xFF;
                uint8_t db = dst & 0xFF;
                
                uint8_t invA = 255 - pixelAlpha;
                uint8_t outA = pixelAlpha + (uint8_t)((da * invA) / 255);
                uint8_t outR = pr + (uint8_t)((dr * invA) / 255);
                uint8_t outG = pg + (uint8_t)((dg * invA) / 255);
                uint8_t outB = pb + (uint8_t)((db * invA) / 255);
                
                pixels[idx] = (outA << 24) | (outR << 16) | (outG << 8) | outB;
            }
        }
    }
}

// ---------------------------------------------------------------
// Drawing: soft outer glow (feathered halo via SDF)
// ---------------------------------------------------------------

void OverlayRenderer::drawSoftGlow(const Vec2& pos, const Vec2& size, float radius,
                                    uint32_t color, float spread)
{
    std::lock_guard<std::mutex> lock(m_renderMutex);

    if (!m_pixels || size.x <= 0 || size.y <= 0 || spread <= 0)
        return;

    float maxR = std::min(size.x, size.y) * 0.5f;
    float r = std::min(radius, maxR);

    uint8_t baseA = (color >> 24) & 0xFF;
    uint8_t sr = (color >> 16) & 0xFF;
    uint8_t sg = (color >> 8) & 0xFF;
    uint8_t sb = color & 0xFF;

    ClipRect clip = activeClip();
    uint32_t* pixels = static_cast<uint32_t*>(m_pixels);

    // SDF center and half-extents (for rounded rect SDF)
    float cx = pos.x + size.x * 0.5f;
    float cy = pos.y + size.y * 0.5f;
    float hx = size.x * 0.5f - r;
    float hy = size.y * 0.5f - r;

    int margin = (int)(spread + 2);
    int x1 = std::max(clip.x1, (int)pos.x - margin);
    int y1 = std::max(clip.y1, (int)pos.y - margin);
    int x2 = std::min(clip.x2, (int)(pos.x + size.x) + margin);
    int y2 = std::min(clip.y2, (int)(pos.y + size.y) + margin);

    for (int y = y1; y < y2; ++y) {
        for (int x = x1; x < x2; ++x) {
            float px = x + 0.5f;
            float py = y + 0.5f;

            float qx = std::abs(px - cx) - hx;
            float qy = std::abs(py - cy) - hy;
            float dist = std::sqrt(std::max(qx, 0.0f) * std::max(qx, 0.0f) +
                                   std::max(qy, 0.0f) * std::max(qy, 0.0f))
                         + std::min(std::max(qx, qy), 0.0f) - r;

            // Only draw OUTSIDE the shape (dist > 0)
            if (dist <= 0.0f || dist > spread)
                continue;

            // Smooth quadratic falloff from shape edge outward
            float t = 1.0f - (dist / spread);
            float fade = t * t;
            uint8_t pixelAlpha = (uint8_t)(baseA * fade);
            if (pixelAlpha == 0) continue;

            uint8_t pr = (uint8_t)((sr * pixelAlpha) / 255);
            uint8_t pg = (uint8_t)((sg * pixelAlpha) / 255);
            uint8_t pb = (uint8_t)((sb * pixelAlpha) / 255);

            int idx = y * m_width + x;
            uint32_t dst = pixels[idx];
            uint8_t da = (dst >> 24) & 0xFF;
            uint8_t dr = (dst >> 16) & 0xFF;
            uint8_t dg = (dst >> 8) & 0xFF;
            uint8_t db = dst & 0xFF;

            uint8_t invA = 255 - pixelAlpha;
            uint8_t outA = pixelAlpha + (uint8_t)((da * invA) / 255);
            uint8_t outR = pr + (uint8_t)((dr * invA) / 255);
            uint8_t outG = pg + (uint8_t)((dg * invA) / 255);
            uint8_t outB = pb + (uint8_t)((db * invA) / 255);

            pixels[idx] = (outA << 24) | (outR << 16) | (outG << 8) | outB;
        }
    }
}

// ---------------------------------------------------------------
// Drawing: glassmorphism panel (blur + shadow + fill + glow + border)
// ---------------------------------------------------------------

void OverlayRenderer::drawGlassPanel(const Vec2& pos, const Vec2& size, float radius,
                                      uint32_t bgColor, uint32_t borderColor, uint32_t glowColor,
                                      int blurRadius, float shadowOffset)
{
    // 1. Frosted blur behind the panel (before drawing anything)
    blurRegion((int)pos.x, (int)pos.y, (int)size.x, (int)size.y, blurRadius);
    
    // 2. Soft outer glow — feathered halo outside the shape for bubble softness
    float glowSpread = 6.0f;
    drawSoftGlow(pos, size, radius, 0x18FFFFFF, glowSpread);
    
    // 3. Drop shadow (offset, darker)
    if (shadowOffset > 0) {
        drawRoundedRect({pos.x + shadowOffset, pos.y + shadowOffset}, size, 0x30000000, radius);
    }
    
    // 4. Glass fill — semi-transparent background over the blurred area
    drawRoundedRect(pos, size, bgColor, radius);
    
    // 5. Smooth per-pixel top-edge highlight (bubble specular — no banding)
    if (glowColor != 0) {
        std::lock_guard<std::mutex> lock(m_renderMutex);
        if (m_pixels) {
            uint8_t ga = (glowColor >> 24) & 0xFF;
            uint8_t gr = (glowColor >> 16) & 0xFF;
            uint8_t gg = (glowColor >> 8) & 0xFF;
            uint8_t gb = glowColor & 0xFF;
            
            float gradientHeight = size.y * 0.3f;
            float maxR = std::min(size.x, size.y) * 0.5f;
            float r = std::min(radius, maxR);
            
            ClipRect clip = activeClip();
            uint32_t* pixels = static_cast<uint32_t*>(m_pixels);
            
            // SDF parameters for rounded rect
            float cx = pos.x + size.x * 0.5f;
            float cy = pos.y + size.y * 0.5f;
            float hx = size.x * 0.5f - r;
            float hy = size.y * 0.5f - r;
            
            int ix1 = std::max(clip.x1, (int)pos.x);
            int iy1 = std::max(clip.y1, (int)pos.y);
            int ix2 = std::min(clip.x2, (int)(pos.x + size.x));
            int iy2 = std::min(clip.y2, (int)(pos.y + gradientHeight));
            
            for (int y = iy1; y < iy2; ++y) {
                float py = y + 0.5f;
                // Vertical fade: 1 at top, 0 at bottom of gradient zone
                float vt = (py - pos.y) / gradientHeight;
                float vFade = (1.0f - vt) * (1.0f - vt); // quadratic falloff
                
                for (int x = ix1; x < ix2; ++x) {
                    float px = x + 0.5f;
                    
                    // SDF test — only draw inside the rounded rect
                    float qx = std::abs(px - cx) - hx;
                    float qy = std::abs(py - cy) - hy;
                    float dist = std::sqrt(std::max(qx, 0.0f) * std::max(qx, 0.0f) +
                                           std::max(qy, 0.0f) * std::max(qy, 0.0f))
                                 + std::min(std::max(qx, qy), 0.0f) - r;
                    
                    if (dist > 0.0f) continue; // outside shape
                    
                    // Anti-alias at edge (smooth 1px transition)
                    float edgeFade = std::min(1.0f, -dist);
                    
                    uint8_t pixelAlpha = (uint8_t)(ga * vFade * edgeFade);
                    if (pixelAlpha == 0) continue;
                    
                    uint8_t pr = (uint8_t)((gr * pixelAlpha) / 255);
                    uint8_t pg = (uint8_t)((gg * pixelAlpha) / 255);
                    uint8_t pb = (uint8_t)((gb * pixelAlpha) / 255);
                    
                    int idx = y * m_width + x;
                    uint32_t dst = pixels[idx];
                    uint8_t da = (dst >> 24) & 0xFF;
                    uint8_t dr = (dst >> 16) & 0xFF;
                    uint8_t dg = (dst >> 8) & 0xFF;
                    uint8_t db = dst & 0xFF;
                    
                    uint8_t invA = 255 - pixelAlpha;
                    uint8_t outA = pixelAlpha + (uint8_t)((da * invA) / 255);
                    uint8_t outR = pr + (uint8_t)((dr * invA) / 255);
                    uint8_t outG = pg + (uint8_t)((dg * invA) / 255);
                    uint8_t outB = pb + (uint8_t)((db * invA) / 255);
                    
                    pixels[idx] = (outA << 24) | (outR << 16) | (outG << 8) | outB;
                }
            }
        }
    }
    
    // 6. Soft border — use lower opacity for subtle glass edge
    uint8_t bA = (borderColor >> 24) & 0xFF;
    uint8_t bR = (borderColor >> 16) & 0xFF;
    uint8_t bG = (borderColor >> 8) & 0xFF;
    uint8_t bB = borderColor & 0xFF;
    // Soften border to ~40% of its original alpha for a frosted, not harsh, edge
    uint8_t softA = (uint8_t)(bA * 0.4f);
    uint32_t softBorder = (softA << 24) | (bR << 16) | (bG << 8) | bB;
    drawRoundedBorder(pos, size, softBorder, radius, 1.0f);
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

// ---------------------------------------------------------------
// Blur: 2-pass separable box blur for frosted glass effect
// ---------------------------------------------------------------

void OverlayRenderer::blurRegion(int x, int y, int w, int h, int radius)
{
    std::lock_guard<std::mutex> lock(m_renderMutex);

    if (!m_pixels || w <= 0 || h <= 0 || radius <= 0)
        return;

    // Clamp region to pixel buffer bounds
    int x1 = std::max(0, x);
    int y1 = std::max(0, y);
    int x2 = std::min(m_width, x + w);
    int y2 = std::min(m_height, y + h);
    int rw = x2 - x1;
    int rh = y2 - y1;
    if (rw <= 0 || rh <= 0) return;

    // Lazy-allocate temp buffer
    size_t needed = static_cast<size_t>(rw) * rh;
    if (m_blurTemp.size() < needed)
        m_blurTemp.resize(needed);

    uint32_t* pixels = static_cast<uint32_t*>(m_pixels);
    int diameter = radius * 2 + 1;

    // --- Pass 1: Horizontal blur → temp buffer ---
    for (int row = 0; row < rh; ++row) {
        int py = y1 + row;
        for (int col = 0; col < rw; ++col) {
            int px = x1 + col;
            uint32_t sumA = 0, sumR = 0, sumG = 0, sumB = 0;
            int count = 0;
            for (int k = -radius; k <= radius; ++k) {
                int sx = std::clamp(px + k, x1, x2 - 1);
                uint32_t c = pixels[py * m_width + sx];
                sumA += (c >> 24) & 0xFF;
                sumR += (c >> 16) & 0xFF;
                sumG += (c >> 8) & 0xFF;
                sumB += c & 0xFF;
                ++count;
            }
            m_blurTemp[row * rw + col] = 
                ((sumA / count) << 24) |
                ((sumR / count) << 16) |
                ((sumG / count) << 8) |
                (sumB / count);
        }
    }

    // --- Pass 2: Vertical blur from temp → back to pixel buffer ---
    for (int col = 0; col < rw; ++col) {
        for (int row = 0; row < rh; ++row) {
            uint32_t sumA = 0, sumR = 0, sumG = 0, sumB = 0;
            int count = 0;
            for (int k = -radius; k <= radius; ++k) {
                int sy = std::clamp(row + k, 0, rh - 1);
                uint32_t c = m_blurTemp[sy * rw + col];
                sumA += (c >> 24) & 0xFF;
                sumR += (c >> 16) & 0xFF;
                sumG += (c >> 8) & 0xFF;
                sumB += c & 0xFF;
                ++count;
            }
            int py = y1 + row;
            int px = x1 + col;
            pixels[py * m_width + px] = 
                ((sumA / count) << 24) |
                ((sumR / count) << 16) |
                ((sumG / count) << 8) |
                (sumB / count);
        }
    }
}
