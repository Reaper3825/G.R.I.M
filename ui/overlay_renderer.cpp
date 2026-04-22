#include "overlay_renderer.hpp"
#include "logger.hpp"
#include "core/platform_window.hpp"
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
    m_backdropPixels = new uint32_t[width * height]();
    m_backdropDirty = true;

#if defined(_WIN32)
    // On Windows we enable DWM blur-behind at the window level, so we skip
    // per-panel synthetic backdrop blurring for better "real desktop" blur.
    m_usePlatformBlur = true;
#endif

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

#if defined(__APPLE__)
    // On macOS we enable vibrancy/NSVisualEffectView behind the overlay window,
    // so we skip per-panel synthetic backdrop blurring.
    m_usePlatformBlur = true;
#endif

    LOG_DEBUG("OverlayRenderer", "Initialized overlay renderer (" +
             std::to_string(width) + "x" + std::to_string(height) + ")");
}
#endif

void OverlayRenderer::init(int width, int height, void* pixelBuffer)
{
    std::lock_guard<std::mutex> lock(m_renderMutex);
    m_width = width;
    m_height = height;

#if defined(_WIN32) || defined(__APPLE__)
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
    m_fontAtlas.clear();
    m_fontLoaded = false;
    m_glassCache.clear();
}

// ---------------------------------------------------------------
// Font loading via stb_truetype
// ---------------------------------------------------------------

static std::vector<uint8_t> readFontFile(const std::string& fontPath)
{
    std::ifstream file(fontPath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LOG_ERROR("OverlayRenderer", "Failed to open font file: " + fontPath);
        return {};
    }
    auto fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(static_cast<size_t>(fileSize));
    file.read(reinterpret_cast<char*>(data.data()), fileSize);
    return data;
}

// Decode one UTF-8 codepoint; advance pos past consumed bytes.
uint32_t OverlayRenderer::decodeUtf8(const std::string& text, size_t& pos)
{
    uint8_t c = static_cast<uint8_t>(text[pos]);
    uint32_t cp = 0;
    int extra = 0;

    if (c < 0x80) {
        cp = c; extra = 0;
    } else if ((c & 0xE0) == 0xC0) {
        cp = c & 0x1F; extra = 1;
    } else if ((c & 0xF0) == 0xE0) {
        cp = c & 0x0F; extra = 2;
    } else if ((c & 0xF8) == 0xF0) {
        cp = c & 0x07; extra = 3;
    } else {
        // Invalid lead byte — skip
        ++pos;
        return 0xFFFD; // replacement character
    }

    ++pos;
    for (int i = 0; i < extra && pos < text.size(); ++i, ++pos) {
        uint8_t cont = static_cast<uint8_t>(text[pos]);
        if ((cont & 0xC0) != 0x80) return 0xFFFD; // broken sequence
        cp = (cp << 6) | (cont & 0x3F);
    }
    return cp;
}

void OverlayRenderer::setFont(const std::string& fontPath, int fontSize)
{
    std::lock_guard<std::mutex> lock(m_renderMutex);

    m_fontFileData = readFontFile(fontPath);
    if (m_fontFileData.empty()) {
        m_fontLoaded = false;
        return;
    }

    m_fontSize = static_cast<float>(fontSize);
    rebuildAtlas();

    if (m_fontLoaded) {
        LOG_DEBUG("OverlayRenderer", "Font loaded: " + fontPath +
                 " (size " + std::to_string(fontSize) +
                 ", atlas " + std::to_string(m_atlasWidth) + "x" + std::to_string(m_atlasHeight) +
                 ", glyphs " + std::to_string(m_glyphMap.size()) + ")");
    }
}

void OverlayRenderer::loadIconFont(const std::string& fontPath)
{
    std::lock_guard<std::mutex> lock(m_renderMutex);

    m_iconFontFileData = readFontFile(fontPath);
    if (m_iconFontFileData.empty()) return;

    rebuildAtlas();

    LOG_DEBUG("OverlayRenderer", "Icon font loaded: " + fontPath +
             " (total glyphs " + std::to_string(m_glyphMap.size()) + ")");
}

void OverlayRenderer::rebuildAtlas()
{
    m_glyphMap.clear();
    m_fontLoaded = false;

    if (m_fontFileData.empty()) return;

    // Define codepoint ranges to bake
    // Range 1: ASCII printable (32-126)
    // Range 2: Latin Extended-A (0x100-0x17F) for accented chars
    // Range 3: General Punctuation (0x2000-0x206F) for em-dash, bullets etc.
    // Icon ranges are added if an icon font is loaded
    struct PackRange {
        int firstCodepoint;
        int count;
        bool isIconFont; // true = use icon font data instead of text font
    };

    std::vector<PackRange> ranges = {
        { 32,    95,   false }, // ASCII 32-126
        { 0x100, 128,  false }, // Latin Extended-A
        { 0x2000, 112, false }, // General Punctuation
    };

    // FontAwesome icon ranges (v4/v5/v6 overlap these)
    if (!m_iconFontFileData.empty()) {
        ranges.push_back({ 0xE000, 256,  true }); // Private Use Area (common icon range)
        ranges.push_back({ 0xE200, 256,  true }); // Extended PUA icons
        ranges.push_back({ 0xF000, 4096, true }); // FontAwesome / Nerd Font main range
    }

    // Total chars across all ranges
    int totalChars = 0;
    for (auto& r : ranges) totalChars += r.count;

    // Try progressively larger atlas sizes
    for (m_atlasWidth = 512, m_atlasHeight = 512;
         m_atlasWidth <= 4096;
         m_atlasWidth *= 2, m_atlasHeight *= 2)
    {
        m_fontAtlas.resize(m_atlasWidth * m_atlasHeight);
        std::fill(m_fontAtlas.begin(), m_fontAtlas.end(), 0);

        stbtt_pack_context pc;
        if (!stbtt_PackBegin(&pc, m_fontAtlas.data(), m_atlasWidth, m_atlasHeight, 0, 1, nullptr)) {
            continue;
        }
        stbtt_PackSetOversampling(&pc, 1, 1); // No oversampling — software blitter draws 1:1

        // Allocate packed char storage
        std::vector<stbtt_packedchar> packedChars(totalChars);

        // Build stbtt_pack_range array
        std::vector<stbtt_pack_range> stbRanges;
        int offset = 0;
        for (auto& r : ranges) {
            stbtt_pack_range pr = {};
            pr.font_size = m_fontSize;
            pr.first_unicode_codepoint_in_range = r.firstCodepoint;
            pr.num_chars = r.count;
            pr.chardata_for_range = &packedChars[offset];
            stbRanges.push_back(pr);
            offset += r.count;
        }

        // Pack text font ranges
        bool allOk = true;
        {
            // Collect text font range indices
            std::vector<stbtt_pack_range> textRanges;
            for (size_t i = 0; i < ranges.size(); ++i) {
                if (!ranges[i].isIconFont) textRanges.push_back(stbRanges[i]);
            }
            if (!textRanges.empty()) {
                int ret = stbtt_PackFontRanges(&pc, m_fontFileData.data(), 0,
                                               textRanges.data(), (int)textRanges.size());
                if (ret == 0) allOk = false;
            }
        }

        // Pack icon font ranges
        if (allOk && !m_iconFontFileData.empty()) {
            std::vector<stbtt_pack_range> iconRanges;
            for (size_t i = 0; i < ranges.size(); ++i) {
                if (ranges[i].isIconFont) iconRanges.push_back(stbRanges[i]);
            }
            if (!iconRanges.empty()) {
                int ret = stbtt_PackFontRanges(&pc, m_iconFontFileData.data(), 0,
                                               iconRanges.data(), (int)iconRanges.size());
                if (ret == 0) {
                    LOG_DEBUG("OverlayRenderer", "Some icon glyphs failed to pack — atlas may be too small");
                }
            }
        }

        stbtt_PackEnd(&pc);

        if (!allOk) continue; // try larger atlas

        // Transfer packed chars into glyph map
        offset = 0;
        for (auto& r : ranges) {
            for (int i = 0; i < r.count; ++i) {
                const stbtt_packedchar& pc2 = packedChars[offset + i];
                // Skip glyphs that weren't successfully packed (empty)
                if (pc2.x1 == 0 && pc2.y1 == 0 && pc2.x0 == 0 && pc2.y0 == 0
                    && pc2.xadvance == 0.0f) {
                    continue;
                }
                uint32_t cp = r.firstCodepoint + i;
                BakedChar bc;
                bc.x0 = pc2.x0;
                bc.y0 = pc2.y0;
                bc.x1 = pc2.x1;
                bc.y1 = pc2.y1;
                bc.xoff = pc2.xoff;
                bc.yoff = pc2.yoff;
                bc.xadvance = pc2.xadvance;
                m_glyphMap[cp] = bc;
            }
            offset += r.count;
        }

        m_fontLoaded = true;
        return;
    }

    LOG_ERROR("OverlayRenderer", "Failed to bake font atlas (tried up to 4096x4096)");
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
    int x2 = x + w;
    int y2 = y + h;
    if (m_dirtyX1 >= m_dirtyX2) {
        // First dirty rect this frame
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

    // Advance grain seed each frame (xorshift32)
    m_grainSeed ^= (m_grainSeed << 13);
    m_grainSeed ^= (m_grainSeed >> 17);
    m_grainSeed ^= (m_grainSeed << 5);

    m_clipStack.clear();

    // Only clear the region that was drawn last frame (union of prev + current dirty).
    // For a multi-monitor overlay this avoids memset on millions of untouched pixels.
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

    // Reset dirty rect for this frame
    m_dirtyX1 = m_dirtyX2 = m_dirtyY1 = m_dirtyY2 = 0;
}

void OverlayRenderer::endFrame()
{
    std::lock_guard<std::mutex> lock(m_renderMutex);

    // Save dirty rect for next frame's clear pass
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

// ---------------------------------------------------------------
// Backdrop: draw to separate buffer so blobs are only visible through blurred panels
// ---------------------------------------------------------------

void OverlayRenderer::drawRoundedRectToBuffer(uint32_t* buf, int bufW, int bufH,
                                              const Vec2& pos, const Vec2& size, uint32_t color, float radius)
{
    if (!buf || bufW <= 0 || bufH <= 0 || size.x <= 0 || size.y <= 0)
        return;
    float maxR = std::min(size.x, size.y) * 0.5f;
    float r = std::min(radius, maxR);
    uint8_t a = (color >> 24) & 0xFF;
    uint8_t sr = (color >> 16) & 0xFF;
    uint8_t sg = (color >> 8) & 0xFF;
    uint8_t sb = color & 0xFF;
    int ix1 = (int)pos.x;
    int iy1 = (int)pos.y;
    int ix2 = (int)(pos.x + size.x);
    int iy2 = (int)(pos.y + size.y);
    int ri = (int)(r + 1.0f);
    int clipX1 = 0, clipY1 = 0, clipX2 = bufW, clipY2 = bufH;

    for (int y = std::max(clipY1, iy1 - 1); y < std::min(clipY2, iy2 + 1); ++y) {
        for (int x = std::max(clipX1, ix1 - 1); x < std::min(clipX2, ix2 + 1); ++x) {
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
            float coverage = 1.0f;
            if (dx > 0 && dy > 0) {
                float dist = std::sqrt(dx * dx + dy * dy);
                if (dist > r + 0.5f) continue;
                if (dist > r - 0.5f) coverage = r + 0.5f - dist;
            }
            uint8_t pixelAlpha = (uint8_t)(a * coverage);
            if (pixelAlpha == 0) continue;
            uint8_t pr = (uint8_t)((sr * pixelAlpha) / 255);
            uint8_t pg = (uint8_t)((sg * pixelAlpha) / 255);
            uint8_t pb = (uint8_t)((sb * pixelAlpha) / 255);
            int idx = y * bufW + x;
            uint32_t dst = buf[idx];
            uint8_t da = (dst >> 24) & 0xFF;
            uint8_t dr = (dst >> 16) & 0xFF;
            uint8_t dg = (dst >> 8) & 0xFF;
            uint8_t db = dst & 0xFF;
            uint8_t invA = 255 - pixelAlpha;
            uint8_t outA = pixelAlpha + (uint8_t)((da * invA) / 255);
            uint8_t outR = pr + (uint8_t)((dr * invA) / 255);
            uint8_t outG = pg + (uint8_t)((dg * invA) / 255);
            uint8_t outB = pb + (uint8_t)((db * invA) / 255);
            buf[idx] = (outA << 24) | (outR << 16) | (outG << 8) | outB;
        }
    }
}

void OverlayRenderer::drawBackdrop(int width, int height)
{
    if (!m_backdropPixels || width <= 0 || height <= 0)
        return;

    if (m_usePlatformBlur) {
        // OS blur already provides the "real desktop" backdrop. Keep our own
        // synthetic noise pipeline disabled so we don't mask the blur.
        m_backdropDirty = false;
        return;
    }

    if (!m_backdropDirty)
        return;

    std::lock_guard<std::mutex> lock(m_renderMutex);

    // Frost source texture: per-pixel pseudo-random noise (tinted).
    // We precompute once to avoid dots/shapes and per-panel generation cost.
    // blurRegion() will then turn it into cloudy frosted distortion.
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uint32_t h = (uint32_t)(x * 73856093) ^ (uint32_t)(y * 19349663);
            h ^= (h << 13);
            h ^= (h >> 17);
            h ^= (h << 5);
            float t = static_cast<float>(h & 0xFFFFu) / 65535.0f;

            // Violet-blue tinted noise — stronger range so blur reads as visible frost
            uint8_t r = static_cast<uint8_t>(30.0f + t * 120.0f);
            uint8_t g = static_cast<uint8_t>(20.0f + t * 80.0f);
            uint8_t b = static_cast<uint8_t>(40.0f + t * 130.0f);
            uint8_t a = static_cast<uint8_t>(140.0f + t * 115.0f);

            m_backdropPixels[y * width + x] = (static_cast<uint32_t>(a) << 24) |
                                              (static_cast<uint32_t>(r) << 16) |
                                              (static_cast<uint32_t>(g) << 8)  |
                                              (static_cast<uint32_t>(b));
        }
    }

    m_backdropDirty = false;
}

void OverlayRenderer::copyBackdropRegion(int dstX, int dstY, int w, int h)
{
    std::lock_guard<std::mutex> lock(m_renderMutex);
    if (!m_pixels || !m_backdropPixels || w <= 0 || h <= 0)
        return;

    if (m_usePlatformBlur)
        return;

    // Copy from a fixed source (center of backdrop) so the frost pattern follows the panel
    int srcX = std::max(0, (m_width - w) / 2);
    int srcY = std::max(0, (m_height - h) / 2);
    int copyW = std::min(w, m_width - srcX);
    int copyH = std::min(h, m_height - srcY);
    uint32_t* dst = static_cast<uint32_t*>(m_pixels);
    for (int row = 0; row < copyH; ++row) {
        int sy = srcY + row;
        int dy = dstY + row;
        if (dy < 0 || dy >= m_height) continue;
        for (int col = 0; col < copyW; ++col) {
            int sx = srcX + col;
            int dx = dstX + col;
            if (dx < 0 || dx >= m_width) continue;
            dst[dy * m_width + dx] = m_backdropPixels[sy * m_width + sx];
        }
    }
}

// ---------------------------------------------------------------
// Drawing: filled rectangle
// ---------------------------------------------------------------

void OverlayRenderer::drawRect(const Vec2& pos, const Vec2& size, uint32_t color)
{
    expandDirtyRect((int)pos.x, (int)pos.y, (int)size.x, (int)size.y);
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
    expandDirtyRect((int)pos.x - 1, (int)pos.y - 1, (int)size.x + 2, (int)size.y + 2);
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
    expandDirtyRect((int)pos.x - 1, (int)pos.y - 1, (int)size.x + 2, (int)size.y + 2);
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
    int sp = (int)spread + 1;
    expandDirtyRect((int)pos.x - sp, (int)pos.y - sp, (int)size.x + sp * 2, (int)size.y + sp * 2);
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

// Fast content hash for dirty detection (sample every 16th pixel)
static uint64_t computeContentHash(const uint32_t* pixels, size_t count)
{
    uint64_t hash = 0;
    const int step = 16;
    for (size_t i = 0; i < count; i += step)
        hash = hash * 31u + pixels[i];
    return hash;
}

void OverlayRenderer::blitCachedToPanel(int rx, int ry, int capW, int capH,
                                        const std::vector<uint32_t>& cachedPixels,
                                        const Vec2& pos, const Vec2& size, float radius)
{
    if (!m_pixels || cachedPixels.size() < static_cast<size_t>(capW * capH))
        return;

    std::lock_guard<std::mutex> lock(m_renderMutex);
    uint32_t* pixels = static_cast<uint32_t*>(m_pixels);
    ClipRect clip = activeClip();

    float maxR = std::min(size.x, size.y) * 0.5f;
    float r = std::min(radius, maxR);
    float cx = pos.x + size.x * 0.5f;
    float cy = pos.y + size.y * 0.5f;
    float hx = size.x * 0.5f - r;
    float hy = size.y * 0.5f - r;

    int ix1 = std::max(clip.x1, rx);
    int iy1 = std::max(clip.y1, ry);
    int ix2 = std::min(clip.x2, rx + capW);
    int iy2 = std::min(clip.y2, ry + capH);

    for (int y = iy1; y < iy2; ++y) {
        float py = y + 0.5f;
        int sy0 = y - ry;
        for (int x = ix1; x < ix2; ++x) {
            float px = x + 0.5f;

            float qx = std::abs(px - cx) - hx;
            float qy = std::abs(py - cy) - hy;
            float dist = std::sqrt(std::max(qx, 0.0f) * std::max(qx, 0.0f) +
                                   std::max(qy, 0.0f) * std::max(qy, 0.0f))
                         + std::min(std::max(qx, qy), 0.0f) - r;
            if (dist > 0.0f) continue;

            int sx = x - rx;
            int sy = sy0;
            if (sx >= 0 && sx < capW && sy >= 0 && sy < capH)
                pixels[y * m_width + x] = cachedPixels[sy * capW + sx];
        }
    }
}

void OverlayRenderer::drawGlassPanel(const Vec2& pos, const Vec2& size, float radius,
                                      uint32_t bgColor, uint32_t borderColor, uint32_t glowColor,
                                      int blurRadius, float shadowOffset,
                                      uintptr_t panelId)
{
    int rx = (int)pos.x, ry = (int)pos.y;
    int rw = (int)size.x, rh = (int)size.y;
    // Track the full panel area (including shadow offset) as dirty
    expandDirtyRect(rx - 1, ry - 1, rw + (int)shadowOffset + 2, rh + (int)shadowOffset + 2);

    // 1-2. Real refraction: capture pixels behind the overlay and distort them.
    // This is the only way to get true "desktop refraction" in our software renderer.
    bool haveOverlayHandle = false;
#if defined(__APPLE__)
    haveOverlayHandle = (m_nativeWindow != nullptr);
#elif defined(_WIN32)
    haveOverlayHandle = (m_hwnd != nullptr);
#endif

    if (haveOverlayHandle) {
        const int capW = std::max(1, rw);
        const int capH = std::max(1, rh);
        const size_t needed = (size_t)capW * (size_t)capH;

        // Fast path: if panel hasn't moved/resized and we have a cached result,
        // skip the expensive captureDesktopBehindOverlay + hash entirely.
        // The desktop behind a stationary panel changes (windows moving, video
        // playing), but through a heavy frosted-glass blur the difference is
        // invisible — same approach macOS uses with NSVisualEffectView caching.
        bool usedCache = false;
        if (panelId != 0) {
            auto it = m_glassCache.find(panelId);
            if (it != m_glassCache.end() &&
                it->second.rectX == rx && it->second.rectY == ry &&
                it->second.rectW == capW && it->second.rectH == capH &&
                it->second.pixels.size() >= needed) {
                blitCachedToPanel(rx, ry, capW, capH, it->second.pixels, pos, size, radius);
                usedCache = true;
            }
        }

        if (!usedCache) {
            // Panel moved, resized, or first frame — do the full capture pipeline.
            if (m_captureTemp.size() < needed) m_captureTemp.resize(needed);

            bool ok = PlatformWindow::captureDesktopBehindOverlay(
#if defined(__APPLE__)
                m_nativeWindow,
#else
                (void*)m_hwnd,
#endif
                rx, ry, capW, capH, m_captureTemp.data());

            if (ok) {
                // Distort + write into panel area (inside rounded rect only).
                {
                    std::lock_guard<std::mutex> lock(m_renderMutex);
                    uint32_t* pixels = static_cast<uint32_t*>(m_pixels);
                    ClipRect clip = activeClip();

                    float maxR = std::min(size.x, size.y) * 0.5f;
                    float r = std::min(radius, maxR);
                    float cx = pos.x + size.x * 0.5f;
                    float cy = pos.y + size.y * 0.5f;
                    float hx = size.x * 0.5f - r;
                    float hy = size.y * 0.5f - r;

                    int ix1 = std::max(clip.x1, rx);
                    int iy1 = std::max(clip.y1, ry);
                    int ix2 = std::min(clip.x2, rx + capW);
                    int iy2 = std::min(clip.y2, ry + capH);

                    // Distortion strength in pixels (tuned for "glass" refraction).
                    const float distortPx = float(blurRadius) / 10.0f;

                    // Low-frequency smooth noise (value noise) → gradient for refraction offsets.
                    auto hash01 = [](uint32_t v) -> float {
                        v ^= (v >> 16);
                        v *= 0x7feb352du;
                        v ^= (v >> 15);
                        v *= 0x846ca68bu;
                        v ^= (v >> 16);
                        return (v & 0x00FFFFFFu) / 16777215.0f;
                    };
                    auto gridNoise = [&](int gx, int gy) -> float {
                        uint32_t v = (uint32_t)gx * 374761393u ^ (uint32_t)gy * 668265263u ^ m_grainSeed;
                        return hash01(v);
                    };
                    auto smoothstep = [](float t) -> float { return t * t * (3.0f - 2.0f * t); };
                    auto sampleNoise = [&](float u, float v) -> float {
                        const float scale = 1.0f / 46.0f;
                        float xg = u * scale;
                        float yg = v * scale;
                        int x0 = (int)std::floor(xg);
                        int y0 = (int)std::floor(yg);
                        float tx = xg - x0;
                        float ty = yg - y0;
                        float sx = smoothstep(tx);
                        float sy = smoothstep(ty);
                        float n00 = gridNoise(x0,     y0);
                        float n10 = gridNoise(x0 + 1, y0);
                        float n01 = gridNoise(x0,     y0 + 1);
                        float n11 = gridNoise(x0 + 1, y0 + 1);
                        float nx0 = n00 + (n10 - n00) * sx;
                        float nx1 = n01 + (n11 - n01) * sx;
                        return nx0 + (nx1 - nx0) * sy;
                    };

                    for (int y = iy1; y < iy2; ++y) {
                        float py = y + 0.5f;
                        int sy0 = y - ry;
                        for (int x = ix1; x < ix2; ++x) {
                            float px = x + 0.5f;

                            float qx = std::abs(px - cx) - hx;
                            float qy = std::abs(py - cy) - hy;
                            float dist = std::sqrt(std::max(qx, 0.0f) * std::max(qx, 0.0f) +
                                                   std::max(qy, 0.0f) * std::max(qy, 0.0f))
                                         + std::min(std::max(qx, qy), 0.0f) - r;
                            if (dist > 0.0f) continue;

                            float u = (float)(x - rx);
                            float v = (float)(y - ry);
                            const float eps = 1.35f;
                            float nL = sampleNoise(u - eps, v);
                            float nR = sampleNoise(u + eps, v);
                            float nU = sampleNoise(u, v - eps);
                            float nD = sampleNoise(u, v + eps);
                            float gx = (nR - nL);
                            float gy = (nD - nU);

                            float ox = std::max(-1.0f, std::min(1.0f, gx * 2.0f)) * distortPx;
                            float oy = std::max(-1.0f, std::min(1.0f, gy * 2.0f)) * distortPx;

                            int sx = std::clamp((int)((x - rx) + ox), 0, capW - 1);
                            int sy = std::clamp((int)(sy0 + oy), 0, capH - 1);

                            pixels[y * m_width + x] = m_captureTemp[sy * capW + sx];
                        }
                    }
                }

                // Slight blur to soften distortion and mimic glass diffusion.
                // 2 passes with moderate radius gives good frosted glass appearance
                // without the extreme cost of 4 passes.
                int frostRadius = (int)std::round(blurRadius / 2.5f);
                frostRadius = std::clamp(frostRadius, 8, 32);
                blurRegion(rx, ry, capW, capH, frostRadius);
                blurRegion(rx, ry, capW, capH, frostRadius);

                // Cache result for next frame (when panelId provided)
                if (panelId != 0) {
                    auto& cache = m_glassCache[panelId];
                    cache.contentHash = 0;
                    cache.rectX = rx;
                    cache.rectY = ry;
                    cache.rectW = capW;
                    cache.rectH = capH;
                    cache.pixels.resize(needed);
                    uint32_t* pixels = static_cast<uint32_t*>(m_pixels);
                    for (int row = 0; row < capH; ++row)
                        std::memcpy(cache.pixels.data() + row * capW,
                                    pixels + (ry + row) * m_width + rx,
                                    capW * sizeof(uint32_t));
                }
            }
        }
    }

    // 3. Drop shadow (slightly offset, same size as panel — no halo)
    if (shadowOffset > 0) {
        drawRoundedRect({pos.x + shadowOffset, pos.y + shadowOffset}, size, 0x35000000, radius);
    }

    // 4. Glass fill — tint so OS-blurred backdrop is visible through panel.
    //    Keep alpha relatively low; otherwise the panel reads as an opaque gray.
    uint8_t ba = (bgColor >> 24) & 0xFF;
    uint32_t glassTint = (bgColor & 0x00FFFFFF) | ((uint32_t)(ba * 40 / 100) << 24);  // ~40% of theme alpha
    drawRoundedRect(pos, size, glassTint, radius);

    // 4b. Glass grain/noise — adds "refraction" cue so it doesn't read as flat gray tint.
    // Only needed when OS blur is used (we're not blurring our own backdrop texture then).
    if (m_usePlatformBlur) {
        applyGlassGrain(pos, size, radius, 0.28f);
    }

    // 5. Top-edge highlight (inside panel only, no bleed)
    if (glowColor != 0) {
        std::lock_guard<std::mutex> lock(m_renderMutex);
        if (m_pixels) {
            uint8_t ga = (glowColor >> 24) & 0xFF;
            uint8_t gr = (glowColor >> 16) & 0xFF;
            uint8_t gg = (glowColor >> 8) & 0xFF;
            uint8_t gb = glowColor & 0xFF;

            float gradientHeight = size.y * 0.25f;
            float maxR = std::min(size.x, size.y) * 0.5f;
            float r = std::min(radius, maxR);

            ClipRect clip = activeClip();
            uint32_t* pixels = static_cast<uint32_t*>(m_pixels);

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
                float vt = (py - pos.y) / gradientHeight;
                float vFade = (1.0f - vt) * (1.0f - vt) * (1.0f - vt); // cubic falloff for smoother highlight

                for (int x = ix1; x < ix2; ++x) {
                    float px = x + 0.5f;

                    float qx = std::abs(px - cx) - hx;
                    float qy = std::abs(py - cy) - hy;
                    float dist = std::sqrt(std::max(qx, 0.0f) * std::max(qx, 0.0f) +
                                           std::max(qy, 0.0f) * std::max(qy, 0.0f))
                                 + std::min(std::max(qx, qy), 0.0f) - r;

                    if (dist > 0.0f) continue;

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

    // 6. Clean border — crisp edge, no glow past the panel
    drawRoundedBorder(pos, size, borderColor, radius, 1.0f);
}

void OverlayRenderer::applyGlassGrain(const Vec2& pos, const Vec2& size, float radius, float strength)
{
    std::lock_guard<std::mutex> lock(m_renderMutex);
    if (!m_pixels || strength <= 0.0f || size.x <= 0 || size.y <= 0)
        return;

    strength = std::min(1.0f, std::max(0.0f, strength));

    ClipRect clip = activeClip();
    uint32_t* pixels = static_cast<uint32_t*>(m_pixels);

    float maxR = std::min(size.x, size.y) * 0.5f;
    float r = std::min(radius, maxR);
    float cx = pos.x + size.x * 0.5f;
    float cy = pos.y + size.y * 0.5f;
    float hx = size.x * 0.5f - r;
    float hy = size.y * 0.5f - r;

    int ix1 = std::max(clip.x1, (int)pos.x);
    int iy1 = std::max(clip.y1, (int)pos.y);
    int ix2 = std::min(clip.x2, (int)(pos.x + size.x));
    int iy2 = std::min(clip.y2, (int)(pos.y + size.y));

    auto clamp8 = [](int v) -> uint8_t { return (uint8_t)std::min(255, std::max(0, v)); };

    // Grain amplitude (in RGB units). Keep subtle.
    int amp = (int)(10.0f * strength + 0.5f); // ~2..10
    if (amp < 1) amp = 1;

    for (int y = iy1; y < iy2; ++y) {
        float py = y + 0.5f;
        for (int x = ix1; x < ix2; ++x) {
            float px = x + 0.5f;

            // Rounded-rect SDF inside test (same as highlight)
            float qx = std::abs(px - cx) - hx;
            float qy = std::abs(py - cy) - hy;
            float dist = std::sqrt(std::max(qx, 0.0f) * std::max(qx, 0.0f) +
                                   std::max(qy, 0.0f) * std::max(qy, 0.0f))
                         + std::min(std::max(qx, qy), 0.0f) - r;
            if (dist > 0.0f)
                continue;

            // Fade grain slightly near edges so border stays clean.
            float edge = std::min(1.0f, -dist);
            float edgeFade = 0.35f + 0.65f * std::min(1.0f, edge * 2.0f);

            // Hash noise from (x,y,seed)
            uint32_t h = (uint32_t)(x * 374761393u) ^ (uint32_t)(y * 668265263u) ^ m_grainSeed;
            h ^= (h >> 13);
            h *= 1274126177u;
            h ^= (h >> 16);

            // Map to [-1,1]
            float n = ((int)(h & 0xFFu) - 128) / 128.0f;
            int d = (int)(n * amp * edgeFade);

            int idx = y * m_width + x;
            uint32_t dst = pixels[idx];
            uint8_t a = (dst >> 24) & 0xFF;
            uint8_t r8 = (dst >> 16) & 0xFF;
            uint8_t g8 = (dst >> 8) & 0xFF;
            uint8_t b8 = dst & 0xFF;

            // Slightly color-shift grain for "refractive" feel (cool bias).
            int dr = d;
            int dg = (int)(d * 0.7f);
            int db = (int)(d * 1.1f);

            pixels[idx] = (uint32_t(a) << 24) |
                          (uint32_t(clamp8((int)r8 + dr)) << 16) |
                          (uint32_t(clamp8((int)g8 + dg)) << 8) |
                          (uint32_t(clamp8((int)b8 + db)));
        }
    }
}

// ---------------------------------------------------------------
// Drawing: text via stb_truetype baked atlas
// ---------------------------------------------------------------

void OverlayRenderer::drawText(const Vec2& pos, const std::string& text, uint32_t color)
{
    // Approximate text bounding box for dirty tracking
    float textW = measureTextWidth(text);
    expandDirtyRect((int)pos.x, (int)pos.y, (int)textW + 2, (int)m_fontSize + 2);
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

    size_t i = 0;
    while (i < text.size()) {
        uint32_t cp = decodeUtf8(text, i);

        if (cp == '\n') {
            cursorX = pos.x;
            cursorY += m_fontSize;
            continue;
        }

        // Look up glyph; fall back to '?' for unknown codepoints
        auto it = m_glyphMap.find(cp);
        if (it == m_glyphMap.end()) {
            it = m_glyphMap.find('?');
            if (it == m_glyphMap.end()) continue;
        }

        const BakedChar& bc = it->second;

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
// Text measurement (mirrors drawText advance logic without pixel writes)
// ---------------------------------------------------------------

float OverlayRenderer::measureTextWidth(const std::string& text) const
{
    if (!m_fontLoaded || text.empty()) return 0.0f;
    float width = 0.0f;
    size_t i = 0;
    while (i < text.size()) {
        uint32_t cp = decodeUtf8(text, i);
        if (cp == '\n') break; // single-line measurement
        auto it = m_glyphMap.find(cp);
        if (it == m_glyphMap.end()) {
            it = m_glyphMap.find('?');
            if (it == m_glyphMap.end()) continue;
        }
        width += it->second.xadvance;
    }
    return width;
}

// ---------------------------------------------------------------
// Drawing: line (Bresenham with thickness)
// ---------------------------------------------------------------

void OverlayRenderer::drawLine(const Vec2& start, const Vec2& end, uint32_t color, float thickness)
{
    // Degenerate (zero-length) lines must be handled BEFORE we acquire
    // m_renderMutex, because the fallback path calls drawRect() which
    // re-locks the same non-recursive mutex \u2014 self-deadlock otherwise.
    // This case fires constantly for stationary entity-tracker trails
    // where consecutive centres are identical.
    {
        const float dx0 = end.x - start.x;
        const float dy0 = end.y - start.y;
        if (dx0 * dx0 + dy0 * dy0 < 0.0001f) {
            drawRect(start, {thickness, thickness}, color);
            return;
        }
    }

    int lx1 = (int)std::min(start.x, end.x) - (int)thickness;
    int ly1 = (int)std::min(start.y, end.y) - (int)thickness;
    int lx2 = (int)std::max(start.x, end.x) + (int)thickness + 1;
    int ly2 = (int)std::max(start.y, end.y) + (int)thickness + 1;
    expandDirtyRect(lx1, ly1, lx2 - lx1, ly2 - ly1);
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
// Blur: 2-pass separable Gaussian (per-pixel weighted) for frosted glass
// ---------------------------------------------------------------

void OverlayRenderer::blurRegion(int x, int y, int w, int h, int radius)
{
    std::lock_guard<std::mutex> lock(m_renderMutex);

    if (!m_pixels || w <= 0 || h <= 0 || radius <= 0)
        return;

    // Even when OS blur is enabled, we still use blurRegion for the
    // captured-desktop refraction path (small radius).

    int x1 = std::max(0, x);
    int y1 = std::max(0, y);
    int x2 = std::min(m_width, x + w);
    int y2 = std::min(m_height, y + h);
    int rw = x2 - x1;
    int rh = y2 - y1;
    if (rw <= 0 || rh <= 0) return;

    size_t needed = static_cast<size_t>(rw) * rh;
    if (m_blurTemp.size() < needed)
        m_blurTemp.resize(needed);

    int diameter = radius * 2 + 1;
    if (m_gaussianKernel.size() != static_cast<size_t>(diameter)) {
        m_gaussianKernel.resize(diameter);
        float sigma = radius / 2.0f;
        if (sigma < 0.5f) sigma = 0.5f;
        float sum = 0;
        for (int k = -radius; k <= radius; ++k) {
            float g = std::exp(-(k * k) / (2.0f * sigma * sigma));
            m_gaussianKernel[k + radius] = g;
            sum += g;
        }
        for (int i = 0; i < diameter; ++i)
            m_gaussianKernel[i] /= sum;
    }

    uint32_t* pixels = static_cast<uint32_t*>(m_pixels);

    // Pass 1: Horizontal Gaussian → temp buffer
    for (int row = 0; row < rh; ++row) {
        int py = y1 + row;
        for (int col = 0; col < rw; ++col) {
            int px = x1 + col;
            float sumA = 0, sumR = 0, sumG = 0, sumB = 0;
            for (int k = -radius; k <= radius; ++k) {
                int sx = std::clamp(px + k, x1, x2 - 1);
                float weight = m_gaussianKernel[k + radius];
                uint32_t c = pixels[py * m_width + sx];
                sumA += weight * ((c >> 24) & 0xFF);
                sumR += weight * ((c >> 16) & 0xFF);
                sumG += weight * ((c >> 8) & 0xFF);
                sumB += weight * (c & 0xFF);
            }
            auto clamp8 = [](float v) { return static_cast<uint32_t>(std::min(255, std::max(0, static_cast<int>(v + 0.5f)))); };
            m_blurTemp[row * rw + col] =
                (clamp8(sumA) << 24) | (clamp8(sumR) << 16) | (clamp8(sumG) << 8) | clamp8(sumB);
        }
    }
    // Pass 2: Vertical Gaussian from temp → pixel buffer
    auto clamp8 = [](float v) { return static_cast<uint32_t>(std::min(255, std::max(0, static_cast<int>(v + 0.5f)))); };
    for (int col = 0; col < rw; ++col) {
        for (int row = 0; row < rh; ++row) {
            float sumA = 0, sumR = 0, sumG = 0, sumB = 0;
            for (int k = -radius; k <= radius; ++k) {
                int sy = std::clamp(row + k, 0, rh - 1);
                float weight = m_gaussianKernel[k + radius];
                uint32_t c = m_blurTemp[sy * rw + col];
                sumA += weight * ((c >> 24) & 0xFF);
                sumR += weight * ((c >> 16) & 0xFF);
                sumG += weight * ((c >> 8) & 0xFF);
                sumB += weight * (c & 0xFF);
            }
            int py = y1 + row;
            int px = x1 + col;
            pixels[py * m_width + px] =
                (clamp8(sumA) << 24) | (clamp8(sumR) << 16) | (clamp8(sumG) << 8) | clamp8(sumB);
        }
    }
}
