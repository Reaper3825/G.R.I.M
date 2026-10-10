#include "overlay_renderer.hpp"
#include "logger.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

#define STB_TRUETYPE_IMPLEMENTATION
#include <stb/stb_truetype.h>

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

uint32_t OverlayRenderer::decodeUtf8(const std::string& text, size_t& pos)
{
    const size_t start = pos;
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
        ++pos;
        return 0xFFFD;
    }

    ++pos;
    if (pos + static_cast<size_t>(extra) > text.size()) {
        pos = text.size();
        return 0xFFFD;
    }

    for (int i = 0; i < extra && pos < text.size(); ++i, ++pos) {
        uint8_t cont = static_cast<uint8_t>(text[pos]);
        if ((cont & 0xC0) != 0x80) {
            pos = start + 1;
            return 0xFFFD;
        }
        cp = (cp << 6) | (cont & 0x3F);
    }
    return cp;
}

void OverlayRenderer::setFont(const std::string& fontPath, int fontSize)
{
    std::lock_guard<std::mutex> lock(m_renderMutex);

    m_fontFileData = readFontFile(fontPath);
    m_sizedGlyphs.clear();
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
    m_sizedGlyphs.clear();
    m_glyphMap.clear();
    m_fontLoaded = false;

    if (m_fontFileData.empty()) return;

    // Validate the font BEFORE handing it to the packer. stbtt_PackFontRanges
    // does only minimal validation and will happily dereference garbage tables
    // on a malformed/unsupported font, which manifests as an access violation
    // (0xC0000005) that takes down the whole process. stbtt_InitFont performs
    // the real table validation, so bail cleanly if it rejects the font.
    {
        const unsigned char* fdata = m_fontFileData.data();
        int numFonts = stbtt_GetNumberOfFonts(fdata);
        int fontOffset = stbtt_GetFontOffsetForIndex(fdata, 0);
        LOG_DEBUG("OverlayRenderer", "rebuildAtlas: fontBytes=" +
                  std::to_string(m_fontFileData.size()) +
                  " numFonts=" + std::to_string(numFonts) +
                  " offset0=" + std::to_string(fontOffset) +
                  " fontSize=" + std::to_string(m_fontSize));
        if (fontOffset < 0) {
            LOG_ERROR("OverlayRenderer", "rebuildAtlas: invalid font offset — font unusable");
            return;
        }
        stbtt_fontinfo probe;
        if (!stbtt_InitFont(&probe, fdata, fontOffset)) {
            LOG_ERROR("OverlayRenderer", "rebuildAtlas: stbtt_InitFont rejected font — skipping atlas build");
            return;
        }
    }

    struct PackRange {
        int firstCodepoint;
        int count;
        bool isIconFont;
    };

    std::vector<PackRange> ranges = {
        { 32,    95,   false },
        { 0x100, 128,  false },
        { 0x2000, 112, false },
    };

    if (!m_iconFontFileData.empty()) {
        ranges.push_back({ 0xE000, 256,  true });
        ranges.push_back({ 0xE200, 256,  true });
        ranges.push_back({ 0xF000, 4096, true });
    }

    int totalChars = 0;
    for (auto& r : ranges) totalChars += r.count;

    for (m_atlasWidth = 512, m_atlasHeight = 512;
         m_atlasWidth <= 4096;
         m_atlasWidth *= 2, m_atlasHeight *= 2)
    {
        m_fontAtlas.resize(m_atlasWidth * m_atlasHeight);
        std::fill(m_fontAtlas.begin(), m_fontAtlas.end(), 0);

        LOG_DEBUG("OverlayRenderer", "rebuildAtlas: trying atlas " +
                  std::to_string(m_atlasWidth) + "x" + std::to_string(m_atlasHeight) +
                  " totalChars=" + std::to_string(totalChars));

        stbtt_pack_context pc;
        if (!stbtt_PackBegin(&pc, m_fontAtlas.data(), m_atlasWidth, m_atlasHeight, 0, 1, nullptr)) {
            LOG_DEBUG("OverlayRenderer", "rebuildAtlas: PackBegin failed at this size, growing");
            continue;
        }
        stbtt_PackSetOversampling(&pc, 1, 1);

        std::vector<stbtt_packedchar> packedChars(totalChars);
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

        bool allOk = true;
        {
            std::vector<stbtt_pack_range> textRanges;
            for (size_t i = 0; i < ranges.size(); ++i) {
                if (!ranges[i].isIconFont) textRanges.push_back(stbRanges[i]);
            }
            if (!textRanges.empty()) {
                LOG_DEBUG("OverlayRenderer", "rebuildAtlas: packing " +
                          std::to_string(textRanges.size()) + " text range(s)");
                int ret = stbtt_PackFontRanges(&pc, m_fontFileData.data(), 0,
                                               textRanges.data(), (int)textRanges.size());
                LOG_DEBUG("OverlayRenderer", "rebuildAtlas: text PackFontRanges ret=" +
                          std::to_string(ret));
                if (ret == 0) allOk = false;
            }
        }

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

        if (!allOk) continue;

        offset = 0;
        for (auto& r : ranges) {
            for (int i = 0; i < r.count; ++i) {
                const stbtt_packedchar& pc2 = packedChars[offset + i];
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

const OverlayRenderer::RasterGlyph& OverlayRenderer::sizedGlyph(uint32_t codepoint, float pixelSize)
{
    auto& glyphs = m_sizedGlyphs[pixelSize];
    const auto cached = glyphs.find(codepoint);
    if (cached != glyphs.end()) return cached->second;

    const bool icon = codepoint >= 0xE000 && codepoint <= 0xF8FF && !m_iconFontFileData.empty();
    const auto& data = icon ? m_iconFontFileData : m_fontFileData;
    stbtt_fontinfo font{};
    const int offset = stbtt_GetFontOffsetForIndex(data.data(), 0);
    if (offset < 0 || !stbtt_InitFont(&font, data.data(), offset))
        throw std::runtime_error("Cannot rasterize glyph from invalid font");
    const float fontScale = stbtt_ScaleForPixelHeight(&font, pixelSize);
    RasterGlyph glyph;
    int advance = 0, bearing = 0;
    stbtt_GetCodepointHMetrics(&font, codepoint, &advance, &bearing);
    glyph.advance = advance * fontScale;
    unsigned char* bitmap = stbtt_GetCodepointBitmap(&font, fontScale, fontScale, codepoint,
        &glyph.width, &glyph.height, &glyph.xoff, &glyph.yoff);
    if (bitmap) {
        try {
            glyph.coverage.assign(bitmap, bitmap + glyph.width * glyph.height);
        } catch (...) {
            stbtt_FreeBitmap(bitmap, nullptr);
            throw;
        }
        stbtt_FreeBitmap(bitmap, nullptr);
    } else if (glyph.width > 0 && glyph.height > 0) {
        throw std::runtime_error("Failed to rasterize font glyph");
    }
    return glyphs.emplace(codepoint, std::move(glyph)).first->second;
}

// Keep drawing, measurement, and wrapping on the same theme-relative scale.
static float typographyScale(float size)
{
    if (!std::isfinite(size) || size <= 0.0f) {
        throw std::invalid_argument("Text size must be finite and > 0");
    }
    static_assert(UITheme::Typography::ReferenceSize > 0.0f,
                  "Theme reference size must be > 0");
    return size / UITheme::Typography::ReferenceSize;
}

void OverlayRenderer::drawText(const Vec2& pos, const std::string& text,
                               uint32_t color, float size)
{
    drawTextScaled(pos, text, color, typographyScale(size));
}

void OverlayRenderer::drawTextScaled(
    const Vec2& pos,
    const std::string& text,
    uint32_t color,
    float scale)
{
    if (!std::isfinite(scale) || scale <= 0.0f) {
        throw std::invalid_argument("OverlayRenderer::drawTextScaled requires finite scale > 0");
    }

    float textW = measureTextWidth(text, UITheme::Typography::ReferenceSize) * scale;
    expandDirtyRect(
        static_cast<int>(pos.x),
        static_cast<int>(pos.y),
        static_cast<int>(std::ceil(textW)) + 2,
        static_cast<int>(std::ceil(m_fontSize * scale)) + 2);
    std::lock_guard<std::mutex> lock(m_renderMutex);

    if (!m_pixels || !m_fontLoaded || text.empty())
        return;

    const float pixelSize = m_fontSize * scale;
    if (!std::isfinite(pixelSize) || pixelSize <= 0.0f)
        throw std::invalid_argument("Rendered font size must be finite and > 0");
    // Bound the number of size variants if a caller animates text size.
    if (scale != 1.0f && m_sizedGlyphs.size() >= 32 && m_sizedGlyphs.find(pixelSize) == m_sizedGlyphs.end())
        m_sizedGlyphs.clear();
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
            cursorY += m_fontSize * scale;
            continue;
        }

        auto it = m_glyphMap.find(cp);
        if (it == m_glyphMap.end()) {
            it = m_glyphMap.find('?');
            if (it == m_glyphMap.end()) continue;
        }

        const BakedChar& bc = it->second;

        const RasterGlyph* raster = scale == 1.0f ? nullptr : &sizedGlyph(it->first, pixelSize);
        const int glyphW = raster ? raster->width : bc.x1 - bc.x0;
        const int glyphH = raster ? raster->height : bc.y1 - bc.y0;
        const float advance = raster ? raster->advance : bc.xadvance;
        if (glyphW <= 0 || glyphH <= 0) {
            cursorX += advance;
            continue;
        }

        const int dstX0 = static_cast<int>(std::floor(cursorX + (raster ? raster->xoff : bc.xoff)));
        const int dstY0 = static_cast<int>(std::floor(cursorY + pixelSize + (raster ? raster->yoff : bc.yoff)));
        for (int gy = 0; gy < glyphH; ++gy) {
            const int dy = dstY0 + gy;
            if (dy < clip.y1 || dy >= clip.y2) continue;
            for (int gx = 0; gx < glyphW; ++gx) {
                const int dx = dstX0 + gx;
                if (dx < clip.x1 || dx >= clip.x2) continue;
                const uint8_t coverage = raster ? raster->coverage[gy * glyphW + gx] :
                    m_fontAtlas[(bc.y0 + gy) * m_atlasWidth + bc.x0 + gx];
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

        cursorX += advance;
    }
}

float OverlayRenderer::measureTextWidth(const std::string& text, float size) const
{
    const float scale = typographyScale(size);
    if (!m_fontLoaded || text.empty()) return 0.0f;
    float width = 0.0f;
    size_t i = 0;
    while (i < text.size()) {
        uint32_t cp = decodeUtf8(text, i);
        if (cp == '\n') break;
        auto it = m_glyphMap.find(cp);
        if (it == m_glyphMap.end()) {
            it = m_glyphMap.find('?');
            if (it == m_glyphMap.end()) continue;
        }
        width += it->second.xadvance;
    }
    return width * scale;
}

std::vector<std::string> OverlayRenderer::wrapText(const std::string& text,
                                                   float maxWidth, float size) const
{
    typographyScale(size); // Validate even when wrapping empty text.
    if (!std::isfinite(maxWidth) || maxWidth <= 0.0f)
        throw std::invalid_argument("OverlayRenderer::wrapText requires finite maxWidth > 0");

    std::vector<std::string> wrappedLines;
    size_t logicalStart = 0;
    while (logicalStart <= text.size()) {
        const size_t newline = text.find('\n', logicalStart);
        const size_t logicalEnd = newline == std::string::npos ? text.size() : newline;
        const std::string logicalLine = text.substr(logicalStart, logicalEnd - logicalStart);

        if (logicalLine.empty()) {
            wrappedLines.push_back("");
        } else {
            size_t lineStart = 0;
            while (lineStart < logicalLine.size()) {
                size_t cursor = lineStart;
                size_t lastSpace = std::string::npos;
                size_t fittingEnd = lineStart;

                while (cursor < logicalLine.size()) {
                    const size_t codepointStart = cursor;
                    decodeUtf8(logicalLine, cursor);
                    if (logicalLine[codepointStart] == ' ')
                        lastSpace = codepointStart;
                    if (measureTextWidth(logicalLine.substr(lineStart, cursor - lineStart), size)
                        > maxWidth) {
                        break;
                    }
                    fittingEnd = cursor;
                }

                if (fittingEnd == lineStart) {
                    decodeUtf8(logicalLine, fittingEnd);
                } else if (cursor < logicalLine.size()
                           && lastSpace != std::string::npos
                           && lastSpace > lineStart) {
                    fittingEnd = lastSpace;
                }

                wrappedLines.push_back(logicalLine.substr(lineStart, fittingEnd - lineStart));
                lineStart = fittingEnd;
                while (lineStart < logicalLine.size() && logicalLine[lineStart] == ' ')
                    ++lineStart;
            }
        }

        if (newline == std::string::npos) break;
        logicalStart = newline + 1;
    }
    return wrappedLines;
}
