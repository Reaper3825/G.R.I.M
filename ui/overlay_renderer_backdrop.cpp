#include "overlay_renderer.hpp"
#include <algorithm>
#include <cmath>

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
        m_backdropDirty = false;
        return;
    }

    if (!m_backdropDirty)
        return;

    std::lock_guard<std::mutex> lock(m_renderMutex);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uint32_t h = (uint32_t)(x * 73856093) ^ (uint32_t)(y * 19349663);
            h ^= (h << 13);
            h ^= (h >> 17);
            h ^= (h << 5);
            float t = static_cast<float>(h & 0xFFFFu) / 65535.0f;

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
