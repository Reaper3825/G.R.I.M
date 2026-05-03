#include "overlay_renderer.hpp"
#include <algorithm>
#include <cmath>

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

void OverlayRenderer::drawRoundedRect(const Vec2& pos, const Vec2& size, uint32_t color, float radius)
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

    int ri = (int)(r + 1.0f);

    for (int y = std::max(clip.y1, iy1 - 1); y < std::min(clip.y2, iy2 + 1); ++y) {
        for (int x = std::max(clip.x1, ix1 - 1); x < std::min(clip.x2, ix2 + 1); ++x) {
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
                if (dist > r + 0.5f)
                    continue;
                if (dist > r - 0.5f)
                    coverage = r + 0.5f - dist;
            }

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

    float th = std::max(1.0f, thickness);
    float halfTh = th * 0.5f;
    float cx = pos.x + size.x * 0.5f;
    float cy = pos.y + size.y * 0.5f;
    float hx = size.x * 0.5f - r;
    float hy = size.y * 0.5f - r;

    int ix1 = (int)pos.x;
    int iy1 = (int)pos.y;
    int ix2 = (int)(pos.x + size.x);
    int iy2 = (int)(pos.y + size.y);

    for (int y = std::max(clip.y1, iy1 - 1); y < std::min(clip.y2, iy2 + 1); ++y) {
        for (int x = std::max(clip.x1, ix1 - 1); x < std::min(clip.x2, ix2 + 1); ++x) {
            float px = x + 0.5f;
            float py = y + 0.5f;
            float qx = std::abs(px - cx) - hx;
            float qy = std::abs(py - cy) - hy;
            float outerDist = std::sqrt(std::max(qx, 0.0f) * std::max(qx, 0.0f) +
                                        std::max(qy, 0.0f) * std::max(qy, 0.0f)) 
                              + std::min(std::max(qx, qy), 0.0f) - r;
            float ringDist = std::abs(outerDist) - halfTh;
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

            if (dist <= 0.0f || dist > spread)
                continue;

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

void OverlayRenderer::drawLine(const Vec2& start, const Vec2& end, uint32_t color, float thickness)
{
    const float dx0 = end.x - start.x;
    const float dy0 = end.y - start.y;
    if (dx0 * dx0 + dy0 * dy0 < 0.0001f) {
        drawRect(start, {thickness, thickness}, color);
        return;
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
