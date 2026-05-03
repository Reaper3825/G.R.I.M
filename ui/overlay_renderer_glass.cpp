#include "overlay_renderer.hpp"
#include "core/platform_window.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>

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

static float averageBackdropSampleDelta(const std::vector<uint32_t>& pixels,
                                        size_t count,
                                        const std::vector<uint32_t>& previousSamples)
{
    constexpr size_t kSampleStep = 64;
    if (previousSamples.empty())
        return 255.0f;

    float total = 0.0f;
    size_t sampleIndex = 0;
    for (size_t i = 0; i < count && sampleIndex < previousSamples.size(); i += kSampleStep, ++sampleIndex) {
        uint32_t a = pixels[i];
        uint32_t b = previousSamples[sampleIndex];
        int ar = (a >> 16) & 0xFF;
        int ag = (a >> 8) & 0xFF;
        int ab = a & 0xFF;
        int br = (b >> 16) & 0xFF;
        int bg = (b >> 8) & 0xFF;
        int bb = b & 0xFF;
        total += static_cast<float>(std::abs(ar - br) + std::abs(ag - bg) + std::abs(ab - bb)) / 3.0f;
    }

    if (sampleIndex == 0 || sampleIndex != previousSamples.size())
        return 255.0f;
    return total / static_cast<float>(sampleIndex);
}

static void storeBackdropSamples(const std::vector<uint32_t>& pixels,
                                 size_t count,
                                 std::vector<uint32_t>& samples)
{
    constexpr size_t kSampleStep = 64;
    samples.clear();
    samples.reserve((count + kSampleStep - 1) / kSampleStep);
    for (size_t i = 0; i < count; i += kSampleStep)
        samples.push_back(pixels[i]);
}

void OverlayRenderer::drawGlassPanel(const Vec2& pos, const Vec2& size, float radius,
                                      uint32_t bgColor, uint32_t borderColor, uint32_t glowColor,
                                      int blurRadius, float shadowOffset,
                                      uintptr_t panelId, bool deferGlassRefresh)
{
    int rx = (int)pos.x, ry = (int)pos.y;
    int rw = (int)size.x, rh = (int)size.y;
    expandDirtyRect(rx - 1, ry - 1, rw + (int)shadowOffset + 2, rh + (int)shadowOffset + 2);

    bool haveOverlayHandle = false;
#if defined(__APPLE__)
    haveOverlayHandle = (m_nativeWindow != nullptr);
#elif defined(_WIN32)
    haveOverlayHandle = (m_hwnd != nullptr);
#endif

    const bool useDesktopCapture = haveOverlayHandle && !m_usePlatformBlur;

    if (useDesktopCapture) {
        constexpr uint64_t kGlassRefreshFrames = 6;
        constexpr float kBackdropDeltaThreshold = 10.0f;
        const int capW = std::max(1, rw);
        const int capH = std::max(1, rh);
        const size_t needed = static_cast<size_t>(capW) * static_cast<size_t>(capH);

        bool hasPanelCache = false;
        bool hasCachedPanel = false;
        uint64_t lastRefreshFrame = 0;
        if (panelId != 0) {
            auto it = m_glassCache.find(panelId);
            hasPanelCache = it != m_glassCache.end();
            if (hasPanelCache) {
                lastRefreshFrame = it->second.lastRefreshFrame;
                hasCachedPanel = it->second.rectX == rx && it->second.rectY == ry &&
                                 it->second.rectW == capW && it->second.rectH == capH &&
                                 it->second.pixels.size() >= needed;
            }
        }

        bool shouldRefreshGlass = !hasPanelCache ||
                                  (m_frameIndex - lastRefreshFrame) >= kGlassRefreshFrames;

        if (!shouldRefreshGlass && hasCachedPanel && panelId != 0) {
            auto it = m_glassCache.find(panelId);
            blitCachedToPanel(rx, ry, capW, capH, it->second.pixels, pos, size, radius);
        } else if (deferGlassRefresh) {
            if (hasCachedPanel && panelId != 0) {
                auto it = m_glassCache.find(panelId);
                blitCachedToPanel(rx, ry, capW, capH, it->second.pixels, pos, size, radius);
            }
        } else {
            if (m_captureTemp.size() < needed) m_captureTemp.resize(needed);

            bool ok = PlatformWindow::captureDesktopBehindOverlay(
#if defined(__APPLE__)
                m_nativeWindow,
#else
                (void*)m_hwnd,
#endif
                rx, ry, capW, capH, m_captureTemp.data());

            if (ok && hasCachedPanel && panelId != 0) {
                auto it = m_glassCache.find(panelId);
                float delta = averageBackdropSampleDelta(m_captureTemp, needed, it->second.sourceSamples);
                if (delta < kBackdropDeltaThreshold) {
                    it->second.lastRefreshFrame = m_frameIndex;
                    blitCachedToPanel(rx, ry, capW, capH, it->second.pixels, pos, size, radius);
                    ok = false;
                }
            }

            if (ok) {
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

                    const float distortPx = float(blurRadius) / 10.0f;

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

                int frostRadius = (int)std::round(blurRadius / 2.5f);
                frostRadius = std::clamp(frostRadius, 8, 32);
                blurRegion(rx, ry, capW, capH, frostRadius);
                blurRegion(rx, ry, capW, capH, frostRadius);

                if (panelId != 0) {
                    auto& cache = m_glassCache[panelId];
                    cache.rectX = rx;
                    cache.rectY = ry;
                    cache.rectW = capW;
                    cache.rectH = capH;
                    cache.lastRefreshFrame = m_frameIndex;
                    storeBackdropSamples(m_captureTemp, needed, cache.sourceSamples);
                    cache.pixels.assign(needed, 0);
                    uint32_t* pixels = static_cast<uint32_t*>(m_pixels);
                    int srcX1 = std::max(0, rx);
                    int srcY1 = std::max(0, ry);
                    int srcX2 = std::min(m_width, rx + capW);
                    int srcY2 = std::min(m_height, ry + capH);
                    int copyW = srcX2 - srcX1;
                    int copyH = srcY2 - srcY1;
                    if (copyW > 0 && copyH > 0) {
                        int dstX = srcX1 - rx;
                        int dstY = srcY1 - ry;
                        for (int row = 0; row < copyH; ++row)
                            std::memcpy(cache.pixels.data() + (dstY + row) * capW + dstX,
                                        pixels + (srcY1 + row) * m_width + srcX1,
                                        static_cast<size_t>(copyW) * sizeof(uint32_t));
                    }
                }
            } else if (hasCachedPanel && panelId != 0) {
                auto it = m_glassCache.find(panelId);
                blitCachedToPanel(rx, ry, capW, capH, it->second.pixels, pos, size, radius);
            }
        }
    }

    if (shadowOffset > 0) {
        drawRoundedRect({pos.x + shadowOffset, pos.y + shadowOffset}, size, 0x35000000, radius);
    }

    uint8_t ba = (bgColor >> 24) & 0xFF;
    uint32_t glassTint = (bgColor & 0x00FFFFFF) | ((uint32_t)(ba * 40 / 100) << 24);
    drawRoundedRect(pos, size, glassTint, radius);

    if (useDesktopCapture) {
        applyGlassGrain(pos, size, radius, 0.28f);
    }

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
                float vFade = (1.0f - vt) * (1.0f - vt) * (1.0f - vt);

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

    int amp = (int)(10.0f * strength + 0.5f);
    if (amp < 1) amp = 1;

    for (int y = iy1; y < iy2; ++y) {
        float py = y + 0.5f;
        for (int x = ix1; x < ix2; ++x) {
            float px = x + 0.5f;

            float qx = std::abs(px - cx) - hx;
            float qy = std::abs(py - cy) - hy;
            float dist = std::sqrt(std::max(qx, 0.0f) * std::max(qx, 0.0f) +
                                   std::max(qy, 0.0f) * std::max(qy, 0.0f))
                         + std::min(std::max(qx, qy), 0.0f) - r;
            if (dist > 0.0f)
                continue;

            float edge = std::min(1.0f, -dist);
            float edgeFade = 0.35f + 0.65f * std::min(1.0f, edge * 2.0f);

            uint32_t h = (uint32_t)(x * 374761393u) ^ (uint32_t)(y * 668265263u) ^ m_grainSeed;
            h ^= (h >> 13);
            h *= 1274126177u;
            h ^= (h >> 16);

            float n = ((int)(h & 0xFFu) - 128) / 128.0f;
            int d = (int)(n * amp * edgeFade);

            int idx = y * m_width + x;
            uint32_t dst = pixels[idx];
            uint8_t a = (dst >> 24) & 0xFF;
            uint8_t r8 = (dst >> 16) & 0xFF;
            uint8_t g8 = (dst >> 8) & 0xFF;
            uint8_t b8 = dst & 0xFF;

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
