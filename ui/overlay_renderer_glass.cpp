#include "overlay_renderer.hpp"
#include "core/platform_window.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>

void OverlayRenderer::ensureGlassMask(PanelGlassCache& cache, int width, int height, float radius)
{
    const size_t needed = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (cache.maskW == width && cache.maskH == height && cache.maskRadius == radius &&
        cache.roundedMask.size() >= needed && cache.edgeAlpha.size() >= needed)
        return;

    cache.maskW = width;
    cache.maskH = height;
    cache.maskRadius = radius;
    cache.roundedMask.assign(needed, 0);
    cache.edgeAlpha.assign(needed, 0);

    const float maxR = std::min(static_cast<float>(width), static_cast<float>(height)) * 0.5f;
    const float r = std::min(radius, maxR);
    const float cx = static_cast<float>(width) * 0.5f;
    const float cy = static_cast<float>(height) * 0.5f;
    const float hx = static_cast<float>(width) * 0.5f - r;
    const float hy = static_cast<float>(height) * 0.5f - r;

    for (int y = 0; y < height; ++y) {
        const float py = static_cast<float>(y) + 0.5f;
        for (int x = 0; x < width; ++x) {
            const float px = static_cast<float>(x) + 0.5f;
            const float qx = std::abs(px - cx) - hx;
            const float qy = std::abs(py - cy) - hy;
            const float dist = std::sqrt(std::max(qx, 0.0f) * std::max(qx, 0.0f) +
                                         std::max(qy, 0.0f) * std::max(qy, 0.0f))
                             + std::min(std::max(qx, qy), 0.0f) - r;
            if (dist > 0.0f)
                continue;

            const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
            cache.roundedMask[idx] = 1;
            cache.edgeAlpha[idx] = static_cast<uint8_t>(std::round(std::min(1.0f, -dist) * 255.0f));
        }
    }
}

void OverlayRenderer::ensureGlassDistortionOffsets(PanelGlassCache& cache, int width, int height, int blurRadius)
{
    const size_t needed = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (cache.distortionW == width && cache.distortionH == height &&
        cache.distortionBlurRadius == blurRadius && cache.distortedSourceIndex.size() >= needed)
        return;

    cache.distortionW = width;
    cache.distortionH = height;
    cache.distortionBlurRadius = blurRadius;
    cache.distortedSourceIndex.assign(needed, 0);

    const float distortPx = static_cast<float>(blurRadius) / 10.0f;
    constexpr uint32_t kDistortionSeed = 0xA53C9E17u;

    auto hash01 = [](uint32_t v) -> float {
        v ^= (v >> 16);
        v *= 0x7feb352du;
        v ^= (v >> 15);
        v *= 0x846ca68bu;
        v ^= (v >> 16);
        return (v & 0x00FFFFFFu) / 16777215.0f;
    };
    auto gridNoise = [&](int gx, int gy) -> float {
        uint32_t v = static_cast<uint32_t>(gx) * 374761393u ^
                     static_cast<uint32_t>(gy) * 668265263u ^ kDistortionSeed;
        return hash01(v);
    };
    auto smoothstep = [](float t) -> float { return t * t * (3.0f - 2.0f * t); };
    auto sampleNoise = [&](float u, float v) -> float {
        constexpr float scale = 1.0f / 46.0f;
        float xg = u * scale;
        float yg = v * scale;
        int x0 = static_cast<int>(std::floor(xg));
        int y0 = static_cast<int>(std::floor(yg));
        float tx = xg - static_cast<float>(x0);
        float ty = yg - static_cast<float>(y0);
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

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float u = static_cast<float>(x);
            const float v = static_cast<float>(y);
            constexpr float eps = 1.35f;
            const float nL = sampleNoise(u - eps, v);
            const float nR = sampleNoise(u + eps, v);
            const float nU = sampleNoise(u, v - eps);
            const float nD = sampleNoise(u, v + eps);
            const float gx = nR - nL;
            const float gy = nD - nU;

            const float ox = std::max(-1.0f, std::min(1.0f, gx * 2.0f)) * distortPx;
            const float oy = std::max(-1.0f, std::min(1.0f, gy * 2.0f)) * distortPx;

            const int sx = std::clamp(static_cast<int>(static_cast<float>(x) + ox), 0, width - 1);
            const int sy = std::clamp(static_cast<int>(static_cast<float>(y) + oy), 0, height - 1);
            cache.distortedSourceIndex[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)] =
                sy * width + sx;
        }
    }
}

void OverlayRenderer::blitCachedToPanel(int rx, int ry, int capW, int capH,
                                        const std::vector<uint32_t>& cachedPixels,
                                        const PanelGlassCache& shapeCache)
{
    if (!m_pixels || cachedPixels.size() < static_cast<size_t>(capW * capH))
        return;

    std::lock_guard<std::mutex> lock(m_renderMutex);
    uint32_t* pixels = static_cast<uint32_t*>(m_pixels);
    ClipRect clip = activeClip();

    int ix1 = std::max(clip.x1, rx);
    int iy1 = std::max(clip.y1, ry);
    int ix2 = std::min(clip.x2, rx + capW);
    int iy2 = std::min(clip.y2, ry + capH);

    for (int y = iy1; y < iy2; ++y) {
        int sy0 = y - ry;
        for (int x = ix1; x < ix2; ++x) {
            int sx = x - rx;
            int sy = sy0;
            const size_t maskIndex = static_cast<size_t>(sy) * static_cast<size_t>(capW) + static_cast<size_t>(sx);
            if (shapeCache.roundedMask[maskIndex] != 0)
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
    PanelGlassCache* activeGlassCache = nullptr;
    if (!useDesktopCapture && rw > 0 && rh > 0) {
        ensureGlassMask(m_transientGlassCache, std::max(1, rw), std::max(1, rh), radius);
        activeGlassCache = &m_transientGlassCache;
    }

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

        PanelGlassCache& shapeCache = (panelId != 0) ? m_glassCache[panelId] : m_transientGlassCache;
        ensureGlassMask(shapeCache, capW, capH, radius);
        ensureGlassDistortionOffsets(shapeCache, capW, capH, blurRadius);
        activeGlassCache = &shapeCache;

        if (!shouldRefreshGlass && hasCachedPanel && panelId != 0) {
            auto it = m_glassCache.find(panelId);
            blitCachedToPanel(rx, ry, capW, capH, it->second.pixels, it->second);
        } else if (deferGlassRefresh) {
            if (hasCachedPanel && panelId != 0) {
                auto it = m_glassCache.find(panelId);
                blitCachedToPanel(rx, ry, capW, capH, it->second.pixels, it->second);
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
                    blitCachedToPanel(rx, ry, capW, capH, it->second.pixels, it->second);
                    ok = false;
                }
            }

            if (ok) {
                {
                    std::lock_guard<std::mutex> lock(m_renderMutex);
                    uint32_t* pixels = static_cast<uint32_t*>(m_pixels);
                    ClipRect clip = activeClip();

                    int ix1 = std::max(clip.x1, rx);
                    int iy1 = std::max(clip.y1, ry);
                    int ix2 = std::min(clip.x2, rx + capW);
                    int iy2 = std::min(clip.y2, ry + capH);

                    for (int y = iy1; y < iy2; ++y) {
                        int sy0 = y - ry;
                        for (int x = ix1; x < ix2; ++x) {
                            const int lx = x - rx;
                            const int ly = sy0;
                            const size_t localIndex = static_cast<size_t>(ly) * static_cast<size_t>(capW) + static_cast<size_t>(lx);
                            if (shapeCache.roundedMask[localIndex] == 0)
                                continue;

                            pixels[y * m_width + x] = m_captureTemp[static_cast<size_t>(shapeCache.distortedSourceIndex[localIndex])];
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
                blitCachedToPanel(rx, ry, capW, capH, it->second.pixels, it->second);
            }
        }
    }

    if (shadowOffset > 0) {
        drawRoundedRect({pos.x + shadowOffset, pos.y + shadowOffset}, size, 0x35000000, radius);
    }

    uint8_t ba = (bgColor >> 24) & 0xFF;
    uint32_t glassTint = (bgColor & 0x00FFFFFF) | ((uint32_t)(ba * 40 / 100) << 24);
    drawRoundedRect(pos, size, glassTint, radius);

    if (useDesktopCapture && activeGlassCache) {
        applyGlassGrain(pos, size, *activeGlassCache, 0.28f);
    }

    if (glowColor != 0) {
        std::lock_guard<std::mutex> lock(m_renderMutex);
        if (m_pixels && activeGlassCache) {
            uint8_t ga = (glowColor >> 24) & 0xFF;
            uint8_t gr = (glowColor >> 16) & 0xFF;
            uint8_t gg = (glowColor >> 8) & 0xFF;
            uint8_t gb = glowColor & 0xFF;

            float gradientHeight = size.y * 0.25f;

            ClipRect clip = activeClip();
            uint32_t* pixels = static_cast<uint32_t*>(m_pixels);

            const int gx = (int)pos.x;
            const int gy = (int)pos.y;
            int ix1 = std::max(clip.x1, gx);
            int iy1 = std::max(clip.y1, gy);
            int ix2 = std::min(clip.x2, gx + activeGlassCache->maskW);
            int iy2 = std::min(clip.y2, std::min(gy + activeGlassCache->maskH, (int)(pos.y + gradientHeight)));

            for (int y = iy1; y < iy2; ++y) {
                float vt = ((float)y + 0.5f - pos.y) / gradientHeight;
                float vFade = (1.0f - vt) * (1.0f - vt) * (1.0f - vt);

                for (int x = ix1; x < ix2; ++x) {
                    const int lx = x - gx;
                    const int ly = y - gy;
                    const size_t localIndex = static_cast<size_t>(ly) * static_cast<size_t>(activeGlassCache->maskW) + static_cast<size_t>(lx);
                    if (activeGlassCache->roundedMask[localIndex] == 0)
                        continue;
                    float edgeFade = static_cast<float>(activeGlassCache->edgeAlpha[localIndex]) / 255.0f;
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

void OverlayRenderer::applyGlassGrain(const Vec2& pos, const Vec2& size,
                                      const PanelGlassCache& shapeCache, float strength)
{
    std::lock_guard<std::mutex> lock(m_renderMutex);
    if (!m_pixels || strength <= 0.0f || size.x <= 0 || size.y <= 0)
        return;

    strength = std::min(1.0f, std::max(0.0f, strength));

    ClipRect clip = activeClip();
    uint32_t* pixels = static_cast<uint32_t*>(m_pixels);

    const int gx = (int)pos.x;
    const int gy = (int)pos.y;
    int ix1 = std::max(clip.x1, gx);
    int iy1 = std::max(clip.y1, gy);
    int ix2 = std::min(clip.x2, gx + shapeCache.maskW);
    int iy2 = std::min(clip.y2, gy + shapeCache.maskH);

    auto clamp8 = [](int v) -> uint8_t { return (uint8_t)std::min(255, std::max(0, v)); };

    int amp = (int)(10.0f * strength + 0.5f);
    if (amp < 1) amp = 1;

    for (int y = iy1; y < iy2; ++y) {
        for (int x = ix1; x < ix2; ++x) {
            const int lx = x - gx;
            const int ly = y - gy;
            const size_t localIndex = static_cast<size_t>(ly) * static_cast<size_t>(shapeCache.maskW) + static_cast<size_t>(lx);
            if (shapeCache.roundedMask[localIndex] == 0)
                continue;

            float edge = static_cast<float>(shapeCache.edgeAlpha[localIndex]) / 255.0f;
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
