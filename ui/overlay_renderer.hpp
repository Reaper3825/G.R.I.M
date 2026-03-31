#pragma once

#include "core/grim_platform.h"

#include <vector>
#include <string>
#include <mutex>
#include <cstdint>
#include <unordered_map>
#include "helpers/vector2.hpp"

struct ClipRect {
    int x1, y1, x2, y2;
};

class OverlayRenderer
{
public:
    void init(HWND hwnd, int width, int height);
    void init(int width, int height, void* pixelBuffer = nullptr);
    void shutdown();
    
    void beginFrame();
    void endFrame();

    // Draw scene behind panels so blur has content to distort (frosted glass).
    void drawBackdrop(int width, int height);

    void drawRect(const Vec2& pos, const Vec2& size, uint32_t color);
    void drawRoundedRect(const Vec2& pos, const Vec2& size, uint32_t color, float radius);
    void drawRoundedBorder(const Vec2& pos, const Vec2& size, uint32_t color, float radius, float thickness = 1.0f);
    void drawGlassPanel(const Vec2& pos, const Vec2& size, float radius,
                        uint32_t bgColor, uint32_t borderColor, uint32_t glowColor,
                        int blurRadius = 100, float shadowOffset = 4.0f,
                        uintptr_t panelId = 0);
    void drawSoftGlow(const Vec2& pos, const Vec2& size, float radius,
                      uint32_t color, float spread);
    void drawText(const Vec2& pos, const std::string& text, uint32_t color);
    float measureTextWidth(const std::string& text) const;
    void drawLine(const Vec2& start, const Vec2& end, uint32_t color, float thickness = 1.0f);
    
    // Load a TTF/OTF font from a file path. fontSize is in pixels.
    void setFont(const std::string& fontPath, int fontSize = 16);
    
    void pushClipRect(const Vec2& pos, const Vec2& size);
    void popClipRect();
    
    // Frosted glass blur: 2-pass separable box blur on the pixel buffer region.
    void blurRegion(int x, int y, int w, int h, int radius = 6);

    // Copy from backdrop buffer into main buffer (panel rect); then blur so blobs never show raw.
    void copyBackdropRegion(int x, int y, int w, int h);

    int getWidth() const { return m_width; }
    int getHeight() const { return m_height; }
    void* getPixels() const { return m_pixels; }
    
private:
#ifdef _WIN32
    HWND m_hwnd = nullptr;
    HDC m_hdcScreen = nullptr;
    HDC m_hdcMem = nullptr;
    HBITMAP m_bitmap = nullptr;
    HBITMAP m_oldBitmap = nullptr;
#elif defined(__APPLE__)
    void* m_nativeWindow = nullptr;
#endif

    int m_width = 0;
    int m_height = 0;
    void* m_pixels = nullptr;
    bool m_ownsPixels = false;

    // When enabled, we rely on OS-level blur-backdrop (vibrancy / DWM acrylic)
    // and skip the synthetic "blur the noise we drew" pipeline.
    bool m_usePlatformBlur = false;

    // Animated grain seed (changes per frame) for glass noise.
    uint32_t m_grainSeed = 0xA53C9E17u;

    // Dirty region tracking: only memset/blit the area that was drawn to
    int m_dirtyX1 = 0, m_dirtyY1 = 0, m_dirtyX2 = 0, m_dirtyY2 = 0;
    int m_prevDirtyX1 = 0, m_prevDirtyY1 = 0, m_prevDirtyX2 = 0, m_prevDirtyY2 = 0;
    void expandDirtyRect(int x, int y, int w, int h);

    std::mutex m_renderMutex;
    std::vector<ClipRect> m_clipStack;
    ClipRect activeClip() const;

    // Backdrop buffer: scene to blur (never shown raw; only composited into main in panel rects)
    uint32_t* m_backdropPixels = nullptr;
    bool m_backdropDirty = true;

    // Temp buffer for blur passes (lazily allocated)
    std::vector<uint32_t> m_blurTemp;
    // Separate buffer for desktop capture (not shared with blur)
    std::vector<uint32_t> m_captureTemp;
    // Gaussian kernel for per-pixel blur (1D, half window)
    std::vector<float> m_gaussianKernel;

    // Dirty detection: cache distorted+blurred result per panel; only update when content changes
    struct PanelGlassCache {
        uint64_t contentHash = 0;
        int rectX = 0, rectY = 0, rectW = 0, rectH = 0;
        std::vector<uint32_t> pixels;
    };
    std::unordered_map<uintptr_t, PanelGlassCache> m_glassCache;

    void blitCachedToPanel(int rx, int ry, int capW, int capH,
                          const std::vector<uint32_t>& cachedPixels,
                          const Vec2& pos, const Vec2& size, float radius);

    void drawRoundedRectToBuffer(uint32_t* buf, int bufW, int bufH,
                                  const Vec2& pos, const Vec2& size, uint32_t color, float radius);

    // Subtle grain/noise overlay for glass (visual refraction cue).
    void applyGlassGrain(const Vec2& pos, const Vec2& size, float radius, float strength);

    // stb_truetype font state
    std::vector<uint8_t> m_fontFileData;
    std::vector<uint8_t> m_fontAtlas;
    int m_atlasWidth = 0;
    int m_atlasHeight = 0;
    float m_fontSize = 16.0f;
    bool m_fontLoaded = false;

    // Baked character data for ASCII 32..126
    static constexpr int kFirstChar = 32;
    static constexpr int kCharCount = 95;
    struct BakedChar {
        unsigned short x0, y0, x1, y1;
        float xoff, yoff, xadvance;
    };
    BakedChar m_bakedChars[kCharCount] = {};
};
