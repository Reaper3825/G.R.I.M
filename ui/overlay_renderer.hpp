#pragma once

#include "core/grim_platform.h"

#include <vector>
#include <string>
#include <mutex>
#include <cstdint>
#include "helpers/vector2.hpp"

struct ClipRect {
    int x1, y1, x2, y2;
};

class OverlayRenderer
{
public:
#ifdef _WIN32
    void init(HWND hwnd, int width, int height);
#endif
    void init(int width, int height, void* pixelBuffer = nullptr);
    void shutdown();
    
    void beginFrame();
    void endFrame();
    
    void drawRect(const Vec2& pos, const Vec2& size, uint32_t color);
    void drawRoundedRect(const Vec2& pos, const Vec2& size, uint32_t color, float radius);
    void drawRoundedBorder(const Vec2& pos, const Vec2& size, uint32_t color, float radius, float thickness = 1.0f);
    void drawGlassPanel(const Vec2& pos, const Vec2& size, float radius, 
                        uint32_t bgColor, uint32_t borderColor, uint32_t glowColor,
                        int blurRadius = 12, float shadowOffset = 4.0f);
    void drawSoftGlow(const Vec2& pos, const Vec2& size, float radius,
                      uint32_t color, float spread);
    void drawText(const Vec2& pos, const std::string& text, uint32_t color);
    void drawLine(const Vec2& start, const Vec2& end, uint32_t color, float thickness = 1.0f);
    
    // Load a TTF/OTF font from a file path. fontSize is in pixels.
    void setFont(const std::string& fontPath, int fontSize = 16);
    
    void pushClipRect(const Vec2& pos, const Vec2& size);
    void popClipRect();
    
    // Frosted glass blur: 2-pass separable box blur on the pixel buffer region.
    // Call BEFORE drawing the panel background to blur content behind it.
    void blurRegion(int x, int y, int w, int h, int radius = 6);
    
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
#endif

    int m_width = 0;
    int m_height = 0;
    void* m_pixels = nullptr;
    bool m_ownsPixels = false;

    std::mutex m_renderMutex;
    std::vector<ClipRect> m_clipStack;
    ClipRect activeClip() const;

    // Temp buffer for blur passes (lazily allocated)
    std::vector<uint32_t> m_blurTemp;

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
