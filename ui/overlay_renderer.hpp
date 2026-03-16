#pragma once
#include <windows.h>
#include <vector>
#include <string>
#include <mutex>
#include "helpers/vector2.hpp"

struct ClipRect {
    int x1, y1, x2, y2;
};

// Overlay renderer that uses GDI to draw to a layered window
// Similar to popup_window but for UI panels
class OverlayRenderer
{
public:
    void init(HWND hwnd, int width, int height);
    void shutdown();
    
    // Begin a new frame
    void beginFrame();
    
    // End frame and update the layered window
    void endFrame();
    
    // Drawing primitives
    void drawRect(const Vec2& pos, const Vec2& size, uint32_t color);
    void drawText(const Vec2& pos, const std::string& text, uint32_t color);
    void drawLine(const Vec2& start, const Vec2& end, uint32_t color, float thickness = 1.0f);
    
    // Font management
    void setFont(const std::string& fontName, int fontSize = 16);
    
    // Clip rect stack -- all draw calls are clamped to the active clip rect.
    // Nested pushes intersect with the current clip, so children can only
    // shrink the visible area, never expand it.
    void pushClipRect(const Vec2& pos, const Vec2& size);
    void popClipRect();
    
private:
    HWND m_hwnd = nullptr;
    int m_width = 0;
    int m_height = 0;
    
    HDC m_hdcScreen = nullptr;
    HDC m_hdcMem = nullptr;
    HBITMAP m_bitmap = nullptr;
    HBITMAP m_oldBitmap = nullptr;
    void* m_pixels = nullptr;
    
    HFONT m_font = nullptr;

    std::mutex m_renderMutex;
    
    std::vector<ClipRect> m_clipStack;
    
    // Returns the active clip rect (intersection of stack, or framebuffer bounds)
    ClipRect activeClip() const;
};
