#pragma once
#include <windows.h>
#include <vector>
#include <string>
#include <mutex>
#include "helpers/vector2.hpp"

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
};
