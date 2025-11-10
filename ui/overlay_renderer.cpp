#include "overlay_renderer.hpp"
#include "logger.hpp"
#include <algorithm>

void OverlayRenderer::init(HWND hwnd, int width, int height)
{
    LOG_DEBUG("OverlayRenderer", "Initializing with thread safety");
    std::lock_guard<std::mutex> lock(m_renderMutex);  // ? Lock during init
    
    m_hwnd = hwnd;
    m_width = width;
    m_height = height;
    
    // Create screen DC
    m_hdcScreen = GetDC(nullptr);
    
    // Create memory DC
    m_hdcMem = CreateCompatibleDC(m_hdcScreen);
    
    // Create DIB section for alpha channel
    BITMAPINFO bmi{};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = width;
    bmi.bmiHeader.biHeight = -height; // Top-down
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;
    
    m_bitmap = CreateDIBSection(m_hdcMem, &bmi, DIB_RGB_COLORS, &m_pixels, nullptr, 0);
    m_oldBitmap = (HBITMAP)SelectObject(m_hdcMem, m_bitmap);
    
    // Create font
    m_font = CreateFontW(16, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
                        DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                        CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Consolas");
    
    SelectObject(m_hdcMem, m_font);
    SetBkMode(m_hdcMem, TRANSPARENT);
    
    LOG_DEBUG("OverlayRenderer", "Initialized GDI overlay renderer (" + 
             std::to_string(width) + "x" + std::to_string(height) + ")");
}

void OverlayRenderer::shutdown()
{
    std::lock_guard<std::mutex> lock(m_renderMutex);  // ? Lock during shutdown
    
    if (m_font)
    {
        DeleteObject(m_font);
        m_font = nullptr;
    }
    
    if (m_oldBitmap)
    {
        SelectObject(m_hdcMem, m_oldBitmap);
        m_oldBitmap = nullptr;
    }
    
    if (m_bitmap)
    {
        DeleteObject(m_bitmap);
        m_bitmap = nullptr;
    }
    
    if (m_hdcMem)
    {
        DeleteDC(m_hdcMem);
        m_hdcMem = nullptr;
    }
    
    if (m_hdcScreen)
    {
        ReleaseDC(nullptr, m_hdcScreen);
        m_hdcScreen = nullptr;
    }
    
    m_hwnd = nullptr;
    m_pixels = nullptr;
}

void OverlayRenderer::setFont(const std::string& fontName, int fontSize)
{
    std::lock_guard<std::mutex> lock(m_renderMutex);
    
    if (!m_hdcMem) {
        LOG_ERROR("OverlayRenderer", "Cannot set font - renderer not initialized");
        return;
    }
    
    // Delete old font if it exists
    if (m_font) {
        DeleteObject(m_font);
        m_font = nullptr;
    }
    
    // Convert font name to wide string
    std::wstring wFontName;
    wFontName.reserve(fontName.length());
    for (char c : fontName) {
        wFontName.push_back(static_cast<wchar_t>(c));
    }
    
    // Create new font
    m_font = CreateFontW(fontSize, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
                        DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                        CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, wFontName.c_str());
    
    if (m_font) {
        SelectObject(m_hdcMem, m_font);
        LOG_DEBUG("OverlayRenderer", "Font changed to: " + fontName + " (size " + std::to_string(fontSize) + ")");
    } else {
        LOG_ERROR("OverlayRenderer", "Failed to create font: " + fontName);
        // Fallback to default
        m_font = CreateFontW(fontSize, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
                            DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                            CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Consolas");
        SelectObject(m_hdcMem, m_font);
    }
}

void OverlayRenderer::beginFrame()
{
    std::lock_guard<std::mutex> lock(m_renderMutex);  // ? ADD thread safety
    
    if (!m_pixels)
        return;
    
    // Clear to transparent (alpha = 0)
    uint32_t* pixels = static_cast<uint32_t*>(m_pixels);
    int pixelCount = m_width * m_height;
    for (int i = 0; i < pixelCount; ++i)
    {
        pixels[i] = 0x00000000; // Fully transparent
    }
}

void OverlayRenderer::endFrame()
{
    std::lock_guard<std::mutex> lock(m_renderMutex);  // ? ADD thread safety
    
    if (!m_hwnd || !m_hdcMem || !m_hdcScreen)
        return;
    
    // Get window position
    RECT rect;
    GetWindowRect(m_hwnd, &rect);
    
    SIZE wndSize{ m_width, m_height };
    POINT srcPos{ 0, 0 };
    POINT wndPos{ rect.left, rect.top };
    
    BLENDFUNCTION blend{};
    blend.BlendOp = AC_SRC_OVER;
    blend.SourceConstantAlpha = 255;
    blend.AlphaFormat = AC_SRC_ALPHA;
    
    // Update the layered window
    UpdateLayeredWindow(m_hwnd, m_hdcScreen, &wndPos, &wndSize, m_hdcMem,
                       &srcPos, 0, &blend, ULW_ALPHA);
}

void OverlayRenderer::drawRect(const Vec2& pos, const Vec2& size, uint32_t color)
{
    std::lock_guard<std::mutex> lock(m_renderMutex);  // ? ADD thread safety
    
    if (!m_pixels)
        return;
    
    // Extract RGBA components
    uint8_t a = (color >> 24) & 0xFF;
    uint8_t r = (color >> 16) & 0xFF;
    uint8_t g = (color >> 8) & 0xFF;
    uint8_t b = color & 0xFF;
    
    // Pre-multiply alpha
    r = (r * a) / 255;
    g = (g * a) / 255;
    b = (b * a) / 255;
    
    // Draw filled rectangle
    int x1 = std::max(0, (int)pos.x);
    int y1 = std::max(0, (int)pos.y);
    int x2 = std::min(m_width, (int)(pos.x + size.x));
    int y2 = std::min(m_height, (int)(pos.y + size.y));
    
    uint32_t* pixels = static_cast<uint32_t*>(m_pixels);
    
    for (int y = y1; y < y2; ++y)
    {
        for (int x = x1; x < x2; ++x)
        {
            int idx = y * m_width + x;
            pixels[idx] = (a << 24) | (r << 16) | (g << 8) | b;
        }
    }
}

void OverlayRenderer::drawText(const Vec2& pos, const std::string& text, uint32_t color)
{
    std::lock_guard<std::mutex> lock(m_renderMutex);  // ? ADD thread safety
    
    if (!m_hdcMem || !m_pixels)
        return;
    
    if (text.empty())
        return;
    
    // Extract RGBA components
    uint8_t a = (color >> 24) & 0xFF;
    if (a == 0) a = 255; // Default to opaque if alpha is 0
    uint8_t r = (color >> 16) & 0xFF;
    uint8_t g = (color >> 8) & 0xFF;
    uint8_t b = color & 0xFF;
    
    // Calculate text area (approximate)
    int x1 = std::max(0, (int)pos.x);
    int y1 = std::max(0, (int)pos.y);
    int textWidth = (int)(text.length() * 9); // ~9 pixels per char for Consolas 16pt
    int textHeight = 20; // ~20 pixels height
    int x2 = std::min(m_width, x1 + textWidth);
    int y2 = std::min(m_height, y1 + textHeight);
    
    // Set the text color for GDI
    SetTextColor(m_hdcMem, RGB(r, g, b));
    SetBkMode(m_hdcMem, OPAQUE);
    SetBkColor(m_hdcMem, RGB(0x10, 0x10, 0x10)); // Very dark gray, but not pure black
    
    // Convert to wide string and handle encoding properly
    std::wstring wtext;
    wtext.reserve(text.length());
    for (char c : text) {
        // Only convert valid ASCII characters
        if (c >= 32 && c < 127) {
            wtext.push_back(static_cast<wchar_t>(c));
        } else {
            wtext.push_back(L'?'); // Replace invalid chars
        }
    }
    
    // Draw text using GDI - this writes to the DIB section
    TextOutW(m_hdcMem, (int)pos.x, (int)pos.y, wtext.c_str(), (int)wtext.length());
    
    // Force GDI to flush so the pixels are updated
    GdiFlush();
    
    // Fix the alpha channel for the text area
    // GDI writes RGB but sets alpha to 0, so we need to make these pixels visible
    uint32_t* pixels = static_cast<uint32_t*>(m_pixels);
    
    for (int y = y1; y < y2; ++y)
    {
        for (int x = x1; x < x2; ++x)
        {
            int idx = y * m_width + x;
            uint32_t pixel = pixels[idx];
            
            // Extract current RGB (GDI has written these)
            uint8_t pb = pixel & 0xFF;
            uint8_t pg = (pixel >> 8) & 0xFF;
            uint8_t pr = (pixel >> 16) & 0xFF;
            
            // Set alpha to make pixel visible (preserve the RGB that GDI wrote)
            pixels[idx] = (255 << 24) | (pr << 16) | (pg << 8) | pb;
        }
    }
}

void OverlayRenderer::drawLine(const Vec2& start, const Vec2& end, uint32_t color, float thickness)
{
    std::lock_guard<std::mutex> lock(m_renderMutex);
    
    if (!m_pixels)
        return;
    
    // Extract RGBA components
    uint8_t a = (color >> 24) & 0xFF;
    uint8_t r = (color >> 16) & 0xFF;
    uint8_t g = (color >> 8) & 0xFF;
    uint8_t b = color & 0xFF;
    
    // Pre-multiply alpha
    r = (r * a) / 255;
    g = (g * a) / 255;
    b = (b * a) / 255;
    
    uint32_t premultColor = (a << 24) | (r << 16) | (g << 8) | b;
    
    // Bresenham's line algorithm with thickness
    float dx = end.x - start.x;
    float dy = end.y - start.y;
    float len = std::sqrt(dx * dx + dy * dy);
    
    if (len < 0.01f) {
        // Draw a point
        drawRect(start, {thickness, thickness}, color);
        return;
    }
    
    // Draw line as series of thick points with anti-aliasing
    // Use more segments for smoother lines (at least 2 segments per pixel)
    int segments = static_cast<int>(len * 2.0f) + 1;
    uint32_t* pixels = static_cast<uint32_t*>(m_pixels);
    
    for (int i = 0; i <= segments; ++i) {
        float t = static_cast<float>(i) / segments;
        float fx = start.x + dx * t;
        float fy = start.y + dy * t;
        int x = static_cast<int>(fx);
        int y = static_cast<int>(fy);
        
        // Draw thick point with circular brush for smoother appearance
        int halfThick = static_cast<int>(thickness / 2.0f) + 1;
        float radiusSq = (thickness / 2.0f) * (thickness / 2.0f);
        
        for (int py = -halfThick; py <= halfThick; ++py) {
            for (int px = -halfThick; px <= halfThick; ++px) {
                // Use circular brush instead of square
                float distSq = static_cast<float>(px * px + py * py);
                if (distSq <= radiusSq) {
                    int xx = x + px;
                    int yy = y + py;
                    
                    if (xx >= 0 && xx < m_width && yy >= 0 && yy < m_height) {
                        int idx = yy * m_width + xx;
                        pixels[idx] = premultColor;
                    }
                }
            }
        }
    }
}
