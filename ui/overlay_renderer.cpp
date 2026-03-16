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
    
    m_font = CreateFontW(16, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
                        DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                        ANTIALIASED_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Consolas");
    
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
    
    m_font = CreateFontW(fontSize, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
                        DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                        ANTIALIASED_QUALITY, DEFAULT_PITCH | FF_DONTCARE, wFontName.c_str());
    
    if (m_font) {
        SelectObject(m_hdcMem, m_font);
        LOG_DEBUG("OverlayRenderer", "Font changed to: " + fontName + " (size " + std::to_string(fontSize) + ")");
    } else {
        LOG_ERROR("OverlayRenderer", "Failed to create font: " + fontName);
        m_font = CreateFontW(fontSize, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
                            DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                            ANTIALIASED_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Consolas");
        SelectObject(m_hdcMem, m_font);
    }
}

void OverlayRenderer::pushClipRect(const Vec2& pos, const Vec2& size)
{
    ClipRect incoming;
    incoming.x1 = std::max(0, (int)pos.x);
    incoming.y1 = std::max(0, (int)pos.y);
    incoming.x2 = std::min(m_width,  (int)(pos.x + size.x));
    incoming.y2 = std::min(m_height, (int)(pos.y + size.y));

    if (!m_clipStack.empty()) {
        const ClipRect& cur = m_clipStack.back();
        incoming.x1 = std::max(incoming.x1, cur.x1);
        incoming.y1 = std::max(incoming.y1, cur.y1);
        incoming.x2 = std::min(incoming.x2, cur.x2);
        incoming.y2 = std::min(incoming.y2, cur.y2);
    }

    m_clipStack.push_back(incoming);
}

void OverlayRenderer::popClipRect()
{
    if (!m_clipStack.empty())
        m_clipStack.pop_back();
}

ClipRect OverlayRenderer::activeClip() const
{
    if (!m_clipStack.empty())
        return m_clipStack.back();
    return { 0, 0, m_width, m_height };
}

void OverlayRenderer::beginFrame()
{
    std::lock_guard<std::mutex> lock(m_renderMutex);
    
    if (!m_pixels)
        return;
    
    m_clipStack.clear();
    
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
    
    ClipRect clip = activeClip();
    
    int x1 = std::max(clip.x1, (int)pos.x);
    int y1 = std::max(clip.y1, (int)pos.y);
    int x2 = std::min(clip.x2, (int)(pos.x + size.x));
    int y2 = std::min(clip.y2, (int)(pos.y + size.y));
    
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
    std::lock_guard<std::mutex> lock(m_renderMutex);
    
    if (!m_hdcMem || !m_pixels)
        return;
    
    if (text.empty())
        return;
    
    ClipRect clip = activeClip();
    
    // Extract RGBA components
    uint8_t a = (color >> 24) & 0xFF;
    if (a == 0) a = 255; // Default to opaque if alpha is 0
    uint8_t r = (color >> 16) & 0xFF;
    uint8_t g = (color >> 8) & 0xFF;
    uint8_t b = color & 0xFF;
    
    // Calculate text area (approximate)
    int rawX1 = (int)pos.x;
    int rawY1 = (int)pos.y;
    int textWidth = (int)(text.length() * 9); // ~9 pixels per char for Consolas 16pt
    int textHeight = 20; // ~20 pixels height
    int rawX2 = rawX1 + textWidth;
    int rawY2 = rawY1 + textHeight;
    
    // Early-out if text is entirely outside clip rect
    if (rawX2 <= clip.x1 || rawX1 >= clip.x2 ||
        rawY2 <= clip.y1 || rawY1 >= clip.y2)
        return;
    
    // Clamp the alpha-fix region to the clip rect
    int x1 = std::max(clip.x1, rawX1);
    int y1 = std::max(clip.y1, rawY1);
    int x2 = std::min(clip.x2, rawX2);
    int y2 = std::min(clip.y2, rawY2);
    
    // Use a GDI clip region so TextOut doesn't write outside the panel
    HRGN hClipRgn = CreateRectRgn(clip.x1, clip.y1, clip.x2, clip.y2);
    SelectClipRgn(m_hdcMem, hClipRgn);
    
    SetTextColor(m_hdcMem, RGB(r, g, b));
    SetBkMode(m_hdcMem, TRANSPARENT);
    
    // Convert to wide string and handle encoding properly
    std::wstring wtext;
    wtext.reserve(text.length());
    for (char c : text) {
        if (c >= 32 && c < 127) {
            wtext.push_back(static_cast<wchar_t>(c));
        } else {
            wtext.push_back(L'?');
        }
    }
    
    uint32_t* pixels = static_cast<uint32_t*>(m_pixels);
    
    // Snapshot pixels before TextOut so we can detect which ones GDI changed.
    // Only glyph pixels are written in TRANSPARENT mode.
    int snapW = x2 - x1;
    int snapH = y2 - y1;
    std::vector<uint32_t> snapshot(snapW * snapH);
    for (int sy = 0; sy < snapH; ++sy) {
        for (int sx = 0; sx < snapW; ++sx) {
            snapshot[sy * snapW + sx] = pixels[(y1 + sy) * m_width + (x1 + sx)];
        }
    }
    
    TextOutW(m_hdcMem, (int)pos.x, (int)pos.y, wtext.c_str(), (int)wtext.length());
    
    GdiFlush();
    
    SelectClipRgn(m_hdcMem, nullptr);
    DeleteObject(hClipRgn);
    
    // Fix alpha only on pixels that GDI actually changed (the glyph pixels).
    // GDI writes RGB but zeros alpha, so unchanged pixels keep their original value.
    for (int sy = 0; sy < snapH; ++sy)
    {
        for (int sx = 0; sx < snapW; ++sx)
        {
            int idx = (y1 + sy) * m_width + (x1 + sx);
            if (pixels[idx] != snapshot[sy * snapW + sx]) {
                uint32_t pixel = pixels[idx];
                uint8_t pr = (pixel >> 16) & 0xFF;
                uint8_t pg = (pixel >> 8) & 0xFF;
                uint8_t pb = pixel & 0xFF;
                pixels[idx] = (a << 24) | (pr << 16) | (pg << 8) | pb;
            }
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
    
    ClipRect clip = activeClip();
    
    float dx = end.x - start.x;
    float dy = end.y - start.y;
    float len = std::sqrt(dx * dx + dy * dy);
    
    if (len < 0.01f) {
        drawRect(start, {thickness, thickness}, color);
        return;
    }
    
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
