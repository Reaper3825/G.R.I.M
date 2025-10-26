#include "ui_renderer.hpp"
#include "logger.hpp"
#include <bx/math.h>


void UIRenderer::init()
{
    LOG_DEBUG("UIRenderer", "Initializing UI renderer (debug text mode)");
    
    // Enable debug text rendering in bgfx
    bgfx::setDebug(BGFX_DEBUG_TEXT);
    
    // For now, we'll use debug text rendering only
    // Shader-based rendering can be added later
    m_initialized = true;
}

void UIRenderer::shutdown()
{
    if (bgfx::isValid(m_colorProgram))
    {
        bgfx::destroy(m_colorProgram);
        m_colorProgram = BGFX_INVALID_HANDLE;
    }
    m_initialized = false;
}

void UIRenderer::drawRect(const Vec2& pos, const Vec2& size, uint32_t color, uint16_t viewId)
{
    // For now, draw a simple debug rectangle using bgfx debug primitives
    // This is a temporary solution until proper shaders are set up
    
    // Draw border with debug text characters
    int x1 = (int)(pos.x / 8.0f);
    int y1 = (int)(pos.y / 16.0f);
    int x2 = (int)((pos.x + size.x) / 8.0f);
    int y2 = (int)((pos.y + size.y) / 16.0f);
    
    uint8_t attr = 0x1F; // White on blue background
    
    // Top border
    for (int x = x1; x <= x2; ++x)
        bgfx::dbgTextPrintf(x, y1, attr, "=");
    
    // Bottom border  
    for (int x = x1; x <= x2; ++x)
        bgfx::dbgTextPrintf(x, y2, attr, "=");
    
    // Left border
    for (int y = y1; y <= y2; ++y)
        bgfx::dbgTextPrintf(x1, y, attr, "|");
    
    // Right border
    for (int y = y1; y <= y2; ++y)
        bgfx::dbgTextPrintf(x2, y, attr, "|");
}

void UIRenderer::drawText(const Vec2& pos, const std::string& text, uint32_t color)
{
    int x = (int)(pos.x / 8.0f);
    int y = (int)(pos.y / 16.0f);
    
    // Convert RGBA color to bgfx attribute (simplified)
    uint8_t attr = ((color >> 24) & 0xFF) > 128 ? 0x0F : 0x08;
    
    bgfx::dbgTextPrintf(x, y, attr, "%s", text.c_str());
}
