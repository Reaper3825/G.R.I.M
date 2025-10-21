#pragma once
#include <bgfx/bgfx.h>
#include <string>
#include "helpers/widget.hpp" 


class UIRenderer {
public:
    void init();
    void shutdown();

    // Draws a filled rectangle
    void drawRect(const Vec2& pos, const Vec2& size, uint32_t abgr, uint16_t viewId = 0);

    // Draws text using bgfx debug text (for now)
    void drawText(const Vec2& pos, const std::string& text, uint32_t color = 0xFFFFFFFF);

private:
    bgfx::ProgramHandle m_colorProgram = BGFX_INVALID_HANDLE;
};
