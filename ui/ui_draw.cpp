#include "ui_draw.hpp"
#include "console_history.hpp"
#include "commands/commands_core.hpp"
#include "logger.hpp"
#include <bgfx/bgfx.h>
#include <bx/math.h>
#include <algorithm>
#include <thread>
#include <chrono>

// ============================================================
// Helper: Draw a solid rectangle using BGFX transient buffers
// ============================================================
static void drawQuad(float x, float y, float w, float h, uint32_t color, uint16_t viewId)
{
    struct PosColorVertex { float x, y, z; uint32_t abgr; };
    static const uint16_t indices[6] = { 0,1,2,0,2,3 };
    PosColorVertex verts[4] = {
        { x,     y,     0.0f, color },
        { x + w, y,     0.0f, color },
        { x + w, y + h, 0.0f, color },
        { x,     y + h, 0.0f, color },
    };

    bgfx::VertexLayout layout;
    layout.begin()
        .add(bgfx::Attrib::Position, 3, bgfx::AttribType::Float)
        .add(bgfx::Attrib::Color0, 4, bgfx::AttribType::Uint8, true)
        .end();

    bgfx::TransientVertexBuffer tvb;
    bgfx::TransientIndexBuffer tib;
    bgfx::allocTransientVertexBuffer(&tvb, 4, layout);
    bgfx::allocTransientIndexBuffer(&tib, 6);


    if (!tvb.data || !tib.data)
        return;

    memcpy(tvb.data, verts, sizeof(verts));
    memcpy(tib.data, indices, sizeof(indices));

    static bgfx::ProgramHandle colorProgram = BGFX_INVALID_HANDLE;
    if (!bgfx::isValid(colorProgram))
    {
        const char* vs =
            "#version 330 core\n"
            "layout(location=0) in vec3 a_position;"
            "layout(location=1) in vec4 a_color0;"
            "out vec4 v_color;"
            "void main(){gl_Position=vec4(a_position.xy,0.0,1.0);v_color=a_color0;}";
        const char* fs =
            "#version 330 core\n"
            "in vec4 v_color;out vec4 fragColor;"
            "void main(){fragColor=v_color;}";
        const bgfx::Memory* vsmem = bgfx::copy(vs, (uint32_t)strlen(vs) + 1);
        const bgfx::Memory* fsmem = bgfx::copy(fs, (uint32_t)strlen(fs) + 1);
        bgfx::ShaderHandle vsh = bgfx::createShader(vsmem);
        bgfx::ShaderHandle fsh = bgfx::createShader(fsmem);
        colorProgram = bgfx::createProgram(vsh, fsh, true);
    }

    bgfx::setVertexBuffer(0, &tvb);
    bgfx::setIndexBuffer(&tib);
    bgfx::setState(BGFX_STATE_WRITE_RGB | BGFX_STATE_WRITE_A);
    bgfx::submit(viewId, colorProgram);
}

// ============================================================
// Draw UI (BGFX version)
// ============================================================
void drawUI(const GRIMConsole::ConsoleState& state,
            ConsoleHistory& history,
            uint32_t width,
            uint32_t height)
{
    float titleH = kTitleBarH;
    float inputH = kInputBarH;
    float bodyH  = height - titleH - inputH;

    // Panels
    drawQuad(0, 0, (float)width, titleH, 0xFF202020, 0);
    drawQuad(0, height - inputH, (float)width, inputH, 0xFF1E1E1E, 0);
    drawQuad(0, titleH, (float)width, bodyH, 0xFF181818, 0);

    // Text
    bgfx::dbgTextClear();
    bgfx::dbgTextPrintf(1, 1, 0x0F, "G R I M");

    std::string input = state.inputBuffer;
    if (state.caretVisible) input.push_back('|');
    bgfx::dbgTextPrintf(1, (int)((height / 16) - 2), 0x0F, "> %s", input.c_str());

    auto& lines = history.wrapped();
    int maxLines = (int)((height / 16) - 6);
    int start = std::max(0, (int)lines.size() - maxLines);
    int y = 3;
    for (int i = start; i < (int)lines.size(); ++i)
        bgfx::dbgTextPrintf(1, y++, 0x07, "%s", lines[i].text.c_str());
}
