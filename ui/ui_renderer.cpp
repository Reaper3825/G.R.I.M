#include "ui_renderer.hpp"
#include <bx/math.h>
#include <bgfx/embedded_shader.h>


// Embedded shader definitions (you’ll add these tiny precompiled blobs)
extern const bgfx::EmbeddedShader* s_embeddedShaders;

void UIRenderer::init()
{
    // Load the “vs_ui” and “fs_ui” shaders from embedded set
    bgfx::RendererType::Enum type = bgfx::getRendererType();

    bgfx::ShaderHandle vsh = bgfx::createEmbeddedShader(s_embeddedShaders, type, "vs_ui");
    bgfx::ShaderHandle fsh = bgfx::createEmbeddedShader(s_embeddedShaders, type, "fs_ui");

    if (bgfx::isValid(vsh) && bgfx::isValid(fsh))
        m_colorProgram = bgfx::createProgram(vsh, fsh, true);
}

void UIRenderer::shutdown()
{
    if (bgfx::isValid(m_colorProgram))
        bgfx::destroy(m_colorProgram);
}

void UIRenderer::drawRect(const Vec2& pos, const Vec2& size, uint32_t color, uint16_t viewId)
{
    if (!bgfx::isValid(m_colorProgram))
        return;

    struct PosColorVertex { float x, y, z; uint32_t abgr; };
    static const uint16_t indices[6] = { 0, 1, 2, 0, 2, 3 };
    PosColorVertex verts[4] = {
        { pos.x,          pos.y,           0.0f, color },
        { pos.x + size.x, pos.y,           0.0f, color },
        { pos.x + size.x, pos.y + size.y,  0.0f, color },
        { pos.x,          pos.y + size.y,  0.0f, color },
    };

    bgfx::VertexLayout layout;
    layout.begin()
        .add(bgfx::Attrib::Position, 3, bgfx::AttribType::Float)
        .add(bgfx::Attrib::Color0,   4, bgfx::AttribType::Uint8, true)
        .end();

    // Declare the transient buffers
    bgfx::TransientVertexBuffer tvb;
    bgfx::TransientIndexBuffer tib;

    // Allocate transient buffers (API now returns void)
    bgfx::allocTransientVertexBuffer(&tvb, 4, layout);
    bgfx::allocTransientIndexBuffer(&tib, 6);

    // Validate that they were allocated
    if (tvb.data == nullptr || tib.data == nullptr)
        return;

    // Copy vertex and index data into buffers
    memcpy(tvb.data, verts, sizeof(verts));
    memcpy(tib.data, indices, sizeof(indices));

    // Submit the geometry
    bgfx::setVertexBuffer(0, &tvb);
    bgfx::setIndexBuffer(&tib);
    bgfx::setState(BGFX_STATE_WRITE_RGB | BGFX_STATE_WRITE_A);
    bgfx::submit(viewId, m_colorProgram);
}


void UIRenderer::drawText(const Vec2& pos, const std::string& text, uint32_t color)
{
    bgfx::dbgTextPrintf((int)(pos.x / 8.0f), (int)(pos.y / 16.0f), color & 0x0F, "%s", text.c_str());
}
