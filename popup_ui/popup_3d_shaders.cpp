#include "popup_3d_shaders.hpp"
#include <bgfx/bgfx.h>
#include <bgfx/embedded_shader.h>
#include <stdexcept>

// ===========================================================
// Popup 3D Shaders — embedded metaballs shaders from bgfx
// Provides: MVP transform + world-space normals + Lambertian
// directional lighting with vertex color.
// ===========================================================

// Pre-compiled shader bytecodes for all renderer backends
// (BSD 2-clause license, Copyright 2011-2026 Branimir Karadzic)
#include "../external/bgfx.cmake/bgfx/examples/02-metaballs/vs_metaballs.bin.h"
#include "../external/bgfx.cmake/bgfx/examples/02-metaballs/fs_metaballs.bin.h"

static const bgfx::EmbeddedShader s_popupEmbeddedShaders[] =
{
    BGFX_EMBEDDED_SHADER(vs_metaballs),
    BGFX_EMBEDDED_SHADER(fs_metaballs),
    BGFX_EMBEDDED_SHADER_END()
};

struct PopupShaderState
{
    bgfx::ProgramHandle program = BGFX_INVALID_HANDLE;
};

PopupShaderState* popupShadersCreate()
{
    bgfx::RendererType::Enum type = bgfx::getRendererType();

    bgfx::ShaderHandle vsh = bgfx::createEmbeddedShader(
        s_popupEmbeddedShaders, type, "vs_metaballs");
    if (!bgfx::isValid(vsh))
        throw std::runtime_error("popupShadersCreate: failed to create vertex shader");

    bgfx::ShaderHandle fsh = bgfx::createEmbeddedShader(
        s_popupEmbeddedShaders, type, "fs_metaballs");
    if (!bgfx::isValid(fsh))
    {
        bgfx::destroy(vsh);
        throw std::runtime_error("popupShadersCreate: failed to create fragment shader");
    }

    auto* state = new PopupShaderState();
    state->program = bgfx::createProgram(vsh, fsh, true);
    if (!bgfx::isValid(state->program))
    {
        delete state;
        throw std::runtime_error("popupShadersCreate: bgfx::createProgram failed");
    }

    return state;
}

void popupShadersDestroy(PopupShaderState* shaders)
{
    if (!shaders) return;
    if (bgfx::isValid(shaders->program))
        bgfx::destroy(shaders->program);
    delete shaders;
}

bgfx::ProgramHandle popupShadersGetProgram(const PopupShaderState* shaders)
{
    if (!shaders)
        throw std::runtime_error("popupShadersGetProgram: shaders is NULL");
    return shaders->program;
}
