#include "popup_renderer.hpp"
#include "logger.hpp"
#include <bgfx/bgfx.h>
#include <bgfx/platform.h>
#include <fstream>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

// ===========================================================
// Global texture/sampler handles
// ===========================================================
static bgfx::UniformHandle s_texColor   = BGFX_INVALID_HANDLE;
static bgfx::UniformHandle s_texOpacity = BGFX_INVALID_HANDLE;
static bgfx::TextureHandle g_diffuseTex = BGFX_INVALID_HANDLE;
static bgfx::TextureHandle g_opacityTex = BGFX_INVALID_HANDLE;

// ===========================================================
// Helper to load shader binary into bgfx::ShaderHandle
// ===========================================================
static bgfx::ShaderHandle loadShader(const char* path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        LOG_ERROR("PopupRenderer", std::string("Failed to open shader: ") + path);
        return BGFX_INVALID_HANDLE;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size + 1);
    if (!file.read(buffer.data(), size))
    {
        LOG_ERROR("PopupRenderer", std::string("Failed to read shader: ") + path);
        return BGFX_INVALID_HANDLE;
    }

    buffer[size] = '\0';
    const bgfx::Memory* mem = bgfx::copy(buffer.data(), static_cast<uint32_t>(size + 1));
    return bgfx::createShader(mem);
}

// ===========================================================
// Load popup shader program (vertex + fragment)
// ===========================================================
bgfx::ProgramHandle loadPopupProgram()
{
    bgfx::ShaderHandle vsh = loadShader("D:/G.R.I.M/resources/shaders/vs_sprite.bin");
    bgfx::ShaderHandle fsh = loadShader("D:/G.R.I.M/resources/shaders/fs_sprite.bin");

    if (!bgfx::isValid(vsh) || !bgfx::isValid(fsh))
    {
        LOG_ERROR("PopupRenderer", "Shader load failed");
        return BGFX_INVALID_HANDLE;
    }

    bgfx::ProgramHandle prog = bgfx::createProgram(vsh, fsh, true);
    if (bgfx::isValid(prog))
    {
        LOG_PHASE("Popup program loaded", true);
    }
    else
    {
        LOG_ERROR("PopupRenderer", "Failed to create bgfx program");
    }

    return prog;
}

// ===========================================================
// Load diffuse + opacity textures and create samplers
// ===========================================================
void loadPopupTextures()
{
    if (!bgfx::isValid(s_texColor))
        s_texColor = bgfx::createUniform("s_texColor", bgfx::UniformType::Sampler);

    if (!bgfx::isValid(s_texOpacity))
        s_texOpacity = bgfx::createUniform("s_texOpacity", bgfx::UniformType::Sampler);

    // ---- Diffuse ----
    if (!bgfx::isValid(g_diffuseTex))
    {
        int texW = 0, texH = 0, texC = 0;
        unsigned char* data = stbi_load("D:/G.R.I.M/resources/shaders/g_sprite_Diffuse.png",
                                        &texW, &texH, &texC, 4);

        if (!data)
        {
            LOG_ERROR("PopupRenderer", "Failed to load g_sprite_Diffuse.png");
            return;
        }

        const bgfx::Memory* mem = bgfx::copy(data, texW * texH * 4);
        g_diffuseTex = bgfx::createTexture2D(
            (uint16_t)texW,
            (uint16_t)texH,
            false,
            1,
            bgfx::TextureFormat::RGBA8,
            0,
            mem
        );

        stbi_image_free(data);

        if (bgfx::isValid(g_diffuseTex))
        {
            LOG_PHASE("Popup diffuse texture loaded", true);
            LOG_DEBUG("PopupRenderer", "Loaded diffuse " + std::to_string(texW) + "x" + std::to_string(texH));
        }
        else
        {
            LOG_ERROR("PopupRenderer", "Diffuse texture handle invalid after creation");
        }
    }

    // ---- Opacity (Oreo RGBA alpha channel) ----
    if (!bgfx::isValid(g_opacityTex))
    {
        int texW = 0, texH = 0, texC = 0;
        unsigned char* data = stbi_load("D:/G.R.I.M/resources/shaders/g_sprite_Oreo.png",
                                        &texW, &texH, &texC, 4);

        if (!data)
        {
            LOG_ERROR("PopupRenderer", "Failed to load g_sprite_Oreo.png");
            return;
        }

        // Keep the original RGBA data (use baked alpha directly)
        std::vector<uint8_t> rgba(texW * texH * 4);
        for (int i = 0; i < texW * texH * 4; ++i)
            rgba[i] = data[i];

        const bgfx::Memory* mem = bgfx::copy(rgba.data(), (uint32_t)rgba.size());
        g_opacityTex = bgfx::createTexture2D(
            (uint16_t)texW,
            (uint16_t)texH,
            false,
            1,
            bgfx::TextureFormat::RGBA8,
            BGFX_SAMPLER_U_CLAMP | BGFX_SAMPLER_V_CLAMP,
            mem
        );

        stbi_image_free(data);

        if (bgfx::isValid(g_opacityTex))
        {
            LOG_PHASE("Popup opacity texture loaded (true alpha)", true);
            LOG_DEBUG("PopupRenderer", "Loaded opacity RGBA " + std::to_string(texW) + "x" + std::to_string(texH));
        }
        else
        {
            LOG_ERROR("PopupRenderer", "Opacity texture handle invalid after creation");
        }
    }
}

// ===========================================================
// Accessors
// ===========================================================
bgfx::UniformHandle getPopupSamplerColor()   { return s_texColor; }
bgfx::UniformHandle getPopupSamplerOpacity() { return s_texOpacity; }
bgfx::TextureHandle getPopupTextureColor()   { return g_diffuseTex; }
bgfx::TextureHandle getPopupTextureOpacity() { return g_opacityTex; }
