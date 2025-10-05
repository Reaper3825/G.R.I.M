#include "popup_renderer.hpp"
#include "logger.hpp"
#include <bgfx/bgfx.h>
#include <bgfx/platform.h>
#include <fstream>
#include <vector>
#include <stb_image.h>   // Add this to load PNGs
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

// Globals for popup resources
static bgfx::UniformHandle s_texColor = BGFX_INVALID_HANDLE;
static bgfx::TextureHandle g_diffuseTex = BGFX_INVALID_HANDLE;

// Helper to load a .bin shader file
static bgfx::ShaderHandle loadShader(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LOG_ERROR("PopupRenderer", std::string("Failed to open shader: ") + path);
        return BGFX_INVALID_HANDLE;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size + 1);
    if (!file.read(buffer.data(), size)) {
        LOG_ERROR("PopupRenderer", std::string("Failed to read shader: ") + path);
        return BGFX_INVALID_HANDLE;
    }

    buffer[size] = '\0'; // null terminate
    const bgfx::Memory* mem = bgfx::copy(buffer.data(), static_cast<uint32_t>(size + 1));
    return bgfx::createShader(mem);
}

// Public: load popup program
bgfx::ProgramHandle loadPopupProgram() {
    bgfx::ShaderHandle vsh = loadShader("D:/G.R.I.M/resources/shaders/vs_sprite.bin");
    bgfx::ShaderHandle fsh = loadShader("D:/G.R.I.M/resources/shaders/fs_sprite.bin");

    if (!bgfx::isValid(vsh) || !bgfx::isValid(fsh)) {
        LOG_ERROR("PopupRenderer", "Shader load failed");
        return BGFX_INVALID_HANDLE;
    }

    bgfx::ProgramHandle prog = bgfx::createProgram(vsh, fsh, true);
    if (!bgfx::isValid(prog)) {
        LOG_ERROR("PopupRenderer", "Failed to create bgfx program");
    } else {
        LOG_PHASE("Popup program loaded", true);
    }
    return prog;
}

// Public: load diffuse texture + uniform
void loadPopupTexture() {
    if (!bgfx::isValid(s_texColor)) {
        s_texColor = bgfx::createUniform("s_texColor", bgfx::UniformType::Sampler);
    }

    if (!bgfx::isValid(g_diffuseTex)) {
        int texW, texH, texC;
        unsigned char* data = stbi_load("D:/G.R.I.M/resources/shaders/g_sprite_Diffuse.png",
                                        &texW, &texH, &texC, 4);
        if (!data) {
            LOG_ERROR("PopupRenderer", "Failed to load g_sprite_Diffuse.png");
            return;
        }

        g_diffuseTex = bgfx::createTexture2D(
            (uint16_t)texW,
            (uint16_t)texH,
            false,
            1,
            bgfx::TextureFormat::RGBA8,
            0,
            bgfx::copy(data, texW * texH * 4)
        );

        stbi_image_free(data);
        LOG_PHASE("Popup diffuse texture loaded", true);
    }
}

// Accessors
bgfx::UniformHandle getPopupSampler() { return s_texColor; }
bgfx::TextureHandle getPopupTexture() { return g_diffuseTex; }
