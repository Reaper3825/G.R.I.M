#include "popup_renderer.hpp"
#include "logger.hpp"
#include "pch.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <vector>
#include <mutex>
#include <string>

// ===========================================================
// CPU-side image cache
// ===========================================================
struct PopupTextureCPU
{
    std::vector<uint8_t> pixels;
    int width  = 0;
    int height = 0;
    bool loaded = false;
};

static std::mutex g_textureMutex;
static PopupTextureCPU g_diffuseCPU;
static PopupTextureCPU g_opacityCPU;

// ===========================================================
// Load image file to RGBA8 memory
// ===========================================================
static bool loadImageRGBA(const std::string& path, PopupTextureCPU& out)
{
    std::lock_guard<std::mutex> lock(g_textureMutex);

    int w = 0, h = 0, c = 0;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &c, 4);
    if (!data)
    {
        LOG_ERROR("PopupRenderer", "Failed to load image: " + path);
        return false;
    }

    out.pixels.assign(data, data + (w * h * 4));
    out.width  = w;
    out.height = h;
    out.loaded = true;

    stbi_image_free(data);

    LOG_DEBUG("PopupRenderer", "Loaded image " + path + " (" +
              std::to_string(w) + "x" + std::to_string(h) + ")");
    return true;
}

// ===========================================================
// Public API (CPU-only data preparation)
// ===========================================================
void loadPopupTextures()
{
    LOG_PHASE( "Loading popup textures (CPU-side only)", true);

    if (!g_diffuseCPU.loaded)
        loadImageRGBA("D:/G.R.I.M/resources/shaders/g_sprite_Diffuse.png", g_diffuseCPU);

    if (!g_opacityCPU.loaded)
        loadImageRGBA("D:/G.R.I.M/resources/shaders/g_sprite_Oreo.png", g_opacityCPU);
}

// ===========================================================
// Accessors
// ===========================================================
bool popupTexturesReady()
{
    std::lock_guard<std::mutex> lock(g_textureMutex);
    return g_diffuseCPU.loaded && g_opacityCPU.loaded;
}

void getPopupTextureData(std::vector<uint8_t>& diffuse, std::vector<uint8_t>& opacity,
                         int& w, int& h)
{
    std::lock_guard<std::mutex> lock(g_textureMutex);
    if (g_diffuseCPU.loaded)
    {
        diffuse = g_diffuseCPU.pixels;
        w = g_diffuseCPU.width;
        h = g_diffuseCPU.height;
    }
    if (g_opacityCPU.loaded)
        opacity = g_opacityCPU.pixels;
}

void unloadPopupTextures()
{
    std::lock_guard<std::mutex> lock(g_textureMutex);
    g_diffuseCPU = {};
    g_opacityCPU = {};
    LOG_DEBUG("PopupRenderer", "Popup textures unloaded (CPU cache cleared)");
}
