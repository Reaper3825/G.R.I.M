#include "popup_3d_material.hpp"
#include <bgfx/bgfx.h>
#include <stb/stb_image.h>
#include <stdexcept>
#include <string>

// ===========================================================
// Popup 3D Material — texture load + sampler binding
// ===========================================================

struct PopupMaterialState
{
    bgfx::TextureHandle albedoTex  = BGFX_INVALID_HANDLE;
    bgfx::TextureHandle packedTex  = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle s_albedo   = BGFX_INVALID_HANDLE;  // sampler slot 0
    bgfx::UniformHandle s_packed   = BGFX_INVALID_HANDLE;  // sampler slot 1
    bool hasPacked = false;
};

static bgfx::TextureHandle loadTexture(const std::string& path)
{
    int w = 0, h = 0, c = 0;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &c, 4);  // force RGBA
    if (!data)
        throw std::runtime_error("popupMaterialCreate: failed to load texture: " + path);

    const bgfx::Memory* mem = bgfx::copy(data, w * h * 4);
    stbi_image_free(data);

    bgfx::TextureHandle tex = bgfx::createTexture2D(
        static_cast<uint16_t>(w),
        static_cast<uint16_t>(h),
        false,   // no mipmaps
        1,       // layers
        bgfx::TextureFormat::RGBA8,
        BGFX_SAMPLER_U_CLAMP | BGFX_SAMPLER_V_CLAMP,
        mem
    );

    if (!bgfx::isValid(tex))
        throw std::runtime_error("popupMaterialCreate: bgfx::createTexture2D failed for: " + path);

    return tex;
}

PopupMaterialState* popupMaterialCreate(const char* albedoPath, const char* packedPath)
{
    if (!albedoPath)
        throw std::runtime_error("popupMaterialCreate: albedoPath is NULL");

    auto* mat = new PopupMaterialState();

    mat->albedoTex = loadTexture(albedoPath);
    mat->s_albedo  = bgfx::createUniform("s_albedo", bgfx::UniformType::Sampler);

    if (packedPath)
    {
        mat->packedTex = loadTexture(packedPath);
        mat->s_packed  = bgfx::createUniform("s_packed", bgfx::UniformType::Sampler);
        mat->hasPacked = true;
    }

    return mat;
}

void popupMaterialDestroy(PopupMaterialState* mat)
{
    if (!mat) return;
    if (bgfx::isValid(mat->albedoTex)) bgfx::destroy(mat->albedoTex);
    if (bgfx::isValid(mat->packedTex)) bgfx::destroy(mat->packedTex);
    if (bgfx::isValid(mat->s_albedo))  bgfx::destroy(mat->s_albedo);
    if (bgfx::isValid(mat->s_packed))  bgfx::destroy(mat->s_packed);
    delete mat;
}

void popupMaterialBind(const PopupMaterialState* mat)
{
    if (!mat)
        throw std::runtime_error("popupMaterialBind: mat is NULL");

    bgfx::setTexture(0, mat->s_albedo, mat->albedoTex);
    if (mat->hasPacked)
        bgfx::setTexture(1, mat->s_packed, mat->packedTex);
}
