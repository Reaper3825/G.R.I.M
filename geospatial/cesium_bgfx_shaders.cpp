#include "cesium_bgfx_shaders.hpp"

#include <bgfx/embedded_shader.h>
#include <stdexcept>

#if defined(__has_include)
#  if __has_include("vs_cesium_terrain.bin.h") && __has_include("fs_cesium_terrain.bin.h")
#    include "vs_cesium_terrain.bin.h"
#    include "fs_cesium_terrain.bin.h"
#    define GRIM_HAS_CESIUM_EMBEDDED_SHADERS 1
#  else
#    define GRIM_HAS_CESIUM_EMBEDDED_SHADERS 0
#  endif
#else
#  define GRIM_HAS_CESIUM_EMBEDDED_SHADERS 0
#endif

namespace GRIM::GeoSpatial {

#if GRIM_HAS_CESIUM_EMBEDDED_SHADERS
namespace {
    const bgfx::EmbeddedShader kCesiumEmbeddedShaders[] = {
        BGFX_EMBEDDED_SHADER(vs_cesium_terrain),
        BGFX_EMBEDDED_SHADER(fs_cesium_terrain),
        BGFX_EMBEDDED_SHADER_END()
    };
}
#endif

struct CesiumShaderState {
    bgfx::ProgramHandle program = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle u_lightDir = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle u_lightParams = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle u_baseColorFactor = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle u_overlayTransform = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle u_cameraPos = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle u_colorParams = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle u_surfaceParams = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle s_imagery = BGFX_INVALID_HANDLE;
};

CesiumShaderState* cesiumShadersCreate()
{
#if !GRIM_HAS_CESIUM_EMBEDDED_SHADERS
    throw std::runtime_error(
        "cesiumShadersCreate: generated shader headers are missing. Build bgfx shaderc and re-run CMake so vs_cesium_terrain.bin.h/fs_cesium_terrain.bin.h are generated.");
#else
    const bgfx::RendererType::Enum rendererType = bgfx::getRendererType();

    bgfx::ShaderHandle vertexShader = bgfx::createEmbeddedShader(
        kCesiumEmbeddedShaders, rendererType, "vs_cesium_terrain");
    if (!bgfx::isValid(vertexShader))
        throw std::runtime_error("cesiumShadersCreate: failed to create vertex shader (vs_cesium_terrain)");

    bgfx::ShaderHandle fragmentShader = bgfx::createEmbeddedShader(
        kCesiumEmbeddedShaders, rendererType, "fs_cesium_terrain");
    if (!bgfx::isValid(fragmentShader)) {
        bgfx::destroy(vertexShader);
        throw std::runtime_error("cesiumShadersCreate: failed to create fragment shader (fs_cesium_terrain)");
    }

    auto* state = new CesiumShaderState();
    state->program = bgfx::createProgram(vertexShader, fragmentShader, true);
    if (!bgfx::isValid(state->program)) {
        delete state;
        throw std::runtime_error("cesiumShadersCreate: bgfx::createProgram failed");
    }

    state->u_lightDir = bgfx::createUniform("u_lightDir", bgfx::UniformType::Vec4);
    state->u_lightParams = bgfx::createUniform("u_lightParams", bgfx::UniformType::Vec4);
    state->u_baseColorFactor = bgfx::createUniform("u_baseColorFactor", bgfx::UniformType::Vec4);
    state->u_overlayTransform = bgfx::createUniform("u_overlayTransform", bgfx::UniformType::Vec4);
    state->u_cameraPos = bgfx::createUniform("u_cameraPos", bgfx::UniformType::Vec4);
    state->u_colorParams = bgfx::createUniform("u_colorParams", bgfx::UniformType::Vec4);
    state->u_surfaceParams = bgfx::createUniform("u_surfaceParams", bgfx::UniformType::Vec4);
    state->s_imagery = bgfx::createUniform("s_imagery", bgfx::UniformType::Sampler);

    return state;
#endif
}

void cesiumShadersDestroy(CesiumShaderState* shaders)
{
    if (!shaders)
        return;
    if (bgfx::isValid(shaders->program))
        bgfx::destroy(shaders->program);
    if (bgfx::isValid(shaders->u_lightDir))
        bgfx::destroy(shaders->u_lightDir);
    if (bgfx::isValid(shaders->u_lightParams))
        bgfx::destroy(shaders->u_lightParams);
    if (bgfx::isValid(shaders->u_baseColorFactor))
        bgfx::destroy(shaders->u_baseColorFactor);
    if (bgfx::isValid(shaders->u_overlayTransform))
        bgfx::destroy(shaders->u_overlayTransform);
    if (bgfx::isValid(shaders->u_cameraPos))
        bgfx::destroy(shaders->u_cameraPos);
    if (bgfx::isValid(shaders->u_colorParams))
        bgfx::destroy(shaders->u_colorParams);
    if (bgfx::isValid(shaders->u_surfaceParams))
        bgfx::destroy(shaders->u_surfaceParams);
    if (bgfx::isValid(shaders->s_imagery))
        bgfx::destroy(shaders->s_imagery);
    delete shaders;
}

bgfx::ProgramHandle cesiumShadersGetProgram(const CesiumShaderState* shaders)
{
    if (!shaders)
        throw std::runtime_error("cesiumShadersGetProgram: shaders is NULL");
    return shaders->program;
}

bgfx::UniformHandle cesiumShadersGetUniform(const CesiumShaderState* shaders, CesiumShaderUniform uniform)
{
    if (!shaders)
        throw std::runtime_error("cesiumShadersGetUniform: shaders is NULL");
    switch (uniform) {
        case CesiumShaderUniform::LightDir:
            return shaders->u_lightDir;
        case CesiumShaderUniform::LightParams:
            return shaders->u_lightParams;
        case CesiumShaderUniform::BaseColorFactor:
            return shaders->u_baseColorFactor;
        case CesiumShaderUniform::OverlayTransform:
            return shaders->u_overlayTransform;
        case CesiumShaderUniform::CameraPos:
            return shaders->u_cameraPos;
        case CesiumShaderUniform::ColorParams:
            return shaders->u_colorParams;
        case CesiumShaderUniform::SurfaceParams:
            return shaders->u_surfaceParams;
        case CesiumShaderUniform::ImagerySampler:
            return shaders->s_imagery;
    }
    throw std::runtime_error("cesiumShadersGetUniform: unknown uniform");
}

} // namespace GRIM::GeoSpatial