#pragma once

#include <bgfx/bgfx.h>

namespace GRIM::GeoSpatial {

struct CesiumShaderState;

enum class CesiumShaderUniform {
    LightDir,
    LightParams,
    BaseColorFactor,
    OverlayTransform,
    CameraPos,
    ColorParams,
    SurfaceParams,
    ImagerySampler
};

CesiumShaderState* cesiumShadersCreate();
void cesiumShadersDestroy(CesiumShaderState* shaders);
bgfx::ProgramHandle cesiumShadersGetProgram(const CesiumShaderState* shaders);
bgfx::UniformHandle cesiumShadersGetUniform(const CesiumShaderState* shaders, CesiumShaderUniform uniform);

} // namespace GRIM::GeoSpatial