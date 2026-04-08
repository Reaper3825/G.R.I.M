#pragma once

#include <bgfx/bgfx.h>

// ===========================================================
// Popup 3D Shaders — compiled popup model program + uniforms
// ===========================================================

// Opaque shader state (bgfx details in .cpp)
struct PopupShaderState;

// Uniform identifiers for the popup model shader
enum class PopupShaderUniform
{
    LightDir,        // vec4: xyz = normalized light direction
    LightParams,     // vec4: x = intensity, y = ambient
    Alpha,           // vec4: x = alpha multiplier
    Emissive,        // vec4: x = emissive multiplier
    AlbedoSampler,   // sampler2D: s_albedo  (stage 0)
    NormalSampler,   // sampler2D: s_normal  (stage 1)
    PackedSampler    // sampler2D: s_packed  (stage 2) — R=AO G=roughness B=metallic A=opacity
};

// Create shader program from embedded bytecodes (no file I/O).
PopupShaderState* popupShadersCreate();

// Destroy shader program and all uniforms.
void popupShadersDestroy(PopupShaderState* shaders);

// Get the program handle for bgfx::submit().
bgfx::ProgramHandle popupShadersGetProgram(const PopupShaderState* shaders);

// Get a uniform handle by identifier.
bgfx::UniformHandle popupShadersGetUniform(const PopupShaderState* shaders, PopupShaderUniform u);
