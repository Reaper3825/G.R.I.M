#pragma once

#include <bgfx/bgfx.h>

// ===========================================================
// Popup Renderer — Texture + Shader Interface
// ===========================================================

// Loads and links the BGFX shader program for popup rendering.
bgfx::ProgramHandle loadPopupProgram();


// Loads both diffuse and opacity textures (and their sampler uniforms).
void loadPopupTextures();

// Sampler and texture accessors
bgfx::UniformHandle getPopupSamplerColor();
bgfx::UniformHandle getPopupSamplerOpacity();
bgfx::TextureHandle getPopupTextureColor();
bgfx::TextureHandle getPopupTextureOpacity();
