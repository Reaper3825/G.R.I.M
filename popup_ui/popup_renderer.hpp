#pragma once
#include <bgfx/bgfx.h>

// Returns the compiled shader program
bgfx::ProgramHandle loadPopupProgram();

// Loads diffuse PNG + sampler uniform
void loadPopupTexture();

// Accessors for the sampler and texture handles
bgfx::UniformHandle getPopupSampler();
bgfx::TextureHandle getPopupTexture();
