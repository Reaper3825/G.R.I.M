#pragma once

#include <bgfx/bgfx.h>

// ===========================================================
// Popup 3D Shaders — embedded bgfx program (no shader files)
// ===========================================================

// Opaque shader state (bgfx details in .cpp)
struct PopupShaderState;

// Create shader program from embedded bytecodes (no file I/O).
PopupShaderState* popupShadersCreate();

// Destroy shader program.
void popupShadersDestroy(PopupShaderState* shaders);

// Get the program handle for bgfx::submit().
bgfx::ProgramHandle popupShadersGetProgram(const PopupShaderState* shaders);
