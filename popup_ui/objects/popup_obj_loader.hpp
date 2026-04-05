#pragma once

#include "../popup_3d_types.hpp"
#include <string>

// ===========================================================
// Lightweight Wavefront OBJ loader → PopupObjectDefinition
// Supports: v (position), vn (normal), f (triangulated faces)
// No textures, no materials, no groups — just geometry.
// Faces MUST be triangulated (3 vertices per face).
// ===========================================================

// Load a .obj file and return a PopupObjectDefinition ready for the 3D renderer.
// Throws std::runtime_error on file-not-found, parse error, or empty mesh.
// Default vertex color is applied uniformly (ABGR packed).
PopupObjectDefinition loadPopupObjectFromOBJ(const std::string& objPath,
                                              uint32_t defaultColorABGR = 0xFF804020);  // opaque teal-ish
