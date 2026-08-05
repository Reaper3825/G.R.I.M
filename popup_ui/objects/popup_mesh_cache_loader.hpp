#pragma once

#include "../popup_3d_types.hpp"
#include <string>
#include <cstdint>

// ===========================================================
// GRIM Mesh Cache (.gmc) loader
// ===========================================================
// Loads a baked geometry-node animation (a sequence of mesh frames) exported
// from Blender via tools/blender/bake_popup_mesh_cache.py.
//
// Binary layout (all little-endian):
//
//   char     magic[8]     = "GRIMMC03"
//   float32  fps
//   uint32   frameCount
//   uint32   flags         (bit0 = HAS_NORMALS, bit1 = HAS_UV, bit2 = HAS_COLOR,
//                           bit3 = HAS_MATERIAL_PROGRAM)
//   uint32   maxVertices   (max vertCount across all frames)
//   uint32   maxIndices    (max indexCount across all frames)
//   --- when HAS_MATERIAL_PROGRAM ---
//     uint32 registerCount
//     uint32 instructionCount
//     uint32 parameterCount
//     PopupMaterialInstruction instructions[instructionCount]
//     PopupMaterialParameter parameters[parameterCount]
//   --- repeated frameCount times ---
//     uint32 vertCount
//     uint32 indexCount
//     float32 positions[3 * vertCount]
//     float32 normals[3 * vertCount]    (only if HAS_NORMALS)
//     float32 uvs[2 * vertCount]        (only if HAS_UV)
//     uint32  colorsABGR[vertCount]      (only if HAS_COLOR)
//     uint16  indices[indexCount]
//
// Constraints: vertCount <= 65535 per frame (16-bit index buffer).
// When normals are absent they are auto-generated per frame from face data.

constexpr uint32_t GMC_FLAG_HAS_NORMALS = 1u << 0;
constexpr uint32_t GMC_FLAG_HAS_UV      = 1u << 1;
constexpr uint32_t GMC_FLAG_HAS_COLOR   = 1u << 2;
constexpr uint32_t GMC_FLAG_HAS_MATERIAL_PROGRAM = 1u << 3;

// Uses `defaultColorABGR` as a visible debug color when HAS_COLOR is absent.
// Throws std::runtime_error on file-not-found, bad magic, or malformed data.
PopupMeshCache loadPopupMeshCache(const std::string& gmcPath,
                                  uint32_t defaultColorABGR = 0xFF804020);
