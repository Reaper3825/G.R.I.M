#pragma once

#include "popup_3d_types.hpp"

// ===========================================================
// Popup 3D Mesh — CPU data + bgfx buffer ownership
// ===========================================================

// Opaque mesh handle (bgfx details in .cpp)
struct PopupMeshGPU;

// Create GPU mesh from CPU vertex/index data. Caller owns the returned pointer.
PopupMeshGPU* popupMeshCreate(const PopupVertex* verts, uint32_t vertCount,
                               const uint16_t* indices, uint32_t indexCount);

// Destroy GPU mesh and free memory.
void popupMeshDestroy(PopupMeshGPU* mesh);

// Bind mesh for a bgfx draw call (sets vertex/index buffers).
void popupMeshBind(const PopupMeshGPU* mesh);

// Return index count for draw call.
uint32_t popupMeshIndexCount(const PopupMeshGPU* mesh);
