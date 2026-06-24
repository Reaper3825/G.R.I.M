#pragma once

#include "popup_3d_types.hpp"

// ===========================================================
// Popup 3D Mesh — CPU data + bgfx buffer ownership
// ===========================================================

// Opaque mesh handle (bgfx details in .cpp)
struct PopupMeshGPU;

// Create a STATIC GPU mesh from CPU vertex/index data. Caller owns the returned pointer.
PopupMeshGPU* popupMeshCreate(const PopupVertex* verts, uint32_t vertCount,
                               const uint16_t* indices, uint32_t indexCount);

// Create a DYNAMIC GPU mesh sized for up to maxVerts / maxIndices, to be filled
// per-frame via popupMeshUpdate(). Used for baked geometry-node animation playback.
PopupMeshGPU* popupMeshCreateDynamic(uint32_t maxVerts, uint32_t maxIndices);

// Upload a new frame of geometry into a dynamic mesh. vertCount/indexCount must
// not exceed the maxima the mesh was created with. No-op for static meshes.
void popupMeshUpdate(PopupMeshGPU* mesh,
                     const PopupVertex* verts, uint32_t vertCount,
                     const uint16_t* indices, uint32_t indexCount);

// Destroy GPU mesh and free memory.
void popupMeshDestroy(PopupMeshGPU* mesh);

// Bind mesh for a bgfx draw call (sets vertex/index buffers for the active range).
void popupMeshBind(const PopupMeshGPU* mesh);

// Return index count for draw call (current range for dynamic meshes).
uint32_t popupMeshIndexCount(const PopupMeshGPU* mesh);
