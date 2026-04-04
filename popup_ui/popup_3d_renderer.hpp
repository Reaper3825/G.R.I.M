#pragma once

#include "popup_3d_types.hpp"
#include <bgfx/bgfx.h>

// ===========================================================
// Popup 3D Renderer — offscreen bgfx render + readback
// ===========================================================

// Forward declarations (opaque bgfx internals in .cpp)
struct PopupMeshGPU;
struct PopupShaderState;

struct Popup3DRenderer
{
    // Opaque modules (created during init)
    PopupMeshGPU*      mesh     = nullptr;
    PopupShaderState*  shaders  = nullptr;

    // Readback ring (3 slots)
    static constexpr int kSlotCount = 3;
    PopupReadbackSlot slots[kSlotCount];

    // Mailbox for presenter handoff
    PopupFrameMailbox mailbox;

    // Current render dimensions
    uint32_t renderWidth  = 0;
    uint32_t renderHeight = 0;

    // Generation counter for readback
    uint64_t nextGeneration = 1;

    // Track whether renderer is initialized
    bool initialized = false;
};

// Initialize the renderer with mesh + embedded shaders.
// objDef: popup object definition (vertices, indices)
// width/height: initial offscreen render dimensions
void popup3DRendererInit(Popup3DRenderer& r,
                         const PopupObjectDefinition& objDef,
                         uint32_t width, uint32_t height);

// Submit one frame: render offscreen, queue readback, poll completed readbacks,
// publish newest completed frame to mailbox.
// input: per-frame render parameters (transform, light, alpha, etc.)
// currentBgfxFrame: current bgfx frame number (from bgfx::frame())
void popup3DRendererSubmit(Popup3DRenderer& r,
                           const PopupRenderInput& input,
                           uint32_t currentBgfxFrame);

// Resize offscreen render targets.
void popup3DRendererResize(Popup3DRenderer& r, uint32_t width, uint32_t height);

// Destroy all renderer resources.
void popup3DRendererShutdown(Popup3DRenderer& r);

// Check if any readback slots are still pending (for drain-then-destroy).
bool popup3DRendererHasPending(const Popup3DRenderer& r);
