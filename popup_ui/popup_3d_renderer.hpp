#pragma once

#include "popup_3d_types.hpp"
#include <bgfx/bgfx.h>

// ===========================================================
// Popup 3D Renderer — offscreen bgfx render + readback
// ===========================================================

// Forward declarations (opaque bgfx internals in .cpp)
struct PopupMeshGPU;
struct PopupShaderState;
struct PopupClipEngine;

struct Popup3DRenderer
{
    // Opaque modules (created during init)
    PopupMeshGPU*      mesh     = nullptr;   // static base mesh (from .obj)
    PopupShaderState*  shaders  = nullptr;

    // Animation: optional baked geometry playback + preset clip engine
    PopupMeshGPU*      dynamicMesh = nullptr; // created when a mesh-cache preset is loaded
    PopupClipEngine*   anim        = nullptr; // preset/clip engine (owns presets + caches)

    // DEBUG: when true, the static resting (.obj) model is NOT drawn — only frames
    // supplied by an active baked clip render; otherwise the target stays transparent.
    bool               hideRestingModel = false;

    // Albedo texture (BGFX_INVALID_HANDLE = no texture, use normal-based coloring)
    bgfx::TextureHandle albedoTex = BGFX_INVALID_HANDLE;

    // Normal map (tangent-space, RGB = xyz). INVALID = use geometry normal.
    bgfx::TextureHandle normalTex = BGFX_INVALID_HANDLE;

    // Packed material map: R=AO, G=roughness, B=metallic, A=opacity.
    // INVALID = defaults (AO=1, roughness=0.5, metallic=0, opacity=1).
    bgfx::TextureHandle packedTex = BGFX_INVALID_HANDLE;

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

// Load an albedo texture from an image file (PNG, JPG, TGA, BMP via stb_image).
// Must be called after popup3DRendererInit(). Replaces any existing texture.
void popup3DRendererLoadTexture(Popup3DRenderer& r, const char* imagePath);

// Load a tangent-space normal map. RGB = direction, blue-dominant = flat.
void popup3DRendererLoadNormalMap(Popup3DRenderer& r, const char* imagePath);

// Load a packed material map: R=AO, G=roughness, B=metallic, A=opacity.
void popup3DRendererLoadPackedMap(Popup3DRenderer& r, const char* imagePath);

// Check if any readback slots are still pending (for drain-then-destroy).
bool popup3DRendererHasPending(const Popup3DRenderer& r);

// -----------------------------------------------------------
// Animation presets (load-in / fade-out / baked geometry clips)
// -----------------------------------------------------------
// Initialize the preset clip engine. Loads built-in presets, then overrides
// from `presetsJsonPath` if it exists, preloading any referenced .gmc mesh
// caches from `popup3dDir`. If any preset references baked geometry, a dynamic
// mesh buffer is created sized to the largest cached frame.
// Must be called after popup3DRendererInit().
void popup3DRendererInitAnim(Popup3DRenderer& r,
                             const std::string& presetsJsonPath,
                             const std::string& popup3dDir,
                             uint32_t defaultColorABGR = 0xFF804020);

// Request playback of a named OneShot preset (thread-safe; callable from the UI thread).
void popup3DRendererTriggerPreset(Popup3DRenderer& r, const char* presetName);

// Start (or replace) the persistent blend-pose track (presence / speech / movement).
void popup3DRendererStartPose(Popup3DRenderer& r, const char* presetName);

// Blend the active pose out and return to idle.
void popup3DRendererStopPose(Popup3DRenderer& r);

// DEBUG: hide the static resting (.obj) model so only baked clip frames are drawn.
void popup3DRendererSetHideResting(Popup3DRenderer& r, bool hide);
