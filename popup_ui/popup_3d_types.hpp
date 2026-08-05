#pragma once

#include "popup_material_ir.hpp"
#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <mutex>
#include <atomic>

// ===========================================================
// Popup 3D — POD types only (no bgfx headers)
// ===========================================================

// Vertex layout: position + normal + tangent + texcoord + color
struct PopupVertex
{
    float px, py, pz;   // position
    float nx, ny, nz;   // normal
    float tx, ty, tz;   // tangent (unit vector in UV-U direction)
    float tw;           // tangent handedness (+1 or -1): bitangent = cross(N, T) * tw
    float u, v;          // texture coordinates
    uint32_t abgr;      // packed color (bgfx ABGR: bits [31:24]=A, [23:16]=B, [15:8]=G, [7:0]=R)
};

// Object transform (model space)
struct PopupTransform
{
    float position[3] = { 0.0f, 0.0f, 0.0f };
    float rotation[3] = { 0.0f, 0.0f, 0.0f };  // euler angles (radians)
    float scale[3]    = { 1.0f, 1.0f, 1.0f };
};

// Directional light parameters
struct PopupLightParams
{
    float direction[3] = { -0.5f, 0.7f, 0.5f };  // world space, upper-front-left
    float intensity    = 1.0f;
    float ambient      = 0.15f;
};

// Snapshot of per-frame render inputs (written by UI thread, consumed by submission thread)
struct PopupRenderInput
{
    PopupTransform transform;
    PopupLightParams light;
    float alphaMul    = 1.0f;
    float emissiveMul = 0.0f;
    uint32_t width    = 0;
    uint32_t height   = 0;
    bool visible      = false;
};

// Readback slot state
enum class PopupSlotState : uint8_t
{
    Idle,
    PendingReadback,
    Ready
};

// One readback staging slot (bgfx handles stored in renderer, not here)
struct PopupReadbackSlot
{
    std::vector<uint8_t> rawStraightBgra;    // width * height * 4
    uint32_t readyAfterFrame = 0;
    uint64_t generation      = 0;
    PopupSlotState state     = PopupSlotState::Idle;
    uint32_t width           = 0;
    uint32_t height          = 0;
};

// Thread-safe mailbox: newest completed frame for presenter consumption
struct PopupFrameMailbox
{
    std::vector<uint8_t> buffer;   // straight-alpha BGRA8
    uint32_t width       = 0;
    uint32_t height      = 0;
    uint32_t stride      = 0;     // bytes per row (width * 4)
    uint64_t generation  = 0;
    std::mutex mutex;
};

// Immutable object description
struct PopupObjectDefinition
{
    std::vector<PopupVertex> vertices;
    std::vector<uint16_t> indices;
    PopupTransform defaultTransform;
    PopupLightParams defaultLight;
};

// ===========================================================
// Baked geometry-node animation (mesh cache)
// ===========================================================
// One baked frame of a procedural-geometry animation. Topology may vary
// frame-to-frame (Blender geometry nodes can change vertex/face count), so
// each frame carries its own vertex + index arrays.
struct PopupMeshFrame
{
    std::vector<PopupVertex> vertices;
    std::vector<uint16_t>    indices;
};

// A baked clip: an ordered sequence of mesh frames played at `fps`.
// `maxVertices` / `maxIndices` are the per-frame maxima across the clip and
// size the dynamic GPU buffers used for playback.
struct PopupMeshCache
{
    float    fps         = 30.0f;
    uint32_t maxVertices = 0;
    uint32_t maxIndices  = 0;
    PopupMaterialProgram materialProgram;
    std::vector<PopupMeshFrame> frames;

    bool empty() const { return frames.empty(); }
    float durationSec() const
    {
        return (fps > 0.0f && !frames.empty())
                   ? static_cast<float>(frames.size()) / fps
                   : 0.0f;
    }
};

// A completed frame ready for presentation
struct PopupRenderFrame
{
    const uint8_t* data = nullptr;
    uint32_t width      = 0;
    uint32_t height     = 0;
    uint64_t generation = 0;
};
