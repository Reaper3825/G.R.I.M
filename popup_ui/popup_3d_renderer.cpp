#include "popup_3d_renderer.hpp"
#include "popup_3d_mesh.hpp"
#include "popup_3d_shaders.hpp"
#include "popup_3d_mailbox.hpp"
#include <bgfx/bgfx.h>
#include <bx/math.h>
#include <stdexcept>
#include <string>
#include <cstring>
#include <cmath>

// ===========================================================
// Popup 3D Renderer — offscreen render + async readback
// ===========================================================

static constexpr bgfx::ViewId kPopupViewId = 31;

// Offscreen framebuffer state (per readback slot)
struct SlotGPU
{
    bgfx::TextureHandle colorTex = BGFX_INVALID_HANDLE;
    bgfx::TextureHandle depthTex = BGFX_INVALID_HANDLE;
    bgfx::FrameBufferHandle fb   = BGFX_INVALID_HANDLE;
};

static SlotGPU s_slotGPU[Popup3DRenderer::kSlotCount];

// -------------------------------------------------------
// Slot GPU resource management
// -------------------------------------------------------
static void createSlotGPU(SlotGPU& sg, uint32_t w, uint32_t h)
{
    sg.colorTex = bgfx::createTexture2D(
        static_cast<uint16_t>(w),
        static_cast<uint16_t>(h),
        false, 1,
        bgfx::TextureFormat::BGRA8,
        BGFX_TEXTURE_RT | BGFX_TEXTURE_READ_BACK
    );
    if (!bgfx::isValid(sg.colorTex))
        throw std::runtime_error("Popup3DRenderer: failed to create color texture");

    sg.depthTex = bgfx::createTexture2D(
        static_cast<uint16_t>(w),
        static_cast<uint16_t>(h),
        false, 1,
        bgfx::TextureFormat::D24S8,
        BGFX_TEXTURE_RT
    );
    if (!bgfx::isValid(sg.depthTex))
        throw std::runtime_error("Popup3DRenderer: failed to create depth texture");

    bgfx::TextureHandle attachments[2] = { sg.colorTex, sg.depthTex };
    sg.fb = bgfx::createFrameBuffer(2, attachments, false);
    if (!bgfx::isValid(sg.fb))
        throw std::runtime_error("Popup3DRenderer: failed to create framebuffer");
}

static void destroySlotGPU(SlotGPU& sg)
{
    if (bgfx::isValid(sg.fb))       bgfx::destroy(sg.fb);
    if (bgfx::isValid(sg.colorTex)) bgfx::destroy(sg.colorTex);
    if (bgfx::isValid(sg.depthTex)) bgfx::destroy(sg.depthTex);
    sg.fb       = BGFX_INVALID_HANDLE;
    sg.colorTex = BGFX_INVALID_HANDLE;
    sg.depthTex = BGFX_INVALID_HANDLE;
}

// -------------------------------------------------------
// Capability check
// -------------------------------------------------------
static void validateCaps()
{
    const bgfx::Caps* caps = bgfx::getCaps();
    if (!(caps->supported & BGFX_CAPS_TEXTURE_READ_BACK))
        throw std::runtime_error("Popup3DRenderer: BGFX_CAPS_TEXTURE_READ_BACK not supported");
}

// -------------------------------------------------------
// Init
// -------------------------------------------------------
void popup3DRendererInit(Popup3DRenderer& r,
                         const PopupObjectDefinition& objDef,
                         uint32_t width, uint32_t height)
{
    if (r.initialized)
        throw std::runtime_error("Popup3DRenderer: already initialized");
    if (width == 0 || height == 0)
        throw std::runtime_error("Popup3DRenderer: width/height must be > 0");

    validateCaps();

    // Create mesh
    r.mesh = popupMeshCreate(objDef.vertices.data(),
                              static_cast<uint32_t>(objDef.vertices.size()),
                              objDef.indices.data(),
                              static_cast<uint32_t>(objDef.indices.size()));

    // Create shaders (embedded — no file I/O)
    r.shaders = popupShadersCreate();

    // Create per-slot GPU resources
    for (int i = 0; i < Popup3DRenderer::kSlotCount; i++)
    {
        createSlotGPU(s_slotGPU[i], width, height);
        r.slots[i].rawStraightBgra.resize(width * height * 4, 0);
        r.slots[i].width  = width;
        r.slots[i].height = height;
        r.slots[i].state  = PopupSlotState::Idle;
    }

    r.renderWidth  = width;
    r.renderHeight = height;
    r.initialized  = true;
}

// -------------------------------------------------------
// Submit one frame
// -------------------------------------------------------
void popup3DRendererSubmit(Popup3DRenderer& r,
                           const PopupRenderInput& input,
                           uint32_t currentBgfxFrame)
{
    if (!r.initialized)
        throw std::runtime_error("Popup3DRenderer::submit: not initialized");
    if (!input.visible)
        return;

    // ---- Poll completed readbacks ----
    int newestReady = -1;
    uint64_t newestGen = 0;
    for (int i = 0; i < Popup3DRenderer::kSlotCount; i++)
    {
        auto& slot = r.slots[i];
        if (slot.state == PopupSlotState::PendingReadback &&
            currentBgfxFrame >= slot.readyAfterFrame)
        {
            slot.state = PopupSlotState::Ready;
        }
        if (slot.state == PopupSlotState::Ready && slot.generation > newestGen)
        {
            newestGen = slot.generation;
            newestReady = i;
        }
    }

    // ---- Publish newest ready frame to mailbox ----
    if (newestReady >= 0)
    {
        auto& slot = r.slots[newestReady];
        popupMailboxPublish(r.mailbox,
                            slot.rawStraightBgra.data(),
                            slot.width, slot.height);

        // Mark all ready slots as idle
        for (int i = 0; i < Popup3DRenderer::kSlotCount; i++)
        {
            if (r.slots[i].state == PopupSlotState::Ready)
                r.slots[i].state = PopupSlotState::Idle;
        }
    }

    // ---- Find an idle slot ----
    int idleSlot = -1;
    for (int i = 0; i < Popup3DRenderer::kSlotCount; i++)
    {
        if (r.slots[i].state == PopupSlotState::Idle)
        {
            idleSlot = i;
            break;
        }
    }
    if (idleSlot < 0)
        return;  // all slots busy, skip this frame

    // ---- Build model matrix ----
    float mtxModel[16];
    bx::mtxSRT(mtxModel,
               input.transform.scale[0], input.transform.scale[1], input.transform.scale[2],
               input.transform.rotation[0], input.transform.rotation[1], input.transform.rotation[2],
               input.transform.position[0], input.transform.position[1], input.transform.position[2]);

    // ---- Build view matrix ----
    float mtxView[16];
    {
        const bx::Vec3 eye = { 0.0f, 0.0f, 2.5f };
        const bx::Vec3 at  = { 0.0f, 0.0f, 0.0f };
        const bx::Vec3 up  = { 0.0f, 1.0f, 0.0f };
        bx::mtxLookAt(mtxView, eye, at, up);
    }

    // ---- Build projection matrix ----
    float mtxProj[16];
    {
        float aspect = static_cast<float>(r.renderWidth) / static_cast<float>(r.renderHeight);
        bx::mtxProj(mtxProj,
                     30.0f,    // fov Y degrees
                     aspect,
                     0.05f,    // near
                     10.0f,    // far
                     bgfx::getCaps()->homogeneousDepth);
    }

    // ---- Setup view ----
    bgfx::setViewFrameBuffer(kPopupViewId, s_slotGPU[idleSlot].fb);
    bgfx::setViewRect(kPopupViewId, 0, 0,
                       static_cast<uint16_t>(r.renderWidth),
                       static_cast<uint16_t>(r.renderHeight));
    bgfx::setViewClear(kPopupViewId,
                        BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH,
                        0x00000000,  // transparent BGRA
                        1.0f, 0);
    bgfx::setViewTransform(kPopupViewId, mtxView, mtxProj);

    // ---- Set transform ----
    bgfx::setTransform(mtxModel);

    // ---- Bind mesh ----
    popupMeshBind(r.mesh);

    // ---- Set render state ----
    uint64_t state = 0
        | BGFX_STATE_WRITE_RGB
        | BGFX_STATE_WRITE_A
        | BGFX_STATE_WRITE_Z
        | BGFX_STATE_DEPTH_TEST_LESS
        | BGFX_STATE_CULL_CW
        | BGFX_STATE_BLEND_FUNC(BGFX_STATE_BLEND_SRC_ALPHA, BGFX_STATE_BLEND_INV_SRC_ALPHA)
        ;
    bgfx::setState(state);

    // ---- Submit ----
    bgfx::submit(kPopupViewId, popupShadersGetProgram(r.shaders));

    // ---- Queue readback ----
    auto& slot = r.slots[idleSlot];
    slot.readyAfterFrame = bgfx::readTexture(s_slotGPU[idleSlot].colorTex,
                                              slot.rawStraightBgra.data());
    slot.generation = r.nextGeneration++;
    slot.state = PopupSlotState::PendingReadback;
}

// -------------------------------------------------------
// Resize
// -------------------------------------------------------
void popup3DRendererResize(Popup3DRenderer& r, uint32_t width, uint32_t height)
{
    if (!r.initialized) return;
    if (width == r.renderWidth && height == r.renderHeight) return;

    // Wait for all pending readbacks to drain
    for (int i = 0; i < Popup3DRenderer::kSlotCount; i++)
    {
        if (r.slots[i].state == PopupSlotState::PendingReadback)
            r.slots[i].state = PopupSlotState::Idle;  // force idle on resize
    }

    // Recreate GPU resources
    for (int i = 0; i < Popup3DRenderer::kSlotCount; i++)
    {
        destroySlotGPU(s_slotGPU[i]);
        createSlotGPU(s_slotGPU[i], width, height);
        r.slots[i].rawStraightBgra.resize(width * height * 4, 0);
        r.slots[i].width  = width;
        r.slots[i].height = height;
        r.slots[i].state  = PopupSlotState::Idle;
    }

    r.renderWidth  = width;
    r.renderHeight = height;
}

// -------------------------------------------------------
// Shutdown
// -------------------------------------------------------
void popup3DRendererShutdown(Popup3DRenderer& r)
{
    if (!r.initialized) return;

    for (int i = 0; i < Popup3DRenderer::kSlotCount; i++)
        destroySlotGPU(s_slotGPU[i]);

    if (r.shaders) { popupShadersDestroy(r.shaders); r.shaders = nullptr; }
    if (r.mesh)    { popupMeshDestroy(r.mesh);        r.mesh    = nullptr; }

    r.initialized = false;
}

// -------------------------------------------------------
// Drain check
// -------------------------------------------------------
bool popup3DRendererHasPending(const Popup3DRenderer& r)
{
    for (int i = 0; i < Popup3DRenderer::kSlotCount; i++)
    {
        if (r.slots[i].state == PopupSlotState::PendingReadback)
            return true;
    }
    return false;
}
