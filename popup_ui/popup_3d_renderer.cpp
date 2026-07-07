#include "popup_3d_renderer.hpp"
#include "popup_3d_mesh.hpp"
#include "popup_3d_shaders.hpp"
#include "popup_3d_mailbox.hpp"
#include "popup_anim_presets.hpp"
#include <bgfx/bgfx.h>
#include <bx/math.h>
#include <stdexcept>
#include <string>
#include <cstring>
#include <cmath>
#include <chrono>

#include <stb/stb_image.h>
#include "logger.hpp"

// ===========================================================
// Popup 3D Renderer — offscreen render + async readback
// ===========================================================

static constexpr bgfx::ViewId kPopupViewId = 31;

// Monotonic seconds since first call — drives preset clip playback timing.
static double popupNowSeconds()
{
    static const auto s_start = std::chrono::steady_clock::now();
    auto dt = std::chrono::steady_clock::now() - s_start;
    return std::chrono::duration<double>(dt).count();
}

// Offscreen framebuffer state (per readback slot)
struct SlotGPU
{
    bgfx::TextureHandle colorTex    = BGFX_INVALID_HANDLE; // RT-only
    bgfx::TextureHandle readbackTex = BGFX_INVALID_HANDLE; // blit-dst + readback
    bgfx::TextureHandle depthTex    = BGFX_INVALID_HANDLE;
    bgfx::FrameBufferHandle fb      = BGFX_INVALID_HANDLE;
};

static SlotGPU s_slotGPU[Popup3DRenderer::kSlotCount];

// 1x1 white fallback for unbound texture samplers (avoids sampling zeros on Metal)
static bgfx::TextureHandle s_whiteFallback = BGFX_INVALID_HANDLE;

// -------------------------------------------------------
// Slot GPU resource management
// -------------------------------------------------------
static constexpr bgfx::ViewId kPopupBlitViewId = 32;

static void createSlotGPU(SlotGPU& sg, uint32_t w, uint32_t h)
{
    sg.colorTex = bgfx::createTexture2D(
        static_cast<uint16_t>(w),
        static_cast<uint16_t>(h),
        false, 1,
        bgfx::TextureFormat::BGRA8,
        BGFX_TEXTURE_RT
    );
    if (!bgfx::isValid(sg.colorTex))
        throw std::runtime_error("Popup3DRenderer: failed to create color texture");

    sg.readbackTex = bgfx::createTexture2D(
        static_cast<uint16_t>(w),
        static_cast<uint16_t>(h),
        false, 1,
        bgfx::TextureFormat::BGRA8,
        BGFX_TEXTURE_BLIT_DST | BGFX_TEXTURE_READ_BACK
    );
    if (!bgfx::isValid(sg.readbackTex))
        throw std::runtime_error("Popup3DRenderer: failed to create readback texture");

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
    if (bgfx::isValid(sg.fb))          bgfx::destroy(sg.fb);
    if (bgfx::isValid(sg.colorTex))    bgfx::destroy(sg.colorTex);
    if (bgfx::isValid(sg.readbackTex)) bgfx::destroy(sg.readbackTex);
    if (bgfx::isValid(sg.depthTex))    bgfx::destroy(sg.depthTex);
    sg.fb          = BGFX_INVALID_HANDLE;
    sg.colorTex    = BGFX_INVALID_HANDLE;
    sg.readbackTex = BGFX_INVALID_HANDLE;
    sg.depthTex    = BGFX_INVALID_HANDLE;
}

// -------------------------------------------------------
// Capability check
// -------------------------------------------------------
static void validateCaps()
{
    const bgfx::Caps* caps = bgfx::getCaps();
    if (!(caps->supported & BGFX_CAPS_TEXTURE_BLIT))
        throw std::runtime_error("Popup3DRenderer: BGFX_CAPS_TEXTURE_BLIT not supported");
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

    // Create 1x1 white fallback so unbound samplers return (1,1,1,1) instead of (0,0,0,0)
    if (!bgfx::isValid(s_whiteFallback))
    {
        uint32_t white = 0xFFFFFFFF;
        const bgfx::Memory* mem = bgfx::copy(&white, 4);
        s_whiteFallback = bgfx::createTexture2D(1, 1, false, 1,
            bgfx::TextureFormat::RGBA8, BGFX_SAMPLER_U_CLAMP | BGFX_SAMPLER_V_CLAMP, mem);
    }

    r.initialized  = true;

    LOG_DEBUG("Popup3D", "Init OK: mesh=" + std::to_string((uintptr_t)r.mesh) +
             " shaders=" + std::to_string((uintptr_t)r.shaders) +
             " verts=" + std::to_string(objDef.vertices.size()) +
             " indices=" + std::to_string(objDef.indices.size()) +
             " size=" + std::to_string(width) + "x" + std::to_string(height));
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
    {
        static int sInvisSkipCount = 0;
        if (++sInvisSkipCount <= 3)
            LOG_DEBUG("Popup3D", "submit: input.visible=false, skipping (" + std::to_string(sInvisSkipCount) + ")");
        return;
    }

    static int sSubmitCount = 0;
    ++sSubmitCount;
    bool shouldLog = (sSubmitCount <= 5) || (sSubmitCount % 300 == 0);

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

        if (shouldLog)
            LOG_DEBUG("Popup3D", "Publishing slot " + std::to_string(newestReady) +
                     " gen=" + std::to_string(slot.generation) +
                     " frame#" + std::to_string(sSubmitCount));

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
    {
        if (shouldLog)
            LOG_DEBUG("Popup3D", "No idle slot — skipping frame#" + std::to_string(sSubmitCount));
        return;  // all slots busy, skip this frame
    }

    // ---- Evaluate active animation preset (render thread) ----
    // Preset output is layered multiplicatively on top of the per-frame input,
    // and may select a baked geometry frame to draw instead of the static mesh.
    PopupRenderInput eff = input;
    const PopupMeshFrame* clipFrame = nullptr;
    if (r.anim)
    {
        PopupClipEval ev;
        if (popupClipEngineEvaluate(r.anim, popupNowSeconds(), ev) && ev.active)
        {
            eff.alphaMul          *= ev.alphaMul;
            eff.transform.scale[0] *= ev.scaleMul;
            eff.transform.scale[1] *= ev.scaleMul;
            eff.transform.scale[2] *= ev.scaleMul;
            eff.emissiveMul        += ev.emissiveAdd;
            eff.transform.rotation[1] += ev.spinY;
            eff.transform.position[0] += ev.posOffset[0];
            eff.transform.position[1] += ev.posOffset[1];
            eff.transform.position[2] += ev.posOffset[2];
            clipFrame = ev.frame;
        }
    }

    // ---- Build model matrix ----
    float mtxModel[16];
    bx::mtxSRT(mtxModel,
               eff.transform.scale[0], eff.transform.scale[1], eff.transform.scale[2],
               eff.transform.rotation[0], eff.transform.rotation[1], eff.transform.rotation[2],
               eff.transform.position[0], eff.transform.position[1], eff.transform.position[2]);

    // ---- Build view matrix ----
    float mtxView[16];
    {
        const bx::Vec3 eye = { 0.0f, 0.0f, 3.5f };
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

    // ---- Decide what to draw ----
    const bool haveClip = clipFrame && r.dynamicMesh &&
                          !clipFrame->vertices.empty() && !clipFrame->indices.empty();
    // DEBUG: with hideRestingModel set, only draw while a clip supplies geometry;
    // otherwise the view is just cleared (transparent), hiding the static .obj.
    const bool drawModel = haveClip || !r.hideRestingModel;

    // ---- Bind mesh (baked geometry frame if a clip is active, else static) ----
    if (haveClip)
    {
        popupMeshUpdate(r.dynamicMesh,
                        clipFrame->vertices.data(),
                        static_cast<uint32_t>(clipFrame->vertices.size()),
                        clipFrame->indices.data(),
                        static_cast<uint32_t>(clipFrame->indices.size()));
        popupMeshBind(r.dynamicMesh);
    }
    else if (drawModel)
    {
        popupMeshBind(r.mesh);
    }

    // ---- Set uniforms for popup model shader ----
    {
        // Light direction (normalized)
        float lightDir[4] = {
            eff.light.direction[0],
            eff.light.direction[1],
            eff.light.direction[2],
            0.0f
        };
        // Normalize
        float len = std::sqrt(lightDir[0]*lightDir[0] + lightDir[1]*lightDir[1] + lightDir[2]*lightDir[2]);
        if (len > 1e-8f) { lightDir[0] /= len; lightDir[1] /= len; lightDir[2] /= len; }
        bgfx::setUniform(popupShadersGetUniform(r.shaders, PopupShaderUniform::LightDir), lightDir);

        float lightParams[4] = { eff.light.intensity, eff.light.ambient, 0.0f, 0.0f };
        bgfx::setUniform(popupShadersGetUniform(r.shaders, PopupShaderUniform::LightParams), lightParams);

        float alpha[4] = { eff.alphaMul, 0.0f, 0.0f, 0.0f };
        bgfx::setUniform(popupShadersGetUniform(r.shaders, PopupShaderUniform::Alpha), alpha);

        float emissive[4] = { eff.emissiveMul, 0.0f, 0.0f, 0.0f };
        bgfx::setUniform(popupShadersGetUniform(r.shaders, PopupShaderUniform::Emissive), emissive);

        // Always bind all 3 texture slots — Metal returns (0,0,0,0) for unbound samplers
        bgfx::TextureHandle albedo = bgfx::isValid(r.albedoTex) ? r.albedoTex : s_whiteFallback;
        bgfx::TextureHandle normal = bgfx::isValid(r.normalTex) ? r.normalTex : s_whiteFallback;
        bgfx::TextureHandle packed = bgfx::isValid(r.packedTex) ? r.packedTex : s_whiteFallback;
        bgfx::setTexture(0, popupShadersGetUniform(r.shaders, PopupShaderUniform::AlbedoSampler), albedo);
        bgfx::setTexture(1, popupShadersGetUniform(r.shaders, PopupShaderUniform::NormalSampler), normal);
        bgfx::setTexture(2, popupShadersGetUniform(r.shaders, PopupShaderUniform::PackedSampler), packed);
    }

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

    // ---- Submit (or just clear the view when nothing should draw) ----
    bgfx::ProgramHandle prog = popupShadersGetProgram(r.shaders);
    if (shouldLog)
        LOG_DEBUG("Popup3D", "submit: slot=" + std::to_string(idleSlot) +
                 " alpha=" + std::to_string(eff.alphaMul) +
                 " prog=" + std::to_string(prog.idx) +
                 " fb=" + std::to_string(s_slotGPU[idleSlot].fb.idx) +
                 " draw=" + std::to_string(drawModel ? 1 : 0) +
                 " frame#" + std::to_string(sSubmitCount));
    if (drawModel)
        bgfx::submit(kPopupViewId, prog);
    else
        bgfx::touch(kPopupViewId);  // process the view so it clears transparent

    // ---- Blit RT → readback texture, then queue readback ----
    // Activate the blit view so bgfx processes the blit command.
    bgfx::setViewRect(kPopupBlitViewId, 0, 0,
                       static_cast<uint16_t>(r.renderWidth),
                       static_cast<uint16_t>(r.renderHeight));
    bgfx::touch(kPopupBlitViewId);

    auto& slot = r.slots[idleSlot];
    bgfx::blit(kPopupBlitViewId,
               s_slotGPU[idleSlot].readbackTex, 0, 0,
               s_slotGPU[idleSlot].colorTex,    0, 0,
               static_cast<uint16_t>(r.renderWidth),
               static_cast<uint16_t>(r.renderHeight));
    slot.readyAfterFrame = bgfx::readTexture(s_slotGPU[idleSlot].readbackTex,
                                              slot.rawStraightBgra.data());
    slot.generation = r.nextGeneration++;
    slot.state = PopupSlotState::PendingReadback;

    if (shouldLog)
        LOG_DEBUG("Popup3D", "readback queued: slot=" + std::to_string(idleSlot) +
                 " gen=" + std::to_string(slot.generation) +
                 " readyAfter=" + std::to_string(slot.readyAfterFrame) +
                 " curBgfxFrame=" + std::to_string(currentBgfxFrame));
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
// Animation presets
// -------------------------------------------------------
void popup3DRendererInitAnim(Popup3DRenderer& r,
                             const std::string& presetsJsonPath,
                             const std::string& popup3dDir,
                             uint32_t defaultColorABGR)
{
    if (!r.initialized)
        throw std::runtime_error("popup3DRendererInitAnim: renderer not initialized");

    if (!r.anim)
        r.anim = popupClipEngineCreate();

    // Built-in presets first, then JSON overrides (if present).
    popupClipEngineLoadDefaults(r.anim);
    popupClipEngineLoadJson(r.anim, presetsJsonPath, popup3dDir, defaultColorABGR);

    // If any preset uses baked geometry, allocate a dynamic mesh sized to the
    // largest cached frame.
    if (popupClipEngineHasGeometry(r.anim))
    {
        uint32_t maxV = 0, maxI = 0;
        popupClipEngineMaxBuffer(r.anim, &maxV, &maxI);
        if (maxV > 0 && maxI > 0 && !r.dynamicMesh)
        {
            try { r.dynamicMesh = popupMeshCreateDynamic(maxV, maxI); }
            catch (const std::exception& e)
            {
                LOG_ERROR("Popup3D", std::string("Dynamic mesh alloc failed: ") + e.what());
            }
        }
    }

    LOG_DEBUG("Popup3D", "Anim engine ready (geometry=" +
              std::string(popupClipEngineHasGeometry(r.anim) ? "yes" : "no") + ")");
}

void popup3DRendererTriggerPreset(Popup3DRenderer& r, const char* presetName)
{
    if (r.anim)
        popupClipEngineTrigger(r.anim, presetName);
}

void popup3DRendererStartPose(Popup3DRenderer& r, const char* presetName)
{
    if (r.anim)
        popupClipEngineStartPose(r.anim, presetName);
}

void popup3DRendererStopPose(Popup3DRenderer& r)
{
    if (r.anim)
        popupClipEngineStopPose(r.anim);
}

void popup3DRendererSetHideResting(Popup3DRenderer& r, bool hide)
{
    r.hideRestingModel = hide;
}

// -------------------------------------------------------
// Shutdown
// -------------------------------------------------------
void popup3DRendererShutdown(Popup3DRenderer& r)
{
    if (!r.initialized) return;

    for (int i = 0; i < Popup3DRenderer::kSlotCount; i++)
        destroySlotGPU(s_slotGPU[i]);

    if (bgfx::isValid(r.albedoTex)) { bgfx::destroy(r.albedoTex); r.albedoTex = BGFX_INVALID_HANDLE; }
    if (bgfx::isValid(r.normalTex)) { bgfx::destroy(r.normalTex); r.normalTex = BGFX_INVALID_HANDLE; }
    if (bgfx::isValid(r.packedTex)) { bgfx::destroy(r.packedTex); r.packedTex = BGFX_INVALID_HANDLE; }
    if (bgfx::isValid(s_whiteFallback)) { bgfx::destroy(s_whiteFallback); s_whiteFallback = BGFX_INVALID_HANDLE; }
    if (r.shaders)     { popupShadersDestroy(r.shaders);   r.shaders     = nullptr; }
    if (r.mesh)        { popupMeshDestroy(r.mesh);          r.mesh        = nullptr; }
    if (r.dynamicMesh) { popupMeshDestroy(r.dynamicMesh);   r.dynamicMesh = nullptr; }
    if (r.anim)        { popupClipEngineDestroy(r.anim);    r.anim        = nullptr; }

    r.initialized = false;
}

// -------------------------------------------------------
// Texture loading (stb_image → bgfx RGBA8 texture)
// -------------------------------------------------------
void popup3DRendererLoadTexture(Popup3DRenderer& r, const char* imagePath)
{
    if (!r.initialized)
        throw std::runtime_error("popup3DRendererLoadTexture: renderer not initialized");
    if (!imagePath)
        throw std::runtime_error("popup3DRendererLoadTexture: imagePath is NULL");

    int w = 0, h = 0, channels = 0;
    stbi_uc* pixels = stbi_load(imagePath, &w, &h, &channels, 4);  // force RGBA
    if (!pixels)
        throw std::runtime_error(std::string("popup3DRendererLoadTexture: failed to load ") + imagePath);

    // Destroy previous texture if any
    if (bgfx::isValid(r.albedoTex))
    {
        bgfx::destroy(r.albedoTex);
        r.albedoTex = BGFX_INVALID_HANDLE;
    }

    const bgfx::Memory* mem = bgfx::copy(pixels, static_cast<uint32_t>(w * h * 4));
    stbi_image_free(pixels);

    r.albedoTex = bgfx::createTexture2D(
        static_cast<uint16_t>(w),
        static_cast<uint16_t>(h),
        false, 1,
        bgfx::TextureFormat::RGBA8,
        BGFX_SAMPLER_U_CLAMP | BGFX_SAMPLER_V_CLAMP | BGFX_SAMPLER_MIN_ANISOTROPIC | BGFX_SAMPLER_MAG_ANISOTROPIC,
        mem
    );

    if (!bgfx::isValid(r.albedoTex))
        throw std::runtime_error(std::string("popup3DRendererLoadTexture: bgfx texture creation failed for ") + imagePath);
}

// -------------------------------------------------------
// Normal map loading (stb_image → bgfx RGBA8 texture)
// -------------------------------------------------------
void popup3DRendererLoadNormalMap(Popup3DRenderer& r, const char* imagePath)
{
    if (!r.initialized)
        throw std::runtime_error("popup3DRendererLoadNormalMap: renderer not initialized");
    if (!imagePath)
        throw std::runtime_error("popup3DRendererLoadNormalMap: imagePath is NULL");

    int w = 0, h = 0, channels = 0;
    stbi_uc* pixels = stbi_load(imagePath, &w, &h, &channels, 4);
    if (!pixels)
        throw std::runtime_error(std::string("popup3DRendererLoadNormalMap: failed to load ") + imagePath);

    if (bgfx::isValid(r.normalTex))
    {
        bgfx::destroy(r.normalTex);
        r.normalTex = BGFX_INVALID_HANDLE;
    }

    const bgfx::Memory* mem = bgfx::copy(pixels, static_cast<uint32_t>(w * h * 4));
    stbi_image_free(pixels);

    r.normalTex = bgfx::createTexture2D(
        static_cast<uint16_t>(w),
        static_cast<uint16_t>(h),
        false, 1,
        bgfx::TextureFormat::RGBA8,
        BGFX_SAMPLER_U_CLAMP | BGFX_SAMPLER_V_CLAMP | BGFX_SAMPLER_MIN_ANISOTROPIC | BGFX_SAMPLER_MAG_ANISOTROPIC,
        mem
    );

    if (!bgfx::isValid(r.normalTex))
        throw std::runtime_error(std::string("popup3DRendererLoadNormalMap: bgfx texture creation failed for ") + imagePath);
}

// -------------------------------------------------------
// Packed material map loading (stb_image → bgfx RGBA8 texture)
// -------------------------------------------------------
void popup3DRendererLoadPackedMap(Popup3DRenderer& r, const char* imagePath)
{
    if (!r.initialized)
        throw std::runtime_error("popup3DRendererLoadPackedMap: renderer not initialized");
    if (!imagePath)
        throw std::runtime_error("popup3DRendererLoadPackedMap: imagePath is NULL");

    int w = 0, h = 0, channels = 0;
    stbi_uc* pixels = stbi_load(imagePath, &w, &h, &channels, 4);
    if (!pixels)
        throw std::runtime_error(std::string("popup3DRendererLoadPackedMap: failed to load ") + imagePath);

    if (bgfx::isValid(r.packedTex))
    {
        bgfx::destroy(r.packedTex);
        r.packedTex = BGFX_INVALID_HANDLE;
    }

    const bgfx::Memory* mem = bgfx::copy(pixels, static_cast<uint32_t>(w * h * 4));
    stbi_image_free(pixels);

    r.packedTex = bgfx::createTexture2D(
        static_cast<uint16_t>(w),
        static_cast<uint16_t>(h),
        false, 1,
        bgfx::TextureFormat::RGBA8,
        BGFX_SAMPLER_U_CLAMP | BGFX_SAMPLER_V_CLAMP | BGFX_SAMPLER_MIN_ANISOTROPIC | BGFX_SAMPLER_MAG_ANISOTROPIC,
        mem
    );

    if (!bgfx::isValid(r.packedTex))
        throw std::runtime_error(std::string("popup3DRendererLoadPackedMap: bgfx texture creation failed for ") + imagePath);
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
