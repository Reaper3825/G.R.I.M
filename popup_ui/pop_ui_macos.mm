#include "core/grim_platform.h"
#ifndef _WIN32

#include "popup_ui.hpp"
#include "popup_window.hpp"
#include "popup_anim.hpp"
#include "popup_3d_renderer.hpp"
#include "popup_3d_mailbox.hpp"
#include "objects/popup_obj_loader.hpp"
#include "logger.hpp"
#include "voice/voice_speak.hpp"
#include "ui/ui_root.hpp"
#include "core/window_manager.hpp"
#include <atomic>
#include <thread>
#include <chrono>

// ===========================================================
// Globals
// ===========================================================
static void* g_popupHandle = nullptr;
static PopupAnimState g_anim;
static std::atomic<bool> g_running{ true };
static std::atomic<bool> g_popupVisible{ false };
static std::atomic<bool> g_pendingPopup{ false };
static std::atomic<int> g_idleTimerMs{ 0 };

static std::chrono::steady_clock::time_point g_idleStart = std::chrono::steady_clock::now();
static std::chrono::steady_clock::time_point g_frameStart = std::chrono::steady_clock::now();

// ===========================================================
// 3D Renderer State
// ===========================================================
static Popup3DRenderer g_popup3D;
static PopupRenderInput g_popup3DInput;
static std::atomic<bool> g_popup3DInitialized{ false };

static void popup3DPreFrameCallback(uint32_t bgfxFrame)
{
    if (!g_popup3DInitialized.load())
        return;

    static int sCallCount = 0;
    if (++sCallCount <= 3)
        LOG_DEBUG("PopupUI", "preFrameCallback called: bgfxFrame=" + std::to_string(bgfxFrame) +
                  " call#" + std::to_string(sCallCount));

    popup3DRendererSubmit(g_popup3D, g_popup3DInput, bgfxFrame);
}

// ===========================================================
// Popup UI main thread (macOS version)
// ===========================================================
void runPopupUI(int width, int height)
{
    LOG_DEBUG("PopupUI", "runPopupUI(macOS) with size " +
              std::to_string(width) + "x" + std::to_string(height));

    g_popupHandle = createPopupWindow(width, height);
    if (!g_popupHandle) {
        LOG_ERROR("PopupUI", "Failed to create macOS popup window");
        return;
    }

    // Register with WindowManager
    auto win = std::make_unique<GRIMWindow>();
    win->hwnd = static_cast<HWND>(g_popupHandle);
    win->name = "popup";
    win->visible = true;
    win->isOverlay = true;
    win->width = width;
    win->height = height;
    WindowManager::registerWindow(std::move(win));

    // Show the popup window
    showPopupWindow(g_popupHandle);
    g_popupVisible = true;
    g_idleTimerMs = 10000;
    g_idleStart = std::chrono::steady_clock::now();

    LOG_DEBUG("PopupUI", "PopupUI shown (macOS)");

    // -------------------------------------------------------
    // Initialize 3D renderer (cube test object)
    // Must happen after BGFX is initialized (WindowManager)
    // Poll with timeout — popup thread may start before BGFX
    // -------------------------------------------------------
    {
        constexpr int kMaxWaitMs = 10000;
        constexpr int kPollIntervalMs = 50;
        int waited = 0;
        while (!WindowManager::isInitialized() && waited < kMaxWaitMs)
        {
            if (waited == 0)
                LOG_DEBUG("PopupUI", "Waiting for BGFX initialization...");
            std::this_thread::sleep_for(std::chrono::milliseconds(kPollIntervalMs));
            waited += kPollIntervalMs;
        }

        if (WindowManager::isInitialized())
        {
            if (waited > 0)
                LOG_DEBUG("PopupUI", "BGFX ready after " + std::to_string(waited) + "ms wait");

            try
            {
                std::string objPath = std::string(GRIM_ROOT_DIR) + "/resources/popup_3d/grim_popup.obj";
                PopupObjectDefinition grimDef = loadPopupObjectFromOBJ(objPath);
                LOG_DEBUG("PopupUI", "OBJ loaded: verts=" + std::to_string(grimDef.vertices.size()) +
                          " indices=" + std::to_string(grimDef.indices.size()) +
                          " from " + objPath);
                popup3DRendererInit(g_popup3D, grimDef,
                                    static_cast<uint32_t>(width),
                                    static_cast<uint32_t>(height));

                // Try to load material textures (fallback to defaults on failure)
                {
                    std::string base = std::string(GRIM_ROOT_DIR) + "/resources/popup_3d/";

                    try {
                        std::string albedoPath = base + "grim_popup_albedo.png";
                        popup3DRendererLoadTexture(g_popup3D, albedoPath.c_str());
                        LOG_DEBUG("PopupUI", "Loaded albedo: " + albedoPath);
                    } catch (const std::exception& e) {
                        LOG_DEBUG("PopupUI", std::string("Albedo not loaded (using default): ") + e.what());
                    }

                    try {
                        std::string normalPath = base + "grim_popup_normal.png";
                        popup3DRendererLoadNormalMap(g_popup3D, normalPath.c_str());
                        LOG_DEBUG("PopupUI", "Loaded normal map: " + normalPath);
                    } catch (const std::exception& e) {
                        LOG_DEBUG("PopupUI", std::string("Normal map not loaded (using default): ") + e.what());
                    }

                    try {
                        std::string packedPath = base + "grim_popup_packed.png";
                        popup3DRendererLoadPackedMap(g_popup3D, packedPath.c_str());
                        LOG_DEBUG("PopupUI", "Loaded packed map: " + packedPath);
                    } catch (const std::exception& e) {
                        LOG_DEBUG("PopupUI", std::string("Packed map not loaded (using default): ") + e.what());
                    }
                }

                // Set initial render input
                g_popup3DInput.transform = grimDef.defaultTransform;
                g_popup3DInput.light     = grimDef.defaultLight;
                g_popup3DInput.alphaMul  = 1.0f;
                g_popup3DInput.width     = static_cast<uint32_t>(width);
                g_popup3DInput.height    = static_cast<uint32_t>(height);
                g_popup3DInput.visible   = true;

                // Register pre-frame callback so main thread renders the cube
                WindowManager::registerPreFrameCallback(popup3DPreFrameCallback);
                g_popup3DInitialized = true;

                LOG_DEBUG("PopupUI", "Popup 3D renderer initialized (GRIM object, macOS)");
            }
            catch (const std::exception& e)
            {
                LOG_ERROR("PopupUI", std::string("Failed to init 3D renderer: ") + e.what());
            }
        }
        else
        {
            LOG_ERROR("PopupUI", "BGFX not initialized after " + std::to_string(kMaxWaitMs) +
                      "ms — 3D renderer SKIPPED");
        }
    }

    // Handle queued popup show requests
    if (g_pendingPopup)
    {
        showPopup();
        g_idleTimerMs = 10000;
        g_idleStart = std::chrono::steady_clock::now();
        g_pendingPopup = false;
    }

    g_frameStart = std::chrono::steady_clock::now();
    LOG_DEBUG("PopupUI", "Popup UI Thread Started (macOS)");

    // ======================================================
    // Main popup loop
    // ======================================================
    while (g_running)
    {
        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<float> dtSec = now - g_frameStart;
        float dt = dtSec.count();
        g_frameStart = now;

        // ---------------------------------------------------
        // Idle timer logic
        // ---------------------------------------------------
        if (g_idleTimerMs > 0)
        {
            auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - g_idleStart).count();
            if (elapsedMs > g_idleTimerMs)
            {
                bool isSpeaking = Voice::isSpeaking();
                bool voiceActive = g_anim.voiceIntensity > 0.1f;

                if (!Voice::isPlaying() && !isSpeaking && !voiceActive)
                {
                    hidePopup();
                    g_idleTimerMs = 0;
                }
                else
                {
                    g_idleStart = std::chrono::steady_clock::now();
                    LOG_DEBUG("PopupUI", "Voice still active - delaying hide");
                }
            }
        }

        // ---------------------------------------------------
        // Animation logic
        // ---------------------------------------------------
        bool isSpeaking = Voice::isSpeaking();

        if (g_popupVisible)
            updateAnim(g_anim, g_popupVisible, dt, 0.08f);

        updateVoiceAnim(g_anim, isSpeaking, dt);

        // ---------------------------------------------------
        // 3D renderer: update rotation + consume readback frame
        // ---------------------------------------------------
        if (g_popupVisible && g_popupHandle)
        {
            if (g_popup3DInitialized.load())
            {
                // Idle rotation + voice-reactive animation
                g_popup3DInput.transform.rotation[0] = 0.15f;  // subtle tilt
                g_popup3DInput.transform.rotation[1] = 0.0f;

                // Breathing scale: gentle 2% oscillation
                float breatheScale = 1.0f + g_anim.breathe * 0.02f;
                // Voice pulse: up to 8% scale boost when speaking
                float voiceScale = 1.0f + g_anim.pulse * 0.08f;
                float finalScale = breatheScale * voiceScale;
                g_popup3DInput.transform.scale[0] = finalScale;
                g_popup3DInput.transform.scale[1] = finalScale;
                g_popup3DInput.transform.scale[2] = finalScale;

                g_popup3DInput.alphaMul   = g_anim.alpha;
                g_popup3DInput.emissiveMul = g_anim.voiceIntensity * 0.5f;
                g_popup3DInput.visible = g_popupVisible.load();

                // Consume rendered frame from the mailbox
                static std::vector<uint8_t> frameBuffer;
                static uint64_t lastGen = 0;
                static int sConsumeAttempts = 0;
                static int sConsumeHits = 0;
                uint32_t fw = 0, fh = 0;
                ++sConsumeAttempts;
                if (popupMailboxConsume(g_popup3D.mailbox, frameBuffer, fw, fh, lastGen))
                {
                    ++sConsumeHits;
                    if (sConsumeHits <= 5)
                    {
                        // DIAG: scan consumed pixels
                        size_t total = static_cast<size_t>(fw) * fh;
                        size_t nzA = 0, nzRGB = 0;
                        uint8_t mxR=0,mxG=0,mxB=0,mxA=0;
                        const uint8_t* px = frameBuffer.data();
                        for (size_t ii = 0; ii < total; ++ii) {
                            uint8_t b=px[ii*4+0],g2=px[ii*4+1],r2=px[ii*4+2],a=px[ii*4+3];
                            if(a>0)++nzA; if(r2>0||g2>0||b>0)++nzRGB;
                            if(r2>mxR)mxR=r2;if(g2>mxG)mxG=g2;if(b>mxB)mxB=b;if(a>mxA)mxA=a;
                        }
                        size_t ci = (fh/2*fw+fw/2)*4;
                        LOG_DEBUG("PopupUI", "DIAG-CONSUME hit#" + std::to_string(sConsumeHits)
                            + " " + std::to_string(fw) + "x" + std::to_string(fh)
                            + " gen=" + std::to_string(lastGen)
                            + " nzA=" + std::to_string(nzA)
                            + " nzRGB=" + std::to_string(nzRGB)
                            + " max=(" + std::to_string(mxR)+","+std::to_string(mxG)+","+std::to_string(mxB)+","+std::to_string(mxA)+")"
                            + " center=[" + std::to_string(px[ci])+" "+std::to_string(px[ci+1])+" "+std::to_string(px[ci+2])+" "+std::to_string(px[ci+3])+"]");
                    }
                    presentPopup3DFrame(g_popupHandle, frameBuffer.data(),
                                        static_cast<int>(fw),
                                        static_cast<int>(fh));
                }
                else if (sConsumeAttempts <= 10 || (sConsumeAttempts % 300 == 0))
                {
                    LOG_DEBUG("PopupUI", "Mailbox empty: attempt#" + std::to_string(sConsumeAttempts) +
                              " totalHits=" + std::to_string(sConsumeHits));
                }
            }
        }

        // Show popup automatically when voice starts speaking
        if (isSpeaking && !g_popupVisible)
        {
            LOG_DEBUG("PopupUI", "Voice started speaking - showing popup");
            showPopup();
            g_idleTimerMs = 5000;
            g_idleStart = std::chrono::steady_clock::now();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }

    // Shutdown 3D renderer before exiting
    if (g_popup3DInitialized.load())
    {
        WindowManager::registerPreFrameCallback(nullptr);
        popup3DRendererShutdown(g_popup3D);
        g_popup3DInitialized = false;
    }

    LOG_DEBUG("PopupUI", "Popup UI Thread Exited (macOS)");
}

// ===========================================================
// Popup Controls
// ===========================================================
void showPopup()
{
    if (g_popupHandle)
    {
        showPopupWindow(g_popupHandle);
        g_popupVisible = true;
        LOG_DEBUG("PopupUI", "PopupUI shown — 3DInit=" +
                  std::to_string(g_popup3DInitialized.load()) +
                  " hasPreFrame=" + std::to_string(WindowManager::hasPreFrameCallback()) +
                  " bgfxInit=" + std::to_string(WindowManager::isInitialized()) +
                  " handle=" + std::to_string((uintptr_t)g_popupHandle));
        WindowManager::setVisibility("popup", true);
    }
    else
    {
        LOG_DEBUG("PopupUI", "showPopup() called but g_popupHandle is null");
    }
}

void hidePopup()
{
    if (g_popupHandle)
    {
        hidePopupWindow(g_popupHandle);
        g_popupVisible = false;
        WindowManager::setVisibility("popup", false);
    }
}

void notifyPopupActivity()
{
    LOG_DEBUG("PopupUI", "notifyPopupActivity() called");

    if (!g_popupHandle)
    {
        g_pendingPopup = true;
        LOG_DEBUG("PopupUI", "Popup handle not ready — popup queued");
        return;
    }

    showPopup();

    g_idleTimerMs = 10000;
    g_idleStart = std::chrono::steady_clock::now();
}

PopupAnimState getPopupAnimState()
{
    return g_anim;
}

float getPopupAlpha()
{
    return g_anim.alpha;
}

float getPopupScale()
{
    return g_anim.scale;
}

float getPopupPulse()
{
    return g_anim.pulse;
}

bool isPopupVisible()
{
    return g_popupVisible.load();
}

#endif // !_WIN32
