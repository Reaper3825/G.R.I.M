#include "core/grim_platform.h"
#ifndef _WIN32

#include "popup_ui.hpp"
#include "popup_window.hpp"
#include "popup_anim.hpp"
#include "popup_3d_renderer.hpp"
#include "popup_3d_mailbox.hpp"
#include "objects/cube_test_object.hpp"
#include "logger.hpp"
#include "voice/voice_speak.hpp"
#include "ui/ui_root.hpp"
#include "core/window_manager.hpp"
#include <atomic>
#include <thread>
#include <chrono>

// ===========================================================
// Globals (matching Win32 pop_ui.cpp)
// ===========================================================
extern std::mutex g_alphaMutex;
extern std::vector<uint8_t> g_popupPixels;
extern std::atomic<bool> g_alphaReady;

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
static float g_cubeRotationY = 0.0f;

static std::atomic<uint32_t> s_preFrameCallCount{ 0 };

static void popup3DPreFrameCallback(uint32_t bgfxFrame)
{
    uint32_t count = s_preFrameCallCount.fetch_add(1);
    if (!g_popup3DInitialized.load())
    {
        if (count < 3)
        {
            LOG_DEBUG("PopupUI", "preFrameCallback #" + std::to_string(count) +
                      " but 3D NOT initialized — skipping");
        }
        return;
    }
    if (count < 5 || count % 60 == 0)
    {
        LOG_DEBUG("PopupUI", "preFrameCallback #" + std::to_string(count) +
                  " bgfxFrame=" + std::to_string(bgfxFrame) +
                  " visible=" + std::to_string(g_popup3DInput.visible));
    }
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
                PopupObjectDefinition cubeDef = createCubeTestObject();
                popup3DRendererInit(g_popup3D, cubeDef,
                                    static_cast<uint32_t>(width),
                                    static_cast<uint32_t>(height));

                // Set initial render input
                g_popup3DInput.transform = cubeDef.defaultTransform;
                g_popup3DInput.light     = cubeDef.defaultLight;
                g_popup3DInput.alphaMul  = 1.0f;
                g_popup3DInput.width     = static_cast<uint32_t>(width);
                g_popup3DInput.height    = static_cast<uint32_t>(height);
                g_popup3DInput.visible   = true;

                // Register pre-frame callback so main thread renders the cube
                WindowManager::registerPreFrameCallback(popup3DPreFrameCallback);
                g_popup3DInitialized = true;

                LOG_DEBUG("PopupUI", "Popup 3D renderer initialized (cube test, macOS)");
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
                    LOG_DEBUG("PopupUI", "Idle timeout reached - hiding popup");
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
                g_cubeRotationY += dt * 0.8f;
                g_popup3DInput.transform.rotation[0] = 0.3f;  // slight tilt
                g_popup3DInput.transform.rotation[1] = g_cubeRotationY;
                g_popup3DInput.visible = g_popupVisible.load();

                // Consume rendered frame from the mailbox
                static std::vector<uint8_t> frameBuffer;
                static uint64_t lastGen = 0;
                static uint32_t consumeAttempts = 0;
                static uint32_t consumeSuccesses = 0;
                uint32_t fw = 0, fh = 0;
                consumeAttempts++;
                if (popupMailboxConsume(g_popup3D.mailbox, frameBuffer, fw, fh, lastGen))
                {
                    consumeSuccesses++;
                    if (consumeSuccesses <= 3 || consumeSuccesses % 60 == 0)
                    {
                        LOG_DEBUG("PopupUI", "Mailbox consumed frame #" +
                                  std::to_string(consumeSuccesses) +
                                  " gen=" + std::to_string(lastGen) +
                                  " size=" + std::to_string(fw) + "x" + std::to_string(fh) +
                                  " bytes=" + std::to_string(frameBuffer.size()));
                    }
                    presentPopup3DFrame(g_popupHandle, frameBuffer.data(),
                                        static_cast<int>(fw),
                                        static_cast<int>(fh));
                }
                else if (consumeAttempts <= 5 || consumeAttempts % 120 == 0)
                {
                    LOG_DEBUG("PopupUI", "Mailbox empty (attempt #" +
                              std::to_string(consumeAttempts) +
                              ", successes=" + std::to_string(consumeSuccesses) + ")");
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
        LOG_DEBUG("PopupUI", "PopupUI hidden");
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
