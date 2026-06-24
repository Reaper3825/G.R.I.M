#include "core/grim_platform.h"
#ifdef _WIN32

#include "popup_ui.hpp"
#include "popup_window.hpp"
#include "popup_anim.hpp"
#include "popup_3d_renderer.hpp"
#include "popup_3d_mailbox.hpp"
#include "objects/popup_obj_loader.hpp"
#include "logger.hpp"
#include "pch.hpp"
#include "voice/voice_speak.hpp"
#include "ui/ui_root.hpp"
#include "core/grim_platform.h"
#include <atomic>
#include <thread>
#include <chrono>
#include "core/window_manager.hpp"
#include "helpers/mouse.hpp"
#include "core/ui_sync.hpp"

// ===========================================================
// Forward declarations
// ===========================================================
void showPopup();
void hidePopup();

// ===========================================================
// Globals
// ===========================================================
static HWND g_hwnd = nullptr;
static PopupAnimState g_anim;
static std::atomic<bool> g_running{ true };
static std::atomic<bool> g_popupVisible{ false };
static std::atomic<bool> g_pendingPopup{ false };
static std::atomic<int> g_idleTimerMs{ 0 };

// Animation preset requests (consumed by the popup loop)
static std::atomic<bool> g_showRequested{ false };  // play "load_in"
static std::atomic<bool> g_hideRequested{ false };  // play "fade_out", then hide


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
    popup3DRendererSubmit(g_popup3D, g_popup3DInput, bgfxFrame);
}

// ===========================================================
// Popup UI main thread (logic only — no BGFX here)
// ===========================================================
void runPopupUI(int width, int height)
{
    LOG_TRACE("PopupUI", "runPopupUI with size " + std::to_string(width) + "x" + std::to_string(height));
    if (WindowManager::isInitialized()) {
        LOG_DEBUG("PopupUI", "Deferring popup creation until BGFX idle");
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    // Initialize COM for this thread
    HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    if (FAILED(hr)) {
        LOG_ERROR("PopupUI", "CoInitializeEx failed: " + std::to_string(hr));
        return;
    }

    std::lock_guard<std::mutex> guard(g_uiSafeZone);  // existing mutex
    g_hwnd = createOverlayWindow(width, height);
    if (!g_hwnd)
    {
        CoUninitialize();
        return;
    }

    LOG_DEBUG("PopupUI", "Popup thread ID = " + std::to_string(GetCurrentThreadId()));

    // Register existing HWND with WindowManager manually
    auto win = std::make_unique<GRIMWindow>();
    win->hwnd = g_hwnd;
    win->name = "popup";
    win->visible = true;
    win->isOverlay = true;
    win->width = width;
    win->height = height;
    WindowManager::registerWindow(std::move(win));

    // -------------------------------------------------------
    // Initialize 3D renderer
    // Must happen after BGFX is initialized (WindowManager)
    // -------------------------------------------------------
    if (WindowManager::isInitialized())
    {
        try
        {
            std::string objPath = std::string(GRIM_ROOT_DIR) + "/resources/popup_3d/grim_popup.obj";
            PopupObjectDefinition grimDef = loadPopupObjectFromOBJ(objPath);
            popup3DRendererInit(g_popup3D, grimDef,
                                static_cast<uint32_t>(width),
                                static_cast<uint32_t>(height));

            // Set initial render input
            g_popup3DInput.transform = grimDef.defaultTransform;
            g_popup3DInput.light     = grimDef.defaultLight;
            g_popup3DInput.alphaMul  = 1.0f;
            g_popup3DInput.width     = static_cast<uint32_t>(width);
            g_popup3DInput.height    = static_cast<uint32_t>(height);
            g_popup3DInput.visible   = true;

            // Load animation presets (built-ins + optional JSON overrides) and
            // any baked geometry-node mesh caches they reference.
            {
                std::string popup3dDir = std::string(GRIM_ROOT_DIR) + "/resources/popup_3d";
                std::string presetsJson = popup3dDir + "/anim_presets.json";
                popup3DRendererInitAnim(g_popup3D, presetsJson, popup3dDir);
            }

            // Register pre-frame callback so main thread renders the cube
            WindowManager::registerPreFrameCallback(popup3DPreFrameCallback);
            g_popup3DInitialized = true;

            // Play the load-in preset on first appearance.
            g_showRequested = true;

            // Show window now that 3D renderer is ready
            ShowWindow(g_hwnd, SW_SHOW);
            UpdateWindow(g_hwnd);
            g_popupVisible = true;
            g_idleTimerMs = 10000;
            g_idleStart = std::chrono::steady_clock::now();

            LOG_PHASE("Popup 3D renderer initialized (GRIM object)", true);
        }
        catch (const std::exception& e)
        {
            LOG_ERROR("PopupUI", std::string("Failed to init 3D renderer: ") + e.what());
        }
    }
    else
    {
        LOG_ERROR("PopupUI", "BGFX not initialized — cannot start 3D renderer");
        CoUninitialize();
        return;
    }

    // -------------------------------------------------------
    // Handle queued popup show requests
    // -------------------------------------------------------
    if (g_pendingPopup)
    {
        showPopup();
        g_idleTimerMs = 10000;
        g_idleStart = std::chrono::steady_clock::now();
        g_pendingPopup = false;
    }

    MSG msg{};
    g_frameStart = std::chrono::steady_clock::now();

    LOG_PHASE("Popup UI Thread Started", true);

    // ======================================================
    // Main popup loop - use the actual window dimensions!
    // ======================================================
    while (g_running)
    {
        // ---------------------------------------------------
        // Win32 message loop
        // ---------------------------------------------------
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            if (msg.message == WM_QUIT)
            {
                LOG_DEBUG("PopupUI", "WM_QUIT received — exiting popup thread");
                g_running = false;
            }
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        // ---------------------------------------------------
        // Mouse input handling (using Mouse class)
        // ---------------------------------------------------
        if (Mouse::wasPressed(MouseButton::Left))
        {
            POINT cursorPos = Mouse::getPosition();
            HWND hwndUnderCursor = WindowFromPoint(cursorPos);

            // Only trigger if the popup window itself was clicked
            if (hwndUnderCursor == g_hwnd)
            {
                LOG_DEBUG("PopupUI", "Popup clicked — showing GRIM console");
                auto consolePanel = UIRoot::get().getPanel("Console");
                if (consolePanel) {
                    consolePanel->setVisible(true);
                    LOG_DEBUG("PopupUI", "Console panel shown");
                }
            }
        }
        Mouse::endFrame();

        // ---------------------------------------------------
        // Idle timer logic
        // ---------------------------------------------------
        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<float> dtSec = now - g_frameStart;
        float dt = dtSec.count();
        g_frameStart = now;

        if (g_idleTimerMs > 0)
        {
            auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_idleStart).count();
            if (elapsedMs > g_idleTimerMs)
            {
                // Don't hide if currently speaking OR if voice intensity is high
                bool isSpeaking = Voice::isSpeaking();
                bool voiceActive = g_anim.voiceIntensity > 0.1f; // Still animating from recent speech
                
                if (!Voice::isPlaying() && !isSpeaking && !voiceActive)
                {
                    hidePopup();
                    g_idleTimerMs = 0;
                }
                else
                {
                    // Keep resetting timer while voice is active
                    g_idleStart = std::chrono::steady_clock::now();
                    LOG_DEBUG("PopupUI", "Voice still active - delaying hide");
                }
            }
        }

        // ---------------------------------------------------
        // Animation preset triggers (load_in / fade_out)
        // ---------------------------------------------------
        {
            static bool sFadingOut = false;
            static std::chrono::steady_clock::time_point sFadeStart;
            constexpr float kFadeOutSec = 0.30f;  // must match the "fade_out" preset duration

            // Show: cancel any fade and play load-in.
            if (g_showRequested.exchange(false))
            {
                sFadingOut = false;
                if (g_popup3DInitialized.load())
                    popup3DRendererTriggerPreset(g_popup3D, "load_in");
            }

            // Hide: play fade-out, then actually hide the window once it completes.
            if (g_hideRequested.load())
            {
                if (!sFadingOut)
                {
                    sFadingOut = true;
                    sFadeStart = now;
                    if (g_popup3DInitialized.load())
                        popup3DRendererTriggerPreset(g_popup3D, "fade_out");
                }
                else if (std::chrono::duration<float>(now - sFadeStart).count() >= kFadeOutSec)
                {
                    if (g_hwnd) ShowWindow(g_hwnd, SW_HIDE);
                    g_popupVisible = false;
                    g_popup3DInput.visible = false;  // stop offscreen submission
                    WindowManager::setVisibility("popup", false);
                    sFadingOut = false;
                    g_hideRequested = false;
                }
            }
        }

        // ---------------------------------------------------
        // Animation logic - voice-reactive (OPTIMIZED)
        // Use actual window dimensions!
        // ---------------------------------------------------
        bool isSpeaking = Voice::isSpeaking();
        
        // Update visibility animation
        if (g_popupVisible)
            updateAnim(g_anim, g_popupVisible, dt, 0.08f);
        
        // Update voice-reactive animation (pulse, breathe, scale)
        updateVoiceAnim(g_anim, isSpeaking, dt);
        
        // Apply animation to window visuals at 60 FPS
        if (g_popupVisible && g_hwnd)
        {
            // ---------------------------------------------------
            // 3D renderer: update rotation + consume readback frame
            // ---------------------------------------------------
            g_popup3DInput.transform.rotation[0] = 0.15f;  // subtle tilt
            g_popup3DInput.transform.rotation[1] = 0.0f;

            // Breathing scale: gentle 2% oscillation
            float breatheScale = 1.0f + g_anim.breathe * 0.02f;
            // Voice pulse: up to 8% scale boost when speaking
            float voiceScale = 1.0f + g_anim.pulse * 0.08f;
            float finalScale = breatheScale * voiceScale;
            // Note: the load_in / fade_out presets layer their own scale + alpha
            // multiplicatively on top of these (evaluated render-side), so we keep
            // the per-frame baseline here to just the voice/breathe reaction.
            g_popup3DInput.transform.scale[0] = finalScale;
            g_popup3DInput.transform.scale[1] = finalScale;
            g_popup3DInput.transform.scale[2] = finalScale;

            g_popup3DInput.alphaMul    = 1.0f;  // alpha is driven by the active preset clip
            g_popup3DInput.emissiveMul = g_anim.voiceIntensity * 0.5f;
            g_popup3DInput.visible     = g_popupVisible.load();

            // Try to consume a rendered frame from the mailbox
            static std::vector<uint8_t> frameBuffer;
            static uint64_t lastGen = 0;
            uint32_t fw = 0, fh = 0;
            if (popupMailboxConsume(g_popup3D.mailbox, frameBuffer, fw, fh, lastGen))
            {
                presentPopup3DFrame(g_hwnd, frameBuffer.data(),
                                    static_cast<int>(fw),
                                    static_cast<int>(fh));
            }
        }

        // Show popup automatically when voice starts speaking
        if (isSpeaking && !g_popupVisible)
        {
            LOG_DEBUG("PopupUI", "Voice started speaking - showing popup");
            showPopup();
            g_idleTimerMs = 5000; // Stay visible for 5s after speech
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

    LOG_PHASE("Popup UI Thread Exited", true);
    CoUninitialize();
}

// ===========================================================
// Popup Controls
// ===========================================================
void showPopup()
{
    if (g_hwnd)
    {
        ShowWindow(g_hwnd, SW_SHOW);
        SetForegroundWindow(g_hwnd);
        g_popupVisible = true;
        g_hideRequested = false;  // cancel any in-flight fade-out
        g_showRequested = true;   // play the load-in preset
        LOG_PHASE("PopupUI shown", true);
        WindowManager::setVisibility("popup", true);
    }
}

void hidePopup()
{
    if (g_hwnd)
    {
        // Request a fade-out; the popup loop plays the "fade_out" preset and
        // performs the actual SW_HIDE once the clip finishes. The window keeps
        // rendering during the fade.
        g_hideRequested = true;
    }
}

void notifyPopupActivity()
{
    LOG_DEBUG("PopupUI", "notifyPopupActivity() called");

    if (!g_hwnd)
    {
        g_pendingPopup = true;
        LOG_DEBUG("PopupUI", "HWND not ready — popup queued");
        return;
    }

    showPopup();
    
    g_idleTimerMs = 10000;
    g_idleStart = std::chrono::steady_clock::now();
}

// Get current animation state (for rendering)
PopupAnimState getPopupAnimState()
{
    return g_anim;
}

// Get animation values (convenience functions)
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

#endif // _WIN32
