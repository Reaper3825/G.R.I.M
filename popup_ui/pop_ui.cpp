#include "popup_ui.hpp"
#include "popup_window.hpp"
#include "popup_renderer.hpp"
#include "popup_anim.hpp"
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
extern std::mutex g_alphaMutex;
extern std::vector<uint8_t> g_popupPixels;
extern std::atomic<bool> g_alphaReady;

static HWND g_hwnd = nullptr;
static PopupAnimState g_anim;
static std::atomic<bool> g_running{ true };
static std::atomic<bool> g_popupVisible{ false };
static std::atomic<bool> g_pendingPopup{ false };
static std::atomic<int> g_idleTimerMs{ 0 };


static std::chrono::steady_clock::time_point g_idleStart = std::chrono::steady_clock::now();
static std::chrono::steady_clock::time_point g_frameStart = std::chrono::steady_clock::now();

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
    // Prepare alpha mask (RGBA from Oreo maps) BEFORE showing
    // Use the actual window size, not POPUP_SIZE constant!
    // -------------------------------------------------------
    LOG_DEBUG("PopupUI", "Loading alpha textures for " + std::to_string(width) + "x" + std::to_string(height));
    queueWindowAlphaReadback(width, height);

    // Wait until alpha data ready before showing window
    LOG_DEBUG("PopupUI", "Waiting for alpha data to become ready...");
    int safetyCounter = 0;
    while (!g_alphaReady.load() && safetyCounter < 200) // up to 2 seconds
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        safetyCounter++;
    }

    if (g_alphaReady.load())
    {
        LOG_DEBUG("PopupUI", "Alpha data ready — applying initial transparency");
        applyWindowAlphaIfReady(g_hwnd, width, height, 0);
        
        ShowWindow(g_hwnd, SW_SHOW);
        UpdateWindow(g_hwnd);
        
        g_popupVisible = true;
        g_idleTimerMs = 10000;
        g_idleStart = std::chrono::steady_clock::now();
        
        LOG_PHASE("PopupUI shown with alpha", true);
        
        // Apply initial animation frame
        applyAnimationToWindow(g_hwnd, width, height, 1.0f, 1.0f, g_anim.voiceIntensity);
    }
    else
    {
        LOG_ERROR("PopupUI", "Alpha data not ready in time — showing without transparency");
        ShowWindow(g_hwnd, SW_SHOW);
        g_popupVisible = true;
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
                    LOG_DEBUG("PopupUI", "Idle timeout reached - hiding popup");
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
            applyAnimationToWindow(g_hwnd, width, height,
                                   g_anim.scale, g_anim.alpha, g_anim.voiceIntensity);
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
        LOG_PHASE("PopupUI shown", true);
        WindowManager::setVisibility("popup", true);
    }
}

void hidePopup()
{
    if (g_hwnd)
    {
        ShowWindow(g_hwnd, SW_HIDE);
        g_popupVisible = false;
        LOG_PHASE("PopupUI hidden", true);
        WindowManager::setVisibility("popup", false);
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
