#include "popup_ui.hpp"
#include "popup_window.hpp"
#include "popup_renderer.hpp"
#include "popup_anim.hpp"
#include "logger.hpp"
#include "pch.hpp"
#include "voice/voice_speak.hpp"
#include "ui/console_ui.hpp"
#include <windows.h>
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
// Constants
// ===========================================================
constexpr uint16_t POPUP_SIZE = 256;

// ===========================================================
// Globals
// ===========================================================
extern std::mutex g_alphaMutex;
extern std::vector<uint8_t> g_alphaPixels;
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
    LOG_TRACE("PopupUI", "runPopupUI");
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

    ShowWindow(g_hwnd, SW_SHOW);
    LOG_DEBUG("PopupUI", "Overlay window created");

    // -------------------------------------------------------
    // Prepare alpha mask (RGBA from Oreo maps)
    // -------------------------------------------------------
    queueWindowAlphaReadback(POPUP_SIZE, POPUP_SIZE);

    // Wait until alpha data ready before applying transparency
    LOG_DEBUG("PopupUI", "Waiting for alpha data to become ready...");
    int safetyCounter = 0;
    while (!g_alphaReady.load() && safetyCounter < 200) // up to 2 seconds
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        safetyCounter++;
    }

    if (g_alphaReady.load())
    {
        LOG_DEBUG("PopupUI", "Alpha data ready — applying transparency");
        applyWindowAlphaIfReady(g_hwnd, POPUP_SIZE, POPUP_SIZE, 0);
    }
    else
    {
        LOG_ERROR("PopupUI", "Alpha data not ready in time — popup may not be transparent");
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
    // Main popup loop
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
                GRIMConsole::showConsole();
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
                if (!Voice::isPlaying())
                {
                    hidePopup();
                    g_idleTimerMs = 0;
                }
                else
                {
                    g_idleStart = std::chrono::steady_clock::now();
                }
            }
        }

        // ---------------------------------------------------
        // Animation logic - voice-reactive
        // ---------------------------------------------------
        bool isSpeaking = Voice::isSpeaking();
        
        // Update visibility animation
        if (g_popupVisible)
            updateAnim(g_anim, g_popupVisible, dt, 0.08f);
        
        // Update voice-reactive animation (pulse, breathe, scale)
        updateVoiceAnim(g_anim, isSpeaking, dt);
        
        // Apply animation to window visuals (scale, alpha)
        if (g_popupVisible && g_hwnd)
        {
            applyAnimationToWindow(g_hwnd, POPUP_SIZE, POPUP_SIZE, 
                                   g_anim.scale, g_anim.alpha);
            
            // Debug logging (every 60 frames = ~1 second)
            static int frameCounter = 0;
            if (++frameCounter >= 60)
            {
                LOG_DEBUG("PopupAnim", "scale=" + std::to_string(g_anim.scale) + 
                          ", alpha=" + std::to_string(g_anim.alpha) +
                          ", pulse=" + std::to_string(g_anim.pulse) +
                          ", intensity=" + std::to_string(g_anim.voiceIntensity) +
                          ", speaking=" + std::to_string(isSpeaking));
                frameCounter = 0;
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

    PostMessage(g_hwnd, WM_GRIM_SHOW_POPUP, 0, 0);
    g_idleTimerMs = 3000;
    g_idleStart = std::chrono::steady_clock::now();
    LOG_DEBUG("PopupUI", "Idle timer reset to " + std::to_string(g_idleTimerMs) + "ms");
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
