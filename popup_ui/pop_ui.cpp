#include "popup_ui.hpp"
#include "popup_window.hpp"
#include "popup_renderer.hpp"
#include "popup_anim.hpp"
#include "logger.hpp"
#include "pch.hpp"
#include <SFML/System/Clock.hpp>
#include <SFML/System/Time.hpp>
#include "voice/voice_speak.hpp"
#include <windows.h>
#include <cstdint>
#include <bx/math.h>
#include <bgfx/bgfx.h>
#include <bgfx/platform.h>
#include "core/window_manager.hpp"

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
static sf::Clock g_idleClock;
static std::atomic<bool> g_running{ true };
static std::atomic<bool> g_popupVisible{ false };
static std::atomic<bool> g_pendingPopup{ false };
static std::atomic<int> g_idleTimerMs{ 0 };

// ===========================================================
// Popup UI Thread
// ===========================================================
void runPopupUI(int width, int height)
{
    g_hwnd = createOverlayWindow(POPUP_SIZE, POPUP_SIZE);
    if (!g_hwnd)
        return;

    WindowManager::createOverlay("popup", POPUP_SIZE, POPUP_SIZE, true);

    ShowWindow(g_hwnd, SW_SHOW);
    LOG_DEBUG("PopupUI", "Overlay window created");

    // Apply alpha data if preloaded
    queueWindowAlphaReadback(POPUP_SIZE, POPUP_SIZE);
    applyWindowAlphaIfReady(g_hwnd, POPUP_SIZE, POPUP_SIZE, 0);

    if (g_pendingPopup)
    {
        showPopup();
        g_idleTimerMs = 10000;
        g_idleClock.restart();
        g_pendingPopup = false;
    }

    MSG msg{};
    sf::Clock frameClock;

    LOG_PHASE("Popup UI Thread Started", true);

    while (g_running)
    {
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

        float dt = frameClock.restart().asSeconds();

        if (g_idleTimerMs > 0 && g_idleClock.getElapsedTime().asMilliseconds() > g_idleTimerMs)
        {
            if (!Voice::isPlaying())
            {
                hidePopup();
                g_idleTimerMs = 0;
            }
            else
                g_idleClock.restart();
        }

        // Smooth animation logic (no rendering here)
        if (g_popupVisible)
            updateAnim(g_anim, g_popupVisible, dt, 0.08f);

        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }

    LOG_PHASE("Popup UI Thread Exited", true);
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

        // Notify the WindowManager that popup should be drawn
        GRIMWindow* popup = WindowManager::get("popup");
        if (popup)
            popup->visible = true;
    }
}

void hidePopup()
{
    if (g_hwnd)
    {
        ShowWindow(g_hwnd, SW_HIDE);
        g_popupVisible = false;
        LOG_PHASE("PopupUI hidden", true);

        GRIMWindow* popup = WindowManager::get("popup");
        if (popup)
            popup->visible = false;
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
    g_idleClock.restart();
    LOG_DEBUG("PopupUI", "Idle timer reset to " + std::to_string(g_idleTimerMs) + "ms");
}
