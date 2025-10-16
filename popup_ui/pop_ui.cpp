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
#include <atomic>
#include <thread>
#include "core/window_manager.hpp"
#include "helpers/mouse.hpp"
#include "core/ui_sync.hpp"


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
// Popup UI main thread (logic only — no BGFX here)
// ===========================================================
void runPopupUI(int width, int height)
{LOG_TRACE("PopupUI", "runPopupUI");
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
        g_idleClock.restart();
        g_pendingPopup = false;
    }

    MSG msg{};
    sf::Clock frameClock;

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

        HWND grimConsole = FindWindowW(nullptr, L"G.R.I.M Console");
        if (grimConsole)
        {
            LOG_DEBUG("PopupUI", "Found existing GRIM console window, bringing to front");
            ShowWindow(grimConsole, SW_RESTORE);
            SetForegroundWindow(grimConsole);
        }
        // else do nothing — console not yet created
        }
    }
    Mouse::endFrame();


        // ---------------------------------------------------
        // Idle timer logic
        // ---------------------------------------------------
        float dt = frameClock.restart().asSeconds();

        if (g_idleTimerMs > 0 &&
            g_idleClock.getElapsedTime().asMilliseconds() > g_idleTimerMs)
        {
            if (!Voice::isPlaying())
            {
                hidePopup();
                g_idleTimerMs = 0;
            }
            else
            {
                g_idleClock.restart();
            }
        }

        // ---------------------------------------------------
        // Animation logic (no rendering)
        // ---------------------------------------------------
        if (g_popupVisible)
            updateAnim(g_anim, g_popupVisible, dt, 0.08f);

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
    g_idleClock.restart();
    LOG_DEBUG("PopupUI", "Idle timer reset to " + std::to_string(g_idleTimerMs) + "ms");
}
