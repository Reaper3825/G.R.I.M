#include "popup_ui.hpp"
#include "popup_window.hpp"
#include "popup_renderer.hpp"
#include "popup_anim.hpp"
#include "logger.hpp"

#include "voice/voice_speak.hpp"

#include <thread>
#include <atomic>
#include <chrono>
#include <windows.h>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <bx/math.h>
#include <bgfx/bgfx.h>
#include <bgfx/platform.h>

// ===========================================================
// Globals
// ===========================================================
static std::atomic<bool> g_popupVisible{false};
static std::atomic<bool> g_running{true};
static std::atomic<int>  g_idleTimerMs{0};
static std::atomic<bool> g_pendingPopup{false};
static HWND g_hwnd = nullptr;
static PopupAnimState g_anim;
static sf::Clock g_idleClock; // still used for idle timing

// ===========================================================
// Vertex format for simple colored quad
// ===========================================================
struct PosColorVertex {
    float x, y, z;
    uint32_t abgr;
    static void init()
    {
        ms_decl.begin()
            .add(bgfx::Attrib::Position, 3, bgfx::AttribType::Float)
            .add(bgfx::Attrib::Color0,   4, bgfx::AttribType::Uint8, true)
            .end();
    }
    static bgfx::VertexLayout ms_decl;
};

bgfx::VertexLayout PosColorVertex::ms_decl;

// A simple fullscreen quad (two triangles) with semi-transparent red
static PosColorVertex s_quadVertices[] =
{
    { -1.0f, -1.0f, 0.0f, 0x7FFF0000 }, // ABGR: 0xAABBGGRR -> 0x7F alpha, FF red
    {  1.0f, -1.0f, 0.0f, 0x7FFF0000 },
    { -1.0f,  1.0f, 0.0f, 0x7FFF0000 },
    {  1.0f,  1.0f, 0.0f, 0x7FFF0000 },
};

static const uint16_t s_quadIndices[] = { 0, 1, 2, 1, 3, 2 };

// ===========================================================
// Popup UI main loop
// ===========================================================
void runPopupUI(int width, int height) {
    constexpr unsigned overlayW = 128u;
    constexpr unsigned overlayH = 128u;

    g_hwnd = createOverlayWindow(static_cast<int>(overlayW), static_cast<int>(overlayH));
    if (!g_hwnd) return;

    ShowWindow(g_hwnd, SW_SHOW);
    LOG_DEBUG("PopupUI", "ShowWindow called at " + std::to_string(GetTickCount()));

    if (g_pendingPopup) {
        LOG_DEBUG("PopupUI", "Processing queued popup activity");
        showPopup();
        g_idleTimerMs = 3000;
        g_idleClock.restart();
        g_pendingPopup = false;
    }

    // === bgfx init ===
    bgfx::Init init;
    init.type = bgfx::RendererType::Count; // auto choose
    init.resolution.width  = overlayW;
    init.resolution.height = overlayH;
    init.resolution.reset  = BGFX_RESET_NONE;

    bgfx::PlatformData pd{};
    pd.nwh = g_hwnd;
    init.platformData = pd;

    if (!bgfx::init(init)) {
        LOG_DEBUG("PopupUI", "Failed to init bgfx");
        return;
    }

    PosColorVertex::init();

    // Create buffers
    bgfx::VertexBufferHandle vbh = bgfx::createVertexBuffer(
        bgfx::makeRef(s_quadVertices, sizeof(s_quadVertices)),
        PosColorVertex::ms_decl
    );

    bgfx::IndexBufferHandle ibh = bgfx::createIndexBuffer(
        bgfx::makeRef(s_quadIndices, sizeof(s_quadIndices))
    );

    // Load your already-working shader program (popup_renderer can expose helper)
    bgfx::ProgramHandle program = loadPopupProgram(); // defined in popup_renderer

    MSG msg{};
    sf::Clock frameClock;
    LOG_PHASE("Popup UI launched", true);

    while (g_running) {
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) g_running = false;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        float dt = frameClock.restart().asSeconds();

        // Idle check (respect TTS)
        if (g_idleTimerMs > 0 && g_idleClock.getElapsedTime().asMilliseconds() > g_idleTimerMs) {
            if (!Voice::isPlaying()) {
                hidePopup();
                g_idleTimerMs = 0;
            } else {
                g_idleClock.restart();
            }
        }

        // Animate (for now we just use alpha, not scale)
        updateAnim(g_anim, g_popupVisible, dt, 0.08f);

        // === bgfx frame ===
        bgfx::setViewClear(0, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH, 0x00000000, 1.0f, 0);
        bgfx::setViewRect(0, 0, 0, overlayW, overlayH);

        bgfx::touch(0); // ensure view is cleared

        bgfx::setVertexBuffer(0, vbh);
        bgfx::setIndexBuffer(ibh);

        // Use default state + blending for transparency
        uint64_t state = BGFX_STATE_WRITE_RGB | BGFX_STATE_WRITE_A | BGFX_STATE_BLEND_ALPHA;
        bgfx::setState(state);

        bgfx::submit(0, program);

        bgfx::frame();

        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }

    // Cleanup
    bgfx::destroy(vbh);
    bgfx::destroy(ibh);
    bgfx::destroy(program);
    bgfx::shutdown();
}

// ===========================================================
// Popup UI controls
// ===========================================================
void showPopup() {
    if (g_hwnd) {
        ShowWindow(g_hwnd, SW_SHOW);
        g_popupVisible = true;
        LOG_PHASE("PopupUI shown", true);
    }
}

void hidePopup() {
    if (g_hwnd) {
        ShowWindow(g_hwnd, SW_HIDE);
        g_popupVisible = false;
        LOG_PHASE("PopupUI hidden", true);
        LOG_DEBUG("PopupUI", "hidePopup called at " + std::to_string(GetTickCount()) +
                          " idleTimerMs=" + std::to_string(g_idleTimerMs));
    }
}

void notifyPopupActivity() {
    if (!g_hwnd) {
        g_pendingPopup = true;
        LOG_DEBUG("PopupUI", "notifyPopupActivity called but HWND not ready - queued");
        return;
    }

    showPopup();
    g_idleTimerMs = 3000;
    g_idleClock.restart();
    LOG_DEBUG("PopupUI", "Activity notified, idle timer reset");
}
