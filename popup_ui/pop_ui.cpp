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


// ===========================================================
// Constants
// ===========================================================
constexpr uint16_t POPUP_SIZE = 256; // Temporarily larger for debugging

// ===========================================================
// Globals
// ===========================================================
static bgfx::FrameBufferHandle g_fb = BGFX_INVALID_HANDLE;
static bgfx::TextureHandle g_colorTex = BGFX_INVALID_HANDLE;  // Add global color texture
static bgfx::TextureHandle g_stagingTex = BGFX_INVALID_HANDLE;
static std::vector<uint8_t> g_readbackData(POPUP_SIZE * POPUP_SIZE * 4);

// External globals from popup_window.cpp
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

static bool g_pendingReadback = false;
static uint32_t g_readbackFrame = 0;

// ===========================================================
// Vertex Format
// ===========================================================
struct PosTexcoordVertex
{
    float x, y, z;
    float u, v;

    static void init()
    {
        ms_decl.begin()
            .add(bgfx::Attrib::Position, 3, bgfx::AttribType::Float)
            .add(bgfx::Attrib::TexCoord0, 2, bgfx::AttribType::Float)
            .end();
    }

    static bgfx::VertexLayout ms_decl;
};

bgfx::VertexLayout PosTexcoordVertex::ms_decl;

static PosTexcoordVertex s_quadVertices[] =
{
    { -1.0f, -1.0f, 0.0f, 0.0f, 1.0f },
    {  1.0f, -1.0f, 0.0f, 1.0f, 1.0f },
    { -1.0f,  1.0f, 0.0f, 0.0f, 0.0f },
    {  1.0f,  1.0f, 0.0f, 1.0f, 0.0f },
};

static const uint16_t s_quadIndices[] = { 0, 1, 2, 1, 3, 2 };

// ===========================================================
// Popup UI main loop
// ===========================================================
void runPopupUI(int width, int height)
{
    g_hwnd = createOverlayWindow(POPUP_SIZE, POPUP_SIZE);
    if (!g_hwnd)
        return;

    ShowWindow(g_hwnd, SW_SHOW);
    LOG_DEBUG("PopupUI", "ShowWindow called at " + std::to_string(GetTickCount()));

    // Load combined diffuse + oreo alpha data from disk
    queueWindowAlphaReadback(POPUP_SIZE, POPUP_SIZE);
    
    // Apply alpha data immediately since it's loaded synchronously
    applyWindowAlphaIfReady(g_hwnd, POPUP_SIZE, POPUP_SIZE, 0);

    if (g_pendingPopup)
    {
        showPopup();
        g_idleTimerMs = 10000; // Temporarily increased to 10 seconds for debugging
        g_idleClock.restart();
        g_pendingPopup = false;
    }

    // === BGFX Init ===
    bgfx::Init init;
    init.type = bgfx::RendererType::Direct3D11;
    init.resolution.width = POPUP_SIZE;
    init.resolution.height = POPUP_SIZE;
    init.resolution.reset = BGFX_RESET_NONE;

    bgfx::PlatformData pd{};
    pd.nwh = g_hwnd;
    init.platformData = pd;

    if (!bgfx::init(init))
    {
        LOG_ERROR("PopupUI", "Failed to init bgfx");
        return;
    }

    const bgfx::Caps* caps = bgfx::getCaps();
    LOG_DEBUG("PopupUI", "Renderer: " + std::string(bgfx::getRendererName(caps->rendererType)));
    LOG_DEBUG("PopupUI", std::string("Supports BGFX_TEXTURE_READ_BACK: ") + 
              (caps->supported & BGFX_CAPS_TEXTURE_READ_BACK ? "Yes" : "No"));

    PosTexcoordVertex::init();

    bgfx::VertexBufferHandle vbh = bgfx::createVertexBuffer(
        bgfx::makeRef(s_quadVertices, sizeof(s_quadVertices)),
        PosTexcoordVertex::ms_decl
    );

    bgfx::IndexBufferHandle ibh = bgfx::createIndexBuffer(
        bgfx::makeRef(s_quadIndices, sizeof(s_quadIndices))
    );

    bgfx::ProgramHandle program = loadPopupProgram();
    if (!bgfx::isValid(program)) {
        LOG_ERROR("PopupUI", "Failed to load popup program!");
        return;
    } else {
        LOG_DEBUG("PopupUI", "Popup program loaded successfully");
    }
    
    loadPopupTextures();
    
    // Check if textures loaded
    if (!bgfx::isValid(getPopupTextureColor()) || !bgfx::isValid(getPopupTextureOpacity())) {
        LOG_ERROR("PopupUI", "Failed to load popup textures!");
        return;
    } else {
        LOG_DEBUG("PopupUI", "Popup textures loaded successfully");
    }

    // --- Framebuffer setup ---
    g_colorTex = bgfx::createTexture2D(
        POPUP_SIZE, POPUP_SIZE, false, 1, bgfx::TextureFormat::RGBA8,
        BGFX_TEXTURE_RT | BGFX_SAMPLER_U_CLAMP | BGFX_SAMPLER_V_CLAMP
    );
    g_fb = bgfx::createFrameBuffer(1, &g_colorTex, true);

    LOG_PHASE("Popup UI launched", true);

    MSG msg{};
    sf::Clock frameClock;
    uint32_t frameCounter = 0;

    // =======================================================
    // Main Render Loop
    // =======================================================
    while (g_running)
    {
        // === Win32 messages ===
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            if (msg.message == WM_QUIT)
                g_running = false;
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

        updateAnim(g_anim, g_popupVisible, dt, 0.08f);

        // === Render pass ===
bgfx::setViewFrameBuffer(0, g_fb);
bgfx::setViewClear(0, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH, 0x00000000, 1.0f, 0);
bgfx::setViewRect(0, 0, 0, POPUP_SIZE, POPUP_SIZE);
bgfx::touch(0);

// ===========================================
// Apply animated scale + fade
// ===========================================
float mtx[16];
bx::mtxSRT(
    mtx,                              // destination matrix
    g_anim.scale, g_anim.scale, 1.0f, // scale
    0.0f, 0.0f, 0.0f,                 // rotation
    0.0f, 0.0f, 0.0f                  // translation
);


// --- Create / Update fade uniform ---
static bgfx::UniformHandle u_alpha = BGFX_INVALID_HANDLE;
if (!bgfx::isValid(u_alpha))
    u_alpha = bgfx::createUniform("u_alpha", bgfx::UniformType::Vec4);

// Pack alpha in .w for shader use
float alphaVec[4] = { 1.0f, 1.0f, 1.0f, g_anim.alpha };
bgfx::setUniform(u_alpha, alphaVec);

// ===========================================
// Submit geometry + textures
// ===========================================
bgfx::setVertexBuffer(0, vbh);
bgfx::setIndexBuffer(ibh);

bgfx::setTexture(0, getPopupSamplerColor(),   getPopupTextureColor());
bgfx::setTexture(1, getPopupSamplerOpacity(), getPopupTextureOpacity());

uint64_t state =
    BGFX_STATE_WRITE_RGB |
    BGFX_STATE_WRITE_A |
    BGFX_STATE_BLEND_FUNC(BGFX_STATE_BLEND_SRC_ALPHA, BGFX_STATE_BLEND_INV_SRC_ALPHA);

bgfx::setState(state);
bgfx::submit(0, program);

// Present
uint32_t frameIdx = bgfx::frame();
frameCounter++;


       std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }

    // Cleanup
    bgfx::destroy(vbh);
    bgfx::destroy(ibh);
    bgfx::destroy(program);
    if (bgfx::isValid(g_stagingTex)) bgfx::destroy(g_stagingTex);
    if (bgfx::isValid(g_fb)) bgfx::destroy(g_fb);
    bgfx::shutdown();
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
    }
}

void hidePopup()
{
    if (g_hwnd)
    {
        ShowWindow(g_hwnd, SW_HIDE);
        g_popupVisible = false;
        LOG_PHASE("PopupUI hidden", true);
        LOG_DEBUG("PopupUI", "hidePopup called at " + std::to_string(GetTickCount()) +
                              " idleTimerMs=" + std::to_string(g_idleTimerMs));
    }
}

void notifyPopupActivity()
{
    if (!g_hwnd)
    {
        g_pendingPopup = true;
        LOG_DEBUG("PopupUI", "notifyPopupActivity called but HWND not ready - queued");
        return;
    }

    showPopup();
    g_idleTimerMs = 3000; // Back to normal 3 seconds
    g_idleClock.restart();
    LOG_DEBUG("PopupUI", "Activity notified, idle timer reset");
}
