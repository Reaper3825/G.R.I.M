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
            {
                LOG_DEBUG("PopupUI", "WM_QUIT received - exiting popup thread");
                g_running = false;
            }
            else
            {
                LOG_DEBUG("PopupUI", "Processing message: " + std::to_string(msg.message));
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

        // Only render when popup is visible
        if (g_popupVisible)
        {
            LOG_DEBUG("PopupUI", "Popup is visible, starting render pass");
            
            try
            {
                LOG_DEBUG("PopupUI", "Checking bgfx resources");
                // Validate bgfx resources before using them
                if (!bgfx::isValid(g_fb) || !bgfx::isValid(g_colorTex))
                {
                    LOG_ERROR("PopupUI", "Invalid framebuffer or texture - bgfx resources corrupted");
                    // Don't exit, just skip this frame
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }
                
                LOG_DEBUG("PopupUI", "Updating animation");
                updateAnim(g_anim, g_popupVisible, dt, 0.08f);

                LOG_DEBUG("PopupUI", "Setting up render pass");
                // === Render pass ===
                bgfx::setViewFrameBuffer(0, g_fb);
                bgfx::setViewClear(0, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH, 0x00000000, 1.0f, 0);
                bgfx::setViewRect(0, 0, 0, POPUP_SIZE, POPUP_SIZE);
                bgfx::touch(0);

                LOG_DEBUG("PopupUI", "Setting up matrices");
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

                LOG_DEBUG("PopupUI", "Setting up uniforms");
                // --- Create / Update fade uniform ---
                static bgfx::UniformHandle u_alpha = BGFX_INVALID_HANDLE;
                if (!bgfx::isValid(u_alpha))
                    u_alpha = bgfx::createUniform("u_alpha", bgfx::UniformType::Vec4);

                // Pack alpha in .w for shader use
                float alphaVec[4] = { 1.0f, 1.0f, 1.0f, g_anim.alpha };
                bgfx::setUniform(u_alpha, alphaVec);

                LOG_DEBUG("PopupUI", "Setting up geometry");
                // ===========================================
                // Submit geometry + textures
                // ===========================================
                bgfx::setVertexBuffer(0, vbh);
                bgfx::setIndexBuffer(ibh);

                LOG_DEBUG("PopupUI", "Setting up textures");
                bgfx::setTexture(0, getPopupSamplerColor(),   getPopupTextureColor());
                bgfx::setTexture(1, getPopupSamplerOpacity(), getPopupTextureOpacity());

                if (!bgfx::isValid(getPopupTextureColor()) || !bgfx::isValid(getPopupTextureOpacity()))
                {
                    LOG_ERROR("PopupUI", "Invalid popup textures - skipping frame");
                    // Don't exit, just skip this frame
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }

                LOG_DEBUG("PopupUI", "Setting render state");
                uint64_t state =
                    BGFX_STATE_WRITE_RGB |
                    BGFX_STATE_WRITE_A |
                    BGFX_STATE_BLEND_FUNC(BGFX_STATE_BLEND_SRC_ALPHA, BGFX_STATE_BLEND_INV_SRC_ALPHA);

                bgfx::setState(state);
                bgfx::submit(0, program);

                if (!bgfx::isValid(program))
                {
                    LOG_ERROR("PopupUI", "Invalid shader program - skipping frame");
                    // Don't exit, just skip this frame
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }

                LOG_DEBUG("PopupUI", "Calling bgfx::frame()");
                // Present
                uint32_t frameIdx = UINT32_MAX;
                try
                {
                    frameIdx = bgfx::frame();
                    LOG_DEBUG("PopupUI", "bgfx::frame() returned: " + std::to_string(frameIdx));
                }
                catch (const std::exception& e)
                {
                    LOG_ERROR("PopupUI", "Exception in bgfx::frame(): " + std::string(e.what()));
                    // Don't exit, just skip this frame
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }
                catch (...)
                {
                    LOG_ERROR("PopupUI", "Unknown exception in bgfx::frame()");
                    // Don't exit, just skip this frame
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }
                
                if (frameIdx == UINT32_MAX)
                {
                    LOG_ERROR("PopupUI", "bgfx::frame() returned error - skipping frame");
                    // Don't exit, just skip this frame
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }
                LOG_DEBUG("PopupUI", "Frame completed successfully");
                frameCounter++;
                
                std::this_thread::sleep_for(std::chrono::milliseconds(16));
            }
            catch (const std::exception& e)
            {
                LOG_ERROR("PopupUI", "Exception during rendering: " + std::string(e.what()));
                // Don't exit the thread, just log and continue
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            catch (...)
            {
                LOG_ERROR("PopupUI", "Unknown exception during rendering");
                // Don't exit the thread, just log and continue  
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
        else
        {
    static bool loggedHidden = false;

    // When hidden, skip rendering entirely to prevent issues
    // Still process messages and check timers, but sleep longer
    if (!loggedHidden)
    {
        LOG_DEBUG("PopupUI", "Popup is hidden, skipping render");
        loggedHidden = true;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

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
    LOG_DEBUG("PopupUI", "notifyPopupActivity called - checking window state");

    if (!g_hwnd)
    {
        g_pendingPopup = true;
        LOG_DEBUG("PopupUI", "notifyPopupActivity called but HWND not ready - queued");
        return;
    }

    // Post a message to the popup thread instead of calling ShowWindow() directly
    PostMessage(g_hwnd, WM_GRIM_SHOW_POPUP, 0, 0);

    g_idleTimerMs = 3000;
    g_idleClock.restart();
    LOG_DEBUG("PopupUI", "Activity notified, idle timer reset to " + std::to_string(g_idleTimerMs) + "ms");
}

