#include "window_manager.hpp"
#include "logger.hpp"
#include "popup_ui/popup_ui.hpp"   // For runPopupUI(), showPopup(), etc.
#include <thread>
#include "popup_ui/popup_window.hpp"

// =====================================================
// BGFX Init
// =====================================================
bool WindowManager::initGlobalBGFX(HWND mainHwnd)
{
    std::lock_guard<std::mutex> lock(s_mutex);
    if (s_bgfxInitialized)
        return true;

    LOG_DEBUG("WindowManager", "Initializing global BGFX context");

    bgfx::Init init;
    init.type = bgfx::RendererType::Count; // auto
    init.resolution.width = 1920;
    init.resolution.height = 1080;
    init.resolution.reset = BGFX_RESET_VSYNC;

    bgfx::PlatformData pd{};
    pd.nwh = mainHwnd;
    init.platformData = pd;

    if (!bgfx::init(init))
    {
        LOG_ERROR("WindowManager", "BGFX initialization failed");
        return false;
    }

    s_bgfxInitialized = true;
    LOG_PHASE("Global BGFX initialized (WindowManager)", true);
    return true;
}

// =====================================================
// BGFX Shutdown
// =====================================================
void WindowManager::shutdown()
{
    std::lock_guard<std::mutex> lock(s_mutex);
    if (!s_bgfxInitialized)
        return;

    LOG_DEBUG("WindowManager", "Shutting down all windows and BGFX");

    for (auto& w : s_windows)
    {
        if (w->hwnd && IsWindow(w->hwnd))
            DestroyWindow(w->hwnd);
    }
    s_windows.clear();

    bgfx::shutdown();
    s_bgfxInitialized = false;

    LOG_PHASE("Global BGFX shutdown complete", true);
}

// =====================================================
// Overlay creation
// =====================================================
GRIMWindow* WindowManager::createOverlay(const std::string& name, int w, int h, bool transparent)
{
    std::lock_guard<std::mutex> lock(s_mutex);

    LOG_DEBUG("WindowManager", "Creating overlay window: " + name);

    HWND hwnd = createOverlayWindow(w, h); // uses your popup_window.cpp
    if (!hwnd)
    {
        LOG_ERROR("WindowManager", "Overlay creation failed for: " + name);
        return nullptr;
    }

    auto win = std::make_unique<GRIMWindow>();
    win->hwnd = hwnd;
    win->name = name;
    win->visible = true;
    win->isOverlay = transparent;
    s_windows.push_back(std::move(win));

    LOG_PHASE(("Overlay created: " + name).c_str(), true);
    return s_windows.back().get();
}

// =====================================================
// Accessors + visibility
// =====================================================
GRIMWindow* WindowManager::get(const std::string& name)
{
    std::lock_guard<std::mutex> lock(s_mutex);
    for (auto& w : s_windows)
        if (w->name == name)
            return w.get();
    return nullptr;
}

void WindowManager::show(const std::string& name)
{
    std::lock_guard<std::mutex> lock(s_mutex);
    for (auto& w : s_windows)
    {
        if (w->name == name && w->hwnd)
        {
            ShowWindow(w->hwnd, SW_SHOW);
            w->visible = true;
            LOG_DEBUG("WindowManager", "Showing window: " + name);
        }
    }
}

void WindowManager::hide(const std::string& name)
{
    std::lock_guard<std::mutex> lock(s_mutex);
    for (auto& w : s_windows)
    {
        if (w->name == name && w->hwnd)
        {
            ShowWindow(w->hwnd, SW_HIDE);
            w->visible = false;
            LOG_DEBUG("WindowManager", "Hiding window: " + name);
        }
    }
}

// =====================================================
// Rendering helpers
// =====================================================
void WindowManager::beginFrame(uint16_t viewId, uint32_t clearColor)
{
    if (!s_bgfxInitialized) return;
    bgfx::setViewClear(viewId, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH, clearColor, 1.0f, 0);
    bgfx::touch(viewId);
}

void WindowManager::endFrame()
{
    if (s_bgfxInitialized)
        bgfx::frame();
}

void WindowManager::drawAll()
{
    if (!s_bgfxInitialized)
        return;

    std::thread([]()
    {
        LOG_PHASE("BGFX Render Thread Started", true);

        constexpr int FRAME_TIME_MS = 16; // ~60 FPS

        while (s_bgfxInitialized)
        {
            auto frameStart = std::chrono::high_resolution_clock::now();

            {
                std::lock_guard<std::mutex> lock(s_mutex);

                // ======================================================
                // Base view setup (clear + viewport)
                // ======================================================
                const uint16_t viewId = 0;
                bgfx::setViewClear(viewId, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH,
                                   0xFF121212, 1.0f, 0);
                bgfx::setViewRect(viewId, 0, 0, 1920, 1080);
                bgfx::touch(viewId);

                // ------------------------------------------------------
                // Example: Popup overlay quad (centered red translucent)
                // ------------------------------------------------------
                GRIMWindow* popup = get("popup");
                if (popup && popup->visible)
                {
                    struct PosColorVertex
                    {
                        float x, y, z;
                        uint32_t abgr;
                    };

                    constexpr float size = 0.25f;
                    const PosColorVertex verts[4] = {
                        { -size, -size, 0.0f, 0x88FF0000 },
                        {  size, -size, 0.0f, 0x88FF0000 },
                        { -size,  size, 0.0f, 0x88FF0000 },
                        {  size,  size, 0.0f, 0x88FF0000 },
                    };
                    const uint16_t indices[6] = { 0, 1, 2, 1, 3, 2 };

                    // --- Vertex layout ---
                    bgfx::VertexLayout layout;
                    layout.begin()
                        .add(bgfx::Attrib::Position, 3, bgfx::AttribType::Float)
                        .add(bgfx::Attrib::Color0,   4, bgfx::AttribType::Uint8, true)
                        .end();

                    // --- Allocate transient buffers (modern BGFX syntax) ---
                    bgfx::TransientVertexBuffer tvb;
                    bgfx::TransientIndexBuffer tib;
                    bgfx::allocTransientVertexBuffer(&tvb, 4, layout);
                    bgfx::allocTransientIndexBuffer(&tib, 6);

                    if (tvb.data && tib.data)
                    {
                        std::memcpy(tvb.data, verts, sizeof(verts));
                        std::memcpy(tib.data, indices, sizeof(indices));

                        static bgfx::ProgramHandle prog = BGFX_INVALID_HANDLE;
                        if (!bgfx::isValid(prog))
                        {
                            const char* vsSrc =
                                "#version 330 core\n"
                                "layout(location=0) in vec3 a_pos;\n"
                                "layout(location=1) in vec4 a_col;\n"
                                "out vec4 v_col;\n"
                                "void main(){ gl_Position=vec4(a_pos,1.0); v_col=a_col; }";

                            const char* fsSrc =
                                "#version 330 core\n"
                                "in vec4 v_col;\n"
                                "out vec4 fragColor;\n"
                                "void main(){ fragColor=v_col; }";

                            const bgfx::Memory* vsmem =
                                bgfx::copy(vsSrc, (uint32_t)strlen(vsSrc) + 1);
                            const bgfx::Memory* fsmem =
                                bgfx::copy(fsSrc, (uint32_t)strlen(fsSrc) + 1);
                            bgfx::ShaderHandle vsh = bgfx::createShader(vsmem);
                            bgfx::ShaderHandle fsh = bgfx::createShader(fsmem);
                            prog = bgfx::createProgram(vsh, fsh, true);
                        }

                        bgfx::setVertexBuffer(0, &tvb);
                        bgfx::setIndexBuffer(&tib);
                        bgfx::setState(
                            BGFX_STATE_WRITE_RGB | BGFX_STATE_WRITE_A |
                            BGFX_STATE_BLEND_FUNC(
                                BGFX_STATE_BLEND_SRC_ALPHA,
                                BGFX_STATE_BLEND_INV_SRC_ALPHA));
                        bgfx::submit(viewId, prog);
                    }
                }
            }

            // ======================================================
            // Finalize frame — single thread only
            // ======================================================
            bgfx::frame();

            // Maintain ~60 FPS
            auto frameEnd = std::chrono::high_resolution_clock::now();
            auto elapsed =
                std::chrono::duration_cast<std::chrono::milliseconds>(frameEnd - frameStart)
                    .count();
            if (elapsed < FRAME_TIME_MS)
                std::this_thread::sleep_for(std::chrono::milliseconds(FRAME_TIME_MS - elapsed));
        }

        LOG_PHASE("BGFX Render Thread Exited", true);
    }).detach();
}
