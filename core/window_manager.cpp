#include "window_manager.hpp"
#include "logger.hpp"
#include "popup_ui/popup_window.hpp"

// =====================================================
// Global GPU mutex (shared with popup_window.cpp)
// =====================================================
std::mutex g_gpuMutex;

namespace
{
    constexpr uint32_t kDefaultClearColor = 0xFF121212;
    constexpr uint32_t kFallbackWidth = 1920;
    constexpr uint32_t kFallbackHeight = 1080;
}

// =====================================================
// BGFX Init
// =====================================================
bool WindowManager::initGlobalBGFX(HWND mainHwnd)
{
    {
        std::lock_guard<std::mutex> lock(s_mutex);
        if (s_bgfxInitialized)
            return true;
    }

    LOG_DEBUG("WindowManager", "Initializing global BGFX context");

    bgfx::Init init;
    init.type = bgfx::RendererType::Count;
    init.resolution.width = kFallbackWidth;
    init.resolution.height = kFallbackHeight;
    init.resolution.reset = BGFX_RESET_VSYNC;

    s_backbufferWidth = init.resolution.width;
    s_backbufferHeight = init.resolution.height;
    s_resetFlags = init.resolution.reset;

    bgfx::PlatformData pd{};
    pd.nwh = mainHwnd;
    init.platformData = pd;

    if (!bgfx::init(init))
    {
        LOG_ERROR("WindowManager", "BGFX initialization failed");
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(s_mutex);
        s_bgfxInitialized = true;
        s_mainLoopStop.store(false, std::memory_order_release);
    }

    LOG_PHASE("Global BGFX initialized (WindowManager)", true);
    return true;
}

// =====================================================
// Check if BGFX is initialized
// =====================================================
bool WindowManager::isInitialized()
{
    std::lock_guard<std::mutex> lock(s_mutex);
    return s_bgfxInitialized;
}

// =====================================================
// BGFX Shutdown
// =====================================================
void WindowManager::shutdown()
{
    std::vector<HWND> handles;

    {
        std::lock_guard<std::mutex> lock(s_mutex);
        if (!s_bgfxInitialized)
            return;

        s_bgfxInitialized = false;
        s_mainLoopStop.store(true, std::memory_order_release);
        s_primaryWindow = nullptr;
        s_platformUpdatePending.store(false, std::memory_order_release);
        s_pendingPlatformWindow = nullptr;
        s_pendingPlatformWidth = 0;
        s_pendingPlatformHeight = 0;

        for (auto& w : s_windows)
        {
            if (w->hwnd && IsWindow(w->hwnd))
                handles.push_back(w->hwnd);
        }
        s_windows.clear();
    }

    LOG_DEBUG("WindowManager", "Shutting down all windows and BGFX");

    for (HWND hwnd : handles)
    {
        if (hwnd && IsWindow(hwnd))
            DestroyWindow(hwnd);
    }

    bgfx::shutdown();

    LOG_PHASE("Global BGFX shutdown complete", true);
}

// =====================================================
// Overlay creation
// =====================================================
GRIMWindow* WindowManager::createOverlay(const std::string& name, int w, int h, bool transparent)
{
    LOG_DEBUG("WindowManager", "Creating overlay window: " + name);

    HWND hwnd = nullptr;
    if (transparent)
    {
        hwnd = createOverlayWindow(w, h);
    }
    else
    {
        hwnd = CreateWindowExW(
            0, L"STATIC", std::wstring(name.begin(), name.end()).c_str(),
            WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
            w, h, nullptr, nullptr, GetModuleHandle(nullptr), nullptr);
    }

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
    win->width = w;
    win->height = h;

    registerWindow(std::move(win));
    return get(name);
}

GRIMWindow* WindowManager::get(const std::string& name)
{
    std::lock_guard<std::mutex> lock(s_mutex);
    for (auto& w : s_windows)
    {
        if (w->name == name)
            return w.get();
    }
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

void WindowManager::setVisibility(const std::string& name, bool visible)
{
    std::lock_guard<std::mutex> lock(s_mutex);
    for (auto& w : s_windows)
    {
        if (w->name == name)
        {
            w->visible = visible;
            break;
        }
    }
}

// =====================================================
// Rendering helpers
// =====================================================
void WindowManager::beginFrame(uint16_t viewId, uint32_t clearColor)
{
    if (!s_bgfxInitialized)
        return;

    bgfx::setViewClear(viewId, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH, clearColor, 1.0f, 0);
    bgfx::touch(viewId);
}

void WindowManager::endFrame()
{
    if (s_bgfxInitialized)
        bgfx::frame();
}

void WindowManager::registerWindow(std::unique_ptr<GRIMWindow> win)
{
    if (!win)
        return;

    const std::string name = win->name;
    const uint32_t incomingWidth = win->width > 0 ? static_cast<uint32_t>(win->width) : 0;
    const uint32_t incomingHeight = win->height > 0 ? static_cast<uint32_t>(win->height) : 0;
    bool updated = false;

    {
        std::lock_guard<std::mutex> lock(s_mutex);

        for (auto& existing : s_windows)
        {
            if (existing->name == name)
            {
                existing->hwnd = win->hwnd;
                existing->visible = win->visible;
                existing->isOverlay = win->isOverlay;
                existing->width = win->width;
                existing->height = win->height;

                if (!existing->isOverlay && existing->hwnd)
                {
                    s_primaryWindow = existing->hwnd;
                    if (incomingWidth != 0 && incomingHeight != 0)
                    {
                        s_backbufferWidth = incomingWidth;
                        s_backbufferHeight = incomingHeight;
                    }
                    s_pendingPlatformWindow = existing->hwnd;
                    s_pendingPlatformWidth = incomingWidth;
                    s_pendingPlatformHeight = incomingHeight;
                    s_platformUpdatePending.store(true, std::memory_order_release);
                }

                updated = true;
                break;
            }
        }

        if (!updated)
        {
            if (!win->isOverlay && win->hwnd)
            {
                s_primaryWindow = win->hwnd;
                if (incomingWidth != 0 && incomingHeight != 0)
                {
                    s_backbufferWidth = incomingWidth;
                    s_backbufferHeight = incomingHeight;
                }
                s_pendingPlatformWindow = win->hwnd;
                s_pendingPlatformWidth = incomingWidth;
                s_pendingPlatformHeight = incomingHeight;
                s_platformUpdatePending.store(true, std::memory_order_release);
            }

            s_windows.push_back(std::move(win));
        }
    }

    LOG_DEBUG("WindowManager", (updated ? "Updated window: " : "Registered window: ") + name);
}

bool WindowManager::processMainThreadUpdates()
{
    HWND hwnd = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    bool pending = false;

    {
        std::lock_guard<std::mutex> lock(s_mutex);
        if (s_platformUpdatePending.load(std::memory_order_acquire))
        {
            hwnd = s_pendingPlatformWindow;
            width = s_pendingPlatformWidth != 0 ? s_pendingPlatformWidth : s_backbufferWidth;
            height = s_pendingPlatformHeight != 0 ? s_pendingPlatformHeight : s_backbufferHeight;
            pending = hwnd != nullptr;
            s_pendingPlatformWindow = nullptr;
            s_pendingPlatformWidth = 0;
            s_pendingPlatformHeight = 0;
            s_platformUpdatePending.store(false, std::memory_order_release);
        }
    }

    if (!pending)
        return false;

    bgfx::PlatformData pd{};
    pd.nwh = hwnd;
    bgfx::setPlatformData(pd);

    if (width == 0) width = 1;
    if (height == 0) height = 1;
    bgfx::reset(width, height, s_resetFlags);

    LOG_DEBUG("WindowManager", "Applied platform update on main thread");
    return true;
}

bool WindowManager::hasPendingPlatformUpdate()
{
    return s_platformUpdatePending.load(std::memory_order_acquire);
}

void WindowManager::updateWindowDimensions(const std::string& name, uint32_t width, uint32_t height)
{
    bool found = false;
    bool queuedPlatformUpdate = false;
    bool sizeChanged = false;

    {
        std::lock_guard<std::mutex> lock(s_mutex);
        for (auto& w : s_windows)
        {
            if (w->name != name)
                continue;

            found = true;
            const uint32_t oldW = static_cast<uint32_t>(w->width);
            const uint32_t oldH = static_cast<uint32_t>(w->height);
            w->width = static_cast<int>(width);
            w->height = static_cast<int>(height);
            sizeChanged = (oldW != width || oldH != height);

            if (!w->isOverlay && w->hwnd)
            {
                if (width > 0 && height > 0)
                {
                    s_backbufferWidth = width;
                    s_backbufferHeight = height;
                }
                s_primaryWindow = w->hwnd;
                s_pendingPlatformWindow = w->hwnd;
                s_pendingPlatformWidth = width;
                s_pendingPlatformHeight = height;
                s_platformUpdatePending.store(true, std::memory_order_release);
                queuedPlatformUpdate = true;
            }
            break;
        }
    }

    if (!found)
    {
        LOG_TRACE("WindowManager", "updateWindowDimensions called for unknown window '" + name + "'");
        return;
    }

    if (sizeChanged)
    {
        LOG_DEBUG("WindowManager", "Updated dimensions for window '" + name + "' to " +
            std::to_string(width) + "x" + std::to_string(height));
    }

    if (queuedPlatformUpdate)
    {
        LOG_DEBUG("WindowManager", "Queued platform update for window '" + name + "'");
    }
}

void WindowManager::renderFrame()
{
    if (!s_bgfxInitialized)
        return;

    HWND primary = nullptr;
    uint32_t targetWidth = 0;
    uint32_t targetHeight = 0;

    {
        std::lock_guard<std::mutex> lock(s_mutex);
        primary = s_primaryWindow;
        targetWidth = s_backbufferWidth;
        targetHeight = s_backbufferHeight;
    }

    if (!primary)
        return;

    std::lock_guard<std::mutex> gpuLock(g_gpuMutex);

    const uint16_t viewId = 0;
    bgfx::setViewClear(viewId, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH, kDefaultClearColor, 1.0f, 0);

    if (targetWidth == 0 || targetHeight == 0)
    {
        targetWidth = kFallbackWidth;
        targetHeight = kFallbackHeight;
    }

    bgfx::setViewRect(viewId, 0, 0, targetWidth, targetHeight);
    bgfx::touch(viewId);

    bgfx::frame();
}

void WindowManager::requestMainLoopStop()
{
    s_mainLoopStop.store(true, std::memory_order_release);
}

bool WindowManager::isMainLoopStopRequested()
{
    return s_mainLoopStop.load(std::memory_order_acquire);
}
