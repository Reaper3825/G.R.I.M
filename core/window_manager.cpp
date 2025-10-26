#include "window_manager.hpp"
#include "logger.hpp"
#include "popup_ui/popup_window.hpp"
#include "ui/ui_root.hpp"
#include <windowsx.h>

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
// Overlay window procedure with WM_NCHITTEST for selective input
// =====================================================
static LRESULT CALLBACK OverlayWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    case WM_NCHITTEST:
    {
        // Get screen coordinates from lParam
        POINT screenPt;
        screenPt.x = GET_X_LPARAM(lParam);
        screenPt.y = GET_Y_LPARAM(lParam);
        
        // Convert to client coordinates
        POINT clientPt = screenPt;
        ScreenToClient(hwnd, &clientPt);
        
        // Check with UIRoot if this position should receive input
        if (UIRoot::get().shouldReceiveInputAt(static_cast<float>(clientPt.x), 
                                                static_cast<float>(clientPt.y)))
        {
            return HTCLIENT; // Allow input
        }
        
        // Default to transparent (pass-through)
        return HTTRANSPARENT;
    }
    
    case WM_LBUTTONDOWN:
    {
        // When clicking on the console, set keyboard focus to this window
        SetFocus(hwnd);
        return 0;
    }
    
    case WM_CHAR:
    {
        // Forward text input to UIRoot for processing
        if (wParam >= 32 && wParam < 127) // Printable ASCII
        {
            char ch = static_cast<char>(wParam);
            UIRoot::get().injectTextInput(std::string(1, ch));
        }
        return 0;
    }
    
    case WM_KEYDOWN:
    {
        // Forward special keys (Enter, Backspace, Escape, arrows)
        // These are already captured by GetAsyncKeyState in InputState
        break;
    }
    
    case WM_ACTIVATE:
        // Don't prevent activation when clicking on console
        if (LOWORD(wParam) != WA_INACTIVE)
        {
            SetFocus(hwnd);
        }
        return 0;
    
    case WM_SETFOCUS:
        return 0;
    
    case WM_MOUSEACTIVATE:
        // Allow mouse input and activate when clicking
        return MA_ACTIVATE;
    
    case WM_CLOSE:
        ShowWindow(hwnd, SW_HIDE);
        return 0;
        
    case WM_DESTROY:
        return 0;
        
    default:
        return DefWindowProcW(hwnd, msg, wParam, lParam);
    }
    
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

// =====================================================
// Overlay creation
// =====================================================
GRIMWindow* WindowManager::createOverlay(const std::string& name, int w, int h, bool transparent)
{
    LOG_DEBUG("WindowManager", "Creating overlay window: " + name);

    // If this is a transparent overlay and one already exists, reuse it
    if (transparent)
    {
        for (auto& existing : s_windows)
        {
            if (existing->isOverlay)
            {
                LOG_DEBUG("WindowManager", "Reusing existing overlay window: " + existing->name);
                existing->visible = true;
                return existing.get();
            }
        }

        // Get virtual screen dimensions for multi-monitor support
        int virtualX = GetSystemMetrics(SM_XVIRTUALSCREEN);
        int virtualY = GetSystemMetrics(SM_YVIRTUALSCREEN);
        int virtualWidth = GetSystemMetrics(SM_CXVIRTUALSCREEN);
        int virtualHeight = GetSystemMetrics(SM_CYVIRTUALSCREEN);
        
        // Use virtual dimensions if not specified
        if (w <= 0 || h <= 0)
        {
            w = virtualWidth;
            h = virtualHeight;
        }

        // Register window class for overlay
        HINSTANCE hInstance = GetModuleHandleW(nullptr);
        WNDCLASSEXW wc{};
        wc.cbSize = sizeof(WNDCLASSEXW);
        wc.lpfnWndProc = OverlayWndProc;
        wc.hInstance = hInstance;
        wc.lpszClassName = L"GRIMOverlayClass";
        wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
        wc.hbrBackground = nullptr;  // No background for layered window

        if (!RegisterClassExW(&wc) && GetLastError() != ERROR_CLASS_ALREADY_EXISTS)
        {
            LOG_ERROR("WindowManager", "Failed to register overlay class");
            return nullptr;
        }

        // Create layered window for proper transparency
        // Remove WS_EX_NOACTIVATE so the window can receive keyboard input
        HWND hwnd = CreateWindowExW(
            WS_EX_LAYERED | WS_EX_TOPMOST,  // Removed WS_EX_NOACTIVATE
            L"GRIMOverlayClass",
            L"GRIM Overlay",
            WS_POPUP,
            virtualX, virtualY,
            virtualWidth, virtualHeight,
            nullptr, nullptr, hInstance, nullptr
        );

        if (!hwnd)
        {
            LOG_ERROR("WindowManager", "Failed to create transparent overlay window");
            return nullptr;
        }

        // DON'T use SetLayeredWindowAttributes - we use UpdateLayeredWindow for per-pixel alpha!
        // The OverlayRenderer will handle all transparency via UpdateLayeredWindow in endFrame()
        // Calling SetLayeredWindowAttributes here would make the window 99.6% transparent and override
        // the per-pixel alpha from UpdateLayeredWindow!
        
        // Don't show initially - UIRoot will show it when UI becomes visible
        UpdateWindow(hwnd);

        auto win = std::make_unique<GRIMWindow>();
        win->hwnd = hwnd;
        win->name = name;
        win->visible = true;
        win->isOverlay = false;  // Changed: Mark as non-overlay so BGFX renders to it
        win->width = virtualWidth;
        win->height = virtualHeight;

        registerWindow(std::move(win));
        LOG_PHASE("Transparent multi-monitor overlay created (" + 
                 std::to_string(virtualWidth) + "x" + std::to_string(virtualHeight) + ")", true);
        return get(name);
    }

    // ----- Non-transparent standard window path -----
    HWND hwnd = CreateWindowExW(
        0, L"STATIC", std::wstring(name.begin(), name.end()).c_str(),
        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
        w, h, nullptr, nullptr, GetModuleHandle(nullptr), nullptr);

    if (!hwnd)
    {
        LOG_ERROR("WindowManager", "Standard overlay creation failed for: " + name);
        return nullptr;
    }

    auto win = std::make_unique<GRIMWindow>();
    win->hwnd = hwnd;
    win->name = name;
    win->visible = true;
    win->isOverlay = false;
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
{LOG_TRACE("CU", "beginFrame");
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
