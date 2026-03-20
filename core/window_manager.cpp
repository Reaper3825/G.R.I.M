#include "window_manager.hpp"
#include "logger.hpp"
#include "ui/ui_root.hpp"
#include "platform_window.hpp"

#ifdef _WIN32
#include "popup_ui/popup_window.hpp"
#include "popup_ui/popup_ui.hpp"
#include <windowsx.h>
#endif

// =====================================================
// Global GPU mutex (shared with popup_window.cpp)
// =====================================================
std::mutex g_gpuMutex;

namespace
{
    // BGFX setViewClear expects RGBA (0xRRGGBBAA); ARGB would show as red.
    constexpr uint32_t kDefaultClearColor = 0x121212FF;
    constexpr uint32_t kFallbackWidth = 1920;
    constexpr uint32_t kFallbackHeight = 1080;
}

#ifdef _WIN32
namespace
{
    // Enable DWM blur-behind for layered windows so transparency shows a blurred
    // real desktop backdrop (instead of only our own synthetic frosted noise).
    static void enableDwmBlurBehind(HWND hwnd)
    {
        if (!hwnd)
            return;

        // ACCENT_POLICY and SetWindowCompositionAttribute are available on Windows 10+.
        // We declare minimal versions here to avoid pulling in undocumented headers.
        struct ACCENT_POLICY
        {
            int AccentState;
            int AccentFlags;
            int GradientColor;
            int AnimationId;
        };

        struct WINDOWCOMPOSITIONATTRIBDATA
        {
            int Attrib;
            PVOID pvData;
            SIZE_T cbData;
        };

        enum
        {
            WCA_ACCENT_POLICY = 19,
            ACCENT_ENABLE_BLURBEHIND = 3
        };

        HMODULE user32 = GetModuleHandleW(L"user32.dll");
        if (!user32)
            return;

        using SetWindowCompositionAttributeFn = BOOL(WINAPI*)(HWND, WINDOWCOMPOSITIONATTRIBDATA*);
        auto fn = reinterpret_cast<SetWindowCompositionAttributeFn>(
            GetProcAddress(user32, "SetWindowCompositionAttribute"));
        if (!fn)
            return;

        ACCENT_POLICY policy{};
        policy.AccentState = ACCENT_ENABLE_BLURBEHIND;
        policy.AccentFlags = 0;
        policy.GradientColor = 0;
        policy.AnimationId = 0;

        WINDOWCOMPOSITIONATTRIBDATA data{};
        data.Attrib = WCA_ACCENT_POLICY;
        data.pvData = &policy;
        data.cbData = sizeof(policy);

        fn(hwnd, &data);
    }
} // namespace
#endif

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
#ifdef _WIN32
            if (w->hwnd && IsWindow(w->hwnd))
                handles.push_back(w->hwnd);
#else
            if (w->hwnd)
                handles.push_back(w->hwnd);
#endif
        }
        s_windows.clear();
    }

    LOG_DEBUG("WindowManager", "Shutting down all windows and BGFX");

    for (HWND hwnd : handles)
    {
#ifdef _WIN32
        if (hwnd && IsWindow(hwnd))
            DestroyWindow(hwnd);
#else
        PlatformWindow::destroyBGFXInitWindow(hwnd);
#endif
    }

    bgfx::shutdown();

    LOG_PHASE("Global BGFX shutdown complete", true);
}

// =====================================================
// Overlay window procedure with WM_NCHITTEST for selective input
// =====================================================
#ifdef _WIN32
static LRESULT CALLBACK OverlayWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    case WM_NCHITTEST:
    {
        POINT pt;
        pt.x = GET_X_LPARAM(lParam);
        pt.y = GET_Y_LPARAM(lParam);
        ScreenToClient(hwnd, &pt);

        if (UIRoot::get().shouldReceiveInputAt(static_cast<float>(pt.x), static_cast<float>(pt.y))
            || UIRoot::get().isAnyPanelDragging()
            || isPopupVisible())
        {
            return HTCLIENT;
        }
        return HTTRANSPARENT;
    }
    
    case WM_LBUTTONDOWN:
    {
        SetFocus(hwnd);
        return 0;
    }
    
    case WM_CHAR:
    {
        if (wParam >= 32 && wParam < 127)
        {
            char ch = static_cast<char>(wParam);
            UIRoot::get().injectTextInput(std::string(1, ch));
        }
        return 0;
    }
    
    case WM_KEYDOWN:
    {
        break;
    }
    
    case WM_ACTIVATE:
        if (LOWORD(wParam) != WA_INACTIVE)
        {
            SetFocus(hwnd);
        }
        return 0;
    
    case WM_SETFOCUS:
        return 0;
    
    case WM_MOUSEACTIVATE:
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
#endif // _WIN32

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

#ifdef _WIN32
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
        wc.hbrBackground = nullptr;

        if (!RegisterClassExW(&wc) && GetLastError() != ERROR_CLASS_ALREADY_EXISTS)
        {
            LOG_ERROR("WindowManager", "Failed to register overlay class");
            return nullptr;
        }

        HWND hwnd = CreateWindowExW(
            WS_EX_LAYERED | WS_EX_TOPMOST,
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

        UpdateWindow(hwnd);

#ifdef _WIN32
        // Blur the real desktop behind the overlay window.
        enableDwmBlurBehind(hwnd);
#endif
#else
        int virtualX = 0, virtualY = 0, virtualWidth = 0, virtualHeight = 0;
        PlatformWindow::getVirtualScreenRect(virtualX, virtualY, virtualWidth, virtualHeight);

        if (w <= 0 || h <= 0)
        {
            w = virtualWidth;
            h = virtualHeight;
        }

        HWND hwnd = static_cast<HWND>(
            PlatformWindow::createOverlayWindow(virtualX, virtualY, virtualWidth, virtualHeight));

        if (!hwnd)
        {
            LOG_ERROR("WindowManager", "Failed to create transparent overlay window");
            return nullptr;
        }
#endif

        auto win = std::make_unique<GRIMWindow>();
        win->hwnd = hwnd;
        win->name = name;
        win->visible = true;
#if defined(__APPLE__)
        win->isOverlay = true;  // BGFX stays on main window; overlay is UI-only so background stays transparent
#else
        win->isOverlay = false;
#endif
        win->width = virtualWidth;
        win->height = virtualHeight;

        registerWindow(std::move(win));
        LOG_PHASE("Transparent multi-monitor overlay created (" + 
                 std::to_string(virtualWidth) + "x" + std::to_string(virtualHeight) + ")", true);
        return get(name);
    }

    // ----- Non-transparent standard window path -----
#ifdef _WIN32
    HWND hwnd = CreateWindowExW(
        0, L"STATIC", std::wstring(name.begin(), name.end()).c_str(),
        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
        w, h, nullptr, nullptr, GetModuleHandle(nullptr), nullptr);
#else
    HWND hwnd = static_cast<HWND>(PlatformWindow::createBGFXInitWindow());
#endif

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
#ifdef _WIN32
            ShowWindow(w->hwnd, SW_SHOW);
#else
            PlatformWindow::setWindowVisible(w->hwnd, true);
#endif
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
#ifdef _WIN32
            ShowWindow(w->hwnd, SW_HIDE);
#else
            PlatformWindow::setWindowVisible(w->hwnd, false);
#endif
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
