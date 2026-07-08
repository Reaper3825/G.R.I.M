#pragma once
#include "grim_platform.h"
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <bgfx/bgfx.h>
#include <bgfx/platform.h>

struct GRIMWindow
{
    HWND hwnd = nullptr;
    std::string name;
    bool visible = true;
    bool isOverlay = false;
    int width = 0;
    int height = 0;
};


// window_manager.hpp
class WindowManager
{
public:
    static bool initGlobalBGFX(HWND mainHwnd);
    static bool isInitialized();
    static void shutdown();
    static GRIMWindow* createOverlay(const std::string& name, int w, int h, bool transparent);
    static GRIMWindow* get(const std::string& name);
    static void show(const std::string& name);
    static void hide(const std::string& name);
    static void setVisibility(const std::string& name, bool visible);
    static void beginFrame(uint16_t viewId, uint32_t clearColor);
    static void endFrame();
    static void registerWindow(std::unique_ptr<GRIMWindow> win);
    static bool processMainThreadUpdates();
    static bool hasPendingPlatformUpdate();
    static void updateWindowDimensions(const std::string& name, uint32_t width, uint32_t height);
    static void renderFrame();
    static void requestMainLoopStop();
    static bool isMainLoopStopRequested();
    static HWND getOverlayHWND();
    static GRIMWindow* ensureOverlay(int w, int h);

    enum class ViewIdRange
    {
        DefaultKeepalive,
        PanelViewport,
        UiCompositorTools,
        Popup3D,
        SubsystemReserved
    };

    struct ViewIdBlock
    {
        bgfx::ViewId first = 0;
        uint16_t count = 0;

        bgfx::ViewId at(uint16_t offset) const;
    };

    static bgfx::ViewId defaultViewId();
    static ViewIdBlock reserveViewIds(const std::string& owner, ViewIdRange range, uint16_t count);
    static void releaseViewIds(const std::string& owner);

    using RenderPassCallback = void(*)(uint32_t bgfxFrame);
    static void registerRenderPass(const std::string& name, RenderPassCallback callback, bool enabled = true);
    static void unregisterRenderPass(const std::string& name);
    static void setRenderPassEnabled(const std::string& name, bool enabled);
    static bool hasEnabledRenderPasses();

private:
    struct RenderPass
    {
        std::string name;
        RenderPassCallback callback = nullptr;
        bool enabled = true;
    };

    struct ViewIdReservation
    {
        std::string owner;
        ViewIdRange range = ViewIdRange::SubsystemReserved;
        ViewIdBlock block;
    };

    static inline std::vector<std::unique_ptr<GRIMWindow>> s_windows;
    static inline std::vector<RenderPass> s_renderPasses;
    static inline std::vector<ViewIdReservation> s_viewIdReservations;
    static inline bool s_bgfxInitialized = false;
    static inline std::mutex s_mutex;
    static inline HWND s_primaryWindow = nullptr;
    static inline uint32_t s_backbufferWidth = 1920;
    static inline uint32_t s_backbufferHeight = 1080;
    static inline uint32_t s_resetFlags = BGFX_RESET_VSYNC;
    static inline std::atomic<bool> s_platformUpdatePending{ false };
    static inline HWND s_pendingPlatformWindow = nullptr;
    static inline uint32_t s_pendingPlatformWidth = 0;
    static inline uint32_t s_pendingPlatformHeight = 0;
    static inline std::atomic<bool> s_mainLoopStop{ false };
};

