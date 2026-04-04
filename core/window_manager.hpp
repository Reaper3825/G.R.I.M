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

    // Pre-frame render callback (called before bgfx::frame())
    using PreFrameCallback = void(*)(uint32_t bgfxFrame);
    static void registerPreFrameCallback(PreFrameCallback cb);
    static bool hasPreFrameCallback();

private:
    static inline std::vector<std::unique_ptr<GRIMWindow>> s_windows;
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
    static inline PreFrameCallback s_preFrameCallback;
};

