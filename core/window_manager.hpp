#pragma once
#include <windows.h>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <bgfx/bgfx.h>
#include <bgfx/platform.h>

struct GRIMWindow
{
    HWND hwnd = nullptr;
    std::string name;
    bool visible = true;
    bool isOverlay = false;

    // ✅ Add these:
    int width = 0;
    int height = 0;
};


class WindowManager {
public:
    // =====================================================
    // Lifecycle
    // =====================================================
    static bool initGlobalBGFX(HWND mainHwnd);
    static void shutdown();

    // =====================================================
    // Window management
    // =====================================================
    static GRIMWindow* createOverlay(const std::string& name, int w, int h, bool transparent = true);
    static GRIMWindow* get(const std::string& name);
    static void show(const std::string& name);
    static void hide(const std::string& name);

    // =====================================================
    // Render + update
    // =====================================================
    static void beginFrame(uint16_t viewId, uint32_t clearColor = 0x00000000);
    static void endFrame();
    static void drawAll();

    static bool isInitialized() { return s_bgfxInitialized; }

private:
    static inline std::vector<std::unique_ptr<GRIMWindow>> s_windows{};
    static inline bool s_bgfxInitialized = false;
    static inline std::mutex s_mutex;
};
