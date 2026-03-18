#pragma once
#include "core/grim_platform.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <cstdint>
#include <bgfx/bgfx.h>
#include "helpers/vector2.hpp"

// ======================================================
// Window Type Enumeration
// ======================================================
enum class WindowType {
    PopupIcon,
    Console,
    Viewport,
    Graph,
    Overlay,
    Debug
};

// ======================================================
// Window Metadata
// ======================================================
struct WindowInfo {
    std::string name;
    std::string className;
    std::string tag;
    DWORD pid;

    Vec2 size;
    Vec2 position;
    Vec2 location;
    Vec2 mousePos;


    float alpha = 1.0f;
    bool topmost = false;
    bool acceptsInput = true;
    bool usesBGFX = true;
    bool visibleExternally = false;
    WindowType type;
};

// ======================================================
// Base Window Class
// ======================================================
class Window {
public:
    Window();
    virtual ~Window();

    bool create(const WindowInfo& info, HWND parent = nullptr);
    void destroy();

    void show(bool visible = true);
    void setTitle(const std::string& title);
    void resize(float width, float height);
    void move(float x, float y);
    void setAlpha(float alpha);
    void setLocation(const Vec2& loc);
    void setTopmost(bool enable);

    HWND getHandle() const { return hwnd; }
    HWND getParent() const { return parent; }
    const WindowInfo& getInfo() const { return info; }
    WindowType getType() const { return info.type; }
    bgfx::ViewId getViewId() const { return viewId; }

    // Per-frame and render callbacks
    virtual void onUpdate();
    virtual void onRender();
    virtual void onMessage(UINT msg, WPARAM wParam, LPARAM lParam);

    // Input events
    std::function<void()> onClick;
    std::function<void(float, float)> onMouseMove;

    // External message/event routing
    std::function<void(UINT, WPARAM, LPARAM)> onEvent;

protected:
    // Core handles
    HWND hwnd = nullptr;
    HWND parent = nullptr;
    WindowInfo info{};
    bool visible = false;

    // BGFX
    std::vector<bgfx::TextureHandle> textures;
    bgfx::ProgramHandle program = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle alphaUniform = BGFX_INVALID_HANDLE;
    bgfx::FrameBufferHandle framebuffer = BGFX_INVALID_HANDLE;
    bgfx::ViewId viewId = 0;

    // State flags
    bool dirty = false;
    bool minimized = false;
    bool focused = false;
    bool hover = false;
    bool clicked = false;
    bool autoHide = false;
    bool persistent = false;
    bool usesBGFX = true;

    // Interaction
    Vec2 mousePos;
    uint32_t zOrder = 0;
    uint64_t lastUpdateTime = 0;

    // Manager reference
    std::weak_ptr<class WindowManager> managerRef;

    // WndProc routing
    static LRESULT CALLBACK StaticWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
    LRESULT InstanceWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
};
