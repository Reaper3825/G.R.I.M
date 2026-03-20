#pragma once
#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <atomic>
#include "overlay_renderer.hpp"
#include "ui_panel.hpp"
#include "logger.hpp"
#include "core/grim_platform.h"

// Include plugin.hpp for GRIM_HOST_API macro
#include "core/plugin.hpp"

struct InputState; // Forward declaration

class GRIM_HOST_API UIRoot {
public:
    static UIRoot& get() {
        static UIRoot instance;
        return instance;
    }

    void init(HWND hwnd, uint32_t width, uint32_t height);
    void shutdown();

    void update(const InputState& input, float dt);
    void draw();

    void addPanel(const std::shared_ptr<UIPanel>& panel);
    void removePanel(const std::string& name);
    std::shared_ptr<UIPanel> getPanel(const std::string& name);
    void setVisible(const std::string& name, bool visible);
    void bringToFront(const std::string& name);  // Move panel to top of z-order
    void postTask(std::function<void()> task);

    // Check if a screen position should receive input (is over visible UI)
    bool shouldReceiveInputAt(float x, float y) const;
    
    // Check if any UI panel is currently visible
    bool hasVisiblePanels() const;
    
    // Check if any panel is currently being dragged or resized
    bool isAnyPanelDragging() const;
    
    // Check if UI consumed input this frame (for blocking pass-through)
    bool didConsumeInput() const { return m_inputConsumed; }
    
    // Inject text input from WM_CHAR messages
    void injectTextInput(const std::string& text);
    
    // Get pending text input
    std::string consumeTextInput();

    HWND getHWND() const { return m_hwnd; }
    uint32_t getWidth() const { return m_width; }
    uint32_t getHeight() const { return m_height; }
    
    // Get renderer instance for font updates
    OverlayRenderer& getRenderer() { return m_renderer; }

    struct MonitorRect {
        Vec2 origin{0.0f, 0.0f};
        Vec2 size{0.0f, 0.0f};
    };
    MonitorRect getMonitorRectAt(const Vec2& point) const;

private:
    UIRoot() = default;
    void processPendingTasks();
    std::vector<std::shared_ptr<UIPanel>> snapshotPanels() const;
    bool isUIThread() const { return std::this_thread::get_id() == m_uiThreadId; }
    void updateWindowZOrder();  // Helper to adjust window position based on visibility
    
    HWND m_hwnd = nullptr;
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    std::vector<std::shared_ptr<UIPanel>> m_panels;
    std::unordered_map<std::string, std::shared_ptr<UIPanel>> m_panelMap;
    OverlayRenderer m_renderer;  // Changed from UIRenderer to OverlayRenderer
    static constexpr int m_activePanelIndex = -1;  // Active panel always has zOrder = -1
    
    std::string m_pendingTextInput; // Buffer for WM_CHAR input
    bool m_inputConsumed = false;   // Track if UI consumed input this frame

    std::thread::id m_uiThreadId;
    mutable std::shared_mutex m_panelMutex;
    std::mutex m_taskMutex;
    std::vector<std::function<void()>> m_pendingTasks;
    std::atomic_flag m_drawGuard = ATOMIC_FLAG_INIT;
};
