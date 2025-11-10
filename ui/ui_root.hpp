#pragma once
#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include "overlay_renderer.hpp"
#include "ui_panel.hpp"
#include "logger.hpp"
#include <windows.h>

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
    std::shared_ptr<UIPanel> getPanel(const std::string& name);
    void setVisible(const std::string& name, bool visible);

    // Check if a screen position should receive input (is over visible UI)
    bool shouldReceiveInputAt(float x, float y) const;
    
    // Check if any UI panel is currently visible
    bool hasVisiblePanels() const;
    
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

private:
    UIRoot() = default;
    void updateWindowZOrder();  // Helper to adjust window position based on visibility
    
    HWND m_hwnd = nullptr;
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    std::vector<std::shared_ptr<UIPanel>> m_panels;
    std::unordered_map<std::string, std::shared_ptr<UIPanel>> m_panelMap;
    OverlayRenderer m_renderer;  // Changed from UIRenderer to OverlayRenderer
    
    std::string m_pendingTextInput; // Buffer for WM_CHAR input
    bool m_inputConsumed = false;   // Track if UI consumed input this frame
};
