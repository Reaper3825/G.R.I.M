#include "ui_root.hpp"
#include "logger.hpp"
#include "input_parser.hpp"

void UIRoot::init(HWND hwnd, uint32_t width, uint32_t height)
{
    LOG_DEBUG("UIRoot", "Initializing UIRoot with overlay renderer");
    
    m_hwnd = hwnd;
    m_width = width;
    m_height = height;
    
    // Initialize the overlay renderer
    m_renderer.init(hwnd, width, height);
    
    LOG_PHASE("UIRoot initialized (overlay renderer)", true);
}

void UIRoot::shutdown()
{
    LOG_DEBUG("UIRoot", "Shutting down UIRoot");
    
    // Clear all panels
    m_panels.clear();
    m_panelMap.clear();
    
    // Shutdown renderer
    m_renderer.shutdown();
    
    m_hwnd = nullptr;
    m_width = 0;
    m_height = 0;
    
    LOG_PHASE("UIRoot shutdown complete", true);
}

void UIRoot::update(const InputState& input, float dt)
{
    // Create a modified input state with injected text
    InputState modifiedInput = input;
    modifiedInput.textInput = consumeTextInput();
    
    // Update all visible panels with the modified input
    for (auto& panel : m_panels)
    {
        if (panel && panel->isVisible())
        {
            panel->update(modifiedInput, dt);
        }
    }
}

void UIRoot::draw()
{
    // ? FIX: Lock to prevent re-entrant rendering
    static bool isRendering = false;
    if (isRendering) {
        LOG_DEBUG("UIRoot", "Skipping draw() - already rendering (re-entrant call blocked)");
        return;
    }
    
    isRendering = true;
    
    // Begin frame - clears to transparent
    m_renderer.beginFrame();
    
    // Draw all visible panels using drawOverlay
    for (auto& panel : m_panels)
    {
        if (panel && panel->isVisible())
        {
            panel->drawOverlay(m_renderer);
        }
    }
    
    // End frame - updates layered window
    m_renderer.endFrame();
    
    isRendering = false;
}

void UIRoot::addPanel(const std::shared_ptr<UIPanel>& panel)
{
    if (!panel)
    {
        LOG_ERROR("UIRoot", "Attempted to add null panel");
        return;
    }
    
    m_panels.push_back(panel);
    
    // Use panel title or generate name
    std::string panelName = panel->getTitle();
    if (panelName.empty())
    {
        panelName = "panel_" + std::to_string(m_panels.size());
    }
    
    m_panelMap[panelName] = panel;
    
    LOG_DEBUG("UIRoot", "Added panel: " + panelName);
}

std::shared_ptr<UIPanel> UIRoot::getPanel(const std::string& name)
{
    auto it = m_panelMap.find(name);
    if (it != m_panelMap.end())
    {
        return it->second;
    }
    
    LOG_DEBUG("UIRoot", "Panel not found: " + name);
    return nullptr;
}

void UIRoot::updateWindowZOrder()
{
    if (!m_hwnd)
        return;
    
    // Check if any panel is visible
    bool anyVisible = false;
    for (const auto& panel : m_panels)
    {
        if (panel && panel->isVisible())
        {
            anyVisible = true;
            break;
        }
    }
    
    if (anyVisible)
    {
        // Show window and bring to TOPMOST when UI is visible
        ShowWindow(m_hwnd, SW_SHOWNOACTIVATE);
        SetWindowPos(m_hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                    SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW);
    }
    else
    {
        // Hide window completely when no UI visible
        ShowWindow(m_hwnd, SW_HIDE);
    }
}

void UIRoot::setVisible(const std::string& name, bool visible)
{
    auto panel = getPanel(name);
    if (panel)
    {
        panel->setVisible(visible);
        LOG_DEBUG("UIRoot", "Set panel '" + name + "' visibility to " + (visible ? "true" : "false"));
        
        // Update window Z-order based on new visibility state
        updateWindowZOrder();
    }
}

bool UIRoot::shouldReceiveInputAt(float x, float y) const
{
    // Check if position is over any visible panel
    for (const auto& panel : m_panels)
    {
        if (!panel || !panel->isVisible())
            continue;
            
        Vec2 pos = panel->getPosition();
        Vec2 size = panel->getSize();
        
        if (x >= pos.x && x <= pos.x + size.x &&
            y >= pos.y && y <= pos.y + size.y)
        {
            return true; // Mouse is over a visible panel
        }
    }
    
    return false; // Mouse is not over any UI, pass through
}

void UIRoot::injectTextInput(const std::string& text)
{
    m_pendingTextInput += text;
}

std::string UIRoot::consumeTextInput()
{
    std::string result = m_pendingTextInput;
    m_pendingTextInput.clear();
    return result;
}
