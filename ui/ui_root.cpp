#include "ui_root.hpp"
#include "logger.hpp"
#include "input_parser.hpp"
#include "../MMO/Core/HardwareInventory.hpp"
#include "core/platform_window.hpp"
#include "ui_theme.hpp"
#include <nlohmann/json.hpp>

extern GRIM::MMO::HardwareInventory g_hardwareInventory;
extern nlohmann::json aiConfig;

void UIRoot::init(HWND hwnd, uint32_t width, uint32_t height)
{
    LOG_DEBUG("UIRoot", "Initializing UIRoot with overlay renderer");
    
    m_hwnd = hwnd;
    m_width = width;
    m_height = height;
    m_uiThreadId = std::this_thread::get_id();
    
    // Initialize the overlay renderer
    m_renderer.init(hwnd, width, height);

#if defined(__APPLE__)
    // Apply saved blur settings from config.
    {
        bool blurEnabled = true;
        float blurOpacity = 0.99f;
        int blurIntensity = 2;
        if (aiConfig.contains("blur") && aiConfig["blur"].is_object()) {
            blurEnabled = aiConfig["blur"].value("enabled", true);
            blurOpacity = aiConfig["blur"].value("opacity", 0.99f);
            blurIntensity = aiConfig["blur"].value("intensity", 2);
        }
        PlatformWindow::setOverlayBlurStyle(hwnd, blurEnabled, blurOpacity, blurIntensity);
    }
#endif

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
    // Reset consumption flag each frame
    m_inputConsumed = false;
    
    processPendingTasks();
    UIPanel::setCanvasSize({static_cast<float>(m_width), static_cast<float>(m_height)});
    
    // Create a modified input state with injected text
    InputState modifiedInput = input;
    modifiedInput.textInput = consumeTextInput();
    
    // Check if cursor is over any visible UI
    bool cursorOverUI = shouldReceiveInputAt(input.mousePos.x, input.mousePos.y);
    
    // Update all visible panels with the modified input
    // Iterate in REVERSE order so top panels get input first
    auto panels = snapshotPanels();
    std::shared_ptr<UIPanel> clickedPanel = nullptr;
    bool anyPanelDragging = false;
    
    // First pass: check if any panel is currently dragging or resizing
    for (auto& panel : panels)
    {
        if (panel && panel->isVisible() && (panel->isDragging() || panel->isResizing()))
        {
            anyPanelDragging = true;
            break;
        }
    }
    
    for (int i = static_cast<int>(panels.size()) - 1; i >= 0; --i)
    {
        auto& panel = panels[i];
        if (!panel || !panel->isVisible()) continue;
        
        Vec2 pos = panel->getPosition();
        Vec2 size = panel->getSize();
        bool isOver = (input.mousePos.x >= pos.x && input.mousePos.x <= pos.x + size.x &&
                      input.mousePos.y >= pos.y && input.mousePos.y <= pos.y + size.y);
        
        // If a panel is dragging, only update that panel (the topmost one)
        // Otherwise, only update topmost panel under cursor
        if (!clickedPanel && (!anyPanelDragging || i == static_cast<int>(panels.size()) - 1))
        {
            panel->update(modifiedInput, dt);
            
            if (isOver)
            {
                m_inputConsumed = true;
                
                // On click, mark for bringing to front
                if (input.mousePressed[0])
                {
                    clickedPanel = panel;
                }
            }
        }
    }
    
    // Bring clicked panel to front after iteration
    if (clickedPanel)
    {
        std::unique_lock lock(m_panelMutex);
        
        // Find panel in vector
        auto it = std::find(m_panels.begin(), m_panels.end(), clickedPanel);
        if (it != m_panels.end() && it != m_panels.end() - 1)
        {
            // Increment z-order of all visible panels except clicked one
            for (auto& p : m_panels)
            {
                if (p && p->isVisible() && p != clickedPanel)
                {
                    p->setZOrder(p->getZOrder() + 1);
                }
            }
            
            // Set clicked panel to z-order -1 (active/top panel)
            clickedPanel->setZOrder(-1);
            
            // Move to end (drawn last = on top)
            m_panels.erase(it);
            m_panels.push_back(clickedPanel);
        }
    }
}

void UIRoot::draw()
{
    if (m_drawGuard.test_and_set(std::memory_order_acquire)) {
        LOG_DEBUG("UIRoot", "Skipping draw() - already rendering (re-entrant call blocked)");
        return;
    }
    
    auto panels = snapshotPanels();

#if defined(__APPLE__)
    // macOS OS-blur is applied via NSVisualEffectView, which would otherwise tint
    // the entire transparent overlay window. Constrain the blur to our panel
    // rectangles by updating the blur mask each frame.
    {
        std::vector<float> blurRects;
        blurRects.reserve(panels.size() * 4);

        for (auto& panel : panels) {
            if (!panel || !panel->isVisible())
                continue;

            Vec2 pos = panel->getPosition();
            Vec2 size = panel->getSize();

            blurRects.push_back(pos.x);
            blurRects.push_back(pos.y);
            blurRects.push_back(size.x);
            blurRects.push_back(size.y);
        }

        if (!blurRects.empty()) {
            PlatformWindow::setOverlayBlurMask(m_hwnd,
                                                blurRects.data(),
                                                static_cast<int>(blurRects.size() / 4),
                                                UITheme::Sizes::BorderRadius);
        } else {
            PlatformWindow::setOverlayBlurMask(m_hwnd, nullptr, 0, UITheme::Sizes::BorderRadius);
        }
    }
#endif
    
    // Begin frame - clears to transparent
    m_renderer.beginFrame();

    // Ensure frost texture exists (generated once; precomputed noise has no dots/shapes).
    m_renderer.drawBackdrop(m_width, m_height);

    // Draw all visible panels
    for (auto& panel : panels)
    {
        if (panel && panel->isVisible())
        {
            panel->drawOverlay(m_renderer);
        }
    }
    
    // End frame - updates layered window
    m_renderer.endFrame();
    
    m_drawGuard.clear(std::memory_order_release);
}

void UIRoot::addPanel(const std::shared_ptr<UIPanel>& panel)
{
    auto task = [this, panel]() {
        if (!panel)
        {
            LOG_ERROR("UIRoot", "Attempted to add null panel");
            return;
        }

        std::unique_lock lock(m_panelMutex);
        m_panels.push_back(panel);

        std::string panelName = panel->getTitle();
        if (panelName.empty())
        {
            panelName = "panel_" + std::to_string(m_panels.size());
        }

        m_panelMap[panelName] = panel;
        LOG_DEBUG("UIRoot", "Added panel: " + panelName);
    };

    if (!isUIThread())
    {
        postTask(task);
    }
    else
    {
        task();
    }
}

void UIRoot::removePanel(const std::string& name)
{
    auto task = [this, name]() {
        std::unique_lock lock(m_panelMutex);
        auto mapIt = m_panelMap.find(name);
        if (mapIt == m_panelMap.end()) return;

        auto panel = mapIt->second;
        m_panelMap.erase(mapIt);

        for (auto it = m_panels.begin(); it != m_panels.end(); ++it) {
            if (*it == panel) {
                m_panels.erase(it);
                break;
            }
        }

        LOG_DEBUG("UIRoot", "Removed panel: " + name);
        lock.unlock();
        updateWindowZOrder();
    };

    if (!isUIThread()) {
        postTask(task);
    } else {
        task();
    }
}

std::shared_ptr<UIPanel> UIRoot::getPanel(const std::string& name)
{
    std::shared_lock lock(m_panelMutex);
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
    
#ifdef _WIN32
    if (hasVisiblePanels())
    {
        ShowWindow(m_hwnd, SW_SHOW);
        SetWindowPos(m_hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                    SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW);
        SetForegroundWindow(m_hwnd);
        SetFocus(m_hwnd);
    }
    else
    {
        ShowWindow(m_hwnd, SW_HIDE);
    }
#else
    PlatformWindow::setWindowVisible(m_hwnd, hasVisiblePanels());
#endif
}

void UIRoot::setVisible(const std::string& name, bool visible)
{
    auto task = [this, name, visible]() {
        auto panel = getPanel(name);
        if (panel)
        {
            panel->setVisible(visible);
            LOG_DEBUG("UIRoot", "Set panel '" + name + "' visibility to " + (visible ? "true" : "false"));
            
            // Bring newly visible panels to front
            if (visible)
            {
                bringToFront(name);
            }
            
            updateWindowZOrder();
        }
    };

    if (!isUIThread())
    {
        postTask(std::move(task));
    }
    else
    {
        task();
    }
}

void UIRoot::bringToFront(const std::string& name)
{
    std::unique_lock lock(m_panelMutex);
    
    // Find panel in vector
    for (size_t i = 0; i < m_panels.size(); ++i)
    {
        if (m_panels[i] && m_panels[i]->getTitle() == name)
        {
            // Already at front
            if (i == m_panels.size() - 1)
            {
                return;
            }
            
            // Move to end (drawn last = on top)
            auto panel = m_panels[i];
            
            // Increment z-order of all visible panels except this one
            for (auto& p : m_panels)
            {
                if (p && p->isVisible() && p != panel)
                {
                    p->setZOrder(p->getZOrder() + 1);
                }
            }
            
            // Set this panel to z-order -1 (active/top panel)
            panel->setZOrder(-1);
            
            m_panels.erase(m_panels.begin() + i);
            m_panels.push_back(panel);
            
            LOG_DEBUG("UIRoot", "Brought panel '" + name + "' to front (active z-order: -1, array index: " + 
                      std::to_string(m_panels.size() - 1) + ")");
            return;
        }
    }
}

bool UIRoot::shouldReceiveInputAt(float x, float y) const
{
    // Check if position is over any visible panel
    auto panels = snapshotPanels();
    for (const auto& panel : panels)
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

bool UIRoot::hasVisiblePanels() const
{
    auto panels = snapshotPanels();
    for (const auto& panel : panels)
    {
        if (panel && panel->isVisible())
            return true;
    }
    return false;
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

UIRoot::MonitorRect UIRoot::getMonitorRectAt(const Vec2& point) const
{
    MonitorRect rect;
    rect.origin = {0.0f, 0.0f};
    rect.size = {static_cast<float>(m_width), static_cast<float>(m_height)};

    if (g_hardwareInventory.monitors.empty())
        return rect;

    Vec2 screenPoint{
        point.x + static_cast<float>(g_hardwareInventory.virtual_origin_x),
        point.y + static_cast<float>(g_hardwareInventory.virtual_origin_y)
    };

    const GRIM::MMO::MonitorInfo* chosen = nullptr;
    for (const auto& monitor : g_hardwareInventory.monitors)
    {
        float x1 = static_cast<float>(monitor.x);
        float y1 = static_cast<float>(monitor.y);
        float x2 = x1 + static_cast<float>(monitor.width);
        float y2 = y1 + static_cast<float>(monitor.height);

        if (screenPoint.x >= x1 && screenPoint.x <= x2 &&
            screenPoint.y >= y1 && screenPoint.y <= y2)
        {
            chosen = &monitor;
            break;
        }
    }

    if (!chosen)
    {
        for (const auto& monitor : g_hardwareInventory.monitors)
        {
            if (monitor.is_primary)
            {
                chosen = &monitor;
                break;
            }
        }

        if (!chosen)
        {
            chosen = &g_hardwareInventory.monitors.front();
        }
    }

    rect.origin.x = static_cast<float>(chosen->x - g_hardwareInventory.virtual_origin_x);
    rect.origin.y = static_cast<float>(chosen->y - g_hardwareInventory.virtual_origin_y);
    rect.size.x = static_cast<float>(chosen->width);
    rect.size.y = static_cast<float>(chosen->height);
    return rect;
}

void UIRoot::postTask(std::function<void()> task)
{
    if (!task) return;
    std::lock_guard lock(m_taskMutex);
    m_pendingTasks.push_back(std::move(task));
}

void UIRoot::processPendingTasks()
{
    std::vector<std::function<void()>> tasks;
    {
        std::lock_guard lock(m_taskMutex);
        tasks.swap(m_pendingTasks);
    }

    for (auto& task : tasks)
    {
        if (task) task();
    }
}

std::vector<std::shared_ptr<UIPanel>> UIRoot::snapshotPanels() const
{
    std::shared_lock lock(m_panelMutex);
    return m_panels;
}
