#include "ui_root.hpp"
#include "logger.hpp"
#include "input_parser.hpp"
#include "system_detect.hpp"

extern SystemInfo g_systemInfo;

void UIRoot::init(HWND hwnd, uint32_t width, uint32_t height)
{
    LOG_DEBUG("UIRoot", "Initializing UIRoot with overlay renderer");
    
    m_hwnd = hwnd;
    m_width = width;
    m_height = height;
    m_uiThreadId = std::this_thread::get_id();
    
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
    auto panels = snapshotPanels();
    for (auto& panel : panels)
    {
        if (panel && panel->isVisible())
        {
            panel->update(modifiedInput, dt);
            
            // If cursor is over this panel, mark input as consumed
            Vec2 pos = panel->getPosition();
            Vec2 size = panel->getSize();
            if (input.mousePos.x >= pos.x && input.mousePos.x <= pos.x + size.x &&
                input.mousePos.y >= pos.y && input.mousePos.y <= pos.y + size.y)
            {
                m_inputConsumed = true;
            }
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
    
    // Begin frame - clears to transparent
    m_renderer.beginFrame();
    
    // Draw all visible panels using drawOverlay
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
    
    if (hasVisiblePanels())
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
    auto task = [this, name, visible]() {
        auto panel = getPanel(name);
        if (panel)
        {
            panel->setVisible(visible);
            LOG_DEBUG("UIRoot", "Set panel '" + name + "' visibility to " + (visible ? "true" : "false"));
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

    if (g_systemInfo.monitors.empty())
        return rect;

    Vec2 screenPoint{
        point.x + static_cast<float>(g_systemInfo.virtualOriginX),
        point.y + static_cast<float>(g_systemInfo.virtualOriginY)
    };

    const MonitorInfo* chosen = nullptr;
    for (const auto& monitor : g_systemInfo.monitors)
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
        for (const auto& monitor : g_systemInfo.monitors)
        {
            if (monitor.isPrimary)
            {
                chosen = &monitor;
                break;
            }
        }

        if (!chosen)
        {
            chosen = &g_systemInfo.monitors.front();
        }
    }

    rect.origin.x = static_cast<float>(chosen->x - g_systemInfo.virtualOriginX);
    rect.origin.y = static_cast<float>(chosen->y - g_systemInfo.virtualOriginY);
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
