#include "ui_root.hpp"
#include "ui/input/ui_input_router.hpp"
#include "dynamic_surface_panel.hpp"
#include "MMO/UI/UISurfaceRegistry.hpp"
#include <algorithm>
#include <stdexcept>

void UIRoot::addPanel(const std::shared_ptr<UIPanel>& panel)
{
    auto task = [this, panel]() {
        if (!panel)
        {
            LOG_ERROR("UIRoot", "Attempted to add null panel");
            return;
        }

        std::unique_lock lock(m_panelMutex);
        if (std::find(m_panels.begin(), m_panels.end(), panel) != m_panels.end()) return;
        const auto& key = panel->getPanelKey();
        if (!key.empty() && m_panelKeyMap.count(key)) {
            LOG_ERROR("UIRoot", "Rejected duplicate panel key: " + key);
            return;
        }
        m_panels.push_back(panel);
        if (!key.empty()) m_panelKeyMap.emplace(key, panel);
        LOG_DEBUG("UIRoot", "Added panel: " + key + " / " + panel->getTitle());
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
        auto panel = getPanel(name);
        if (!panel) return;
        std::unique_lock lock(m_panelMutex);
        GRIM::UI::InputRouter::get().forgetPanel(panel->getPanelID());
        m_panelKeyMap.erase(panel->getPanelKey());

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
    const auto keyed = m_panelKeyMap.find(name);
    if (keyed != m_panelKeyMap.end()) return keyed->second;
    // Compatibility for existing host callers. Ambiguous titles are not identities.
    std::shared_ptr<UIPanel> match;
    for (const auto& panel : m_panels) if (panel->getTitle() == name) {
        if (match) return nullptr;
        match = panel;
    }
    if (match) return match;
    
    LOG_DEBUG("UIRoot", "Panel not found: " + name);
    return nullptr;
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
    if (!isUIThread()) {
        postTask([this, name] { bringToFront(name); });
        return;
    }
    auto target = getPanel(name);
    if (!target) return;
    std::unique_lock lock(m_panelMutex);
    
    // Find panel in vector
    for (size_t i = 0; i < m_panels.size(); ++i)
    {
        if (m_panels[i] == target)
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

void UIRoot::postTask(std::function<void()> task)
{
    if (!task) return;
    std::lock_guard lock(m_taskMutex);
    m_pendingTasks.push_back(std::move(task));
}

void UIRoot::processPendingTasks()
{
    if (!isUIThread()) throw std::logic_error("UI tasks must be processed on the UI thread");
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

std::shared_ptr<UIPanel> UIRoot::getPanelByKey(const std::string& key)
{
    std::shared_lock lock(m_panelMutex);
    const auto found = m_panelKeyMap.find(key);
    return found == m_panelKeyMap.end() ? nullptr : found->second;
}

namespace {
UIPanelInfo panelInfo(const UIPanel& panel) {
    return {panel.getPanelKey(), panel.getTitle(), panel.getPanelID(),
            panel.isVisible(), panel.isMinimized(), panel.isMaximized(),
            panel.getPosition(), panel.getSize(), panel.getZOrder()};
}
}

std::future<std::vector<UIPanelInfo>> UIRoot::requestPanelList()
{
    auto task = std::make_shared<std::packaged_task<std::vector<UIPanelInfo>()>>([this] {
        std::vector<UIPanelInfo> result;
        for (const auto& panel : snapshotPanels())
            if (!panel->getPanelKey().empty()) result.push_back(panelInfo(*panel));
        std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) { return a.key < b.key; });
        return result;
    });
    auto future = task->get_future();
    if (isUIThread()) (*task)();
    else postTask([task] { (*task)(); });
    return future;
}

std::future<UIPanelOperationResult> UIRoot::requestPanelOperation(
    const std::string& key, UIPanelOperation operation)
{
    auto task = std::make_shared<std::packaged_task<UIPanelOperationResult()>>([this, key, operation] {
        auto panel = getPanelByKey(key);
        if (!panel) return UIPanelOperationResult{false, "not_found", std::nullopt};
        switch (operation) {
        case UIPanelOperation::Inspect: break;
        case UIPanelOperation::Show:
            panel->setVisible(true);
            if (dynamic_cast<DynamicSurfacePanel*>(panel.get()))
                GRIM::MMO::UISurfaceRegistry::instance().show(key);
            bringToFront(key);
            updateWindowZOrder();
            break;
        case UIPanelOperation::Hide:
            panel->setVisible(false);
            if (dynamic_cast<DynamicSurfacePanel*>(panel.get()))
                GRIM::MMO::UISurfaceRegistry::instance().hide(key);
            updateWindowZOrder();
            break;
        case UIPanelOperation::BringToFront: bringToFront(key); break;
        default: return UIPanelOperationResult{false, "invalid_operation", std::nullopt};
        }
        return UIPanelOperationResult{true, "", panelInfo(*panel)};
    });
    auto future = task->get_future();
    if (isUIThread()) (*task)();
    else postTask([task] { (*task)(); });
    return future;
}

std::vector<std::shared_ptr<UIPanel>> UIRoot::snapshotPanels() const
{
    std::shared_lock lock(m_panelMutex);
    return m_panels;
}
