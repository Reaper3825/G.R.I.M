#include "ui_root.hpp"
#include "ui/input/ui_input_router.hpp"
#include "dynamic_surface_panel.hpp"
#include "MMO/UI/UISurfaceRegistry.hpp"
#include <algorithm>
#include <stdexcept>
#include <utility>
#include "MMO/Core/ToolRegistry.hpp"

void UIRoot::bindPanelResource(const std::string& key, uint64_t runtime,
                              const std::string& provider, const std::string& instance) {
    if (!isUIThread()) throw std::logic_error("Panel resource binding requires the UI thread");
    const auto panel = getPanelByKey(key);
    if (!panel || panel->getPanelID() != runtime || m_panelResources.count(key))
        throw std::logic_error("Panel resource identity is missing, stale, or already bound");
    GRIM::MMO::ResourceDescriptor r;
    r.provider_name = provider; r.provider_instance = instance;
    r.resource_id = "panel:" + key; r.kind = "ui.panel";
    r.name = panel->getTitle(); r.runtime_id = runtime;
    r.operations = {"ui.inspect", "ui.show", "ui.hide", "ui.bring_to_front"};
    m_panelResources.emplace(key, std::move(r));
    synchronizePanelResources();
}

void UIRoot::synchronizePanelResources() {
    auto& registry = GRIM::MMO::ToolRegistry::instance();
    for (auto it = m_panelResources.begin(); it != m_panelResources.end();) {
        auto panel = getPanelByKey(it->first);
        auto& r = it->second;
        if (!panel || panel->getPanelID() != r.runtime_id) {
            registry.removeResource(r); it = m_panelResources.erase(it); continue;
        }
        r.name = panel->getTitle();
        const auto position = panel->getPosition(), size = panel->getSize();
        r.state = {{"visible", panel->isVisible()}, {"minimized", panel->isMinimized()},
            {"maximized", panel->isMaximized()}, {"position", {{"x", position.x}, {"y", position.y}}},
            {"size", {{"width", size.x}, {"height", size.y}}}, {"z_order", panel->getZOrder()}};
        if (!registry.publishResource(r)) { it = m_panelResources.erase(it); continue; }
        ++it;
    }
}

void UIRoot::registerPanelFactory(const std::string& implementationKey,
                                  const std::string& panelKey, PanelFactory factory,
                                  PanelCleanup cleanup)
{
    if (!isUIThread()) throw std::logic_error("Panel factories must be registered on the UI thread");
    const auto validKey = [](const std::string& key) {
        return !key.empty() && key.size() <= 256 &&
            key.find_first_of("\r\n\0", 0, 3) == std::string::npos;
    };
    if (!validKey(implementationKey) || !validKey(panelKey) || !factory)
        throw std::invalid_argument("Panel factory requires valid implementation and panel keys and a constructor");
    if (m_panelFactories.count(implementationKey))
        throw std::logic_error("Duplicate panel implementation: " + implementationKey);
    for (const auto& entry : m_panelFactories)
        if (entry.second.panelKey == panelKey)
            throw std::logic_error("Panel key already assigned to a factory: " + panelKey);
    m_panelFactories.emplace(implementationKey, PanelFactoryEntry{panelKey, std::move(factory), std::move(cleanup)});
}

std::shared_ptr<UIPanel> UIRoot::createPanelFromFactory(
    const std::string& implementationKey, bool initiallyVisible)
{
    if (!isUIThread()) throw std::logic_error("Panel construction must run on the UI thread");
    const auto found = m_panelFactories.find(implementationKey);
    if (found == m_panelFactories.end())
        throw std::invalid_argument("Unknown panel implementation: " + implementationKey);
    // Copy before invoking host code: construction must not retain a map iterator.
    const auto entry = found->second;
    if (getPanelByKey(entry.panelKey))
        throw std::logic_error("Panel already exists: " + entry.panelKey);
    auto panel = entry.construct();
    try {
        if (!panel || panel->getPanelKey() != entry.panelKey)
            throw std::logic_error("Panel factory returned a null panel or an unexpected key: " + implementationKey);
        panel->setVisible(initiallyVisible);
        if (entry.cleanup) m_panelCleanups.emplace(entry.panelKey, entry.cleanup);
        addPanel(panel);
        if (getPanelByKey(entry.panelKey) != panel)
            throw std::runtime_error("Could not attach factory panel: " + entry.panelKey);
    } catch (...) {
        m_panelCleanups.erase(entry.panelKey);
        if (entry.cleanup) {
            try { entry.cleanup(panel); }
            catch (...) { LOG_ERROR("UIRoot", "Factory rollback cleanup failed: " + entry.panelKey); }
        }
        if (panel && getPanelByKey(entry.panelKey) == panel) removePanel(entry.panelKey);
        throw;
    }
    return panel;
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
        // Stop native attachments and panel-owned activity before detaching.
        // Cleanup runs on the UI thread, outside the panel collection lock.
        panel->setVisible(false);
        auto cleanup = m_panelCleanups.extract(panel->getPanelKey());
        if (!cleanup.empty()) {
            try { cleanup.mapped()(panel); }
            catch (...) { LOG_ERROR("UIRoot", "Panel cleanup failed: " + panel->getPanelKey()); }
        }
        std::unique_lock lock(m_panelMutex);
        GRIM::UI::InputRouter::get().forgetPanel(panel->getPanelID());
        m_panelKeyMap.erase(panel->getPanelKey());
        const auto resource = m_panelResources.find(panel->getPanelKey());
        if (resource != m_panelResources.end()) {
            GRIM::MMO::ToolRegistry::instance().removeResource(resource->second);
            m_panelResources.erase(resource);
        }

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
