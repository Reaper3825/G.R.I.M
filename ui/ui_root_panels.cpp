#include "ui_root.hpp"
#include "ui_panel_service.hpp"
#include "ui/input/ui_input_router.hpp"
#include "dynamic_surface_panel.hpp"
#include "MMO/UI/UISurfaceRegistry.hpp"
#include <algorithm>
#include <stdexcept>
#include <utility>
#include "MMO/Core/Registries/ProviderRegistry.hpp"

void UIRoot::bindPanelResource(const std::string& key, uint64_t runtime,
                              const std::string& provider, const std::string& instance) {
    if (!isUIThread()) throw std::logic_error("Panel resource binding requires the UI thread");
    const auto panel = getPanelByKey(key);
    if (!panel || panel->getPanelID() != runtime || m_panelResources.count(key))
        throw std::logic_error("Panel resource identity is missing, stale, or already bound");
    GRIM::MMO::RegisteredResource r;
    r.provider = {GRIM::MMO::ToolProviderType::Plugin, provider, instance, {}};
    r.resource_id = "panel:" + key; r.kind = "ui.panel";
    r.name = panel->getTitle(); r.runtime_id = runtime;
    m_panelResources.emplace(key, std::move(r));
    synchronizePanelResources();
}

void UIRoot::synchronizePanelResources() {
    auto& registry = GRIM::MMO::ProviderRegistry::instance();
    for (auto it = m_panelResources.begin(); it != m_panelResources.end();) {
        auto panel = getPanelByKey(it->first);
        auto& r = it->second;
        if (!panel || panel->getPanelID() != r.runtime_id) {
            registry.removeResource(r.provider, r); it = m_panelResources.erase(it); continue;
        }
        r.name = panel->getTitle();
        const auto position = panel->getPosition(), size = panel->getSize();
        r.state = {{"visible", panel->isVisible()}, {"minimized", panel->isMinimized()},
            {"maximized", panel->isMaximized()}, {"position", {{"x", position.x}, {"y", position.y}}},
            {"size", {{"width", size.x}, {"height", size.y}}}, {"z_order", panel->getZOrder()}};
        if (!registry.publishResource(r.provider, r)) { it = m_panelResources.erase(it); continue; }
        ++it;
    }
}

void UIRoot::requireUIThread() const {
    if (!isUIThread()) throw std::logic_error("Native panel lifecycle requires the UI thread");
}

void UIRoot::registerPanel(UIPanel* panel)
{
    // Never publish a partially constructed panel to another execution thread.
    // Constructors must not reenter the UI pump while constructing derived state.
    if (!isUIThread()) throw std::logic_error("Panel construction requires the UI thread");
    if (!panel) throw std::invalid_argument("Cannot register a null panel");
    std::unique_lock lock(m_panelMutex);
    if (std::find(m_panels.begin(), m_panels.end(), panel) != m_panels.end()) return;
    const auto& key = panel->getPanelKey();
    if (!key.empty() && m_panelKeyMap.count(key))
        throw std::logic_error("Duplicate panel key: " + key);
    // Roll back if allocation fails; a throwing base constructor has no destructor.
    if (!key.empty()) m_panelKeyMap.emplace(key, panel);
    try { m_panels.push_back(panel); }
    catch (...) { if (!key.empty()) m_panelKeyMap.erase(key); throw; }
}

void UIRoot::unregisterPanel(UIPanel* panel) noexcept
{
    if (!isUIThread()) std::terminate(); // Borrowed UI pointers require same-thread lifetime.
    GRIM::UI::forgetNativePanel(panel);
    GRIM::UI::InputRouter::get().forgetPanel(panel->getPanelID());
    std::unique_lock lock(m_panelMutex);
    const auto found = std::find(m_panels.begin(), m_panels.end(), panel);
    if (found == m_panels.end()) return;
    const auto& key = panel->getPanelKey();
    const auto resource = m_panelResources.find(key);
    if (resource != m_panelResources.end()) {
        GRIM::MMO::ProviderRegistry::instance().removeResource(resource->second.provider, resource->second);
        m_panelResources.erase(resource);
    }
    m_panelKeyMap.erase(key);
    m_panels.erase(found);
}

void UIRoot::addPanel(const std::shared_ptr<UIPanel>& panel)
{
    if (!isUIThread()) throw std::logic_error("Panel adoption requires the UI thread");
    if (!panel) throw std::invalid_argument("Cannot adopt a null panel");
    registerPanel(panel.get()); // Idempotent after automatic constructor registration.
    m_ownedPanels.emplace(panel.get(), panel);
}

void UIRoot::removePanel(const std::string& name)
{
    auto task = [this, name]() {
        auto panel = getPanel(name);
        if (!panel) return;
        // Stop native attachments and panel-owned activity before detaching.
        panel->setVisible(false);
        std::unique_lock lock(m_panelMutex);
        GRIM::UI::InputRouter::get().forgetPanel(panel->getPanelID());
        m_panelKeyMap.erase(panel->getPanelKey());
        const auto resource = m_panelResources.find(panel->getPanelKey());
        if (resource != m_panelResources.end()) {
            GRIM::MMO::ProviderRegistry::instance().removeResource(resource->second.provider, resource->second);
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
        m_ownedPanels.erase(panel); // Release only after unlocking; destructor unregisters.
        updateWindowZOrder();
    };

    if (!isUIThread()) {
        postTask(task);
    } else {
        task();
    }
}

UIPanel* UIRoot::getPanel(const std::string& name)
{
    std::shared_lock lock(m_panelMutex);
    const auto keyed = m_panelKeyMap.find(name);
    if (keyed != m_panelKeyMap.end()) return keyed->second;
    // Compatibility for existing host callers. Ambiguous titles are not identities.
    UIPanel* match = nullptr;
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

UIPanel* UIRoot::getPanelByKey(const std::string& key)
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
            if (dynamic_cast<DynamicSurfacePanel*>(panel))
                GRIM::MMO::UISurfaceRegistry::instance().show(key);
            bringToFront(key);
            updateWindowZOrder();
            break;
        case UIPanelOperation::Hide:
            panel->setVisible(false);
            if (dynamic_cast<DynamicSurfacePanel*>(panel))
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

std::vector<UIPanel*> UIRoot::snapshotPanels() const
{
    std::shared_lock lock(m_panelMutex);
    return m_panels;
}
