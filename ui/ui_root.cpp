#include "ui/input/ui_input_router.hpp"
#include "helpers/routed_input_scope.hpp"
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
    m_lastBlurRects.clear();
    m_blurMaskInitialized = false;
    
    // Initialize the overlay renderer
    m_renderer.init(hwnd, width, height);

    // Apply saved blur settings from config on startup so every platform starts
    // with the same runtime glass state before the settings menu is opened.
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

    LOG_PHASE("UIRoot initialized (overlay renderer)", true);
}

void UIRoot::shutdown()
{
    LOG_DEBUG("UIRoot", "Shutting down UIRoot");
    
    // Clear all panels
    m_panels.clear();
    m_panelKeyMap.clear();
    { std::lock_guard lock(m_taskMutex); m_pendingTasks.clear(); }
    
    // Shutdown renderer
    m_renderer.shutdown();
    
    m_hwnd = nullptr;
    m_width = 0;
    m_height = 0;
    m_lastBlurRects.clear();
    m_blurMaskInitialized = false;
    
    LOG_PHASE("UIRoot shutdown complete", true);
}

void UIRoot::update(const InputState& input, float dt)
{
    // Reset consumption flag each frame
    m_inputConsumed = false;
    
    processPendingTasks();
    UIPanel::setCanvasSize({static_cast<float>(m_width), static_cast<float>(m_height)});
    
    // Refresh cached panel rects + dragging state for lock-free WM_NCHITTEST
    updateHitTestCache();
    
    // Create a modified input state with injected text
    InputState modifiedInput = input;
    modifiedInput.textInput = consumeTextInput();
    
    auto panels = snapshotPanels();
    auto& router = GRIM::UI::InputRouter::get();
    for (const auto owner : {router.capture, router.popup, router.focus}) {
        if (!owner) continue;
        const auto found = std::find_if(panels.begin(), panels.end(), [&](const auto& panel) {
            return panel && panel->getPanelID() == owner.panel && panel->isVisible();
        });
        if (found == panels.end()) router.forgetPanel(owner.panel);
    }
    const bool cancelGesture = input.keyPressed[VK_ESCAPE] && bool(router.capture);
    router.beginFrame(input.interactionCancelled || cancelGesture);
    const uint64_t pointerPanel = router.pointerTarget(panels, [&](const auto& panel) -> uint64_t {
            if (!panel || !panel->isVisible()) return 0;
            const auto p = panel->getPosition(), extent = panel->getSize();
            if (input.mousePos.x >= p.x && input.mousePos.x < p.x + extent.x &&
                input.mousePos.y >= p.y && input.mousePos.y < p.y + extent.y &&
                !panel->shouldPassThroughAt(input.mousePos.x, input.mousePos.y)) {
                return panel->getPanelID();
            }
            return 0;
    });
    // A click away relinquishes keyboard ownership before routing this frame.
    if ((input.mousePressed[0] || input.mousePressed[1]) && !router.popup &&
        router.focus.panel != pointerPanel) router.focus = {};
    const uint64_t keyboardPanel = router.popup ? router.popup.panel :
        router.focus ? router.focus.panel : pointerPanel;
    std::shared_ptr<UIPanel> clickedPanel;
    for (auto it = panels.rbegin(); it != panels.rend(); ++it) {
        const auto& panel = *it;
        if (!panel || !panel->isVisible()) continue;
        InputState routed = modifiedInput;
        if (router.cancelled || panel->getPanelID() != pointerPanel) routed.suppressPointer();
        if (router.cancelled || panel->getPanelID() != keyboardPanel) routed.suppressKeyboard();
        routed.interactionCancelled = router.cancelled;
        router.currentPanel = panel->getPanelID();
        RoutedInputScope legacyScope(routed);
        // Always tick visible panels. Polling/animations must not depend on hover.
        panel->update(routed, dt);
        if (panel->getPanelID() == pointerPanel) {
            m_inputConsumed = true;
            if (input.mousePressed[0] && !router.cancelled) clickedPanel = panel;
        }
    }
    router.endFrame();
    PlatformWindow::setInputPointerCapture(m_hwnd, bool(router.capture));
    updateHitTestCache();

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

void UIRoot::updateOverlayInteraction(const InputState& input)
{
    const bool overUI = shouldReceiveInputAtCached(input.mousePos.x, input.mousePos.y) ||
                        isAnyPanelDraggingCached();
    PlatformWindow::setOverlayClickThrough(m_hwnd, !overUI);
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
        blurRects.reserve(panels.size() * 5);

        for (auto& panel : panels) {
            if (!panel || !panel->isVisible())
                continue;

            Vec2 pos = panel->getPosition();
            Vec2 size = panel->getSize();
            float cr = panel->isMaximized() ? 0.0f : UITheme::Sizes::BorderRadius;

            blurRects.push_back(pos.x);
            blurRects.push_back(pos.y);
            blurRects.push_back(size.x);
            blurRects.push_back(size.y);
            blurRects.push_back(cr);
        }

        if (!m_blurMaskInitialized || blurRects != m_lastBlurRects) {
            if (!blurRects.empty()) {
                PlatformWindow::setOverlayBlurMask(m_hwnd,
                                                    blurRects.data(),
                                                    static_cast<int>(blurRects.size() / 5));
            } else {
                PlatformWindow::setOverlayBlurMask(m_hwnd, nullptr, 0);
            }
            m_lastBlurRects = std::move(blurRects);
            m_blurMaskInitialized = true;
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

bool UIRoot::shouldReceiveInputAt(float x, float y) const
{
    return shouldReceiveInputAtCached(x, y);
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

bool UIRoot::isAnyPanelDragging() const
{
    if (GRIM::UI::InputRouter::get().capture) return true;
    auto panels = snapshotPanels();
    for (const auto& panel : panels)
    {
        if (panel && panel->isVisible() && (panel->isDragging() || panel->isResizing()))
            return true;
    }
    return false;
}

// ---------------------------------------------------------------
// Cached hit-test: updated once per frame, read lock-free from
// WM_NCHITTEST which Windows calls per mouse message (~100s/sec).
// ---------------------------------------------------------------
void UIRoot::updateHitTestCache()
{
    auto panels = snapshotPanels();
    int count = 0;
    int passThroughCount = 0;
    bool dragging = bool(GRIM::UI::InputRouter::get().capture);

    for (const auto& panel : panels)
    {
        if (!panel || !panel->isVisible()) continue;
        if (panel->isDragging() || panel->isResizing()) dragging = true;
        if (count < kMaxCachedPanels)
        {
            Vec2 pos = panel->getPosition();
            Vec2 size = panel->getSize();
            m_cachedRects[count] = {pos.x, pos.y, size.x, size.y};
            ++count;
        }

        std::vector<PanelRect> passThroughRects;
        panel->collectPassThroughRects(passThroughRects);
        for (const PanelRect& rect : passThroughRects) {
            if (passThroughCount >= kMaxCachedPassThroughRects)
                break;

            m_cachedPassThroughRects[passThroughCount] = {
                rect.origin.x,
                rect.origin.y,
                rect.size.x,
                rect.size.y
            };
            ++passThroughCount;
        }
    }

    m_cachedPassThroughRectCount.store(passThroughCount, std::memory_order_release);
    m_cachedRectCount.store(count, std::memory_order_release);
    m_cachedDragging.store(dragging, std::memory_order_release);
}

bool UIRoot::shouldReceiveInputAtCached(float x, float y) const
{
    int passThroughCount = m_cachedPassThroughRectCount.load(std::memory_order_acquire);
    for (int i = 0; i < passThroughCount; ++i)
    {
        const auto& r = m_cachedPassThroughRects[i];
        if (x >= r.x && x <= r.x + r.w &&
            y >= r.y && y <= r.y + r.h)
            return false;
    }

    int count = m_cachedRectCount.load(std::memory_order_acquire);
    for (int i = 0; i < count; ++i)
    {
        const auto& r = m_cachedRects[i];
        if (x >= r.x && x <= r.x + r.w &&
            y >= r.y && y <= r.y + r.h)
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
