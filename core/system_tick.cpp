#include "system_tick.hpp"
#include "plugin_manager.hpp"
#include "plugin_api_impl.hpp"

#include "../geospatial/geospatial_runtime.hpp"
#include "../helpers/key.hpp"
#include "../helpers/mouse.hpp"
#include "../ui/ui_root.hpp"
#include "../ui/ui_shortcuts.hpp"
#include "../wake/wake_key.hpp"
#include "input_parser.hpp"
#include "platform_window.hpp"
#include "window_manager.hpp"

#include <chrono>
#include <thread>

namespace GRIM {

void tickApplicationFrame(
    GeoSpatial::GeoSpatialRuntime& geoSpatialRuntime,
    const GRIMWindow& overlayWindow)
{
    constexpr float kDeltaSeconds = 0.016f;
    constexpr auto kFrameDuration = std::chrono::milliseconds(16);
    const auto frameStart = std::chrono::steady_clock::now();

    PlatformWindow::ScrollDelta wheelDelta;
    bool quitRequested = false;
    PlatformWindow::pumpEvents(wheelDelta, quitRequested);
    if (quitRequested) {
        WindowManager::requestMainLoopStop();
    }

    InputState input;
    input.captureFromHWND(overlayWindow.hwnd);
    input.mouseWheelDelta = wheelDelta.y * 120.0f;
    input.mouseWheelDeltaX = wheelDelta.x * 120.0f;

    Mouse::updateFromInput(input);
    Key::updateFromInput(input);
    WakeKey::update();

    UI::processPanelShortcuts();
    PluginManager::tick();
    PluginAPI::processTimers(); // Dispatch gated plugin callbacks; providers own due-work policy.

    geoSpatialRuntime.tick(kDeltaSeconds);

    // Global bindings above retain raw hardware input. UI gestures stop when
    // another application owns activation; they cannot restart from a held key.
    input.interactionCancelled = !PlatformWindow::isInputWindowActive(overlayWindow.hwnd);
    UIRoot::get().update(input, kDeltaSeconds);
    UIRoot::get().updateOverlayInteraction(input);
    UIRoot::get().draw();
    WindowManager::processMainThreadUpdates();
    WindowManager::renderFrame();

    Key::endFrame();
    Mouse::endFrame();

    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - frameStart);
    if (elapsed < kFrameDuration) {
        std::this_thread::sleep_for(kFrameDuration - elapsed);
    }
}

} // namespace GRIM
