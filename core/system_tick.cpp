#include "system_tick.hpp"
#include "plugin_api_impl.hpp"
#include "../MMO/Core/Registries/ProviderRegistry.hpp"

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

void tickApplicationFrame(const GRIMWindow& overlayWindow)
{
    constexpr float kDeltaSeconds = 0.016f;
    constexpr auto kFrameDuration = std::chrono::milliseconds(16);
    const auto frameStart = std::chrono::steady_clock::now();
    static const auto firstFrameStart = frameStart;
    static auto previousFrameStart = frameStart;
    const auto deltaSeconds = std::chrono::duration<double>(frameStart - previousFrameStart).count();
    const auto offsetSeconds = std::chrono::duration<double>(frameStart - firstFrameStart).count();
    previousFrameStart = frameStart;

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
    PluginAPI::dispatchSystemTick(deltaSeconds, offsetSeconds);

    // Global bindings above retain raw hardware input. UI gestures stop when
    // another application owns activation; they cannot restart from a held key.
    input.interactionCancelled = !PlatformWindow::isInputWindowActive(overlayWindow.hwnd);
    MMO::ProviderRegistry::instance().dispatchProviderUpdates();
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
