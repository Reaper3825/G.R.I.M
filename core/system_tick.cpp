#include "system_tick.hpp"

#include "../geospatial/geospatial_runtime.hpp"
#include "../helpers/key.hpp"
#include "../helpers/mouse.hpp"
#include "../perception/digital/DigitalContextProjector.hpp"
#include "../perception/physical/PhysicalEnvironmentLoop.hpp"
#include "../perception/physical/PhysicalGestureControlLoop.hpp"
#include "../perception/physical/PhysicalInteractionLoop.hpp"
#include "../perception/physical/PhysicalLocalizationLoop.hpp"
#include "../perception/physical/PhysicalPerceptionPrimitivesLoop.hpp"
#include "../perception/physical/PhysicalSpatialGroundingLoop.hpp"
#include "../perception/physical/PhysicalWorldStateContextProjector.hpp"
#include "../perception/physical/PhysicalWorldStateLoop.hpp"
#include "../perception/physical/PhysicalWorldStateMemoryWriter.hpp"
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

    float wheelDelta = 0.0f;
    bool quitRequested = false;
    PlatformWindow::pumpEvents(wheelDelta, quitRequested);
    if (quitRequested) {
        WindowManager::requestMainLoopStop();
    }

    InputState input;
    input.captureFromHWND(overlayWindow.hwnd);
    input.mouseWheelDelta = wheelDelta;

    Mouse::updateFromInput(input);
    Key::updateFromInput(input);
    WakeKey::update();

    UI::processPanelShortcuts();

    Perception::Physical::TickPhysicalEnvironment();
    Perception::Physical::TickPhysicalInteraction();
    Perception::Physical::TickPhysicalGestureControl();
    Perception::Physical::TickPhysicalPerceptionPrimitives();
    Perception::Physical::TickPhysicalSpatialGrounding();
    Perception::Physical::TickPhysicalLocalization();
    Perception::Physical::TickPhysicalWorldState();
    Perception::Physical::TickPhysicalWorldStateContextProjector();
    Perception::Physical::TickPhysicalWorldStateMemoryWriter();
    Perception::Digital::TickDigitalContextProjector();
    geoSpatialRuntime.tick(kDeltaSeconds);

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