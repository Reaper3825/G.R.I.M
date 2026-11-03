#include "system_tick.hpp"
#include "plugin_manager.hpp"
#include "scheduling_service.hpp"
#include "ui/console_panel.hpp"
#include "popup_ui/popup_ui.hpp"

#include "../geospatial/geospatial_runtime.hpp"
#include "../helpers/key.hpp"
#include "../helpers/mouse.hpp"
#include "../perception/digital/DigitalContextProjector.hpp"
#include "../perception/physical/PhysicalCameraCalibrator.hpp"
#include "../perception/physical/PhysicalEnvironmentLoop.hpp"
#include "../perception/physical/PhysicalFrameBus.hpp"
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
    static auto nextSchedulePoll=std::chrono::steady_clock::time_point{};
    if(frameStart>=nextSchedulePoll) {
        nextSchedulePoll=frameStart+std::chrono::seconds(1);
        if(auto console=std::dynamic_pointer_cast<ConsolePanel>(UIRoot::get().getPanelByKey("console"))) {
            const auto notifications=grim::scheduling::schedulingService().takeDue(grim::scheduling::scheduleNow());
            for(const auto& message:notifications) console->getHistory().push("[Schedule] "+message,0xFF66CCFF,"system");
            if(!notifications.empty()) {UIRoot::get().setVisible("console",true);showPopup();notifyPopupActivity();}
        }
    }

    Perception::Physical::TickPhysicalEnvironment();
    // Calibration is an exclusive raw-camera workflow. Do not schedule the
    // competing interaction/inference/world pipeline while it is collecting
    // samples; this also prevents calibration-board observations from becoming
    // durable environmental memories.
    if (!Perception::Physical::IsPhysicalCameraCalibrationCaptureActive()
        && Perception::Physical::PhysicalFrameBus::Instance().HasCalibratedModelFrame()) {
        Perception::Physical::TickPhysicalInteraction();
        Perception::Physical::TickPhysicalGestureControl();
        Perception::Physical::TickPhysicalPerceptionPrimitives();
        Perception::Physical::TickPhysicalSpatialGrounding();
        Perception::Physical::TickPhysicalLocalization();
        Perception::Physical::TickPhysicalWorldState();
        Perception::Physical::TickPhysicalWorldStateContextProjector();
        Perception::Physical::TickPhysicalWorldStateMemoryWriter();
    }
    Perception::Digital::TickDigitalContextProjector();
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
