#include "bootstrap.hpp"

#include "../control/devices/server/device_comm_server.hpp"
#include "../core/input/InputBindings.hpp"
#include "../core/platform_clipboard.hpp"
#include "../core/platform_input.hpp"
#include "../core/platform_window.hpp"
#include "../core/window_manager.hpp"
#include "../geospatial/geospatial_runtime.hpp"
#include "../helpers/mouse.hpp"
#include "../logger.hpp"
#include "../perception/digital/DigitalCaptureSource.hpp"
#include "../perception/digital/DigitalEnvironmentLoop.hpp"
#include "../perception/digital/DigitalPerceptionPrimitivesLoop.hpp"
#include "../perception/physical/PhysicalEnvironmentLoop.hpp"
#include "../popup_ui/popup_ui.hpp"
#include "../resources.hpp"
#include "../ui/console_panel.hpp"
#include "../ui/primitives/ui_native_3d_viewport_attachment.hpp"
#include "../ui/primitives/ui_native_3d_viewport_clear_pass.hpp"
#include "../ui/ui_data_hub.hpp"
#include "../ui/ui_digital_environment_panel.hpp"
#include "../ui/ui_geospatial_panel.hpp"
#include "../ui/ui_physical_environment_panel.hpp"
#include "../ui/ui_root.hpp"
#include "../ui/ui_settings_menu.hpp"
#include "../ui/ui_storage_panel.hpp"
#include "../ui/ui_surface_renderer_bridge.hpp"
#include "../ui/ui_training_panel.hpp"

#include <filesystem>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

std::shared_ptr<UITrainingPanel> g_trainingPanel;

namespace {

void loadUIFonts(int argc, char** argv)
{
    std::string fontPath = findAnyFontInResources(argc, argv);
    if (!fontPath.empty()) {
        UIRoot::get().getRenderer().setFont(fontPath, 16);
    } else {
        LOG_ERROR("UIRoot", "No font file found - text will not render");
    }

    std::string iconFontName;
    if (aiConfig.contains("ui") && aiConfig["ui"].contains("icon_font")) {
        iconFontName = aiConfig["ui"]["icon_font"].get<std::string>();
    }
    if (iconFontName.empty()) {
        return;
    }

    std::string iconFontPath;
    const std::filesystem::path fontsDir =
        std::filesystem::path(argv[0]).parent_path() / "resources" / "fonts";
    if (std::filesystem::exists(fontsDir)) {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(fontsDir)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            const std::string stem = entry.path().stem().string();
            if (stem == iconFontName || entry.path().filename().string() == iconFontName) {
                iconFontPath = entry.path().string();
                break;
            }
        }
    }

    if (!iconFontPath.empty()) {
        UIRoot::get().getRenderer().loadIconFont(iconFontPath);
    } else {
        LOG_DEBUG("UIRoot", "Icon font '" + iconFontName + "' not found in resources/fonts/");
    }
}

} // namespace

UIBootstrapResult bootstrapUI(
    GRIM::DeviceCommServer& deviceCommServer,
    int argc,
    char** argv)
{
    PlatformInput::initialize();
    LOG_PHASE("Platform input initialized", true);
    PlatformClipboard::initialize();
    LOG_PHASE("Platform clipboard initialized", true);
    Mouse::initialize();
    LOG_PHASE("Mouse initialized", true);

    LOG_PHASE("Initializing BGFX on main thread", true);
    void* bgfxWindow = PlatformWindow::createBGFXInitWindow();
    if (!bgfxWindow) {
        throw std::runtime_error(
            "bootstrapUI: platform layer returned NULL BGFX initialization window");
    }
    if (!WindowManager::initGlobalBGFX(static_cast<HWND>(bgfxWindow))) {
        PlatformWindow::destroyBGFXInitWindow(bgfxWindow);
        throw std::runtime_error("bootstrapUI: failed to initialize BGFX");
    }

    auto mainWindow = std::make_unique<GRIMWindow>();
    mainWindow->hwnd = static_cast<HWND>(bgfxWindow);
    mainWindow->name = "main";
    mainWindow->visible = false;
    mainWindow->isOverlay = false;
    mainWindow->width = 1;
    mainWindow->height = 1;
    WindowManager::registerWindow(std::move(mainWindow));
    WindowManager::processMainThreadUpdates();
    PlatformWindow::setWindowVisible(bgfxWindow, false);
    LOG_PHASE("BGFX initialized successfully", true);

    int virtualX = 0;
    int virtualY = 0;
    int virtualWidth = 0;
    int virtualHeight = 0;
    PlatformWindow::getVirtualScreenRect(
        virtualX, virtualY, virtualWidth, virtualHeight);
    LOG_DEBUG("UIBootstrap", "Virtual screen: " + std::to_string(virtualWidth) + "x" +
              std::to_string(virtualHeight) + " at (" + std::to_string(virtualX) + "," +
              std::to_string(virtualY) + ")");

    GRIMWindow* overlayWindow = WindowManager::createOverlay(
        "overlay", virtualWidth, virtualHeight, true);
    if (!overlayWindow) {
        throw std::runtime_error("bootstrapUI: failed to create transparent overlay window");
    }

    std::string captureExclusionError;
    if (!GRIM::Perception::Digital::SetDigitalCaptureExcludedWindow(
            overlayWindow->hwnd, &captureExclusionError)) {
        LOG_ERROR("DigitalCapture", "Could not exclude GRIM overlay from capture: " +
                  captureExclusionError);
    } else {
        LOG_PHASE("GRIM overlay excluded from digital capture", true);
    }

    GRIM::Perception::Digital::DigitalEnvironmentConfig digitalCaptureConfig;
    digitalCaptureConfig.request.mode =
        GRIM::Perception::Digital::DigitalCaptureMode::ActiveMonitor;
    digitalCaptureConfig.capture_interval = std::chrono::milliseconds(1000);
    GRIM::Perception::Digital::StartDigitalEnvironment(digitalCaptureConfig);
    GRIM::Perception::Digital::StartDigitalPerceptionPrimitives();
    LOG_PHASE("Digital environment capture started", true);

    WindowManager::updateWindowDimensions("overlay", virtualWidth, virtualHeight);
    WindowManager::processMainThreadUpdates();
    LOG_PHASE("Unified overlay initialized successfully (multi-monitor)", true);

    UIRoot::get().init(
        overlayWindow->hwnd, overlayWindow->width, overlayWindow->height);
    loadUIFonts(argc, argv);

    auto consolePanel = std::make_shared<ConsolePanel>();
    auto settingsPanel = std::make_shared<UISettingsMenu>();
    auto trainingPanel = std::make_shared<UITrainingPanel>();
    auto dataHubPanel = std::make_shared<UIDataHubPanel>();
    auto storagePanel = std::make_shared<UIStoragePanel>();
    auto geoSpatialPanel = std::make_shared<UIGeoSpatialPanel>();
    auto geoSpatialRuntime = std::make_shared<GRIM::GeoSpatial::GeoSpatialRuntime>();
    auto geoSpatialViewportAttachment = std::make_shared<UINative3DViewportAttachment>(
        UIRoot::get().getHWND(), "GeoSpatialViewport");

    UINative3DViewportClearPass::registerClearPass(
        "GeoSpatialViewportClear", geoSpatialViewportAttachment, 0x153A4AFF);
    geoSpatialRuntime->setViewportReady(true);
    geoSpatialRuntime->setViewportAttachment(geoSpatialViewportAttachment);
    geoSpatialPanel->setController(geoSpatialRuntime.get());
    geoSpatialPanel->setViewportAttachment(geoSpatialViewportAttachment);

    GRIM::Perception::Physical::RegisterPhysicalEnvironmentDeviceServer(&deviceCommServer);
    auto physicalEnvironmentPanel = std::make_shared<UIPhysicalEnvironmentPanel>();
    auto digitalEnvironmentPanel = std::make_shared<UIDigitalEnvironmentPanel>();
    storagePanel->setServer(&deviceCommServer);

    g_trainingPanel = trainingPanel;

    const std::shared_ptr<UIPanel> panels[] = {
        consolePanel,
        settingsPanel,
        trainingPanel,
        dataHubPanel,
        storagePanel,
        geoSpatialPanel,
        physicalEnvironmentPanel,
        digitalEnvironmentPanel,
    };
    for (const auto& panel : panels) {
        panel->setVisible(false);
        UIRoot::get().addPanel(panel);
    }

    GRIM::InputBindings::loadRuntimeConfig(aiConfig);

#if defined(__APPLE__)
    PlatformWindow::setTextInputCallback([](const std::string& text) {
        UIRoot::get().injectTextInput(text);
    });
#endif

    UISurfaceRendererBridge::install();
    LOG_PHASE("UIRoot and panels initialized (hidden)", true);
    LOG_PHASE("UISurfaceRendererBridge installed", true);

    std::thread([]() {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        runPopupUI(256, 256);
    }).detach();
#ifdef _WIN32
    LOG_PHASE("Popup UI launched (layered window)", true);
#else
    LOG_PHASE("Popup UI launched (macOS NSWindow)", true);
#endif
    WindowManager::processMainThreadUpdates();

    return {*overlayWindow, std::move(geoSpatialRuntime)};
}