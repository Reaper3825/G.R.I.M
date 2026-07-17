#include "voice/voice_speak.hpp"
#include "bootstrap/bootstrap.hpp"
#include "aliases.hpp"
#include "popup_ui/popup_ui.hpp"
#include "logger.hpp"
#include "wake/wake_key.hpp"
#include "wake/wake_voice.hpp"
#include "memory/unified_memory.hpp"
#include "memory/memory_buffer_rotation.hpp"
#include "ai/ai_rl.hpp"
#include "ai/intent_gate.hpp"
#include "ai/task_planner.hpp" 
#include "ai/grim_text_server_manager.hpp"  
#include "ai/training_server_manager.hpp"  
#include "core/window_manager.hpp"
#include "core/plugin_manager.hpp"
#include "core/input_parser.hpp"
#include "core/input/InputBindings.hpp"
#include "core/crash_dump.hpp"
#include "core/platform_input.hpp"
#include "core/platform_clipboard.hpp"
#include "core/platform_window.hpp"  
#include "helpers/mouse.hpp"
#include "helpers/key.hpp"
#include "helpers/cerr_suppressor.hpp"
#include "net/websocket_server.hpp"
#include "ui/ui_root.hpp"
#include "ui/console_panel.hpp"
#include "ui/ui_settings_menu.hpp"
#include "ui/ui_training_panel.hpp"
#include "ui/ui_data_hub.hpp"
#include "ui/ui_storage_panel.hpp"
#include "ui/ui_geospatial_panel.hpp"
#include "ui/primitives/ui_native_3d_viewport_attachment.hpp"
#include "ui/primitives/ui_native_3d_viewport_clear_pass.hpp"
#include "ui/ui_surface_renderer_bridge.hpp"
#include "geospatial/geospatial_runtime.hpp"
#include "control/devices/server/device_comm_server.hpp"
#include "resources.hpp"
#include "nlp/nlp.hpp"
#include "timer.hpp"
#include "perception/perception.hpp"
#include "perception/perception_context.hpp" 
#include "perception/digital/DigitalCaptureProbe.hpp"
#include "perception/digital/DigitalCaptureSource.hpp"
#include "perception/digital/DigitalContextProjector.hpp"
#include "perception/digital/DigitalEnvironmentLoop.hpp"
#include "perception/digital/DigitalPerceptionPrimitivesLoop.hpp"
#include "perception/physical/PhysicalEnvironmentLoop.hpp"
#include "perception/physical/PhysicalInteractionLoop.hpp"
#include "perception/physical/PhysicalGestureControlLoop.hpp"
#include "perception/physical/PhysicalGestureControlConfigIO.hpp"
#include "perception/physical/PhysicalPerceptionPrimitivesLoop.hpp"
#include "perception/physical/PhysicalSpatialGroundingLoop.hpp"
#include "perception/physical/PhysicalLocalizationLoop.hpp"
#include "perception/physical/PhysicalWorldStateLoop.hpp"
#include "perception/physical/PhysicalWorldStateContextProjector.hpp"
#include "perception/physical/PhysicalWorldStateMemoryWriter.hpp"
#include "ui/ui_physical_environment_panel.hpp"
#include "ui/ui_digital_environment_panel.hpp"
#include "MMO/UI/UISurfaceRegistry.hpp"
#ifdef _WIN32
#include <crtdbg.h>
#endif
#include <chrono>
#include <thread>
#include <csignal>
#include <atomic>

#include "core/grim_platform.h"

#ifdef _WIN32
#define CHECK_HEAP() _CrtCheckMemory()
#else
#define CHECK_HEAP() ((void)0)
#endif

GRIM::UnifiedMemoryStorage g_memoryStorage;
static GRIM::WebSocketServer wsServer;
static std::unique_ptr<GRIM::DeviceCommServer> g_deviceCommServer;
namespace fs = std::filesystem;

// External hardware inventory (defined in bootstrap.cpp)
#include "MMO/Core/HardwareInventory.hpp"
extern GRIM::MMO::HardwareInventory g_hardwareInventory;

// Global training panel for commands
std::shared_ptr<UITrainingPanel> g_trainingPanel;

// ============================================================
// Signal Handler for Clean Shutdown
// ============================================================

std::atomic<bool> g_shutdownRequested{false};

#ifdef _WIN32
BOOL WINAPI consoleHandler(DWORD signal) {
    if (signal == CTRL_C_EVENT || signal == CTRL_CLOSE_EVENT || signal == CTRL_BREAK_EVENT) {
        LOG_PHASE("Shutdown signal received, cleaning up...", true);
        g_shutdownRequested = true;

        // Stop idle-tick thread first
        stopMMOIdleTick();

        GRIM::Perception::Digital::ShutdownDigitalPerceptionPrimitives();
        GRIM::Perception::Digital::ShutdownDigitalEnvironment();
        GRIM::Perception::Digital::ShutdownDigitalContextProjector();
        GRIM::Perception::Physical::ShutdownPhysicalGestureControl();
        GRIM::Perception::Physical::ShutdownPhysicalInteraction();

        // Stop perception-side world-state consumers BEFORE the facade is
        // freed — the memory writer flushes long-dwell entity summaries on
        // shutdown and needs g_memoryFacade alive to do so.
        GRIM::Perception::Physical::ShutdownPhysicalWorldStateMemoryWriter();
        GRIM::Perception::Physical::ShutdownPhysicalWorldStateContextProjector();

        // Flush rotation pipeline to long-term storage
        GRIM::MemoryBufferRotation::instance().mergeToWorking();
        GRIM::MemoryBufferRotation::instance().syncToLongTerm(g_memoryStorage);
        GRIM::MemoryBufferRotation::instance().clear();

        // Tear down MMO orchestration layer (top-down)
        if (g_orchestrator) {
            g_orchestrator->shutdown();
            delete g_orchestrator;
            g_orchestrator = nullptr;
        }
        if (g_memoryFacade) {
            delete g_memoryFacade;
            g_memoryFacade = nullptr;
        }
        if (g_modelLoader) {
            g_modelLoader->unloadAll();
            delete g_modelLoader;
            g_modelLoader = nullptr;
        }
        // ProcessManager — stop all model server processes
        if (g_processManager) {
            LOG_DEBUG("Shutdown", "Stopping model server processes...");
            g_processManager->stopAll();
            delete g_processManager;
            g_processManager = nullptr;
        }
        // UISurfaceRegistry — tear down all managed surfaces
        GRIM::MMO::UISurfaceRegistry::instance().destroyAll();
        if (g_resourceCoordinator) {

            delete g_resourceCoordinator;
            g_resourceCoordinator = nullptr;
        }
        if (g_resourceSignal) {
            g_resourceSignal->stop();
            delete g_resourceSignal;
            g_resourceSignal = nullptr;
        }
        
        LOG_DEBUG("Shutdown", "Stopping training control server...");
        GRIM::stopTrainingServer();
        
        // Give a moment for servers to shut down
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        LOG_PHASE("Cleanup complete, exiting...", true);
        std::exit(0);  // Exit cleanly after cleanup
    }
    return FALSE;
}
#else
void signalHandler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        LOG_PHASE("Shutdown signal received, cleaning up...", true);
        g_shutdownRequested = true;

        // Stop idle-tick thread first
        stopMMOIdleTick();

        GRIM::Perception::Digital::ShutdownDigitalPerceptionPrimitives();
        GRIM::Perception::Digital::ShutdownDigitalEnvironment();
        GRIM::Perception::Digital::ShutdownDigitalContextProjector();
        GRIM::Perception::Physical::ShutdownPhysicalGestureControl();
        GRIM::Perception::Physical::ShutdownPhysicalInteraction();

        // Stop perception-side world-state consumers BEFORE the facade is
        // freed — the memory writer flushes long-dwell entity summaries on
        // shutdown and needs g_memoryFacade alive to do so.
        GRIM::Perception::Physical::ShutdownPhysicalWorldStateMemoryWriter();
        GRIM::Perception::Physical::ShutdownPhysicalWorldStateContextProjector();

        // Flush rotation pipeline to long-term storage
        GRIM::MemoryBufferRotation::instance().mergeToWorking();
        GRIM::MemoryBufferRotation::instance().syncToLongTerm(g_memoryStorage);
        GRIM::MemoryBufferRotation::instance().clear();

        // Tear down MMO orchestration layer (top-down)
        if (g_orchestrator) {
            g_orchestrator->shutdown();
            delete g_orchestrator;
            g_orchestrator = nullptr;
        }
        if (g_memoryFacade) {
            delete g_memoryFacade;
            g_memoryFacade = nullptr;
        }
        if (g_modelLoader) {
            g_modelLoader->unloadAll();
            delete g_modelLoader;
            g_modelLoader = nullptr;
        }
        // ProcessManager — stop all model server processes
        if (g_processManager) {
            g_processManager->stopAll();
            delete g_processManager;
            g_processManager = nullptr;
        }
        // UISurfaceRegistry — tear down all managed surfaces
        GRIM::MMO::UISurfaceRegistry::instance().destroyAll();
        if (g_resourceCoordinator) {
            delete g_resourceCoordinator;
            g_resourceCoordinator = nullptr;
        }
        if (g_resourceSignal) {
            g_resourceSignal->stop();
            delete g_resourceSignal;
            g_resourceSignal = nullptr;
        }
        
        GRIM::stopTrainingServer();
        
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        std::exit(0);
    }
}
#endif

// ============================================================
// Main entry point
// ============================================================
int main(int argc, char* argv[])
{
    GRIM::CerrSuppressor cerrFilter;
    initLogger("grim.log");
    GRIM::InstallCrashDumpHandler();
    LOG_PHASE("Initializing G.R.I.M", true);

    std::string dpiAwarenessError;
    if (!GRIM::Perception::Digital::EnsureDigitalCaptureDpiAwareness(&dpiAwarenessError)) {
        LOG_ERROR("DigitalCapture", "Failed to enable per-monitor DPI awareness: " +
                  dpiAwarenessError);
    }

    int digitalProbeExitCode = 0;
    if (GRIM::Perception::Digital::TryRunDigitalCaptureProbe(
            argc, argv, digitalProbeExitCode)) {
        return digitalProbeExitCode;
    }

    // ======================================================
    // 0. Install signal handlers for clean shutdown
    // ======================================================
#ifdef _WIN32
    if (!SetConsoleCtrlHandler(consoleHandler, TRUE)) {
        LOG_ERROR("Main", "Failed to set console handler");
    } else {
        LOG_DEBUG("Main", "Signal handler installed (Ctrl+C will clean up child processes)");
    }
#else
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
    LOG_DEBUG("Main", "Signal handlers installed");
#endif

    // ======================================================
    // 1. Start WebSocket + logger
    // ======================================================
    wsServer.start();
    LOG_PHASE("WebSocket server started", true);

    // Start Device Communication server
    {
        GRIM::DeviceCommServer::Config dcConfig;
        dcConfig.port                 = 11437;
        dcConfig.heartbeat_timeout_sec = 30;
        dcConfig.max_chunk_size_bytes  = 65536;
        dcConfig.transfer_timeout_sec  = 300;

        std::string home;
#ifdef _WIN32
        const char* userProfile = std::getenv("USERPROFILE");
        if (!userProfile) throw std::runtime_error("USERPROFILE not set");
        home = userProfile;
#else
        const char* homeDir = std::getenv("HOME");
        if (!homeDir) throw std::runtime_error("HOME not set");
        home = homeDir;
#endif
        dcConfig.storage_root = home + "/.grim/shared_storage/";
        dcConfig.registry_dir = home + "/.grim/devices/";

        g_deviceCommServer = std::make_unique<GRIM::DeviceCommServer>(dcConfig);
        g_deviceCommServer->start();
        LOG_PHASE("Device comm server started", true);
    }

    
    
    LOG_DEBUG("Main", "Heap check mode: Manual (removed CHECK_ALWAYS for debugging)");

    // ✅ NEW: Initialize cross-platform input system
    PlatformInput::initialize();
    LOG_PHASE("Platform input initialized", true);

    // ✅ NEW: Initialize cross-platform clipboard system
    PlatformClipboard::initialize();
    LOG_PHASE("Platform clipboard initialized", true);

    Mouse::initialize();
    LOG_PHASE("Mouse initialized", true);

    // ======================================================
    // 2. Bootstrap + aliases + system detection
    // ======================================================
    runBootstrapChecks(argc, argv);
    LOG_PHASE("Bootstrap checks complete", true);

    {
        std::string gestureConfigError;
        if (!GRIM::Perception::Physical::ApplyPhysicalGestureControlConfigFromRuntime(
                aiConfig, gestureConfigError)) {
            LOG_ERROR("PhysicalGestureControl",
                "Gesture-control config rejected; using safe defaults: " +
                gestureConfigError);
        }
    }

    aliases::init();
    LOG_PHASE("Aliases initialized", true);

    PluginManager::initialize((std::filesystem::path(getGrimRootDir()) / "plugins").string());
    LOG_PHASE("Plugin manager initialized", true);

    // ======================================================
    // 3. Voice / Queue
    // ======================================================
    Voice::initQueue();
    // Wait up to 10s for TTS to become ready (avoids blocking startup indefinitely)
    {
        const int kWaitTimeoutMs = 10000;
        auto tStart = std::chrono::steady_clock::now();
        while (!Voice::isReady()) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - tStart).count();
            if (elapsed > kWaitTimeoutMs) {
                LOG_ERROR("Main", "TTS not ready after 10s - continuing without voice");
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    if (Voice::isReady()) {
        Voice::speak("Welcome back, Austin. Grim is online.", "system");
        LOG_PHASE("Startup greeting spoken", true);
    } else {
        LOG_DEBUG("Main", "Skipping startup greeting (TTS unavailable)");
        LOG_PHASE("Startup greeting spoken", false);
    }

    // ======================================================
    // 4.   Memory system
    // ======================================================
    g_memoryStorage.initialize((std::filesystem::path(getGrimRootDir()) / "data" / "memories.fb").string());
    LOG_PHASE("Memory system initialized", true);

    // Wire MemoryFacade into the MMO orchestrator (memory init is after bootstrap)
    g_memoryFacade = new GRIM::MMO::MemoryFacade(g_memoryStorage);
    if (g_orchestrator) {
        g_orchestrator->setMemoryFacade(g_memoryFacade);
    }
    LOG_PHASE("MemoryFacade wired to orchestrator", true);
    
    // ======================================================
    // 5.   Initialize TTS pre-cache in background
    // ======================================================
    Voice::initPreCache();
    LOG_PHASE("TTS pre-cache started (background)", true);
    
    // ======================================================
    // 6.   Initialize Intent Classification System
    // ======================================================
    GRIM::IntentGate::init();
    LOG_PHASE("Intent classification system initialized", true);

    // ======================================================
    // 6.5  Initialize Perception System
    // ======================================================
    GRIM::Perception::init();
    LOG_PHASE("Perception system initialized", true);
    
    // ✅ NEW: Initialize context-aware perception manager
    GRIM::Perception::initContextManager();
    LOG_PHASE("Perception context manager initialized", true);
    
    // ======================================================
    // 6.7  Initialize Task Planner
    // ======================================================
    GRIM::TaskPlanner::init();
    LOG_PHASE("Task planner initialized", true);
    
    // ======================================================
    // 7. Initialize BGFX global context (platform window)
    // ======================================================
    LOG_PHASE("Initializing BGFX on main thread", true);

    void* tempHwnd = PlatformWindow::createBGFXInitWindow();

    if (!tempHwnd) {
        LOG_ERROR("Main", "Failed to create temporary window for BGFX (platform layer returned null)");
        return 1;
    }

    if (!WindowManager::initGlobalBGFX(static_cast<HWND>(tempHwnd))) {
        LOG_ERROR("Main", "Failed to initialize BGFX on main thread");
        PlatformWindow::destroyBGFXInitWindow(tempHwnd);
        return 1;
    }

#if defined(__APPLE__)
    // On macOS: BGFX host window is a hidden 1x1 borderless window.
    // Key input comes through NSApp event queue, not window focus.
    // The overlay is software-rendered via grimOverlayBlit and stays transparent.
    {
        auto mainWin = std::make_unique<GRIMWindow>();
        mainWin->hwnd = tempHwnd;
        mainWin->name = "main";
        mainWin->visible = false;
        mainWin->isOverlay = false;
        mainWin->width = 1;
        mainWin->height = 1;
        WindowManager::registerWindow(std::move(mainWin));
        WindowManager::processMainThreadUpdates();
    }
    PlatformWindow::setWindowVisible(tempHwnd, false);
#else
    // On Windows: register tempHwnd as the BGFX primary window (hidden) so
    // bgfx::frame() has a valid render target. The overlay is software-rendered
    // via OverlayRenderer → UpdateLayeredWindow and must NOT host BGFX.
    {
        auto mainWin = std::make_unique<GRIMWindow>();
        mainWin->hwnd = static_cast<HWND>(tempHwnd);
        mainWin->name = "main";
        mainWin->visible = false;
        mainWin->isOverlay = false;
        mainWin->width = 1;
        mainWin->height = 1;
        WindowManager::registerWindow(std::move(mainWin));
        WindowManager::processMainThreadUpdates();
    }
    PlatformWindow::setWindowVisible(tempHwnd, false);
#endif
    LOG_PHASE("BGFX initialized successfully", true);

    // ======================================================
    // 8. Create unified transparent overlay (multi-monitor)
    // ======================================================
    LOG_PHASE("Creating GRIM unified overlay window", true);

    int virtualX = 0, virtualY = 0, virtualWidth = 0, virtualHeight = 0;
    PlatformWindow::getVirtualScreenRect(virtualX, virtualY, virtualWidth, virtualHeight);
    
    LOG_DEBUG("Main", "Virtual screen: " + std::to_string(virtualWidth) + "x" + 
              std::to_string(virtualHeight) + " at (" + 
              std::to_string(virtualX) + "," + std::to_string(virtualY) + ")");
    // Virtual screen dims already captured in g_hardwareInventory during bootstrap.
    // If changed at runtime (hot-plug), re-detect would be needed.

    GRIMWindow* overlayWin = WindowManager::createOverlay("overlay", virtualWidth, virtualHeight, true);
    if (!overlayWin) {
        LOG_ERROR("Main", "Failed to create transparent overlay window");
        return 1;
    }

    std::string captureExclusionError;
    if (!GRIM::Perception::Digital::SetDigitalCaptureExcludedWindow(
            overlayWin->hwnd, &captureExclusionError)) {
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

    // BGFX stays on tempHwnd (hidden on Windows, visible on macOS).
    // The overlay is software-rendered via OverlayRenderer → UpdateLayeredWindow.
    // Do NOT redirect BGFX to the overlay — it conflicts with layered window rendering.
    WindowManager::updateWindowDimensions("overlay", virtualWidth, virtualHeight);
    WindowManager::processMainThreadUpdates();
    
    LOG_PHASE("Unified overlay initialized successfully (multi-monitor)", true);

    // ======================================================
    // 9. Initialize unified UI system (UIRoot)
    // ======================================================
    UIRoot::get().init(overlayWin->hwnd, overlayWin->width, overlayWin->height);

    {
        std::string fontPath = findAnyFontInResources(argc, argv);
        if (!fontPath.empty()) {
            UIRoot::get().getRenderer().setFont(fontPath, 16);
        } else {
            LOG_ERROR("UIRoot", "No font file found — text will not render");
        }

        // Load icon font if configured (FontAwesome, Material Icons, Nerd Font, etc.)
        std::string iconFontName;
        if (aiConfig.contains("ui") && aiConfig["ui"].contains("icon_font")) {
            iconFontName = aiConfig["ui"]["icon_font"].get<std::string>();
        }
        if (!iconFontName.empty()) {
            std::string iconFontPath;
            // Search resources/fonts/ for the icon font file
            std::filesystem::path fontsDir = std::filesystem::path(argv[0]).parent_path() / "resources" / "fonts";
            if (std::filesystem::exists(fontsDir)) {
                for (auto& entry : std::filesystem::recursive_directory_iterator(fontsDir)) {
                    if (entry.is_regular_file()) {
                        std::string stem = entry.path().stem().string();
                        if (stem == iconFontName || entry.path().filename().string() == iconFontName) {
                            iconFontPath = entry.path().string();
                            break;
                        }
                    }
                }
            }
            if (!iconFontPath.empty()) {
                UIRoot::get().getRenderer().loadIconFont(iconFontPath);
            } else {
                LOG_DEBUG("UIRoot", "Icon font '" + iconFontName + "' not found in resources/fonts/");
            }
        }
    }

    auto consolePanel  = std::make_shared<ConsolePanel>();
    auto settingsPanel = std::make_shared<UISettingsMenu>();
    auto trainingPanel = std::make_shared<UITrainingPanel>();
    auto dataHubPanel = std::make_shared<UIDataHubPanel>();
    auto storagePanel = std::make_shared<UIStoragePanel>();
    auto geoSpatialPanel = std::make_shared<UIGeoSpatialPanel>();
    auto geoSpatialRuntime = std::make_shared<GRIM::GeoSpatial::GeoSpatialRuntime>();
    auto geoSpatialViewportAttachment = std::make_shared<UINative3DViewportAttachment>(
        UIRoot::get().getHWND(), "GeoSpatialViewport");
    UINative3DViewportClearPass::registerClearPass("GeoSpatialViewportClear",
                                                   geoSpatialViewportAttachment,
                                                   0x153A4AFF);
    geoSpatialRuntime->setViewportReady(true);
    geoSpatialRuntime->setViewportAttachment(geoSpatialViewportAttachment);
    geoSpatialPanel->setController(geoSpatialRuntime.get());
    geoSpatialPanel->setViewportAttachment(geoSpatialViewportAttachment);
    // Stage 1: hand the device server to the physical environment subsystem so the
    // camera directory can include hub-registered devices that advertise "camera".
    // MUST be registered BEFORE constructing the panel so its initial directory
    // refresh sees hub candidates.
    GRIM::Perception::Physical::RegisterPhysicalEnvironmentDeviceServer(g_deviceCommServer.get());
    auto physicalEnvPanel = std::make_shared<UIPhysicalEnvironmentPanel>();
    auto digitalEnvPanel = std::make_shared<UIDigitalEnvironmentPanel>();
    // Stage 2 perception-primitives surface lives as a tab inside
    // UIPhysicalEnvironmentPanel; the standalone panel was removed.
    if (g_deviceCommServer) {
        storagePanel->setServer(g_deviceCommServer.get());
    }
    
    // Assign to global for command access
    g_trainingPanel = trainingPanel;

    // Panels use window-relative coordinates set in their constructors
    // No need to reposition them here
    
    // Hide panels initially - they should only show when explicitly toggled
    consolePanel->setVisible(false);
    settingsPanel->setVisible(false);
    trainingPanel->setVisible(false);
    dataHubPanel->setVisible(false);
    storagePanel->setVisible(false);
    geoSpatialPanel->setVisible(false);
    physicalEnvPanel->setVisible(false);
    digitalEnvPanel->setVisible(false);

    UIRoot::get().addPanel(consolePanel);
    UIRoot::get().addPanel(settingsPanel);
    UIRoot::get().addPanel(trainingPanel);
    UIRoot::get().addPanel(dataHubPanel);
    UIRoot::get().addPanel(storagePanel);
    UIRoot::get().addPanel(geoSpatialPanel);
    UIRoot::get().addPanel(physicalEnvPanel);
    UIRoot::get().addPanel(digitalEnvPanel);

    GRIM::InputBindings::loadRuntimeConfig(aiConfig);

#if defined(__APPLE__)
    // macOS: inject typed characters into text input (Windows uses WM_CHAR in OverlayWndProc)
    PlatformWindow::setTextInputCallback([](const std::string& text) {
        UIRoot::get().injectTextInput(text);
    });
#endif

    LOG_PHASE("UIRoot and panels initialized (hidden)", true);

    // Wire UISurfaceRegistry → UIRoot bridge so tool-created surfaces render
    UISurfaceRendererBridge::install();
    LOG_PHASE("UISurfaceRendererBridge installed", true);

    // ======================================================
    // 10. Launch popup UI (still separate layered window)
    // ======================================================
#ifdef _WIN32
    std::thread([]() {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        runPopupUI(256, 256);
    }).detach();
    LOG_PHASE("Popup UI launched (layered window)", true);
#else
    std::thread([]() {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        runPopupUI(256, 256);
    }).detach();
    LOG_PHASE("Popup UI launched (macOS NSWindow)", true);
#endif

    WindowManager::processMainThreadUpdates();

    // ======================================================
    // 11. Wake systems - Initialize with nullptr for unused parameters
    // ======================================================
    static std::vector<Timer> emptyTimers;
    static nlohmann::json emptyMemory;
    static NLP dummyNLP;
    
    WakeKey::start(nullptr, emptyTimers, emptyMemory, dummyNLP);
    LOG_PHASE("WakeKey listener started", true);

    WakeVoice::start(nullptr, emptyTimers, emptyMemory, dummyNLP);
    LOG_PHASE("WakeVoice listener started", true);

    // ======================================================
    // 12. Main loop
    // ======================================================
    LOG_PHASE("Entering main thread render loop", true);
    constexpr auto kFrameDuration = std::chrono::milliseconds(16);

    while (!WindowManager::isMainLoopStopRequested())
    {
        auto frameStart = std::chrono::steady_clock::now();

        // Pump OS messages first. Windows bindings use global asynchronous
        // state; macOS updates its global event-backed state here.
        float wheelDelta = 0.0f;
        bool quitRequested = false;
        PlatformWindow::pumpEvents(wheelDelta, quitRequested);
        if (quitRequested)
            WindowManager::requestMainLoopStop();

        // Capture input after message pump so keyboard state is current
        InputState input;
        input.captureFromHWND(overlayWin->hwnd);
        input.mouseWheelDelta = wheelDelta;

        // Update Mouse class state from InputState for better reliability
        Mouse::updateFromInput(input);
        Key::updateFromInput(input);
        WakeKey::update();

        // Resolve configurable actions only after the low-level input snapshot
        // has been normalized. Binding capture suppresses runtime shortcuts so
        // the key being assigned cannot also trigger its previous action.
        auto toggleBoundPanel = [](const char* actionId, const char* panelName) {
            if (!GRIM::InputBindings::wasPressed(actionId)) return;
            auto panel = UIRoot::get().getPanel(panelName);
            if (!panel) return;
            const bool show = !panel->isVisible();
            UIRoot::get().setVisible(panelName, show);
        };
        toggleBoundPanel("toggle_console", "Console");
        toggleBoundPanel("toggle_settings", "Settings");
        toggleBoundPanel("toggle_training", "GRIM-text Training Control");
        toggleBoundPanel("toggle_data_hub", "DataHub");
        toggleBoundPanel("toggle_storage", "Shared Storage");
        toggleBoundPanel("toggle_geospatial", "GeoSpatial");
        toggleBoundPanel("toggle_physical_environment", "Physical Environment");
        toggleBoundPanel("toggle_digital_environment", "Digital Environment");

        // Single integration point for the physical-environment perception subsystem.
        // Lazy-inits on first call; pumps the active IP camera stream into the FrameBus.
        GRIM::Perception::Physical::TickPhysicalEnvironment();
        // Stage 2 auxiliary: non-blocking local hand/gesture interaction branch.
        // It owns a one-frame worker queue and never performs inference here.
        GRIM::Perception::Physical::TickPhysicalInteraction();
        // Phase 2 controller: stabilizes hand results into semantic events,
        // then routes only guarded mouse/wake mappings on the main thread.
        GRIM::Perception::Physical::TickPhysicalGestureControl();
        // Stage 2: issue a non-blocking, coalesced latest-frame request. Its
        // worker runs enabled perception primitives and publishes the aggregate
        // without stalling capture, UI texture upload, or rendering.
        GRIM::Perception::Physical::TickPhysicalPerceptionPrimitives();
        // Stage 3: issue a non-blocking, coalesced latest-result request. Its
        // worker uses Stage 2's pinned source frame for coherent depth/grounding.
        GRIM::Perception::Physical::TickPhysicalSpatialGrounding();
        // Stage 5: issue a non-blocking, coalesced latest-frame request. Its
        // worker runs visual odometry, updates a Nav2-style 2D occupancy grid,
        // and publishes a
        // PhysicalLocalizationSnapshot (T_world_camera, velocity, trajectory,
        // grid) to PhysicalLocalizationBus. Stage 4 below can later stamp the
        // world frame onto every entity it publishes.
        GRIM::Perception::Physical::TickPhysicalLocalization();
        // Stage 4: fuses the perception-primitive bus + spatial-grounding bus into
        // a single identity-keyed PhysicalWorldStateSnapshot (object_id, class,
        // position, velocity, visibility, depth, text_on_object, relations) and
        // publishes it to PhysicalWorldStateBus. This is what the model reads.
        GRIM::Perception::Physical::TickPhysicalWorldState();
        // Stage 4 consumers: project the latest world-state snapshot into the
        // live router context (SessionContextManager.VisualContext.physical)
        // and diff it into MemoryFacade as durable state-change events for
        // the personality LoRA training corpus. Both are cheap pull-only
        // consumers of PhysicalWorldStateBus and no-op when the bus has not
        // advanced. Order: projector first so router sees the freshest scene.
        GRIM::Perception::Physical::TickPhysicalWorldStateContextProjector();
        GRIM::Perception::Physical::TickPhysicalWorldStateMemoryWriter();

        // Capture runs on a low-frequency worker; this is a cheap bus pull.
        GRIM::Perception::Digital::TickDigitalContextProjector();

        geoSpatialRuntime->tick(0.016f);

        UIRoot::get().update(input, 0.016f);

        // Per-frame overlay click-through: allow clicks to pass through to
        // other apps when the cursor is not over any visible GRIM panel.
        // On macOS this toggles NSWindow.ignoresMouseEvents; on Windows the
        // WM_NCHITTEST handler in OverlayWndProc does position-based testing.
        // Uses cached panel rects (populated by update()) to avoid redundant
        // mutex lock + vector copy.
        {
            bool overUI = UIRoot::get().shouldReceiveInputAtCached(input.mousePos.x, input.mousePos.y)
                          || UIRoot::get().isAnyPanelDraggingCached();
            PlatformWindow::setOverlayClickThrough(overlayWin->hwnd, !overUI);
        }

        UIRoot::get().draw();
        WindowManager::processMainThreadUpdates();
        WindowManager::renderFrame();
        
        // Clear per-frame input states
        Key::endFrame();
        Mouse::endFrame();

        auto frameEnd = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(frameEnd - frameStart);
        
        if (elapsed < kFrameDuration)
            std::this_thread::sleep_for(kFrameDuration - elapsed);
    }

    LOG_PHASE("Main thread render loop exited", true);

    // ======================================================
    // 13. Shutdown
    // ======================================================
    LOG_PHASE("Shutting down subsystems", true);

    // Stop idle-tick thread first
    stopMMOIdleTick();

    // Stop perception-side world-state consumers BEFORE the facade is freed —
    // the memory writer flushes long-dwell entity summaries on shutdown and
    // needs g_memoryFacade alive to do so. The bus producer
    // (ShutdownPhysicalWorldState) runs later in the perception teardown.
    GRIM::Perception::Physical::ShutdownPhysicalWorldStateMemoryWriter();
    GRIM::Perception::Physical::ShutdownPhysicalWorldStateContextProjector();

    // Flush rotation pipeline to long-term storage before teardown
    GRIM::MemoryBufferRotation::instance().mergeToWorking();
    GRIM::MemoryBufferRotation::instance().syncToLongTerm(g_memoryStorage);
    GRIM::MemoryBufferRotation::instance().clear();

    // Tear down MMO orchestration layer (top-down, mirrors bootstrap Phase 4 in reverse)
    if (g_orchestrator) {
        g_orchestrator->shutdown();
        delete g_orchestrator;
        g_orchestrator = nullptr;
    }
    if (g_memoryFacade) {
        delete g_memoryFacade;
        g_memoryFacade = nullptr;
    }
    if (g_modelLoader) {
        g_modelLoader->unloadAll();
        delete g_modelLoader;
        g_modelLoader = nullptr;
    }
    if (g_processManager) {
        g_processManager->stopAll();
        delete g_processManager;
        g_processManager = nullptr;
    }
    // UISurfaceRegistry — tear down all managed surfaces
    GRIM::MMO::UISurfaceRegistry::instance().destroyAll();
    if (g_resourceCoordinator) {
        delete g_resourceCoordinator;
        g_resourceCoordinator = nullptr;
    }
    if (g_resourceSignal) {
        g_resourceSignal->stop();
        delete g_resourceSignal;
        g_resourceSignal = nullptr;
    }

    // Shutdown training control server
    GRIM::stopTrainingServer();
    
    WakeKey::stop();
    WakeVoice::stop();
    Voice::shutdownQueue();
    Voice::shutdownTTS();
    GRIM::RL::shutdown();
    GRIM::IntentGate::shutdown(); 
    GRIM::Perception::Digital::ShutdownDigitalPerceptionPrimitives();
    GRIM::Perception::Digital::ShutdownDigitalEnvironment();
    GRIM::Perception::Digital::ShutdownDigitalContextProjector();
    GRIM::Perception::Physical::ShutdownPhysicalWorldStateMemoryWriter();
    GRIM::Perception::Physical::ShutdownPhysicalWorldStateContextProjector();
    GRIM::Perception::Physical::ShutdownPhysicalWorldState();
    GRIM::Perception::Physical::ShutdownPhysicalLocalization();
    GRIM::Perception::Physical::ShutdownPhysicalSpatialGrounding();
    GRIM::Perception::Physical::ShutdownPhysicalPerceptionPrimitives();
    GRIM::Perception::Physical::ShutdownPhysicalGestureControl();
    GRIM::Perception::Physical::ShutdownPhysicalInteraction();
    GRIM::Perception::Physical::ShutdownPhysicalEnvironment();
    GRIM::Perception::shutdown();  
    Mouse::shutdown();
    
    PlatformInput::shutdown();
    PlatformClipboard::shutdown();

    UIRoot::get().shutdown();
    WindowManager::shutdown();

    wsServer.stop();
    if (g_deviceCommServer) {
        g_deviceCommServer->stop();
        g_deviceCommServer.reset();
    }

    LOG_PHASE("All subsystems shut down", true);
    LOG_PHASE("G.R.I.M terminated successfully", true);
    shutdownLogger();
    return 0;
}
