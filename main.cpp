#include "voice/voice_speak.hpp"

#include "bootstrap/bootstrap.hpp"
#include "aliases.hpp"
#include "popup_ui/popup_ui.hpp"
#include "logger.hpp"
#include "wake/wake_key.hpp"
#include "wake/wake_voice.hpp"
#include "memory/memory_storage.hpp"
#include "memory/context_manager.hpp"
#include "ai/ai_rl.hpp"
#include "ai/intent_gate.hpp"
#include "ai/task_planner.hpp"  // ✅ NEW: Multi-step task planning
#include "ai/grim_text_server_manager.hpp"  // ✅ NEW: GRIM-text server lifecycle
#include "ai/training_server_manager.hpp"  // ✅ NEW: Training control server lifecycle
#include "core/window_manager.hpp"
#include "core/plugin_manager.hpp"
#include "core/input_parser.hpp"
#include "core/platform_input.hpp"  // ✅ NEW: Cross-platform input
#include "helpers/mouse.hpp"
#include "helpers/key.hpp"
#include "helpers/cerr_suppressor.hpp"
#include "net/websocket_server.hpp"
#include "system_detect.hpp"
#include "ui/ui_root.hpp"
#include "ui/console_panel.hpp"
#include "ui/ui_settings_menu.hpp"
#include "ui/ui_training_panel.hpp"
#include "nlp/nlp.hpp"
#include "timer.hpp"
#include "perception/perception.hpp"
#include "perception/perception_context.hpp" 
#include <crtdbg.h>
#include <chrono>
#include <thread>


#define CHECK_HEAP() _CrtCheckMemory()

GRIM::MemoryStorage g_memoryStorage;
static GRIM::WebSocketServer wsServer;
namespace fs = std::filesystem;

// External system info (defined in bootstrap.cpp)
extern SystemInfo g_systemInfo;

// Global training panel for commands
std::shared_ptr<UITrainingPanel> g_trainingPanel;

// ============================================================
// Main entry point
// ============================================================
int main(int argc, char* argv[])
{
    GRIM::CerrSuppressor cerrFilter;
    initLogger("grim.log");
    LOG_PHASE("Initializing G.R.I.M", true);


    // ======================================================
    // 1. Start WebSocket + logger
    // ======================================================
    wsServer.start();
    LOG_PHASE("WebSocket server started", true);

    
    
    LOG_DEBUG("Main", "Heap check mode: Manual (removed CHECK_ALWAYS for debugging)");

    // ✅ NEW: Initialize cross-platform input system
    PlatformInput::initialize();
    LOG_PHASE("Platform input initialized", true);

    Mouse::initialize();
    LOG_PHASE("Mouse initialized", true);

    // ======================================================
    // 2. Bootstrap + aliases + system detection
    // ======================================================
    runBootstrapChecks(argc, argv);
    LOG_PHASE("Bootstrap checks complete", true);

    aliases::init();
    LOG_PHASE("Aliases initialized", true);

    // System info is now populated by bootstrap
    LOG_PHASE("System detection complete", true);

    PluginManager::initialize("D:/G.R.I.M/plugins");
    LOG_PHASE("Plugin manager initialized", true);

    // ======================================================
    // 3. Voice / Queue
    // ======================================================
    Voice::initQueue();
    while (!Voice::isReady())
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    Voice::speak("Welcome back, Austin. Grim is online.", "system");
    LOG_PHASE("Startup greeting spoken", true);

    // ======================================================
    // 4.   Memory system
    // ======================================================
    g_memoryStorage.initialize("D:/G.R.I.M/data/memories.fb");
    GRIM::ContextManager::setMemoryStorage(&g_memoryStorage);
    LOG_PHASE("Memory system initialized", true);
    
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
    
    // ✅ NEW: Start continuous screen awareness
    GRIM::Perception::ContinuousCaptureConfig captureConfig;
    captureConfig.frameSkip = 30;              // Capture every 30 frames (~1/sec at 30fps)
    captureConfig.captureIntervalMs = 1000;    // Or every 1000ms
    captureConfig.useFrameSkip = false;        // Use time-based for consistency
    captureConfig.captureAllMonitors = false;  // Just active monitor
    captureConfig.changeThreshold = 0.05f;     // 5% change detection
    captureConfig.useVisionAI = false;         // ✅ Vision AI too slow for background capture (only on-demand)
    GRIM::Perception::g_contextManager->startContinuousCapture(captureConfig);
    LOG_PHASE("Continuous screen capture started", true);

    // ======================================================
    // 7. Initialize BGFX global context
    // ======================================================
    LOG_PHASE("Initializing BGFX on main thread", true);

    HWND tempHwnd = CreateWindowExW(
        0, L"STATIC", L"TempBGFXWindow",
        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
        1, 1, nullptr, nullptr, GetModuleHandle(nullptr), nullptr);

    if (!tempHwnd) {
        LOG_ERROR("Main", "Failed to create temporary window for BGFX");
        return 1;
    }

    if (!WindowManager::initGlobalBGFX(tempHwnd)) {
        LOG_ERROR("Main", "Failed to initialize BGFX on main thread");
        DestroyWindow(tempHwnd);
        return 1;
    }

    ShowWindow(tempHwnd, SW_HIDE);
    LOG_PHASE("BGFX initialized successfully", true);

    // ======================================================
    // 8. Create unified transparent overlay (multi-monitor)
    // ======================================================
    LOG_PHASE("Creating GRIM unified overlay window", true);
    
    // Use virtual screen dimensions to span all monitors
    int virtualX = GetSystemMetrics(SM_XVIRTUALSCREEN);
    int virtualY = GetSystemMetrics(SM_YVIRTUALSCREEN);
    int virtualWidth = GetSystemMetrics(SM_CXVIRTUALSCREEN);
    int virtualHeight = GetSystemMetrics(SM_CYVIRTUALSCREEN);
    
    LOG_DEBUG("Main", "Virtual screen: " + std::to_string(virtualWidth) + "x" + 
              std::to_string(virtualHeight) + " at (" + 
              std::to_string(virtualX) + "," + std::to_string(virtualY) + ")");

    GRIMWindow* overlayWin = WindowManager::createOverlay("overlay", virtualWidth, virtualHeight, true);
    if (!overlayWin) {
        LOG_ERROR("Main", "Failed to create transparent overlay window");
        return 1;
    }

    WindowManager::initGlobalBGFX(overlayWin->hwnd);
    
    // Set the overlay as the BGFX render target
    WindowManager::updateWindowDimensions("overlay", virtualWidth, virtualHeight);
    WindowManager::processMainThreadUpdates();
    
    LOG_PHASE("Unified overlay initialized successfully (multi-monitor)", true);

    // ======================================================
    // 9. Initialize unified UI system (UIRoot)
    // ======================================================
    UIRoot::get().init(overlayWin->hwnd, overlayWin->width, overlayWin->height);

    auto consolePanel  = std::make_shared<ConsolePanel>();
    auto settingsPanel = std::make_shared<UISettingsMenu>();
    auto trainingPanel = std::make_shared<UITrainingPanel>();
    
    // Assign to global for command access
    g_trainingPanel = trainingPanel;

    // Panels use window-relative coordinates set in their constructors
    // No need to reposition them here
    
    // Hide panels initially - they should only show when explicitly toggled
    consolePanel->setVisible(false);
    settingsPanel->setVisible(false);
    trainingPanel->setVisible(false);

    UIRoot::get().addPanel(consolePanel);
    UIRoot::get().addPanel(settingsPanel);
    UIRoot::get().addPanel(trainingPanel);

    LOG_PHASE("UIRoot and panels initialized (hidden)", true);

    // ======================================================
    // 10. Launch popup UI (still separate layered window)
    // ======================================================
    std::thread([]() {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        runPopupUI(256, 256); // Layered window popup (separate)
    }).detach();
    LOG_PHASE("Popup UI launched (layered window)", true);

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

        // Capture input and convert to client coordinates for UI
        auto inputCaptureStart = std::chrono::steady_clock::now();
        InputState input;
        input.captureFromHWND(overlayWin->hwnd);
        
        // ✅ NEW: Update Mouse class state from InputState for better reliability
        Mouse::updateFromInput(input);
        auto inputCaptureEnd = std::chrono::steady_clock::now();
        
        MSG msg{};
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            // Capture mouse wheel before dispatching
            if (msg.message == WM_MOUSEWHEEL) {
                // WM_MOUSEWHEEL provides delta in HIWORD(wParam)
                // Positive = scroll up, Negative = scroll down
                short delta = GET_WHEEL_DELTA_WPARAM(msg.wParam);
                input.mouseWheelDelta = static_cast<float>(delta);
            }
            
            if (msg.message == WM_QUIT)
                WindowManager::requestMainLoopStop();
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        auto uiUpdateStart = std::chrono::steady_clock::now();
        UIRoot::get().update(input, 0.016f);
        auto uiUpdateEnd = std::chrono::steady_clock::now();
        
        auto uiDrawStart = std::chrono::steady_clock::now();
        UIRoot::get().draw();
        auto uiDrawEnd = std::chrono::steady_clock::now();

        auto wmUpdateStart = std::chrono::steady_clock::now();
        WindowManager::processMainThreadUpdates();
        WindowManager::renderFrame();
        auto wmUpdateEnd = std::chrono::steady_clock::now();
        
        // Clear per-frame input states
        Key::endFrame();
        Mouse::endFrame();

        auto frameEnd = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(frameEnd - frameStart);
        
        // ✅ LOG: Frame timing breakdown if frame is slow
        if (elapsed.count() > 20) { // More than 20ms = potential stutter
            auto inputMs = std::chrono::duration_cast<std::chrono::microseconds>(inputCaptureEnd - inputCaptureStart).count() / 1000.0;
            auto uiUpdateMs = std::chrono::duration_cast<std::chrono::microseconds>(uiUpdateEnd - uiUpdateStart).count() / 1000.0;
            auto uiDrawMs = std::chrono::duration_cast<std::chrono::microseconds>(uiDrawEnd - uiDrawStart).count() / 1000.0;
            auto wmMs = std::chrono::duration_cast<std::chrono::microseconds>(wmUpdateEnd - wmUpdateStart).count() / 1000.0;
            
            LOG_DEBUG("MainLoop", "SLOW FRAME (" + std::to_string(elapsed.count()) + "ms): " +
                      "Input=" + std::to_string(inputMs) + "ms, " +
                      "UI Update=" + std::to_string(uiUpdateMs) + "ms, " +
                      "UI Draw=" + std::to_string(uiDrawMs) + "ms, " +
                      "WM=" + std::to_string(wmMs) + "ms");
        }
        
        if (elapsed < kFrameDuration)
            std::this_thread::sleep_for(kFrameDuration - elapsed);
    }

    LOG_PHASE("Main thread render loop exited", true);

    // ======================================================
    // 13. Shutdown
    // ======================================================
    LOG_PHASE("Shutting down subsystems", true);

    // Shutdown GRIM-text server (inference)
    GRIM::stopGRIMTextServer();
    
    // Shutdown training control server
    GRIM::stopTrainingServer();
    
    WakeKey::stop();
    WakeVoice::stop();
    Voice::shutdownQueue();
    Voice::shutdownTTS();
    GRIM::RL::shutdown();
    GRIM::IntentGate::shutdown(); 
    GRIM::Perception::shutdown();  // ? Added
    Mouse::shutdown();
    
    // ✅ NEW: Shutdown cross-platform input system
    PlatformInput::shutdown();

    UIRoot::get().shutdown();
    WindowManager::shutdown();

    wsServer.stop();

    LOG_PHASE("All subsystems shut down", true);
    LOG_PHASE("G.R.I.M terminated successfully", true);
    shutdownLogger();
    return 0;
}
