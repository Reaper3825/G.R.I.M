#include "pch.hpp"
#include "commands/commands_core.hpp"
#include "voice/voice.hpp"
#include "voice/voice_speak.hpp"
#include "voice/voice_stream.hpp"
#include "response_manager.hpp"
#include "resources.hpp"
#include "console_history.hpp"
#include "ui/ui_events.hpp"
#include "error_manager.hpp"
#include "bootstrap/bootstrap.hpp"
#include "aliases.hpp"
#include "popup_ui/popup_ui.hpp"
#include "logger.hpp"
#include "wake/wake.hpp"
#include "wake/wake_key.hpp"
#include "wake/wake_voice.hpp"
#include "memory/memory_storage.hpp"
#include "memory/context_manager.hpp"
#include "ai/ai_rl.hpp"
#include "ui/console_ui.hpp"
#include <bgfx/bgfx.h>
#include <bgfx/platform.h>
#include <chrono>
#include <thread>
#include "core/window_manager.hpp"
#include "helpers/mouse.hpp"
#include "system_detect.hpp"
#include "core/plugin_manager.hpp"
#include <crtdbg.h>

#define CHECK_HEAP() _CrtCheckMemory()

GRIM::MemoryStorage g_memoryStorage;

namespace fs = std::filesystem;

// ============================================================
// Global state
// ============================================================
std::string g_inputBuffer;
ConsoleHistory g_consoleHistory;
std::vector<Timer> g_uiTimers;
nlohmann::json g_longTermMemory;

// ============================================================
// Main entry point
// ============================================================
int main(int argc, char* argv[])
{

    // ======================================================
    // 1. Logger + mouse
    // ======================================================
    initLogger("grim.log");
    LOG_PHASE("Initializing G.R.I.M", true);
_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF | _CRTDBG_CHECK_ALWAYS_DF);
    Mouse::initialize();
    LOG_PHASE("Mouse initialized", true);

    // ======================================================
    // 3. Bootstrap + aliases
    // ======================================================
    
    runBootstrapChecks(argc, argv);
    LOG_PHASE("Bootstrap checks complete", true);

    aliases::init();
    LOG_PHASE("Aliases initialized", true);

    PluginManager::initialize("D:/G.R.I.M/plugins");
    LOG_PHASE("Plugin manager initialized", true);

    // ======================================================
    // 4. Voice TTS + queue
    // ======================================================
    Voice::initQueue();

    while (!Voice::isReady())
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

    Voice::speak("Welcome back, Austin. Grim is online.", "system");
    LOG_PHASE("Startup greeting spoken", true);

    // ======================================================
    // 5. Memory system
    // ======================================================
    g_memoryStorage.initialize("D:/G.R.I.M/data/memories.json");
    GRIM::ContextManager::setMemoryStorage(&g_memoryStorage);
    LOG_PHASE("Memory system initialized", true);

    // ======================================================
    // 6. Initialize BGFX on main thread (required)
    // ======================================================
    LOG_PHASE("Initializing BGFX on main thread", true);
    // Create a temporary window for BGFX initialization
    HWND tempHwnd = CreateWindowExW(
        0, L"STATIC", L"TempBGFXWindow",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
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
    LOG_PHASE("BGFX initialized successfully", true);

// Keep temp window hidden instead of destroying it
    ShowWindow(tempHwnd, SW_HIDE);
    
 


    // ======================================================
    // 7. Launch popup overlay once
    // ======================================================
    LOG_PHASE("Launching GRIM Console (BGFX)", true);
    std::thread consoleThread([]() {
        GRIMConsole::runConsoleUI(512, 720);
    });
    consoleThread.detach();

    // Allow the console thread to register its window and apply BGFX updates on this thread.
    bool applied = false;
    for (int i = 0; i < 200; ++i)
    {
        if (WindowManager::processMainThreadUpdates())
            applied = true;
        if (applied && !WindowManager::hasPendingPlatformUpdate())
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    WindowManager::processMainThreadUpdates();

    // ======================================================
    // 5. Launch popup overlay after BGFX context exists
    // ======================================================
    std::thread([]() {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    runPopupUI(400, 400);
    }).detach();


    LOG_PHASE("Popup UI launched", true);

    // Process any late BGFX updates (e.g., popup attaching overlays)
    WindowManager::processMainThreadUpdates();

    // ======================================================
    // 8. Wake systems
    // ======================================================
    WakeKey::start(&g_consoleHistory, g_uiTimers, g_longTermMemory, g_nlp);
    LOG_PHASE("WakeKey listener started", true);

    WakeVoice::start(&g_consoleHistory, g_uiTimers, g_longTermMemory, g_nlp);
    LOG_PHASE("WakeVoice listener started", true);

    // ======================================================
    // 9. Main render loop (runs on main thread)
    // ======================================================
    LOG_PHASE("Entering main thread render loop", true);

    constexpr auto kFrameDuration = std::chrono::milliseconds(16);
    while (!WindowManager::isMainLoopStopRequested())
    {
        auto frameStart = std::chrono::steady_clock::now();

        WindowManager::processMainThreadUpdates();
        WindowManager::renderFrame();

        auto frameEnd = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(frameEnd - frameStart);
        if (elapsed < kFrameDuration)
        {
            std::this_thread::sleep_for(kFrameDuration - elapsed);
        }
    }

    LOG_PHASE("Main thread render loop exited", true);

    // ======================================================
    // 10. Cleanup + shutdown
    // ======================================================
    LOG_PHASE("Shutting down subsystems", true);

    WakeKey::stop();
    WakeVoice::stop();
    Voice::shutdownQueue();
    Voice::shutdownTTS();
    GRIM::RL::shutdown();
    Mouse::shutdown();
    WindowManager::shutdown();

    LOG_PHASE("All subsystems shut down", true);
    shutdownLogger();
    LOG_PHASE("G.R.I.M terminated successfully", true);
    return 0;




}
