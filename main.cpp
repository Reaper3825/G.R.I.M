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
#include "core/window_manager.hpp"

GRIM::MemoryStorage g_memoryStorage;

namespace fs = std::filesystem;

// Global console-related data
std::string g_inputBuffer;
ConsoleHistory g_consoleHistory;
std::vector<Timer> g_uiTimers;
nlohmann::json g_longTermMemory;

// ============================================================
// Main entry point
// ============================================================
int main(int argc, char* argv[])
{
    // ====================================================
    // Logger + startup phase
    // ====================================================
    initLogger("grim.log");
    LOG_PHASE("Startup begin", true);
    SetConsoleTitleW(L"G.R.I.M Console");

    // ====================================================
    // Bootstrap and initialize subsystems
    // ====================================================
    runBootstrapChecks(argc, argv);
    LOG_PHASE("Bootstrap checks complete", true);

    aliases::init();
    LOG_PHASE("Aliases initialized", true);

    // ====================================================
    // Initialize TTS and queue
    // ====================================================
    Voice::initQueue();

    if (!Voice::isReady())
    {
        LOG_DEBUG("Voice", "Waiting for TTS bridge to be ready...");
        while (!Voice::isReady())
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    Voice::speak("Welcome back, Austin. Grim is online.", "system");
    LOG_PHASE("Startup greeting spoken", true);

    // ====================================================
    // Initialize long-term memory system
    // ====================================================
    g_memoryStorage.initialize("D:/G.R.I.M/data/memories.json");
    GRIM::ContextManager::setMemoryStorage(&g_memoryStorage);
    LOG_PHASE("Memory system initialized", true);

    
    // ====================================================
    // Initialize BGFX globally (via WindowManager)
    // ====================================================
    HWND hwndConsole = GetConsoleWindow();
    if (!WindowManager::initGlobalBGFX(hwndConsole))
        return -1;
    LOG_PHASE("Global BGFX initialized (via WindowManager)", true);


    // Register popup window in WindowManager (track visibility, HWND)
    WindowManager::createOverlay("popup", 400, 400, true);


    // ====================================================
    // Launch popup overlay (uses shared BGFX view)
    // ====================================================
    std::thread([]()
    {
        LOG_DEBUG("PopupUI", "Launching overlay window (shared BGFX)...");
        runPopupUI(400, 400); // now uses view 1
    }).detach();
    LOG_PHASE("Popup UI launched", true);
    WindowManager::createOverlay("popup", 400, 400, true);

    // ====================================================
    // Start wakeword systems
    // ====================================================
    WakeKey::start(&g_consoleHistory, g_uiTimers, g_longTermMemory, g_nlp);
    LOG_PHASE("WakeKey listener started", true);

    WakeVoice::start(&g_consoleHistory, g_uiTimers, g_longTermMemory, g_nlp);
    LOG_PHASE("WakeVoice listener started", true);

    // ====================================================
    // Start Console UI (main window + render loop)
    // ====================================================
    LOG_PHASE("Launching GRIM Console (BGFX)", true);
    GRIMConsole::runConsoleUI(1280, 720);

    // ====================================================
    // Cleanup on shutdown
    // ====================================================
    LOG_PHASE("Shutting down subsystems", true);
    WakeKey::stop();
    Voice::shutdownQueue();
    Voice::shutdownTTS();
    GRIM::RL::shutdown();

    WindowManager::shutdown();
    LOG_PHASE("Shutdown complete", true);

    shutdownLogger();
    return 0;
}
