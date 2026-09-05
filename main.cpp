#include "voice/voice_speak.hpp"
#include "bootstrap/bootstrap.hpp"
#include "aliases.hpp"
#include "logger.hpp"
#include "wake/wake_key.hpp"
#include "wake/wake_voice.hpp"
#include "memory/unified_memory.hpp"
#include "ai/grim_text_server_manager.hpp"  
#include "core/window_manager.hpp"
#include "core/system_tick.hpp"
#include "core/system_teardown.hpp"
#include "core/plugin_manager.hpp"
#include "core/crash_dump.hpp"
#include "net/websocket_server.hpp"
#include "control/devices/server/device_comm_server.hpp"
#include "MMO/Core/SessionContextManager.hpp"
#include "resources.hpp"
#include "timer.hpp"
#include "perception/perception.hpp"
#include "perception/perception_context.hpp" 
#include "perception/digital/DigitalCaptureProbe.hpp"
#include "perception/digital/DigitalCaptureSource.hpp"
#include "perception/physical/PhysicalGestureControlConfigIO.hpp"
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

// ============================================================
// Signal Handler for Clean Shutdown
// ============================================================

std::atomic<bool> g_shutdownRequested{false};
std::atomic<bool> g_shutdownComplete{false};

#ifdef _WIN32
BOOL WINAPI consoleHandler(DWORD signal) {
    if (signal != CTRL_C_EVENT &&
        signal != CTRL_CLOSE_EVENT &&
        signal != CTRL_BREAK_EVENT) {
        return FALSE;
    }

    // Console control handlers run on a Windows-owned thread. Only request
    // shutdown here; the main thread owns subsystem teardown and its ordering.
    g_shutdownRequested.store(true, std::memory_order_release);
    WindowManager::requestMainLoopStop();

    // Windows may terminate the process as soon as a CTRL_CLOSE handler
    // returns. Give the main thread most of that window to drain cleanly.
    if (signal == CTRL_CLOSE_EVENT) {
        constexpr DWORD kCloseDrainTimeoutMs = 4500;
        constexpr DWORD kPollIntervalMs = 10;
        for (DWORD waitedMs = 0;
             waitedMs < kCloseDrainTimeoutMs &&
                 !g_shutdownComplete.load(std::memory_order_acquire);
             waitedMs += kPollIntervalMs) {
            ::Sleep(kPollIntervalMs);
        }
    }

    return TRUE;
}
#else
void signalHandler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        g_shutdownRequested.store(true, std::memory_order_release);
    }
}
#endif

// ============================================================
// Main entry point
// ============================================================
int main(int argc, char* argv[])
{
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

        g_deviceCommServer = bootstrapNetworkServices(wsServer);

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
    LOG_PHASE("Intent classification system initialized", true);

    // ======================================================
    // 6.5  Initialize Perception System
    // ======================================================
    GRIM::Perception::init();
    LOG_PHASE("Perception system initialized", true);
    
    // ✅ NEW: Initialize context-aware perception manager
    GRIM::Perception::initContextManager();
    LOG_PHASE("Perception context manager initialized", true);
    
    auto ui = bootstrapUI(*g_deviceCommServer, argc, argv);

    // ======================================================
    // 11. Wake systems - Initialize with nullptr for unused parameters
    // ======================================================
    static std::vector<Timer> emptyTimers;
    static nlohmann::json emptyMemory;
    auto& defaultSessionHistory = GRIM::MMO::SessionContextManager::instance()
        .displayHistory("default");
    WakeKey::start(&defaultSessionHistory, emptyTimers, emptyMemory);
    LOG_PHASE("WakeKey listener started", true);

    WakeVoice::start(nullptr, emptyTimers, emptyMemory);
    LOG_PHASE("WakeVoice listener started", true);

    // ======================================================
    // 12. Main loop
    // ======================================================
    LOG_PHASE("Entering main thread render loop", true);

    while (!WindowManager::isMainLoopStopRequested() &&
           !g_shutdownRequested.load(std::memory_order_acquire))
    {
        GRIM::tickApplicationFrame(*ui.geoSpatialRuntime, ui.overlayWindow);
    }

    LOG_PHASE("Main thread render loop exited", true);

    GRIM::shutdownApplication(g_memoryStorage, wsServer, g_deviceCommServer);
    g_shutdownComplete.store(true, std::memory_order_release);
    return 0;
}
