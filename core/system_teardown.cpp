#include "system_teardown.hpp"

#include "../MMO/UI/UISurfaceRegistry.hpp"
#include "../ai/ai_rl.hpp"
#include "../ai/training_server_manager.hpp"
#include "../bootstrap/bootstrap.hpp"
#include "../control/devices/server/device_comm_server.hpp"
#include "../logger.hpp"
#include "../memory/memory_buffer_rotation.hpp"
#include "../memory/unified_memory.hpp"
#include "../net/websocket_server.hpp"
#include "../perception/digital/DigitalContextProjector.hpp"
#include "../perception/digital/DigitalEnvironmentLoop.hpp"
#include "../perception/digital/DigitalPerceptionPrimitivesLoop.hpp"
#include "../perception/perception.hpp"
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
#include "../voice/voice_speak.hpp"
#include "../wake/wake_key.hpp"
#include "../wake/wake_voice.hpp"
#include "platform_clipboard.hpp"
#include "platform_input.hpp"
#include "window_manager.hpp"
#include "grim_platform.h"
#include "../helpers/mouse.hpp"

#include <atomic>
#include <csignal>
#include <stdexcept>

namespace {

std::atomic<bool> s_shutdownRequested{false};
std::atomic<bool> s_shutdownComplete{false};

#ifdef _WIN32
BOOL WINAPI consoleHandler(DWORD signal)
{
    if (signal != CTRL_C_EVENT &&
        signal != CTRL_CLOSE_EVENT &&
        signal != CTRL_BREAK_EVENT) {
        return FALSE;
    }

    s_shutdownRequested.store(true, std::memory_order_release);
    WindowManager::requestMainLoopStop();

    if (signal == CTRL_CLOSE_EVENT) {
        constexpr DWORD kCloseDrainTimeoutMs = 4500;
        constexpr DWORD kPollIntervalMs = 10;
        for (DWORD waitedMs = 0;
             waitedMs < kCloseDrainTimeoutMs &&
                 !s_shutdownComplete.load(std::memory_order_acquire);
             waitedMs += kPollIntervalMs) {
            ::Sleep(kPollIntervalMs);
        }
    }

    return TRUE;
}
#else
void signalHandler(int signal)
{
    if (signal == SIGINT || signal == SIGTERM) {
        s_shutdownRequested.store(true, std::memory_order_release);
    }
}
#endif

} // namespace

namespace GRIM {

void installApplicationShutdownHandlers()
{
#ifdef _WIN32
    if (!SetConsoleCtrlHandler(consoleHandler, TRUE)) {
        throw std::runtime_error(
            "installApplicationShutdownHandlers: SetConsoleCtrlHandler failed");
    }
    LOG_DEBUG("Teardown", "Signal handler installed (Ctrl+C will request shutdown)");
#else
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
    LOG_DEBUG("Teardown", "Signal handlers installed");
#endif
}

bool isApplicationShutdownRequested()
{
    return s_shutdownRequested.load(std::memory_order_acquire);
}

void shutdownApplication(
    UnifiedMemoryStorage& memoryStorage,
    WebSocketServer& webSocketServer,
    std::unique_ptr<DeviceCommServer>& deviceCommServer)
{
    LOG_PHASE("Shutting down subsystems", true);

    stopMMOIdleTick();

    Perception::Physical::ShutdownPhysicalWorldStateMemoryWriter();
    Perception::Physical::ShutdownPhysicalWorldStateContextProjector();

    MemoryBufferRotation::instance().mergeToWorking();
    MemoryBufferRotation::instance().syncToLongTerm(memoryStorage);
    MemoryBufferRotation::instance().clear();

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
    MMO::UISurfaceRegistry::instance().destroyAll();
    if (g_resourceCoordinator) {
        delete g_resourceCoordinator;
        g_resourceCoordinator = nullptr;
    }
    if (g_resourceSignal) {
        g_resourceSignal->stop();
        delete g_resourceSignal;
        g_resourceSignal = nullptr;
    }

    stopTrainingServer();

    WakeKey::stop();
    WakeVoice::stop();
    Voice::shutdownQueue();
    Voice::shutdownTTS();
    RL::shutdown();
    Perception::Digital::ShutdownDigitalPerceptionPrimitives();
    Perception::Digital::ShutdownDigitalEnvironment();
    Perception::Digital::ShutdownDigitalContextProjector();
    Perception::Physical::ShutdownPhysicalWorldStateMemoryWriter();
    Perception::Physical::ShutdownPhysicalWorldStateContextProjector();
    Perception::Physical::ShutdownPhysicalWorldState();
    Perception::Physical::ShutdownPhysicalLocalization();
    Perception::Physical::ShutdownPhysicalSpatialGrounding();
    Perception::Physical::ShutdownPhysicalPerceptionPrimitives();
    Perception::Physical::ShutdownPhysicalGestureControl();
    Perception::Physical::ShutdownPhysicalInteraction();
    Perception::Physical::ShutdownPhysicalEnvironment();
    Perception::shutdown();
    Mouse::shutdown();

    PlatformInput::shutdown();
    PlatformClipboard::shutdown();

    UIRoot::get().shutdown();
    WindowManager::shutdown();

    webSocketServer.stop();
    if (deviceCommServer) {
        deviceCommServer->stop();
        deviceCommServer.reset();
    }

    memoryStorage.shutdown();

    LOG_PHASE("All subsystems shut down", true);
    LOG_PHASE("G.R.I.M terminated successfully", true);
    shutdownLogger();
    s_shutdownComplete.store(true, std::memory_order_release);
}

} // namespace GRIM