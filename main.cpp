#include "bootstrap/bootstrap.hpp"
#include "logger.hpp"
#include "wake/wake_key.hpp"
#include "wake/wake_voice.hpp"
#include "memory/unified_memory.hpp"
#include "core/window_manager.hpp"
#include "core/system_tick.hpp"
#include "core/system_teardown.hpp"
#include "core/plugin_manager.hpp"
#include "core/crash_dump.hpp"
#include "net/websocket_server.hpp"
#include "control/devices/server/device_comm_server.hpp"
#include "resources.hpp"

GRIM::UnifiedMemoryStorage g_memoryStorage;
static GRIM::WebSocketServer wsServer;
static std::unique_ptr<GRIM::DeviceCommServer> g_deviceCommServer;

int main(int argc, char* argv[])
{
    initLogger("grim.log");
    GRIM::InstallCrashDumpHandler();
    LOG_PHASE("Initializing G.R.I.M", true);

#ifdef _WIN32
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
#endif

    GRIM::installApplicationShutdownHandlers();

    g_deviceCommServer = bootstrapNetworkServices(wsServer);

    runBootstrapChecks(argc, argv);

    PluginManager::initialize((std::filesystem::path(getGrimRootDir()) / "plugins").string());
    LOG_PHASE("Plugin manager initialized", true);

    bootstrapVoiceSubsystem();

    bootstrapMemorySubsystem(g_memoryStorage);

    auto ui = bootstrapUI(*g_deviceCommServer, argc, argv);

    WakeKey::start();
    LOG_PHASE("WakeKey listener started", true);

    WakeVoice::start();
    LOG_PHASE("WakeVoice listener started", true);

    LOG_PHASE("Entering main thread render loop", true);

    while (!WindowManager::isMainLoopStopRequested() &&
            !GRIM::isApplicationShutdownRequested())
    {
        GRIM::tickApplicationFrame(*ui.geoSpatialRuntime, ui.overlayWindow);
    }

    LOG_PHASE("Main thread render loop exited", true);

    GRIM::shutdownApplication(g_memoryStorage, wsServer, g_deviceCommServer);
    return 0;
}
