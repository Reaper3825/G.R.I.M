#include "bootstrap.hpp"

#include "../control/devices/server/device_comm_server.hpp"
#include "../logger.hpp"
#include "../net/websocket_server.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>

std::unique_ptr<GRIM::DeviceCommServer> bootstrapNetworkServices(
    GRIM::WebSocketServer& webSocketServer)
{
    if (!webSocketServer.start()) {
        throw std::runtime_error("bootstrapNetworkServices: WebSocket server failed to start");
    }
    LOG_PHASE("WebSocket server started", true);

    GRIM::DeviceCommServer::Config config;
    config.port = 11437;
    config.heartbeat_timeout_sec = 30;
    config.max_chunk_size_bytes = 65536;
    config.transfer_timeout_sec = 300;

#ifdef _WIN32
    const char* home = std::getenv("USERPROFILE");
    if (!home) {
        throw std::runtime_error("bootstrapNetworkServices: USERPROFILE is not set");
    }
#else
    const char* home = std::getenv("HOME");
    if (!home) {
        throw std::runtime_error("bootstrapNetworkServices: HOME is not set");
    }
#endif

    config.storage_root = std::string(home) + "/.grim/shared_storage/";
    config.registry_dir = std::string(home) + "/.grim/devices/";

    auto deviceCommServer = std::make_unique<GRIM::DeviceCommServer>(config);
    if (!deviceCommServer->start()) {
        throw std::runtime_error(
            "bootstrapNetworkServices: device communication server failed to start");
    }
    LOG_PHASE("Device comm server started", true);
    LOG_DEBUG("Bootstrap", "Heap check mode: Manual (removed CHECK_ALWAYS for debugging)");
    return deviceCommServer;
}