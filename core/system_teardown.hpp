#pragma once

#include <memory>

namespace GRIM {

class DeviceCommServer;
class UnifiedMemoryStorage;
class WebSocketServer;

void installApplicationShutdownHandlers();
bool isApplicationShutdownRequested();

void shutdownApplication(
    UnifiedMemoryStorage& memoryStorage,
    WebSocketServer& webSocketServer,
    std::unique_ptr<DeviceCommServer>& deviceCommServer);

} // namespace GRIM