#pragma once

#include <memory>

namespace GRIM {

class DeviceCommServer;
class WebSocketServer;

void installApplicationShutdownHandlers();
bool isApplicationShutdownRequested();

void shutdownApplication(
    WebSocketServer& webSocketServer,
    std::unique_ptr<DeviceCommServer>& deviceCommServer);

} // namespace GRIM