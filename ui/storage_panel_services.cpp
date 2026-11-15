#include "storage_panel_services.hpp"
#include "control/devices/server/device_comm_server.hpp"
#include "ui_root.hpp"

#include <stdexcept>

namespace GRIM::UI::Storage {
namespace {
DeviceCommServer* boundServer = nullptr;

DeviceCommServer& server() {
    UIRoot::get().requireUIThread();
    if (!boundServer)
        throw std::runtime_error("Storage panel service requires a bound host device server");
    return *boundServer;
}
}

void bindServer(DeviceCommServer& value) {
    UIRoot::get().requireUIThread();
    if (boundServer) throw std::logic_error("Storage panel device server is already bound");
    boundServer = &value;
}

void unbindServer(DeviceCommServer& value) {
    UIRoot::get().requireUIThread();
    if (boundServer != &value)
        throw std::logic_error("Storage panel device server binding does not match");
    boundServer = nullptr;
}

Status status() {
    auto& value = server();
    return {value.storageManager().getDiskUsage(), value.localDeviceCode(),
            value.isConnectedToHub(), value.connectedHubHost(), value.connectedHubPort()};
}
std::vector<DirectoryEntry> listDirectory(const std::string& path) {
    return server().storageManager().index().listDirectory(path);
}
std::vector<DeviceSnapshot> listDevices() { return server().listDeviceSnapshots(); }
std::vector<ActiveTransfer> listTransfers() { return server().transferManager().listActive(); }
void regenerateCode() { server().regenerateLocalCode(); }
bool linkDevice(const std::string& code) {
    if (code.empty()) throw std::invalid_argument("Storage device code must not be empty");
    return server().addPendingDeviceWithCode(code);
}
HubResult connectHub(const std::string& host, uint16_t port) {
    if (host.empty() || port == 0)
        throw std::invalid_argument("Storage hub requires a host and a nonzero port");
    auto result = server().registerWithHub(host, port);
    return {result.success, std::move(result.message)};
}
void disconnectHub() { server().disconnectFromHub(); }
}
