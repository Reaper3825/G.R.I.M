#pragma once

#include "core/host_exports.hpp"
#include "control/devices/registry/device_snapshot.hpp"
#include "control/devices/storage/storage_index.hpp"
#include "control/devices/transfer/file_transfer_manager.hpp"

namespace GRIM { class DeviceCommServer; }

namespace GRIM::UI::Storage {
struct Status {
    uint64_t diskUsage = 0;
    std::string localCode;
    bool connected = false;
    std::string hubHost;
    uint16_t hubPort = 0;
};

struct HubResult {
    bool success;
    std::string message;
};

// Host lifecycle only; bind before plugin initialization, unbind after retirement.
void bindServer(DeviceCommServer& server);
void unbindServer(DeviceCommServer& server);

// UI-thread operations return copies; the server and its subsystems stay host-owned.
GRIM_HOST_API Status status();
GRIM_HOST_API std::vector<DirectoryEntry> listDirectory(const std::string& path);
GRIM_HOST_API std::vector<DeviceSnapshot> listDevices();
GRIM_HOST_API std::vector<ActiveTransfer> listTransfers();
GRIM_HOST_API void regenerateCode();
GRIM_HOST_API bool linkDevice(const std::string& code);
GRIM_HOST_API HubResult connectHub(const std::string& host, uint16_t port);
GRIM_HOST_API void disconnectHub();
}
