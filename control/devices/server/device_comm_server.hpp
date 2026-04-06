#pragma once

#include "../registry/device_registry.hpp"
#include "../registry/device_snapshot.hpp"
#include "../session/session_manager.hpp"
#include "../storage/storage_manager.hpp"
#include "../transfer/file_transfer_manager.hpp"

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

namespace GRIM {

// ─────────────────────────────────────────────────────────
//  DeviceCommServer — WebSocket hub for device communication
//
//  Port 11437 (configurable).  Owns all subsystems:
//    DeviceRegistry, SessionManager, StorageManager,
//    FileTransferManager.
//
//  Text frames  → JSON protocol (device_protocol.hpp)
//  Binary frames → chunked file transfer
// ─────────────────────────────────────────────────────────

class DeviceCommServer {
public:
    struct Config {
        uint16_t    port                 = 11437;
        uint32_t    heartbeat_timeout_sec = 30;
        uint32_t    max_chunk_size_bytes  = 65536;
        uint32_t    transfer_timeout_sec  = 300;
        std::string storage_root; // required — no default
        std::string registry_dir; // required — no default
    };

    explicit DeviceCommServer(const Config& config);
    ~DeviceCommServer();

    bool start();
    void stop();

    // Snapshot list for UI consumption (thread-safe)
    std::vector<DeviceSnapshot> listDeviceSnapshots() const;

    // Generate a pairing code and create a Pending device record.
    // Returns the XXXX-XXXX code to display in UI.
    std::string createPendingDevice();

    // Accept a user-entered pairing code and create a Pending device record.
    // Returns true if the code was accepted (valid format, not already in use).
    bool addPendingDeviceWithCode(const std::string& code);

    // Access subsystems (for UI queries)
    const StorageManager& storageManager() const { return storage_; }
    StorageManager& storageManager() { return storage_; }
    const FileTransferManager& transferManager() const { return transfer_; }

private:
    Config config_;

    DeviceRegistry     registry_;
    SessionManager     sessions_;
    StorageManager     storage_;
    FileTransferManager transfer_;

    std::atomic<bool> running_{false};
    std::thread       server_thread_;
};

} // namespace GRIM
