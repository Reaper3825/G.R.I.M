#pragma once

#include "../registry/device_registry.hpp"
#include "../registry/device_snapshot.hpp"
#include "../session/session_manager.hpp"
#include "../storage/storage_manager.hpp"
#include "../transfer/file_transfer_manager.hpp"

#include <uwebsockets/App.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
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

    // Local instance code — shown in UI so the user can enter it
    // on the hub to link this device.
    const std::string& localDeviceCode() const { return local_device_code_; }
    void regenerateLocalCode();

    // Accept a user-entered device code and create a Pending device record.
    // Returns true if the code was accepted (not empty, not already in use).
    bool addPendingDeviceWithCode(const std::string& code);

    // Connect to a remote hub and register this instance using its local code.
    // hub_host: IP or hostname, hub_port: port (default 11437).
    // Returns { success, message } for UI feedback.
    struct RegisterResult { bool success; std::string message; };
    RegisterResult registerWithHub(const std::string& hub_host, uint16_t hub_port);

    // Disconnect from the current hub (clears saved connection).
    void disconnectFromHub();

    // Hub connection state (persisted across restarts)
    bool        isConnectedToHub() const { return connected_to_hub_; }
    std::string connectedHubHost() const { return hub_host_; }
    uint16_t    connectedHubPort() const { return hub_port_; }

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

    std::string local_device_code_; // this instance's pairing code

    // Persisted hub connection state (satellite → hub)
    bool        connected_to_hub_ = false;
    std::string hub_host_;
    uint16_t    hub_port_ = 0;

    void loadHubConnection();
    void saveHubConnection() const;
    std::string hubConnectionPath() const;

    std::atomic<bool> running_{false};
    std::thread       server_thread_;
    std::mutex lifecycle_mutex_;
    std::condition_variable startup_cv_;
    uWS::Loop* server_loop_ = nullptr;
    uWS::App* server_app_ = nullptr;
    bool startup_complete_ = false;
    bool listen_succeeded_ = false;
    bool stop_requested_ = false;
};

} // namespace GRIM
