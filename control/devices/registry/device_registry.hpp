#pragma once

#include "device_record.hpp"
#include <mutex>
#include <string>
#include <vector>

namespace GRIM {

// ─────────────────────────────────────────────────────────
//  DeviceRegistry — persistent device metadata store
//
//  Thread-safe.  Persists to ~/.grim/devices/device_registry.json
//  via AtomicWriter.  All mutations auto-persist.
// ─────────────────────────────────────────────────────────

class DeviceRegistry {
public:
    explicit DeviceRegistry(const std::string& storage_dir);

    // Code generation
    std::string generatePairingCode();

    // CRUD
    void addDevice(DeviceRecord record);
    void removeDevice(const std::string& device_id);

    // Lookups
    const DeviceRecord* findByDeviceId(const std::string& device_id) const;
    const DeviceRecord* findByPairingCode(const std::string& code) const;

    // State transitions
    DeviceRecord completePairing(const std::string& pairing_code,
                                 const std::string& device_name,
                                 DeviceType device_type,
                                 Platform platform);
    void setPairingState(const std::string& device_id, PairingState state);
    void blockDevice(const std::string& device_id);
    void updateLastSeen(const std::string& device_id, const std::string& iso_time);

    // Listings
    std::vector<DeviceRecord> listAll() const;
    std::vector<DeviceRecord> listPaired() const;

private:
    void load();
    void persist() const;

    std::string registry_path_;
    mutable std::mutex mutex_;
    std::vector<DeviceRecord> devices_;
};

} // namespace GRIM
