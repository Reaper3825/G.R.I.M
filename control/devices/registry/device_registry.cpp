#include "device_registry.hpp"
#include "memory/atomic_writer.hpp"
#include "logger.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <nlohmann/json.hpp>

namespace GRIM {

static constexpr const char* TAG = "DeviceRegistry";
static constexpr const char* REGISTRY_FILENAME = "device_registry.json";

// ─── Helpers ─────────────────────────────────────────────

static std::string generateUUID() {
    static std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<uint32_t> dist(0, 15);

    const char hex[] = "0123456789abcdef";
    const char* pattern = "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx";

    std::string uuid;
    uuid.reserve(36);
    for (const char* p = pattern; *p; ++p) {
        if (*p == 'x') {
            uuid += hex[dist(gen)];
        } else if (*p == 'y') {
            uuid += hex[(dist(gen) & 0x3) | 0x8];
        } else {
            uuid += *p;
        }
    }
    return uuid;
}

static std::string nowISO8601() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &time);
#else
    gmtime_r(&time, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

// ─── Construction ────────────────────────────────────────

DeviceRegistry::DeviceRegistry(const std::string& storage_dir) {
    std::filesystem::path dir(storage_dir);
    std::filesystem::create_directories(dir);
    registry_path_ = (dir / REGISTRY_FILENAME).string();
    load();
    LOG_DEBUG(TAG, "Initialized with " + std::to_string(devices_.size()) + " devices");
}

// ─── Pairing code generation ─────────────────────────────

std::string DeviceRegistry::generatePairingCode() {
    static const char charset[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    static constexpr int charset_len = sizeof(charset) - 1;
    static std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, charset_len - 1);

    std::string code;
    code.reserve(9); // XXXX-XXXX
    for (int i = 0; i < 8; ++i) {
        if (i == 4) code += '-';
        code += charset[dist(gen)];
    }

    // Ensure uniqueness within registry
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& d : devices_) {
        if (d.pairing_code == code)
            throw std::runtime_error("Pairing code collision — retry");
    }
    return code;
}

// ─── CRUD ────────────────────────────────────────────────

void DeviceRegistry::addDevice(DeviceRecord record) {
    std::lock_guard<std::mutex> lock(mutex_);

    for (const auto& d : devices_) {
        if (d.device_id == record.device_id)
            throw std::runtime_error(
                "DeviceRegistry: duplicate device_id '" + record.device_id + "'");
    }

    if (record.device_id.empty())
        record.device_id = generateUUID();
    if (record.registered_at.empty())
        record.registered_at = nowISO8601();
    if (record.last_seen_at.empty())
        record.last_seen_at = record.registered_at;

    devices_.push_back(std::move(record));
    persist();
    LOG_DEBUG(TAG, "Added device: " + devices_.back().device_id);
}

void DeviceRegistry::removeDevice(const std::string& device_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = std::find_if(devices_.begin(), devices_.end(),
                           [&](const DeviceRecord& d) { return d.device_id == device_id; });
    if (it == devices_.end())
        throw std::runtime_error("DeviceRegistry: device_id '" + device_id + "' not found");

    devices_.erase(it);
    persist();
    LOG_DEBUG(TAG, "Removed device: " + device_id);
}

// ─── Lookups ─────────────────────────────────────────────

const DeviceRecord* DeviceRegistry::findByDeviceId(const std::string& device_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& d : devices_) {
        if (d.device_id == device_id) return &d;
    }
    return nullptr;
}

const DeviceRecord* DeviceRegistry::findByPairingCode(const std::string& code) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& d : devices_) {
        if (d.pairing_code == code) return &d;
    }
    return nullptr;
}

// ─── State transitions ──────────────────────────────────

DeviceRecord DeviceRegistry::completePairing(const std::string& pairing_code,
                                             const std::string& device_name,
                                             DeviceType device_type,
                                             Platform platform) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& d : devices_) {
        if (d.pairing_code != pairing_code) continue;

        if (d.pairing_state != PairingState::Pending) {
            throw std::runtime_error(
                "DeviceRegistry::completePairing: pairing code '" + pairing_code +
                "' is not pending");
        }

        d.device_name = device_name;
        d.device_type = device_type;
        d.platform = platform;
        d.pairing_state = PairingState::Paired;
        d.last_seen_at = nowISO8601();
        persist();
        return d;
    }

    throw std::runtime_error(
        "DeviceRegistry::completePairing: pairing code '" + pairing_code + "' not found");
}

void DeviceRegistry::setPairingState(const std::string& device_id, PairingState state) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& d : devices_) {
        if (d.device_id == device_id) {
            d.pairing_state = state;
            persist();
            return;
        }
    }
    throw std::runtime_error("DeviceRegistry::setPairingState: device_id '" + device_id + "' not found");
}

void DeviceRegistry::blockDevice(const std::string& device_id) {
    setPairingState(device_id, PairingState::Blocked);
}

void DeviceRegistry::updateLastSeen(const std::string& device_id, const std::string& iso_time) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& d : devices_) {
        if (d.device_id == device_id) {
            d.last_seen_at = iso_time;
            persist();
            return;
        }
    }
}

// ─── Listings ────────────────────────────────────────────

std::vector<DeviceRecord> DeviceRegistry::listAll() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return devices_;
}

std::vector<DeviceRecord> DeviceRegistry::listPaired() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<DeviceRecord> result;
    for (const auto& d : devices_) {
        if (d.pairing_state == PairingState::Paired)
            result.push_back(d);
    }
    return result;
}

// ─── Persistence ─────────────────────────────────────────

void DeviceRegistry::load() {
    if (!std::filesystem::exists(registry_path_)) return;

    std::ifstream in(registry_path_);
    if (!in)
        throw std::runtime_error("DeviceRegistry: cannot open " + registry_path_);

    nlohmann::json j;
    in >> j;

    if (!j.is_array())
        throw std::runtime_error("DeviceRegistry: expected JSON array in " + registry_path_);

    devices_.clear();
    for (const auto& entry : j) {
        devices_.push_back(entry.get<DeviceRecord>());
    }
}

void DeviceRegistry::persist() const {
    nlohmann::json j = nlohmann::json::array();
    for (const auto& d : devices_) {
        j.push_back(d);
    }
    AtomicWriter::writeString(registry_path_, j.dump(2));
}

} // namespace GRIM
