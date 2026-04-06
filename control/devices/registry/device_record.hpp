#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace GRIM {

// ─────────────────────────────────────────────────────────
//  Enums
// ─────────────────────────────────────────────────────────

enum class DeviceType : uint8_t {
    Desktop = 0,
    Laptop  = 1,
    Phone   = 2,
    Tablet  = 3,
    Server  = 4
};

enum class Platform : uint8_t {
    Windows = 0,
    macOS   = 1,
    Linux   = 2,
    iOS     = 3,
    Android = 4
};

enum class PairingState : uint8_t {
    Pending = 0,
    Paired  = 1,
    Blocked = 2
};

// Bitfield for device roles
enum DeviceRole : uint32_t {
    Role_None    = 0,
    Role_Storage = 1 << 0,
    Role_Compute = 1 << 1,
    Role_Display = 1 << 2
};

// ─────────────────────────────────────────────────────────
//  JSON helpers for enums
// ─────────────────────────────────────────────────────────

NLOHMANN_JSON_SERIALIZE_ENUM(DeviceType, {
    { DeviceType::Desktop, "desktop" },
    { DeviceType::Laptop,  "laptop"  },
    { DeviceType::Phone,   "phone"   },
    { DeviceType::Tablet,  "tablet"  },
    { DeviceType::Server,  "server"  }
})

NLOHMANN_JSON_SERIALIZE_ENUM(Platform, {
    { Platform::Windows, "windows" },
    { Platform::macOS,   "macos"   },
    { Platform::Linux,   "linux"   },
    { Platform::iOS,     "ios"     },
    { Platform::Android, "android" }
})

NLOHMANN_JSON_SERIALIZE_ENUM(PairingState, {
    { PairingState::Pending, "pending" },
    { PairingState::Paired,  "paired"  },
    { PairingState::Blocked, "blocked" }
})

// ─────────────────────────────────────────────────────────
//  DeviceRecord — persistent hub-owned device metadata
// ─────────────────────────────────────────────────────────

struct DeviceRecord {
    std::string              device_id;         // UUID
    std::string              device_name;
    DeviceType               device_type;
    Platform                 platform;
    PairingState             pairing_state = PairingState::Pending;
    std::string              pairing_code;       // XXXX-XXXX
    uint32_t                 roles = Role_None;  // bitfield
    std::vector<std::string> allowed_actions;
    std::string              registered_at;      // ISO 8601
    std::string              last_seen_at;       // ISO 8601
};

inline void to_json(nlohmann::json& j, const DeviceRecord& r) {
    j = nlohmann::json{
        { "device_id",      r.device_id      },
        { "device_name",    r.device_name    },
        { "device_type",    r.device_type    },
        { "platform",       r.platform       },
        { "pairing_state",  r.pairing_state  },
        { "pairing_code",   r.pairing_code   },
        { "roles",          r.roles          },
        { "allowed_actions",r.allowed_actions},
        { "registered_at",  r.registered_at  },
        { "last_seen_at",   r.last_seen_at   }
    };
}

inline void from_json(const nlohmann::json& j, DeviceRecord& r) {
    auto require = [&](const char* field) {
        if (!j.contains(field))
            throw std::runtime_error(
                std::string("DeviceRecord: missing required field '") + field + "'");
    };

    require("device_id");
    require("device_name");
    require("device_type");
    require("platform");
    require("pairing_state");
    require("pairing_code");
    require("roles");
    require("allowed_actions");
    require("registered_at");
    require("last_seen_at");

    j.at("device_id").get_to(r.device_id);
    j.at("device_name").get_to(r.device_name);
    j.at("device_type").get_to(r.device_type);
    j.at("platform").get_to(r.platform);
    j.at("pairing_state").get_to(r.pairing_state);
    j.at("pairing_code").get_to(r.pairing_code);
    j.at("roles").get_to(r.roles);
    j.at("allowed_actions").get_to(r.allowed_actions);
    j.at("registered_at").get_to(r.registered_at);
    j.at("last_seen_at").get_to(r.last_seen_at);
}

} // namespace GRIM
