#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace GRIM {

// ─────────────────────────────────────────────────────────
//  TransferState — what the device is currently doing
// ─────────────────────────────────────────────────────────

enum class TransferState : uint8_t {
    Idle        = 0,
    Uploading   = 1,
    Downloading = 2
};

// ─────────────────────────────────────────────────────────
//  DeviceSession — runtime-only, NOT persisted
// ─────────────────────────────────────────────────────────

struct DeviceSession {
    std::string              session_id;
    std::string              device_id;
    std::string              remote_address;
    void*                    ws_handle = nullptr; // uWS::WebSocket<...>*
    std::chrono::steady_clock::time_point connected_at;
    std::chrono::steady_clock::time_point last_heartbeat_at;
    std::vector<std::string> available_capabilities;
    TransferState            transfer_state = TransferState::Idle;
};

} // namespace GRIM
