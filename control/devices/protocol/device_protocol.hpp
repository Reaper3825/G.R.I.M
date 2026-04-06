#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <nlohmann/json.hpp>

namespace GRIM {

// ─────────────────────────────────────────────────────────
//  MessageType — all device-comm protocol message types
// ─────────────────────────────────────────────────────────

enum class MessageType : uint8_t {
    Register         = 0,
    PairRequest      = 1,
    PairResponse     = 2,
    Heartbeat        = 3,
    HeartbeatAck     = 4,
    ListFiles        = 5,
    FileMetadata     = 6,
    TransferRequest  = 7,
    TransferResponse = 8,
    TransferComplete = 9,
    Error            = 10
};

NLOHMANN_JSON_SERIALIZE_ENUM(MessageType, {
    { MessageType::Register,         "register"          },
    { MessageType::PairRequest,      "pair_request"      },
    { MessageType::PairResponse,     "pair_response"     },
    { MessageType::Heartbeat,        "heartbeat"         },
    { MessageType::HeartbeatAck,     "heartbeat_ack"     },
    { MessageType::ListFiles,        "list_files"        },
    { MessageType::FileMetadata,     "file_metadata"     },
    { MessageType::TransferRequest,  "transfer_request"  },
    { MessageType::TransferResponse, "transfer_response" },
    { MessageType::TransferComplete, "transfer_complete" },
    { MessageType::Error,            "error"             }
})

// ─────────────────────────────────────────────────────────
//  Parse top-level envelope: { "type": "...", "payload": {...} }
// ─────────────────────────────────────────────────────────

inline MessageType parseMessageType(const nlohmann::json& msg) {
    if (!msg.contains("type"))
        throw std::runtime_error("DeviceProtocol: message missing 'type' field");
    return msg.at("type").get<MessageType>();
}

inline const nlohmann::json& getPayload(const nlohmann::json& msg) {
    if (!msg.contains("payload"))
        throw std::runtime_error("DeviceProtocol: message missing 'payload' field");
    return msg.at("payload");
}

inline nlohmann::json makeEnvelope(MessageType type, const nlohmann::json& payload) {
    return { { "type", type }, { "payload", payload } };
}

} // namespace GRIM
