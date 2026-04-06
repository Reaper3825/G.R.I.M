#pragma once

#include "device_protocol.hpp"
#include "../registry/device_record.hpp"
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace GRIM {

// ─────────────────────────────────────────────────────────
//  Register — device announces itself with a pairing code
// ─────────────────────────────────────────────────────────

struct RegisterMessage {
    std::string pairing_code;   // XXXX-XXXX
    std::string device_name;
    DeviceType  device_type;
    Platform    platform;
    std::vector<std::string> capabilities;
};

inline void to_json(nlohmann::json& j, const RegisterMessage& m) {
    j = {
        { "pairing_code", m.pairing_code },
        { "device_name",  m.device_name  },
        { "device_type",  m.device_type  },
        { "platform",     m.platform     },
        { "capabilities", m.capabilities }
    };
}

inline void from_json(const nlohmann::json& j, RegisterMessage& m) {
    if (!j.contains("pairing_code"))
        throw std::runtime_error("RegisterMessage: missing 'pairing_code'");
    if (!j.contains("device_name"))
        throw std::runtime_error("RegisterMessage: missing 'device_name'");
    if (!j.contains("device_type"))
        throw std::runtime_error("RegisterMessage: missing 'device_type'");
    if (!j.contains("platform"))
        throw std::runtime_error("RegisterMessage: missing 'platform'");
    if (!j.contains("capabilities"))
        throw std::runtime_error("RegisterMessage: missing 'capabilities'");

    j.at("pairing_code").get_to(m.pairing_code);
    j.at("device_name").get_to(m.device_name);
    j.at("device_type").get_to(m.device_type);
    j.at("platform").get_to(m.platform);
    j.at("capabilities").get_to(m.capabilities);
}

// ─────────────────────────────────────────────────────────
//  PairResponse — hub replies to Register
// ─────────────────────────────────────────────────────────

struct PairResponseMessage {
    bool        success;
    std::string device_id;      // assigned UUID (empty on failure)
    std::string error_message;  // empty on success
};

inline void to_json(nlohmann::json& j, const PairResponseMessage& m) {
    j = {
        { "success",       m.success       },
        { "device_id",     m.device_id     },
        { "error_message", m.error_message }
    };
}

inline void from_json(const nlohmann::json& j, PairResponseMessage& m) {
    if (!j.contains("success"))
        throw std::runtime_error("PairResponseMessage: missing 'success'");
    j.at("success").get_to(m.success);
    j.at("device_id").get_to(m.device_id);
    j.at("error_message").get_to(m.error_message);
}

// ─────────────────────────────────────────────────────────
//  Heartbeat — device sends periodically to stay alive
// ─────────────────────────────────────────────────────────

struct HeartbeatMessage {
    std::string device_id;
    std::vector<std::string> available_capabilities;
};

inline void to_json(nlohmann::json& j, const HeartbeatMessage& m) {
    j = {
        { "device_id",               m.device_id              },
        { "available_capabilities",   m.available_capabilities }
    };
}

inline void from_json(const nlohmann::json& j, HeartbeatMessage& m) {
    if (!j.contains("device_id"))
        throw std::runtime_error("HeartbeatMessage: missing 'device_id'");
    if (!j.contains("available_capabilities"))
        throw std::runtime_error("HeartbeatMessage: missing 'available_capabilities'");
    j.at("device_id").get_to(m.device_id);
    j.at("available_capabilities").get_to(m.available_capabilities);
}

// ─────────────────────────────────────────────────────────
//  HeartbeatAck — hub acknowledges heartbeat
// ─────────────────────────────────────────────────────────

struct HeartbeatAckMessage {
    std::string device_id;
};

inline void to_json(nlohmann::json& j, const HeartbeatAckMessage& m) {
    j = { { "device_id", m.device_id } };
}

inline void from_json(const nlohmann::json& j, HeartbeatAckMessage& m) {
    if (!j.contains("device_id"))
        throw std::runtime_error("HeartbeatAckMessage: missing 'device_id'");
    j.at("device_id").get_to(m.device_id);
}

// ─────────────────────────────────────────────────────────
//  ListFiles — request directory listing
// ─────────────────────────────────────────────────────────

struct ListFilesMessage {
    std::string directory_path; // relative path, empty = root
};

inline void to_json(nlohmann::json& j, const ListFilesMessage& m) {
    j = { { "directory_path", m.directory_path } };
}

inline void from_json(const nlohmann::json& j, ListFilesMessage& m) {
    if (!j.contains("directory_path"))
        throw std::runtime_error("ListFilesMessage: missing 'directory_path'");
    j.at("directory_path").get_to(m.directory_path);
}

// ─────────────────────────────────────────────────────────
//  FileMetadata — single file entry in a listing response
// ─────────────────────────────────────────────────────────

struct FileMetadataMessage {
    std::string file_id;
    std::string relative_path;
    uint64_t    size_bytes;
    std::string sha256_hash;
    std::string content_type;
    std::string source_device_id;
    std::string modified_at;
    bool        is_directory;
};

inline void to_json(nlohmann::json& j, const FileMetadataMessage& m) {
    j = {
        { "file_id",          m.file_id          },
        { "relative_path",    m.relative_path    },
        { "size_bytes",       m.size_bytes       },
        { "sha256_hash",      m.sha256_hash      },
        { "content_type",     m.content_type     },
        { "source_device_id", m.source_device_id },
        { "modified_at",      m.modified_at      },
        { "is_directory",     m.is_directory     }
    };
}

inline void from_json(const nlohmann::json& j, FileMetadataMessage& m) {
    if (!j.contains("file_id"))
        throw std::runtime_error("FileMetadataMessage: missing 'file_id'");
    if (!j.contains("relative_path"))
        throw std::runtime_error("FileMetadataMessage: missing 'relative_path'");
    j.at("file_id").get_to(m.file_id);
    j.at("relative_path").get_to(m.relative_path);
    j.at("size_bytes").get_to(m.size_bytes);
    j.at("sha256_hash").get_to(m.sha256_hash);
    j.at("content_type").get_to(m.content_type);
    j.at("source_device_id").get_to(m.source_device_id);
    j.at("modified_at").get_to(m.modified_at);
    j.at("is_directory").get_to(m.is_directory);
}

// ─────────────────────────────────────────────────────────
//  TransferRequest — initiate a file transfer
// ─────────────────────────────────────────────────────────

struct TransferRequestMessage {
    std::string relative_path;
    uint64_t    size_bytes;
    std::string sha256;
    uint32_t    chunk_size;     // bytes per chunk (default 65536)
};

inline void to_json(nlohmann::json& j, const TransferRequestMessage& m) {
    j = {
        { "relative_path", m.relative_path },
        { "size_bytes",    m.size_bytes    },
        { "sha256",        m.sha256        },
        { "chunk_size",    m.chunk_size    }
    };
}

inline void from_json(const nlohmann::json& j, TransferRequestMessage& m) {
    if (!j.contains("relative_path"))
        throw std::runtime_error("TransferRequestMessage: missing 'relative_path'");
    if (!j.contains("size_bytes"))
        throw std::runtime_error("TransferRequestMessage: missing 'size_bytes'");
    if (!j.contains("sha256"))
        throw std::runtime_error("TransferRequestMessage: missing 'sha256'");
    if (!j.contains("chunk_size"))
        throw std::runtime_error("TransferRequestMessage: missing 'chunk_size'");
    j.at("relative_path").get_to(m.relative_path);
    j.at("size_bytes").get_to(m.size_bytes);
    j.at("sha256").get_to(m.sha256);
    j.at("chunk_size").get_to(m.chunk_size);
}

// ─────────────────────────────────────────────────────────
//  TransferResponse — hub accepts/rejects transfer
// ─────────────────────────────────────────────────────────

struct TransferResponseMessage {
    bool        accepted;
    std::string transfer_id;    // UUID assigned by hub
    std::string error_message;
};

inline void to_json(nlohmann::json& j, const TransferResponseMessage& m) {
    j = {
        { "accepted",      m.accepted      },
        { "transfer_id",   m.transfer_id   },
        { "error_message", m.error_message }
    };
}

inline void from_json(const nlohmann::json& j, TransferResponseMessage& m) {
    if (!j.contains("accepted"))
        throw std::runtime_error("TransferResponseMessage: missing 'accepted'");
    j.at("accepted").get_to(m.accepted);
    j.at("transfer_id").get_to(m.transfer_id);
    j.at("error_message").get_to(m.error_message);
}

// ─────────────────────────────────────────────────────────
//  TransferComplete — hub confirms transfer result
// ─────────────────────────────────────────────────────────

struct TransferCompleteMessage {
    std::string transfer_id;
    bool        success;
    std::string file_id;        // assigned file ID in storage index
    std::string error_message;
};

inline void to_json(nlohmann::json& j, const TransferCompleteMessage& m) {
    j = {
        { "transfer_id",   m.transfer_id   },
        { "success",       m.success       },
        { "file_id",       m.file_id       },
        { "error_message", m.error_message }
    };
}

inline void from_json(const nlohmann::json& j, TransferCompleteMessage& m) {
    if (!j.contains("transfer_id"))
        throw std::runtime_error("TransferCompleteMessage: missing 'transfer_id'");
    if (!j.contains("success"))
        throw std::runtime_error("TransferCompleteMessage: missing 'success'");
    j.at("transfer_id").get_to(m.transfer_id);
    j.at("success").get_to(m.success);
    j.at("file_id").get_to(m.file_id);
    j.at("error_message").get_to(m.error_message);
}

// ─────────────────────────────────────────────────────────
//  ErrorMessage — generic error response
// ─────────────────────────────────────────────────────────

struct ErrorMessage {
    int         code;
    std::string message;
};

inline void to_json(nlohmann::json& j, const ErrorMessage& m) {
    j = { { "code", m.code }, { "message", m.message } };
}

inline void from_json(const nlohmann::json& j, ErrorMessage& m) {
    if (!j.contains("code"))
        throw std::runtime_error("ErrorMessage: missing 'code'");
    if (!j.contains("message"))
        throw std::runtime_error("ErrorMessage: missing 'message'");
    j.at("code").get_to(m.code);
    j.at("message").get_to(m.message);
}

} // namespace GRIM
