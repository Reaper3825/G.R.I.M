#include "device_comm_server.hpp"
#include "control/devices/protocol/device_protocol.hpp"
#include "control/devices/protocol/device_messages.hpp"
#include "logger.hpp"

#include <uwebsockets/App.h>
#include <nlohmann/json.hpp>
#include <chrono>
#include <limits>
#include <sstream>

namespace GRIM {

static constexpr const char* TAG = "DeviceComm";

template <typename TWebSocket>
void sendEnvelope(TWebSocket* ws, MessageType type, const nlohmann::json& payload) {
    const std::string serialized = makeEnvelope(type, payload).dump();
    ws->send(serialized, uWS::OpCode::TEXT);
}

// Per-connection data stored by uWebSockets
struct DevicePerSocketData {
    std::string session_id;
    std::string device_id;   // empty until registered
    bool        authenticated = false;
};

// ─── Construction ────────────────────────────────────────

DeviceCommServer::DeviceCommServer(const Config& config)
    : config_(config)
    , registry_(config.registry_dir)
    , sessions_(config.heartbeat_timeout_sec)
    , storage_(config.storage_root)
    , transfer_(storage_, config.max_chunk_size_bytes, config.transfer_timeout_sec)
{
    if (config.storage_root.empty())
        throw std::runtime_error("DeviceCommServer: storage_root is required");
    if (config.registry_dir.empty())
        throw std::runtime_error("DeviceCommServer: registry_dir is required");
}

DeviceCommServer::~DeviceCommServer() {
    stop();
}

// ─── Snapshot query ──────────────────────────────────────

std::vector<DeviceSnapshot> DeviceCommServer::listDeviceSnapshots() const {
    auto records = registry_.listAll();
    std::vector<DeviceSnapshot> snapshots;
    snapshots.reserve(records.size());

    for (const auto& rec : records) {
        const DeviceSession* session = sessions_.findByDeviceId(rec.device_id);
        if (session) {
            snapshots.push_back(DeviceSnapshot::fromJoin(rec, *session));
        } else {
            snapshots.push_back(DeviceSnapshot::fromRecord(rec));
        }
    }
    return snapshots;
}

// ─── Server lifecycle ────────────────────────────────────

bool DeviceCommServer::start() {
    if (running_) {
        LOG_ERROR(TAG, "Server already running");
        return false;
    }

    const uint32_t idle_timeout_wide = config_.heartbeat_timeout_sec * 2u;
    if (idle_timeout_wide > std::numeric_limits<unsigned short>::max()) {
        throw std::runtime_error(
            "DeviceCommServer: heartbeat_timeout_sec*2 exceeds uWebSockets idleTimeout limit");
    }
    const unsigned short idle_timeout = static_cast<unsigned short>(idle_timeout_wide);

    running_ = true;
    uint16_t port = config_.port;

    server_thread_ = std::thread([this, port, idle_timeout]() {
        LOG_DEBUG(TAG, "Starting device comm server on port " + std::to_string(port));

        try {
            uWS::App()
            .ws<DevicePerSocketData>("/*", {
                .compression = uWS::DISABLED,
                .maxPayloadLength = config_.max_chunk_size_bytes + 1024, // chunk + overhead
                .idleTimeout = idle_timeout,
                .maxBackpressure = 4 * 1024 * 1024,

                .upgrade = nullptr,

                // ─── Open ────────────────────────────
                .open = [this](auto* ws) {
                    auto* data = ws->getUserData();

                    static int counter = 0;
                    std::stringstream ss;
                    ss << "dev_" << ++counter << "_"
                       << std::chrono::system_clock::now().time_since_epoch().count();
                    data->session_id = ss.str();

                    LOG_DEBUG(TAG, "Connection opened: " + data->session_id);
                },

                // ─── Text message (JSON protocol) ────
                .message = [this](auto* ws, std::string_view message, uWS::OpCode opCode) {
                    auto* data = ws->getUserData();

                    if (opCode == uWS::OpCode::BINARY) {
                        // Binary frame → chunked transfer
                        if (!data->authenticated || data->device_id.empty()) {
                            ws->close();
                            return;
                        }

                        // Route to transfer manager
                        // The device embeds transfer_id in binary header — for now
                        // we use the device's single active transfer
                        auto transfers = transfer_.listActive();
                        for (const auto& t : transfers) {
                            if (t.device_id == data->device_id &&
                                t.direction == TransferDirection::Upload) {
                                bool done = transfer_.processChunk(
                                    t.transfer_id,
                                    reinterpret_cast<const uint8_t*>(message.data()),
                                    message.size());
                                if (done) {
                                    nlohmann::json payload;
                                    payload["transfer_id"] = t.transfer_id;
                                    payload["success"] = true;
                                    sendEnvelope(ws, MessageType::TransferComplete, payload);
                                }
                                return;
                            }
                        }

                        LOG_ERROR(TAG, "Binary frame from " + data->device_id +
                                       " but no active upload transfer");
                        return;
                    }

                    // Text frame → JSON protocol
                    nlohmann::json envelope;
                    try {
                        envelope = nlohmann::json::parse(message);
                    } catch (const nlohmann::json::parse_error& e) {
                        nlohmann::json err;
                        err["code"] = "PARSE_ERROR";
                        err["message"] = e.what();
                        sendEnvelope(ws, MessageType::Error, err);
                        return;
                    }

                    MessageType type = parseMessageType(envelope);

                    switch (type) {
                    case MessageType::Register: {
                        auto msg = envelope["payload"].get<RegisterMessage>();

                        DeviceRecord rec;
                        try {
                            rec = registry_.completePairing(
                                msg.pairing_code,
                                msg.device_name,
                                msg.device_type,
                                msg.platform);
                        } catch (const std::exception& e) {
                            nlohmann::json err;
                            err["code"] = "PAIRING_FAILED";
                            err["message"] = e.what();
                            sendEnvelope(ws, MessageType::Error, err);
                            return;
                        }

                        // Create session
                        DeviceSession session;
                        session.session_id = data->session_id;
                        session.device_id = rec.device_id;
                        session.ws_handle = static_cast<void*>(ws);
                        session.connected_at = std::chrono::steady_clock::now();
                        session.last_heartbeat_at = session.connected_at;
                        session.available_capabilities = msg.capabilities;

                        sessions_.addSession(std::move(session));

                        data->device_id     = rec.device_id;
                        data->authenticated = true;

                        // Respond with PairResponse
                        PairResponseMessage resp;
                        resp.success = true;
                        resp.device_id = rec.device_id;
                        resp.error_message.clear();

                        nlohmann::json resp_json = resp;
                        sendEnvelope(ws, MessageType::PairResponse, resp_json);

                        LOG_DEBUG(TAG, "Device registered: " + rec.device_name +
                                       " (id=" + rec.device_id + ")");
                        break;
                    }

                    case MessageType::Heartbeat: {
                        if (!data->authenticated) { ws->close(); return; }
                        auto msg = envelope["payload"].get<HeartbeatMessage>();
                        sessions_.updateHeartbeat(data->device_id, msg.available_capabilities);

                        HeartbeatAckMessage ack;
                        ack.device_id = data->device_id;
                        nlohmann::json ack_json = ack;
                        sendEnvelope(ws, MessageType::HeartbeatAck, ack_json);
                        break;
                    }

                    case MessageType::ListFiles: {
                        if (!data->authenticated) { ws->close(); return; }
                        auto msg = envelope["payload"].get<ListFilesMessage>();

                        auto listing = storage_.index().listDirectory(msg.directory_path);
                        nlohmann::json entries = nlohmann::json::array();
                        for (const auto& entry : listing) {
                            nlohmann::json e;
                            e["name"]     = entry.name;
                            e["is_dir"]   = entry.is_directory;
                            e["size"]     = entry.size_bytes;
                            e["modified"] = entry.modified_at;
                            entries.push_back(std::move(e));
                        }

                        nlohmann::json resp;
                        resp["directory"] = msg.directory_path;
                        resp["entries"]   = entries;
                        sendEnvelope(ws, MessageType::FileMetadata, resp);
                        break;
                    }

                    case MessageType::TransferRequest: {
                        if (!data->authenticated) { ws->close(); return; }
                        auto msg = envelope["payload"].get<TransferRequestMessage>();

                        try {
                            std::string tid = transfer_.beginUpload(
                                data->device_id,
                                msg.relative_path,
                                msg.size_bytes,
                                msg.sha256,
                                msg.chunk_size);

                            TransferResponseMessage resp;
                            resp.transfer_id = tid;
                            resp.accepted    = true;
                            resp.error_message.clear();
                            nlohmann::json resp_json = resp;
                            sendEnvelope(ws, MessageType::TransferResponse, resp_json);
                        } catch (const std::exception& e) {
                            TransferResponseMessage resp;
                            resp.accepted = false;
                            resp.transfer_id.clear();
                            resp.error_message = e.what();
                            nlohmann::json resp_json = resp;
                            sendEnvelope(ws, MessageType::TransferResponse, resp_json);
                        }
                        break;
                    }

                    default: {
                        nlohmann::json err;
                        err["code"] = "UNKNOWN_MESSAGE_TYPE";
                        err["message"] = "Unhandled message type";
                        sendEnvelope(ws, MessageType::Error, err);
                        break;
                    }
                    } // switch
                },

                // ─── Close ───────────────────────────
                .close = [this](auto* ws, int code, std::string_view message) {
                    auto* data = ws->getUserData();
                    if (!data->device_id.empty()) {
                        sessions_.removeByDeviceId(data->device_id);
                        LOG_DEBUG(TAG, "Device disconnected: " + data->device_id +
                                       " (code=" + std::to_string(code) + ")");
                    }
                }
            })
            .listen(port, [port](auto* listen_socket) {
                if (listen_socket) {
                    LOG_DEBUG(TAG, "Device comm server listening on port " + std::to_string(port));
                } else {
                    LOG_ERROR(TAG, "Failed to listen on port " + std::to_string(port));
                }
            })
            .run();

        } catch (const std::exception& e) {
            LOG_ERROR(TAG, "Device comm server error: " + std::string(e.what()));
        }

        running_ = false;
    });

    return true;
}

void DeviceCommServer::stop() {
    if (!running_) return;
    running_ = false;

    // uWebSockets doesn't have a built-in stop from outside the event loop.
    // The thread will exit when the event loop stops (e.g., on process shutdown).
    if (server_thread_.joinable()) {
        server_thread_.detach();
    }
    LOG_DEBUG(TAG, "Device comm server stopped");
}

} // namespace GRIM
