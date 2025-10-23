#include "websocket_server.hpp"
#include "logger.hpp"
#include "commands_core.hpp"  // ✅ Added for handleCommand
#include <uwebsockets/App.h>
#include <string>
#include <mutex>
#include <chrono>
#include <sstream>

namespace GRIM {

struct PerSocketData {
    std::string sessionId;
    std::string remoteAddress;
    std::chrono::steady_clock::time_point connectedAt;
};

bool WebSocketServer::start(uint16_t port) {
    LOG_DEBUG("WebSocket", "=== WebSocket Server Startup Sequence ===");
    LOG_DEBUG("WebSocket", "Requested port: " + std::to_string(port));
    
    if (running) {
        LOG_ERROR("WebSocket", "Server is already running - ignoring start request");
        return false;
    }

    running = true;

    LOG_DEBUG("WebSocket", "Creating server thread...");
    serverThread = std::thread([this, port]() {
        LOG_DEBUG("WebSocket", "Server thread started (thread ID: " + 
                  std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id())) + ")");

        try {
            LOG_DEBUG("WebSocket", "Initializing uWebSockets App...");

            uWS::App()
             .ws<PerSocketData>("/*", {
                    /* Settings */
                    .compression = uWS::SHARED_COMPRESSOR,
                    .maxPayloadLength = 16 * 1024 * 1024,
                    .idleTimeout = 120,
                    .maxBackpressure = 1 * 1024 * 1024,

                    /* Handlers */
                    .upgrade = nullptr,

                    .open = [](auto *ws) {
                        auto *data = ws->getUserData();
                        data->connectedAt = std::chrono::steady_clock::now();

                        std::string_view address = ws->getRemoteAddressAsText();
                        data->remoteAddress = std::string(address);

                        static int sessionCounter = 0;
                        std::stringstream ss;
                        ss << "ws_" << ++sessionCounter << "_"
                           << std::chrono::system_clock::now().time_since_epoch().count();
                        data->sessionId = ss.str();

                        LOG_DEBUG("WebSocket", "Connection opened");
                        LOG_DEBUG("WebSocket", "  Session ID: " + data->sessionId);
                        LOG_DEBUG("WebSocket", "  Remote addr: " + data->remoteAddress);
                    },

                    // ✅ Modified: message now goes through handleCommand()
                    .message = [](auto *ws, std::string_view message, uWS::OpCode opCode) {
    auto *data = ws->getUserData();

    std::string opCodeStr = (opCode == uWS::OpCode::TEXT) ? "TEXT" :
                            (opCode == uWS::OpCode::BINARY) ? "BINARY" : "OTHER";

    LOG_DEBUG("WebSocket", "Message received");
    LOG_DEBUG("WebSocket", "  Session: " + data->sessionId);
    LOG_DEBUG("WebSocket", "  OpCode: " + opCodeStr);
    LOG_DEBUG("WebSocket", "  Length: " + std::to_string(message.length()) + " bytes");

    std::string preview = std::string(message.substr(0, std::min<size_t>(200, message.length())));
    if (message.length() > 200) preview += "...";
    LOG_DEBUG("WebSocket", "  Content: " + preview);

    try {
        std::string input = std::string(message);

        // Capture what handleCommand prints
        std::ostringstream capture;
        std::streambuf* oldBuf = std::cout.rdbuf(capture.rdbuf());

        handleCommand(input);  // your existing void version

        // Restore output
        std::cout.rdbuf(oldBuf);

        std::string output = capture.str();
        if (output.empty())
            output = " "; // send at least one character to keep the socket open

        // ✅ Send only GRIM’s response text — nothing else
        ws->send(output, uWS::OpCode::TEXT);

        LOG_DEBUG("WebSocket", "Raw GRIM response sent to client");

    } catch (const std::exception &e) {
        LOG_ERROR("WebSocket", std::string("Exception processing command: ") + e.what());
        ws->send("Internal error.", uWS::OpCode::TEXT);
    }
}
,

                    .drain = [](auto *ws) {
                        auto *data = ws->getUserData();
                        unsigned int buffered = ws->getBufferedAmount();
                        if (buffered > 0) {
                            LOG_DEBUG("WebSocket", "Drain event for " + data->sessionId +
                                                   " (" + std::to_string(buffered) + " bytes buffered)");
                        }
                    },

                    .ping = [](auto *ws, std::string_view) {
                        auto *data = ws->getUserData();
                        LOG_TRACE("WebSocket", "Ping received from " + data->sessionId);
                    },

                    .pong = [](auto *ws, std::string_view) {
                        auto *data = ws->getUserData();
                        LOG_TRACE("WebSocket", "Pong received from " + data->sessionId);
                    },

                    .close = [](auto *ws, int code, std::string_view message) {
                        auto *data = ws->getUserData();

                        auto now = std::chrono::steady_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::seconds>(
                                            now - data->connectedAt).count();

                        LOG_DEBUG("WebSocket", "Connection closed");
                        LOG_DEBUG("WebSocket", "  Session: " + data->sessionId);
                        LOG_DEBUG("WebSocket", "  Code: " + std::to_string(code));
                        LOG_DEBUG("WebSocket", "  Reason: " + std::string(message));
                        LOG_DEBUG("WebSocket", "  Duration: " + std::to_string(duration) + "s");
                    }
                })
                .listen(port, [this, port](auto *listen_socket) {
                    if (listen_socket) {
                        LOG_PHASE("WebSocket server started", true);
                        LOG_DEBUG("WebSocket", "Listening on port " + std::to_string(port));
                        LOG_DEBUG("WebSocket", "Endpoint: ws://localhost:" + std::to_string(port));
                    } else {
                        LOG_ERROR("WebSocket", "Failed to bind to port " + std::to_string(port));
                        LOG_ERROR("WebSocket", "Port may be in use or require admin privileges");
                        running = false;
                    }
                })
                .run();

            LOG_DEBUG("WebSocket", "Server event loop exited");
            LOG_PHASE("WebSocket server stopped", true);

        } catch (const std::exception &e) {
            LOG_ERROR("WebSocket", "Exception in server thread: " + std::string(e.what()));
            running = false;
        } catch (...) {
            LOG_ERROR("WebSocket", "Unknown exception in server thread");
            running = false;
        }

        LOG_DEBUG("WebSocket", "Server thread exiting");
    });

    serverThread.detach();
    LOG_DEBUG("WebSocket", "Server thread detached; running asynchronously");
    return true;
}

void WebSocketServer::stop() {
    LOG_DEBUG("WebSocket", "=== WebSocket Server Shutdown Sequence ===");

    if (!running) {
        LOG_DEBUG("WebSocket", "Server not running - nothing to stop");
        return;
    }

    running = false;

    if (serverThread.joinable()) {
        LOG_DEBUG("WebSocket", "Detaching server thread (uWS::App().run() blocks indefinitely)");
        serverThread.detach();
    }

    LOG_PHASE("WebSocket server shutdown", true);
    LOG_DEBUG("WebSocket", "uWebSockets event loop may still be running");
    LOG_DEBUG("WebSocket", "=== WebSocket Server Shutdown Complete ===");
}

} // namespace GRIM
