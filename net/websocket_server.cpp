#include "websocket_server.hpp"
#include "commands/commands_core.hpp"
#include "logger.hpp"
#include "httplib.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace GRIM;

void WebSocketServer::start(int port) {
    if (running) return;
    running = true;

    serverThread = std::thread([this, port]() {
        httplib::Server svr;

        // Simple WebSocket handler
        svr.set_logger([](const auto &req, const auto &res) {
            LOG_DEBUG("[HTTP] %s -> %d", req.path.c_str() + res.status);
        });

        svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("GRIM WebSocket online", "text/plain");
        });

        svr.set_ws_message_handler([&](const httplib::Request &req,
                                       std::shared_ptr<httplib::WebSocket> ws,
                                       const std::string &message,
                                       bool is_text) {
            try {
                LOG_DEBUG("[WS] Received: %s", message.c_str());

                // Handle command
                CommandResult result = handleCommand(message);
                json reply = {
                    {"type", "response"},
                    {"success", result.success},
                    {"message", result.message},
                    {"errorCode", result.errorCode},
                    {"category", result.category}
                };
                ws->send(reply.dump());
            } catch (const std::exception &e) {
                json err = {{"type", "error"}, {"message", e.what()}};
                ws->send(err.dump());
            }
        });

        LOG_INFO("[WS] Listening on port %d", port);
        svr.listen("0.0.0.0", port);
        running = false;
    });
}

void WebSocketServer::stop() {
    running = false;
    // cpp-httplib doesn't expose direct stop from outside, so you'd track server and call svr.stop()
    if (serverThread.joinable()) serverThread.detach();
}
