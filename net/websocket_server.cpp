#include "websocket_server.hpp"
#include <uwebsockets/App.h>
#include <iostream>
#include <string>
#include <mutex>

namespace GRIM {

struct PerSocketData {
    // You can store per-connection data here if needed
    std::string sessionId;
};

void WebSocketServer::start(int port) {
    if (running) {
        std::cerr << "WebSocket server is already running" << std::endl;
        return;
    }

    running = true;

    serverThread = std::thread([this, port]() {
        try {
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
                        std::cout << "WebSocket connection opened" << std::endl;
                    },
                    
                    .message = [](auto *ws, std::string_view message, uWS::OpCode opCode) {
                        std::cout << "Received message: " << message << std::endl;
                        
                        // Echo the message back to the client
                        ws->send(message, opCode);
                    },
                    
                    .drain = [](auto *ws) {
                        // Check ws->getBufferedAmount() here
                    },
                    
                    .ping = [](auto *ws, std::string_view) {
                        // Handle ping if needed
                    },
                    
                    .pong = [](auto *ws, std::string_view) {
                        // Handle pong if needed
                    },
                    
                    .close = [](auto *ws, int code, std::string_view message) {
                        std::cout << "WebSocket connection closed with code: " << code << std::endl;
                    }
                })
                .listen(port, [this, port](auto *listen_socket) {
                    if (listen_socket) {
                        std::cout << "WebSocket server listening on port " << port << std::endl;
                    } else {
                        std::cerr << "Failed to listen on port " << port << std::endl;
                        running = false;
                    }
                })
                .run();
                
            std::cout << "WebSocket server stopped" << std::endl;
        } catch (const std::exception &e) {
            std::cerr << "WebSocket server error: " << e.what() << std::endl;
            running = false;
        }
    });
}

void WebSocketServer::stop() {
    if (!running) {
        return;
    }

    running = false;

    // Note: uWebSockets doesn't have a built-in graceful shutdown mechanism
    // You may need to implement a custom solution if graceful shutdown is required
    
    if (serverThread.joinable()) {
        serverThread.detach(); // Detach instead of join since uWS::App().run() blocks
    }

    std::cout << "WebSocket server shutdown initiated" << std::endl;
}

} // namespace GRIM
