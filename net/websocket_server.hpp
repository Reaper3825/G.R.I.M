#pragma once
#include <uwebsockets/App.h>
#include <thread>
#include <atomic>
#include <string>

namespace GRIM {

class WebSocketServer {
public:
    WebSocketServer() : running(false) {}
    ~WebSocketServer() { stop(); }

    bool start(uint16_t port = 8080);
    void stop();

private:
    std::atomic<bool> running;
    std::thread serverThread;
};

} // namespace GRIM
