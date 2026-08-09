#pragma once
#include <uwebsockets/App.h>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

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
    std::mutex lifecycleMutex_;
    std::condition_variable startupCv_;
    uWS::Loop* serverLoop_ = nullptr;
    uWS::App* serverApp_ = nullptr;
    bool startupComplete_ = false;
    bool listenSucceeded_ = false;
    bool stopRequested_ = false;
};

} // namespace GRIM
