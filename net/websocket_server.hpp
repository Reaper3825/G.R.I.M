#pragma once
#include <thread>
#include <atomic>

namespace GRIM {
    class WebSocketServer {
    public:
        void start(int port = 8080);
        void stop();
        bool isRunning() const { return running; }

    private:
        std::thread serverThread;
        std::atomic<bool> running{false};
    };
}
