#include "wake.hpp"
#include "logger.hpp"
#include "popup_ui.hpp"

#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>

namespace Wake {

std::atomic<bool> g_awake{false};
static std::queue<WakeEvent> eventQueue;
static std::mutex queueMutex;
static std::condition_variable cv;
static std::thread dispatcherThread;
static bool running = true;

// Push event into queue (called by stimulants)
void pushEvent(const WakeEvent& ev) {
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        eventQueue.push(ev);
    }
    cv.notify_one();
}

// Dispatcher
static void dispatcherLoop() {
    while (running) {
        WakeEvent ev;
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            cv.wait(lock, [] { return !eventQueue.empty() || !running; });
            if (!running) break;
            ev = eventQueue.front();
            eventQueue.pop();
        }

        triggerWake(ev);
    }
}

void triggerWake(const WakeEvent& ev) {
    g_awake = true;
    LOG_DEBUG("Wake", "Triggered by " + ev.source);
    notifyPopupActivity();
}

void init() {
    running = true;
    dispatcherThread = std::thread(dispatcherLoop);
    LOG_PHASE("Wake system initialized", true);
}

void shutdown() {
    running = false;
    cv.notify_all();
    if (dispatcherThread.joinable())
        dispatcherThread.join();
    LOG_PHASE("Wake system shutdown", true);
}

} // namespace Wake
