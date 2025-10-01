#include "wake.hpp"
#include "logger.hpp"

#include <atomic>

namespace Wake {
    std::atomic<bool> g_awake{false};

    void init() {
        logDebug("Wake", "Init called");
        g_awake = true;
    }

    void shutdown() {
        logDebug("Wake", "Shutdown called");
        g_awake = false;
    }
}
