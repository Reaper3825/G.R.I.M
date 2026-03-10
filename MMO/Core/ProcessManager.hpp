// Multi-Model Orchestration (MMO) - Model-Keyed Process Manager
//
// Manages one server process per model. Each model gets its own
// process handle, port, mutex identity, and health-check lifecycle.
//
// This replaces the single-instance GRIMTextServerManager for MMO
// use cases where multiple models may need concurrent server processes.
//
// Design rules:
//   - One process per model ID, one port per process
//   - Two models MUST NOT share the same port or mutex identity
//   - Start/stop are model-keyed, not global
//   - Health checks are per-process
//   - Thread-safe under internal mutex
//======================================================//
#pragma once

#include "../Shared/MMD.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>

#ifdef _WIN32
#include <windows.h>
#endif

namespace GRIM::MMO {

// =========================================================
// Per-model process record (internal to ProcessManager)
// =========================================================
struct ProcessSlot {
    std::string model_id;
    std::string executable_path;
    std::string url;               // e.g. "http://127.0.0.1:11435"
    uint16_t    port          = 0;
    bool        running       = false;

#ifdef _WIN32
    PROCESS_INFORMATION process_info{};
    HANDLE              h_process = nullptr;
    HANDLE              h_mutex   = nullptr;
#endif
};

// =========================================================
// ProcessManager — model-keyed server process lifecycle
//
// Usage:
//   ProcessManager pm;
//   bool ok = pm.start(modelInfo);    // spawn process
//   pm.checkHealth("model-a", 5000);  // check if alive
//   pm.stop("model-a");              // tear down
//   pm.stopAll();                    // shutdown
// =========================================================
class ProcessManager {
public:
    ProcessManager();
    ~ProcessManager();

    // Start a server process for the given model.
    // Returns true if the server is running and healthy after startup.
    // Throws std::runtime_error on port/identity collision with another model.
    bool start(const ModelInfo& model);

    // Stop a specific model's server process. No-op if not running.
    void stop(const std::string& model_id);

    // Stop all managed server processes.
    void stopAll();

    // Check if a model's server process is still alive.
    bool isRunning(const std::string& model_id) const;

    // HTTP health check for a model's server.
    bool checkHealth(const std::string& model_id, int timeout_ms = 5000) const;

    // Get the URL for a running model's server (empty string if not found).
    std::string getUrl(const std::string& model_id) const;

    // How many processes are currently running.
    int runningCount() const;

private:
    // Launch the OS process for a GrimTextServer model.
    bool launchGrimTextServer(ProcessSlot& slot, const ModelInfo& model);

    // Terminate the OS process in a slot.
    void terminateSlot(ProcessSlot& slot);

    // Validate that no other model already claims the same port.
    void validatePortUniqueness(const std::string& model_id, uint16_t port) const;

    mutable std::mutex mutex_;
    std::unordered_map<std::string, ProcessSlot> slots_;
};

} // namespace GRIM::MMO
