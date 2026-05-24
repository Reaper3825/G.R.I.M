// Multi-Model Orchestration (MMO) - Model-Keyed Process Manager
// Implementation.
//======================================================//

#include "ProcessManager.hpp"
#include "../../logger.hpp"
#include "../../resources.hpp"

#include <cpr/cpr.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <thread>

#ifndef _WIN32
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;
#endif

#include "core/grim_platform.h"

namespace fs = std::filesystem;

namespace GRIM::MMO {

static bool hasSuffix(const std::string& value, const char* suffix) {
    const size_t suffix_len = std::char_traits<char>::length(suffix);
    return value.size() >= suffix_len &&
           value.compare(value.size() - suffix_len, suffix_len, suffix) == 0;
}

// =========================================================
// Helpers
// =========================================================

// Extract port from a URL like "http://127.0.0.1:11435"
static uint16_t extractPort(const std::string& url) {
    // Find last colon (after the host)
    auto scheme_end = url.find("://");
    size_t search_start = (scheme_end != std::string::npos) ? scheme_end + 3 : 0;
    auto colon = url.rfind(':');
    if (colon == std::string::npos || colon < search_start) {
        return 0;
    }
    try {
        return static_cast<uint16_t>(std::stoi(url.substr(colon + 1)));
    } catch (...) {
        return 0;
    }
}

// =========================================================
// ProcessManager
// =========================================================

ProcessManager::ProcessManager() = default;

ProcessManager::~ProcessManager() {
    stopAll();
}

bool ProcessManager::start(const ModelInfo& model) {
    std::lock_guard<std::mutex> lock(mutex_);

    // If already running, just health-check
    auto it = slots_.find(model.id);
    if (it != slots_.end() && it->second.running) {
        LOG_DEBUG("ProcessManager", "Model '" + model.id + "' already has a running process");
        return true;
    }

    uint16_t port = extractPort(model.url);

    // Validate port uniqueness across all other running models
    validatePortUniqueness(model.id, port);

    // Build or update slot
    ProcessSlot& slot = slots_[model.id];
    slot.model_id = model.id;
    slot.url      = model.url;
    slot.port     = port;

    if (model.backend_type == BackendType::GrimTextServer) {
        return launchGrimTextServer(slot, model);
    }

    // Non-GrimTextServer backends (Ollama, LlamaCpp, External) are
    // managed externally — we just mark them as "running = true" and
    // rely on health checks.
    slot.running = true;
    return true;
}

void ProcessManager::stop(const std::string& model_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = slots_.find(model_id);
    if (it == slots_.end()) return;

    terminateSlot(it->second);
}

void ProcessManager::stopAll() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& [id, slot] : slots_) {
        terminateSlot(slot);
    }
    slots_.clear();
}

bool ProcessManager::isRunning(const std::string& model_id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = slots_.find(model_id);
    if (it == slots_.end()) return false;

#ifdef _WIN32
    const auto& slot = it->second;
    if (!slot.running || !slot.h_process) return false;

    DWORD exit_code = 0;
    if (GetExitCodeProcess(slot.h_process, &exit_code)) {
        return exit_code == STILL_ACTIVE;
    }
#else
    const auto& slot = it->second;
    if (!slot.running || slot.pid <= 0) return false;

    // kill(pid, 0) checks if process exists without sending a signal
    if (kill(slot.pid, 0) != 0) return false;
#endif
    return it->second.running;
}

bool ProcessManager::checkHealth(const std::string& model_id, int timeout_ms) const {
    std::string url;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = slots_.find(model_id);
        if (it == slots_.end() || !it->second.running) return false;
        url = it->second.url;
    }

    try {
        auto resp = cpr::Get(cpr::Url{url}, cpr::Timeout{timeout_ms});
        return resp.status_code != 0;
    } catch (...) {
        return false;
    }
}

std::string ProcessManager::getUrl(const std::string& model_id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = slots_.find(model_id);
    if (it == slots_.end()) return {};
    return it->second.url;
}

int ProcessManager::runningCount() const {
    std::lock_guard<std::mutex> lock(mutex_);

    int count = 0;
    for (const auto& [id, slot] : slots_) {
        if (slot.running) ++count;
    }
    return count;
}

// =========================================================
// Port uniqueness validation
// =========================================================

void ProcessManager::validatePortUniqueness(const std::string& model_id, uint16_t port) const {
    if (port == 0) return; // no port to check

    for (const auto& [id, slot] : slots_) {
        if (id == model_id) continue; // same model — OK
        if (slot.running && slot.port == port) {
            throw std::runtime_error(
                "ProcessManager: port " + std::to_string(port) +
                " already claimed by model '" + id +
                "' — model '" + model_id + "' cannot start on the same port");
        }
    }
}

// =========================================================
// GrimTextServer process launch
// =========================================================

bool ProcessManager::launchGrimTextServer(ProcessSlot& slot, const ModelInfo& model) {
#ifdef _WIN32
    // Mutex name per model to prevent duplicate instances of the same model
    std::string mutex_name = "Global\\GRIMTextServer_" + model.id;
    slot.h_mutex = CreateMutexA(nullptr, FALSE, mutex_name.c_str());
    if (GetLastError() == ERROR_ALREADY_EXISTS) {
        LOG_DEBUG("ProcessManager", "Another instance of model '" + model.id + "' already running");
        // Check if the existing instance is healthy
        if (checkHealth(model.id, 2000)) {
            LOG_DEBUG("ProcessManager", "Existing server for '" + model.id + "' is healthy, reusing");
            slot.running = true;
            return true;
        }
        LOG_DEBUG("ProcessManager", "Existing server for '" + model.id + "' not responding");
        // Release the stale mutex and try again
        CloseHandle(slot.h_mutex);
        slot.h_mutex = nullptr;
        // Note: we do NOT taskkill all grim_text_server.exe — that would kill
        // other models' processes. The stale process should time out or be
        // cleaned up by OS.
    }

    const auto resolveFromGrimRoot = [](const std::string& rawPath) {
        fs::path path(rawPath);
        if (path.is_absolute()) {
            return path;
        }
        return fs::path(getGrimRootDir()) / path;
    };

    const nlohmann::json& grimTextPaths = aiConfig.at("paths").at("grim_text");

    // Determine executable path — use model-specific path if provided, else default
    fs::path server_exe;
    if (!model.model_path.empty() && fs::exists(model.model_path)) {
        // If model_path points to the exe itself (unlikely but supported)
        if (hasSuffix(model.model_path, ".exe")) {
            server_exe = fs::absolute(model.model_path);
        } else {
            // Default exe location
            server_exe = fs::absolute("resources/models/GRIM-text/training/build/Release/grim_text_server.exe");
        }
    } else {
        server_exe = resolveFromGrimRoot("resources/models/GRIM-text/training/build/Release/grim_text_server.exe");
    }

    fs::path vocab_path = resolveFromGrimRoot(grimTextPaths.at("vocab").get<std::string>());
    fs::path model_weights = resolveFromGrimRoot(grimTextPaths.at("model").get<std::string>());

    if (!fs::exists(server_exe)) {
        LOG_ERROR("ProcessManager", "Server executable not found: " + server_exe.string());
        return false;
    }
    if (!fs::exists(vocab_path)) {
        LOG_ERROR("ProcessManager", "Vocabulary file not found: " + vocab_path.string());
        return false;
    }
    if (!fs::exists(model_weights)) {
        LOG_ERROR("ProcessManager", "Model file not found: " + model_weights.string());
        return false;
    }

    slot.executable_path = server_exe.string();

    // Build command line
    std::string port_str = std::to_string(slot.port > 0 ? slot.port : 11435);
    std::string cmd_line = "\"" + server_exe.string() + "\" \"" +
                           vocab_path.string() + "\" \"" +
                           model_weights.string() + "\" " + port_str;

    LOG_DEBUG("ProcessManager", "Starting model '" + model.id + "': " + cmd_line);

    STARTUPINFOA si{};
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;

    ZeroMemory(&slot.process_info, sizeof(slot.process_info));

    std::vector<char> mutable_cmd(cmd_line.begin(), cmd_line.end());
    mutable_cmd.push_back('\0');

    BOOL ok = CreateProcessA(
        nullptr,
        mutable_cmd.data(),
        nullptr, nullptr,
        FALSE,
        CREATE_NO_WINDOW,
        nullptr,
        server_exe.parent_path().string().c_str(),
        &si,
        &slot.process_info
    );

    if (!ok) {
        DWORD err = GetLastError();
        LOG_ERROR("ProcessManager", "Failed to launch model '" + model.id +
                  "': CreateProcess error " + std::to_string(err));
        return false;
    }

    slot.h_process = slot.process_info.hProcess;
    slot.running = true;

    LOG_DEBUG("ProcessManager", "Model '" + model.id + "' process started (PID: " +
              std::to_string(slot.process_info.dwProcessId) + ")");

    // Poll for health
    const int max_wait_ms = 30000;
    const int poll_ms = 500;
    int elapsed = 0;

    // Release lock during poll (we already set running=true)
    // Note: caller holds our mutex, but checkHealth acquires it too
    // so we need to call the URL directly here
    while (elapsed < max_wait_ms) {
        try {
            auto resp = cpr::Get(cpr::Url{slot.url}, cpr::Timeout{1000});
            if (resp.status_code != 0) {
                LOG_DEBUG("ProcessManager", "Model '" + model.id + "' server ready at " +
                          slot.url + " (took " + std::to_string(elapsed) + "ms)");
                return true;
            }
        } catch (...) {}

        // Check if process crashed
        DWORD exit_code = 0;
        if (GetExitCodeProcess(slot.h_process, &exit_code) && exit_code != STILL_ACTIVE) {
            LOG_ERROR("ProcessManager", "Model '" + model.id +
                      "' server terminated unexpectedly (exit code: " +
                      std::to_string(exit_code) + ")");
            slot.running = false;
            return false;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(poll_ms));
        elapsed += poll_ms;
    }

    LOG_ERROR("ProcessManager", "Model '" + model.id +
              "' server failed to respond within " + std::to_string(max_wait_ms) + "ms");
    terminateSlot(slot);
    return false;

#else
    // ── macOS / POSIX implementation ──

    // Check if an existing instance is already serving on this port
    // NOTE: Cannot call checkHealth() here — caller (start()) holds mutex_
    // and checkHealth() also acquires mutex_, causing deadlock.
    try {
        auto resp = cpr::Get(cpr::Url{slot.url}, cpr::Timeout{2000});
        if (resp.status_code != 0) {
            LOG_DEBUG("ProcessManager", "Existing server for '" + model.id + "' is healthy, reusing");
            slot.running = true;
            return true;
        }
    } catch (...) {}

    const auto resolveFromGrimRoot = [](const std::string& rawPath) {
        fs::path path(rawPath);
        if (path.is_absolute()) {
            return path;
        }
        return fs::path(getGrimRootDir()) / path;
    };

    const nlohmann::json& grimTextPaths = aiConfig.at("paths").at("grim_text");

    // Determine executable path
    fs::path server_exe;
    if (!model.model_path.empty() && fs::exists(model.model_path) && model.model_path.ends_with(".exe")) {
        server_exe = fs::absolute(model.model_path);
    } else {
        server_exe = resolveFromGrimRoot("resources/models/GRIM-text/training/build/Release/grim_text_server");
    }

    fs::path vocab_path = resolveFromGrimRoot(grimTextPaths.at("vocab").get<std::string>());
    fs::path model_weights = resolveFromGrimRoot(grimTextPaths.at("model").get<std::string>());

    if (!fs::exists(server_exe)) {
        LOG_ERROR("ProcessManager", "Server executable not found: " + server_exe.string());
        return false;
    }
    if (!fs::exists(vocab_path)) {
        LOG_ERROR("ProcessManager", "Vocabulary file not found: " + vocab_path.string());
        return false;
    }
    if (!fs::exists(model_weights)) {
        LOG_ERROR("ProcessManager", "Model file not found: " + model_weights.string());
        return false;
    }

    slot.executable_path = server_exe.string();

    std::string port_str = std::to_string(slot.port > 0 ? slot.port : 11435);

    LOG_DEBUG("ProcessManager", "Starting model '" + model.id + "': "
              + server_exe.string() + " " + vocab_path.string()
              + " " + model_weights.string() + " " + port_str);

    // Build argv for posix_spawn
    std::string exe_str   = server_exe.string();
    std::string vocab_str = vocab_path.string();
    std::string model_str = model_weights.string();

    char* argv[] = {
        const_cast<char*>(exe_str.c_str()),
        const_cast<char*>(vocab_str.c_str()),
        const_cast<char*>(model_str.c_str()),
        const_cast<char*>(port_str.c_str()),
        nullptr
    };

    // Set working directory via file actions (chdir)
    posix_spawn_file_actions_t file_actions;
    posix_spawn_file_actions_init(&file_actions);
    std::string cwd = server_exe.parent_path().string();
    posix_spawn_file_actions_addchdir_np(&file_actions, cwd.c_str());

    pid_t pid = 0;
    int spawn_err = posix_spawn(&pid, exe_str.c_str(), &file_actions, nullptr, argv, environ);
    posix_spawn_file_actions_destroy(&file_actions);

    if (spawn_err != 0) {
        LOG_ERROR("ProcessManager", "Failed to launch model '" + model.id
                  + "': posix_spawn error " + std::to_string(spawn_err));
        return false;
    }

    slot.pid = pid;
    slot.running = true;

    LOG_DEBUG("ProcessManager", "Model '" + model.id + "' process started (PID: "
              + std::to_string(pid) + ")");

    // Poll for health
    const int max_wait_ms = 30000;
    const int poll_ms = 500;
    int elapsed = 0;

    while (elapsed < max_wait_ms) {
        try {
            auto resp = cpr::Get(cpr::Url{slot.url}, cpr::Timeout{1000});
            if (resp.status_code != 0) {
                LOG_DEBUG("ProcessManager", "Model '" + model.id + "' server ready at "
                          + slot.url + " (took " + std::to_string(elapsed) + "ms)");
                return true;
            }
        } catch (...) {}

        // Check if process crashed
        int status = 0;
        pid_t result = waitpid(pid, &status, WNOHANG);
        if (result == pid) {
            int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
            LOG_ERROR("ProcessManager", "Model '" + model.id
                      + "' server terminated unexpectedly (exit code: "
                      + std::to_string(exit_code) + ")");
            slot.running = false;
            slot.pid = 0;
            return false;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(poll_ms));
        elapsed += poll_ms;
    }

    LOG_ERROR("ProcessManager", "Model '" + model.id
              + "' server failed to respond within " + std::to_string(max_wait_ms) + "ms");
    terminateSlot(slot);
    return false;
#endif
}

// =========================================================
// Process termination
// =========================================================

void ProcessManager::terminateSlot(ProcessSlot& slot) {
    if (!slot.running) return;

    LOG_DEBUG("ProcessManager", "Stopping model '" + slot.model_id + "'...");

#ifdef _WIN32
    if (slot.h_process) {
        // Try graceful shutdown
        if (!GenerateConsoleCtrlEvent(CTRL_C_EVENT, slot.process_info.dwProcessId)) {
            LOG_DEBUG("ProcessManager", "Forcefully terminating model '" + slot.model_id + "'");
            TerminateProcess(slot.h_process, 0);
        }
        WaitForSingleObject(slot.h_process, 5000);

        CloseHandle(slot.process_info.hProcess);
        CloseHandle(slot.process_info.hThread);
        slot.h_process = nullptr;
        ZeroMemory(&slot.process_info, sizeof(slot.process_info));
    }

    if (slot.h_mutex) {
        CloseHandle(slot.h_mutex);
        slot.h_mutex = nullptr;
    }
#else
    if (slot.pid > 0) {
        // Try graceful shutdown (SIGTERM)
        if (kill(slot.pid, SIGTERM) == 0) {
            // Wait up to 5 seconds for graceful exit
            const int max_wait_ms = 5000;
            const int poll_ms = 100;
            int elapsed = 0;
            bool exited = false;

            while (elapsed < max_wait_ms) {
                int status = 0;
                pid_t result = waitpid(slot.pid, &status, WNOHANG);
                if (result == slot.pid) {
                    exited = true;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(poll_ms));
                elapsed += poll_ms;
            }

            if (!exited) {
                LOG_DEBUG("ProcessManager", "Forcefully terminating model '" + slot.model_id + "'");
                kill(slot.pid, SIGKILL);
                waitpid(slot.pid, nullptr, 0);
            }
        }
        slot.pid = 0;
    }
#endif

    slot.running = false;
    LOG_DEBUG("ProcessManager", "Model '" + slot.model_id + "' stopped");
}

} // namespace GRIM::MMO
