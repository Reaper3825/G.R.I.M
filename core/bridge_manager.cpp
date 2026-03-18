#include "bridge_manager.hpp"

#ifdef _WIN32
#include "logger.hpp"
#include <unordered_map>
#include <string>
#include <thread>
#include <mutex>
#include <iostream>
#include <sstream>
#include <nlohmann/json.hpp>
#include "grim_platform.h"

namespace {

struct BridgeProcess {
    HANDLE hWrite = nullptr;
    HANDLE hRead = nullptr;
    PROCESS_INFORMATION procInfo{};
    std::mutex ioMutex;
};

std::unordered_map<std::string, BridgeProcess> bridges;

} // anon namespace

// ===================================================
// Launch Python bridge
// ===================================================
bool BridgeManager::start(const std::string& id, const std::string& scriptPath) {
    if (bridges.find(id) != bridges.end()) return true;

    SECURITY_ATTRIBUTES sa{ sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE };
    HANDLE stdinRead, stdoutWrite;

    if (!CreatePipe(&stdinRead, &bridges[id].hWrite, &sa, 0)) return false;
    if (!CreatePipe(&bridges[id].hRead, &stdoutWrite, &sa, 0)) return false;

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.hStdInput = stdinRead;
    si.hStdOutput = stdoutWrite;
    si.hStdError = stdoutWrite;
    si.dwFlags |= STARTF_USESTDHANDLES;

    std::string cmd = "python " + scriptPath;
    if (!CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, TRUE,
                        CREATE_NO_WINDOW, nullptr, nullptr,
                        &si, &bridges[id].procInfo))
    {
        LOG_ERROR("BridgeManager", "Failed to start " + scriptPath);
        return false;
    }

    CloseHandle(stdinRead);
    CloseHandle(stdoutWrite);
    LOG_DEBUG("BridgeManager", "Started bridge: " + id);
    return true;
}

// ===================================================
// Send JSON → Receive JSON
// ===================================================
nlohmann::json BridgeManager::send(const std::string& id, const nlohmann::json& msg) {
    if (bridges.find(id) == bridges.end()) return {{"error", "bridge not running"}};

    BridgeProcess& bp = bridges[id];
    std::lock_guard<std::mutex> lock(bp.ioMutex);

    std::string input = msg.dump() + "\n";
    DWORD written;
    WriteFile(bp.hWrite, input.c_str(), (DWORD)input.size(), &written, nullptr);

    char buffer[2048];
    DWORD read = 0;
    if (!ReadFile(bp.hRead, buffer, sizeof(buffer) - 1, &read, nullptr))
        return {{"error", "read failed"}};

    buffer[read] = 0;
    try {
        return nlohmann::json::parse(buffer);
    } catch (...) {
        return {{"error", "invalid json from bridge"}};
    }
}

// ===================================================
// Stop a bridge
// ===================================================
void BridgeManager::stop(const std::string& id) {
    if (bridges.find(id) == bridges.end()) return;

    BridgeProcess& bp = bridges[id];
    if (bp.procInfo.hProcess) {
        TerminateProcess(bp.procInfo.hProcess, 0);
        CloseHandle(bp.procInfo.hProcess);
        CloseHandle(bp.procInfo.hThread);
    }
    if (bp.hWrite) CloseHandle(bp.hWrite);
    if (bp.hRead) CloseHandle(bp.hRead);
    bridges.erase(id);

    LOG_DEBUG("BridgeManager", "Stopped bridge: " + id);
}
#else // !_WIN32

bool BridgeManager::start(const std::string&, const std::string&) { return false; }
nlohmann::json BridgeManager::send(const std::string&, const nlohmann::json&) { return {}; }
void BridgeManager::stop(const std::string&) {}

#endif // _WIN32
