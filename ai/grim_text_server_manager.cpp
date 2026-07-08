//======================================================//
//  GRIM-text Server Manager Implementation
//======================================================//

#include "grim_text_server_manager.hpp"
#include "../logger.hpp"
#include "../resources.hpp"
#include <cpr/cpr.h>
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <thread>
#include <chrono>
#include <sstream>
#include <iomanip>

#ifdef _WIN32
#include "core/grim_platform.h"
#endif

namespace fs = std::filesystem;
using json = nlohmann::json;

#ifdef _WIN32
namespace {

constexpr DWORD kStatusDllNotFound = 0xC0000135u;

std::string formatWindowsStatusCode(DWORD code) {
    std::ostringstream oss;
    oss << "0x" << std::uppercase << std::hex << std::setw(8) << std::setfill('0') << code;
    return oss.str();
}

void logServerTerminationDiagnosis(DWORD exitCode, const fs::path& serverExe) {
    if (exitCode != kStatusDllNotFound) {
        return;
    }

    LOG_ERROR("GRIMTextServer",
        "Windows loader status " + formatWindowsStatusCode(exitCode) +
        " (STATUS_DLL_NOT_FOUND): missing runtime dependency while loading " + serverExe.string());
    LOG_ERROR("GRIMTextServer",
        "Install/repair Microsoft Visual C++ Redistributable (x64), verify CUDA runtime DLLs are on PATH, then inspect grim_text_server.exe dependencies.");
}

} // namespace
#endif

namespace GRIM {

std::unique_ptr<GRIMTextServerManager> g_grimTextServerManager;

GRIMTextServerManager& GRIMTextServerManager::getInstance() {
    if (!g_grimTextServerManager) {
        g_grimTextServerManager = std::unique_ptr<GRIMTextServerManager>(new GRIMTextServerManager());
    }
    return *g_grimTextServerManager;
}

GRIMTextServerManager::GRIMTextServerManager() 
    : serverPath_("resources/models/GRIM-text/training/build/Release/grim_text_server.exe"),
      serverURL_("http://127.0.0.1:11435"),
      running_(false)
#ifdef _WIN32
      , hProcess_(nullptr)
      , hMutex_(nullptr)
#endif
{
#ifdef _WIN32
    ZeroMemory(&processInfo_, sizeof(processInfo_));
#endif
}

GRIMTextServerManager::~GRIMTextServerManager() {
    shutdown();
}

void GRIMTextServerManager::setServerPath(const std::string& path) {
    serverPath_ = path;
}

void GRIMTextServerManager::setServerURL(const std::string& url) {
    serverURL_ = url;
}

bool GRIMTextServerManager::checkHealth(int timeoutMs) {
    try {
        // Try to reach the server's health endpoint or root
        auto resp = cpr::Get(
            cpr::Url{serverURL_},
            cpr::Timeout{timeoutMs}
        );
        
        // Server is considered healthy if it responds (even with 404 is ok, means it's alive)
        return resp.status_code != 0;
    } catch (...) {
        return false;
    }
}

bool GRIMTextServerManager::start() {
    if (running_) {
        LOG_DEBUG("GRIMTextServer", "Server already running");
        return true;
    }
    
    // Check if another instance is already running
#ifdef _WIN32
    hMutex_ = CreateMutexA(nullptr, FALSE, "Global\\GRIMTextServerMutex");
    if (GetLastError() == ERROR_ALREADY_EXISTS) {
        LOG_DEBUG("GRIMTextServer", "Another GRIM-text server is already running");
        if (checkHealth(2000)) {
            LOG_DEBUG("GRIMTextServer", "Existing server is healthy, reusing it");
            running_ = true;  // Mark as running so we don't try to stop it
            return true;
        }
        LOG_DEBUG("GRIMTextServer", "Existing server not responding, killing stale processes");
        // Kill any orphaned grim_text_server processes
        system("taskkill /F /IM grim_text_server.exe >nul 2>&1");
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
#endif
    
    LOG_DEBUG("GRIMTextServer", "Starting GRIM-text server...");

    const auto resolveFromGrimRoot = [](const std::string& rawPath) {
        fs::path path(rawPath);
        if (path.is_absolute()) {
            return path;
        }
        return fs::path(getGrimRootDir()) / path;
    };

    fs::path grimRoot = fs::path(getGrimRootDir());
    fs::path serverExe = resolveFromGrimRoot(serverPath_);
    
    if (!fs::exists(serverExe)) {
        LOG_ERROR("GRIMTextServer", "Server executable not found: " + serverExe.string());
        LOG_ERROR("GRIMTextServer", "Build GRIM-text first: cmake --build resources/models/GRIM-text/training/build --config Release");
        return false;
    }
    
    LOG_DEBUG("GRIMTextServer", "Server path: " + serverExe.string());
    LOG_DEBUG("GRIMTextServer", "Working directory: " + grimRoot.string());
    
#ifdef _WIN32
    // Setup startup info
    STARTUPINFOA si{};
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;  // Run in background
    
    ZeroMemory(&processInfo_, sizeof(processInfo_));
    
    // The server now reads config internally; no CLI arguments are required.
    std::string cmdLine = "\"" + serverExe.string() + "\"";
    
    std::vector<char> mutableCmd(cmdLine.begin(), cmdLine.end());
    mutableCmd.push_back('\0');
    
    LOG_DEBUG("GRIMTextServer", "Command: " + cmdLine);
    
    BOOL success = CreateProcessA(
        nullptr,
        mutableCmd.data(),
        nullptr,
        nullptr,
        FALSE,
        CREATE_NO_WINDOW,  // Don't create new console - inherit parent's process group
        nullptr,
        grimRoot.string().c_str(),
        &si,
        &processInfo_
    );
    
    if (!success) {
        DWORD error = GetLastError();
        LOG_ERROR("GRIMTextServer", "Failed to launch server: " + std::to_string(error));
        return false;
    }
    
    hProcess_ = processInfo_.hProcess;
    running_ = true;
    
    LOG_DEBUG("GRIMTextServer", "Server process started (PID: " + std::to_string(processInfo_.dwProcessId) + ")");
    
    // Wait for server to become ready
    LOG_DEBUG("GRIMTextServer", "Waiting for server to respond...");
    
    const int maxWaitMs = 30000;  // 30 seconds
    const int pollIntervalMs = 500;
    int elapsed = 0;
    
    while (elapsed < maxWaitMs) {
        if (checkHealth(1000)) {
            LOG_PHASE("GRIM-text server ready", true);
            LOG_DEBUG("GRIMTextServer", "Server ready at " + serverURL_ + " (took " + std::to_string(elapsed) + "ms)");
            return true;
        }
        
        // Check if process crashed
        DWORD exitCode = 0;
        if (GetExitCodeProcess(hProcess_, &exitCode) && exitCode != STILL_ACTIVE) {
            LOG_ERROR("GRIMTextServer",
                "Server process terminated unexpectedly (exit code: " + std::to_string(exitCode) +
                ", status: " + formatWindowsStatusCode(exitCode) + ")");
            logServerTerminationDiagnosis(exitCode, serverExe);
            running_ = false;
            return false;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(pollIntervalMs));
        elapsed += pollIntervalMs;
    }
    
    LOG_ERROR("GRIMTextServer", "Server failed to respond within " + std::to_string(maxWaitMs) + "ms");
    shutdown();  // Kill unresponsive process
    return false;
    
#else
    LOG_ERROR("GRIMTextServer", "Server management only supported on Windows");
    return false;
#endif
}

void GRIMTextServerManager::shutdown() {
    if (!running_) {
        return;
    }
    
    LOG_DEBUG("GRIMTextServer", "Shutting down GRIM-text server...");
    
#ifdef _WIN32
    if (hProcess_) {
        // Try graceful shutdown first (server should handle CTRL+C)
        if (!GenerateConsoleCtrlEvent(CTRL_C_EVENT, processInfo_.dwProcessId)) {
            // If that fails, terminate forcefully
            LOG_DEBUG("GRIMTextServer", "Forcefully terminating server process");
            TerminateProcess(hProcess_, 0);
        }
        
        // Wait for process to exit
        WaitForSingleObject(hProcess_, 5000);
        
        CloseHandle(processInfo_.hProcess);
        CloseHandle(processInfo_.hThread);
        
        hProcess_ = nullptr;
        ZeroMemory(&processInfo_, sizeof(processInfo_));
    }
    
    // Release the mutex
    if (hMutex_) {
        CloseHandle(hMutex_);
        hMutex_ = nullptr;
    }
#endif
    
    running_ = false;
    LOG_DEBUG("GRIMTextServer", "Server shutdown complete");
}

bool GRIMTextServerManager::isRunning() const {
#ifdef _WIN32
    if (!running_ || !hProcess_) {
        return false;
    }
    
    DWORD exitCode = 0;
    if (GetExitCodeProcess(hProcess_, &exitCode)) {
        return exitCode == STILL_ACTIVE;
    }
#endif
    
    return running_;
}

// Helper functions
bool startGRIMTextServer() {
    return GRIMTextServerManager::getInstance().start();
}

void stopGRIMTextServer() {
    GRIMTextServerManager::getInstance().shutdown();
}

bool isGRIMTextServerRunning() {
    return GRIMTextServerManager::getInstance().isRunning();
}

} // namespace GRIM
