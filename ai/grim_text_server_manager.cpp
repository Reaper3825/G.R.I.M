//======================================================//
//  GRIM-text Server Manager Implementation
//======================================================//

#include "grim_text_server_manager.hpp"
#include "../core/grim_text_server_api.hpp"
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
#include <stdexcept>
#include <vector>

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

int extractConfiguredPublicPort(const std::string& url) {
    const auto scheme_end = url.find("://");
    const std::size_t host_start = scheme_end == std::string::npos ? 0 : scheme_end + 3;
    const auto colon = url.rfind(':');
    if (colon == std::string::npos || colon < host_start) {
        throw std::runtime_error(
            "GRIMTextServer: configured server URL must contain an explicit port: " + url);
    }
    const int port = std::stoi(url.substr(colon + 1));
    if (port <= 0 || port > 65535) {
        throw std::runtime_error(
            "GRIMTextServer: configured public port must be in 1..65535");
    }
    return port;
}

int configuredInferenceWorkerPort(int publicPort) {
    if (!aiConfig.contains("training") || !aiConfig["training"].is_object() ||
        !aiConfig["training"].contains("server_port")) {
        throw std::runtime_error(
            "GRIMTextServer: ai_config.json is missing training.server_port");
    }
    const int port = aiConfig["training"].at("server_port").get<int>();
    if (port <= 0 || port > 65535) {
        throw std::runtime_error(
            "GRIMTextServer: training.server_port must be in 1..65535");
    }
    if (port == publicPort) {
        throw std::runtime_error(
            "GRIMTextServer: public port and training.server_port must differ");
    }
    return port;
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
      serverURL_(GRIM::ServerAPI::defaultPublicURL()),
      running_(false)
{
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
    if (isRunning()) {
        LOG_DEBUG("GRIMTextServer", "Server already running");
        return true;
    }
    shutdown();
    
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
    // train_gpu reads model/vocabulary settings from canonical ai_config.json.
    // Only process-scoped bridge ports are passed on the command line.
    const int publicPort = extractConfiguredPublicPort(serverURL_);
    const int workerPort = configuredInferenceWorkerPort(publicPort);
    std::string cmdLine = "\"" + serverExe.string() +
        "\" --public-port " + std::to_string(publicPort) +
        " --worker-port " + std::to_string(workerPort);
    
    std::vector<char> mutableCmd(cmdLine.begin(), cmdLine.end());
    mutableCmd.push_back('\0');
    const std::string workingDirectory = grimRoot.string();
    
    LOG_DEBUG("GRIMTextServer", "Command: " + cmdLine);

    STARTUPINFOEXA startup{};
    startup.StartupInfo.cb = sizeof(startup);
    SIZE_T attributeBytes = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &attributeBytes);
    if (attributeBytes == 0) {
        LOG_ERROR("GRIMTextServer", "Cannot size the bridge launch attributes");
        return false;
    }
    std::vector<unsigned char> attributeStorage(attributeBytes);
    startup.lpAttributeList = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attributeStorage.data());

    hMutex_ = CreateMutexA(nullptr, FALSE, "Global\\GRIMTextServerMutex");
    const DWORD mutexStatus = GetLastError();
    if (!hMutex_) {
        LOG_ERROR("GRIMTextServer", "Cannot open the bridge launch guard: " + std::to_string(mutexStatus));
        return false;
    }
    if (mutexStatus == ERROR_ALREADY_EXISTS) {
        shutdown();
        if (checkHealth(2000)) {
            LOG_DEBUG("GRIMTextServer", "Reusing the bridge owned by another GRIM host");
            return true;
        }
        LOG_ERROR("GRIMTextServer", "Another GRIM host owns the bridge launch guard and the server did not respond");
        return false;
    }

    hJob_ = CreateJobObjectW(nullptr, nullptr);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (!hJob_ ||
        !SetInformationJobObject(hJob_, JobObjectExtendedLimitInformation, &limits, sizeof(limits)) ||
        !InitializeProcThreadAttributeList(startup.lpAttributeList, 1, 0, &attributeBytes)) {
        LOG_ERROR("GRIMTextServer", "Cannot prepare bridge process ownership: " + std::to_string(GetLastError()));
        shutdown();
        return false;
    }

    // Assign the job at creation so even an interrupted launch cannot orphan the bridge.
    if (!UpdateProcThreadAttribute(startup.lpAttributeList, 0, PROC_THREAD_ATTRIBUTE_JOB_LIST,
                                  &hJob_, sizeof(hJob_), nullptr, nullptr)) {
        const DWORD error = GetLastError();
        DeleteProcThreadAttributeList(startup.lpAttributeList);
        LOG_ERROR("GRIMTextServer", "Cannot set bridge launch job: " + std::to_string(error));
        shutdown();
        return false;
    }

    PROCESS_INFORMATION child{};
    
    BOOL success = CreateProcessA(
        nullptr,
        mutableCmd.data(),
        nullptr,
        nullptr,
        FALSE,
        CREATE_NO_WINDOW | EXTENDED_STARTUPINFO_PRESENT,
        nullptr,
        workingDirectory.c_str(),
        &startup.StartupInfo,
        &child
    );
    const DWORD launchError = success ? ERROR_SUCCESS : GetLastError();
    DeleteProcThreadAttributeList(startup.lpAttributeList);
    
    if (!success) {
        LOG_ERROR("GRIMTextServer", "Failed to launch server: " + std::to_string(launchError));
        shutdown();
        return false;
    }
    
    CloseHandle(child.hThread);
    hProcess_ = child.hProcess;
    running_ = true;
    
    LOG_DEBUG("GRIMTextServer", "Server process started (PID: " + std::to_string(child.dwProcessId) + ")");
    
    // Wait for server to become ready
    LOG_DEBUG("GRIMTextServer", "Waiting for server to respond...");
    
    const int maxWaitMs = 30000;  // 30 seconds
    const int pollIntervalMs = 500;
    int elapsed = 0;
    
    while (elapsed < maxWaitMs) {
        if (checkHealth(1000) && isRunning()) {
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
            shutdown();
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
#ifdef _WIN32
    if (hJob_) {
        CloseHandle(hJob_);
        hJob_ = nullptr;
    }
    if (hProcess_) {
        const DWORD result = WaitForSingleObject(hProcess_, 5000);
        if (result != WAIT_OBJECT_0) {
            const DWORD error = result == WAIT_FAILED ? GetLastError() : ERROR_TIMEOUT;
            LOG_ERROR("GRIMTextServer", "Could not confirm bridge exit: " + std::to_string(error));
        }
        CloseHandle(hProcess_);
        hProcess_ = nullptr;
    }
    
    // Release the mutex
    if (hMutex_) {
        CloseHandle(hMutex_);
        hMutex_ = nullptr;
    }
#endif
    
    running_ = false;
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
    if (g_grimTextServerManager) {
        g_grimTextServerManager->shutdown();
    }
}

bool isGRIMTextServerRunning() {
    return GRIMTextServerManager::getInstance().isRunning();
}

} // namespace GRIM
