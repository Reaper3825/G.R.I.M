#include "lm_intent.hpp"
#include "intent_gate.hpp"
#include "logger.hpp"
#include <nlohmann/json.hpp>

#ifdef _WIN32
#include <windows.h>
#include <thread>
#include <chrono>
#endif

namespace GRIM {

bool LMIntent::initialized_ = false;
bool LMIntent::available_ = false;

#ifdef _WIN32
static HANDLE hBridgeStdinWr = nullptr;
static HANDLE hBridgeStdoutRd = nullptr;
static PROCESS_INFORMATION bridgeProc{};
#endif

// Helper to read a line from the bridge (similar to Coqui)
static std::string readLineFromBridge(int timeoutMs = 5000) {
#ifdef _WIN32
    std::string result;
    char ch;
    DWORD read = 0, avail = 0;
    
    auto start = std::chrono::steady_clock::now();
    
    while (true) {
        if (!PeekNamedPipe(hBridgeStdoutRd, nullptr, 0, nullptr, &avail, nullptr)) {
            LOG_ERROR("LMIntent/Bridge", "PeekNamedPipe failed");
            break;
        }
        
        if (avail == 0) {
            // Check timeout
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start).count();
            if (elapsed > timeoutMs) {
                LOG_ERROR("LMIntent/Bridge", "Read timeout");
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        if (!ReadFile(hBridgeStdoutRd, &ch, 1, &read, nullptr) || read == 0) {
            break;
        }
        
        if (ch == '\r') continue;
        if (ch == '\n') break;
        result.push_back(ch);
    }
    
    return result;
#else
    return "";
#endif
}

void LMIntent::init() {
    if (initialized_) return;
    
    LOG_DEBUG("LMIntent", "Initializing LLM intent bridge...");
    
#ifdef _WIN32
    // Create pipes for stdin/stdout communication (same pattern as Coqui)
    SECURITY_ATTRIBUTES saAttr{};
    saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
    saAttr.bInheritHandle = TRUE;
    saAttr.lpSecurityDescriptor = nullptr;
    
    HANDLE hStdoutWr = nullptr;
    HANDLE hStdinRd = nullptr;
    
    // Create pipe for child's STDOUT
    if (!CreatePipe(&hBridgeStdoutRd, &hStdoutWr, &saAttr, 0)) {
        LOG_ERROR("LMIntent", "Failed to create stdout pipe");
        initialized_ = true;
        available_ = false;
        return;
    }
    SetHandleInformation(hBridgeStdoutRd, HANDLE_FLAG_INHERIT, 0);
    
    // Create pipe for child's STDIN
    if (!CreatePipe(&hStdinRd, &hBridgeStdinWr, &saAttr, 0)) {
        LOG_ERROR("LMIntent", "Failed to create stdin pipe");
        CloseHandle(hBridgeStdoutRd);
        CloseHandle(hStdoutWr);
        initialized_ = true;
        available_ = false;
        return;
    }
    SetHandleInformation(hBridgeStdinWr, HANDLE_FLAG_INHERIT, 0);
    
    // Setup process startup info
    STARTUPINFOA si{};
    ZeroMemory(&bridgeProc, sizeof(bridgeProc));
    si.cb = sizeof(STARTUPINFOA);
    si.hStdError = hStdoutWr;
    si.hStdOutput = hStdoutWr;
    si.hStdInput = hStdinRd;
    si.dwFlags |= STARTF_USESTDHANDLES;
    
    // Launch Python bridge (uses Ollama with configured model)
    std::string cmd = "python D:/G.R.I.M/resources/python/mistral_bridge.py";
    std::vector<char> mutableCmd(cmd.begin(), cmd.end());
    mutableCmd.push_back('\0');
    
    BOOL success = CreateProcessA(
        nullptr,
        mutableCmd.data(),
        nullptr,
        nullptr,
        TRUE,
        CREATE_NO_WINDOW,  // Hide console window
        nullptr,
        "D:/G.R.I.M/resources/python",
        &si,
        &bridgeProc
    );
    
    // Close unused pipe ends
    CloseHandle(hStdoutWr);
    CloseHandle(hStdinRd);
    
    if (!success) {
        LOG_ERROR("LMIntent", "Failed to launch intent bridge: " + std::to_string(GetLastError()));
        CloseHandle(hBridgeStdinWr);
        CloseHandle(hBridgeStdoutRd);
        initialized_ = true;
        available_ = false;
        return;
    }
    
    LOG_DEBUG("LMIntent", "Waiting for bridge handshake...");
    
    // Wait for "ready" handshake (with timeout)
    std::string response = readLineFromBridge(15000);  // 15s timeout for model loading
    
    if (response.empty()) {
        LOG_ERROR("LMIntent", "No handshake received from intent bridge");
        initialized_ = true;
        available_ = false;
        return;
    }
    
    try {
        auto j = nlohmann::json::parse(response);
        if (j.value("status", "") == "ready") {
            available_ = true;
            LOG_PHASE("LMIntent bridge ready", true);
        } else {
            LOG_ERROR("LMIntent", "Unexpected handshake: " + response);
            available_ = false;
        }
    } catch (const std::exception& e) {
        LOG_ERROR("LMIntent", std::string("Handshake parse failed: ") + e.what());
        available_ = false;
    }
#else
    // Non-Windows: use existing AI backend
    available_ = true;
    LOG_DEBUG("LMIntent", "Using fallback AI backend (non-Windows)");
#endif
    
    initialized_ = true;
}

void LMIntent::shutdown() {
#ifdef _WIN32
    if (hBridgeStdinWr) {
        // Send exit command
        std::string exitCmd = R"({"command":"exit"})" "\n";
        DWORD written;
        WriteFile(hBridgeStdinWr, exitCmd.c_str(), (DWORD)exitCmd.size(), &written, nullptr);
        
        CloseHandle(hBridgeStdinWr);
        hBridgeStdinWr = nullptr;
    }
    
    if (hBridgeStdoutRd) {
        CloseHandle(hBridgeStdoutRd);
        hBridgeStdoutRd = nullptr;
    }
    
    if (bridgeProc.hProcess) {
        WaitForSingleObject(bridgeProc.hProcess, 2000);
        CloseHandle(bridgeProc.hProcess);
        CloseHandle(bridgeProc.hThread);
    }
#endif
    
    initialized_ = false;
    available_ = false;
    LOG_DEBUG("LMIntent", "Bridge shutdown complete");
}

bool LMIntent::isAvailable() {
    return available_;
}

IntentType LMIntent::askOllama(const std::string& line) {
    if (!available_) {
        LOG_DEBUG("LMIntent", "Bridge not available");
        return IntentType::Unknown;
    }
    
#ifdef _WIN32
    try {
        // Build JSON request
        nlohmann::json req = {
            {"command", "classify"},
            {"text", line}
        };
        std::string request = req.dump() + "\n";
        
        LOG_DEBUG("LMIntent", "Sending classification request: " + line);
        
        // Send to bridge
        DWORD written = 0;
        if (!WriteFile(hBridgeStdinWr, request.c_str(), (DWORD)request.size(), &written, nullptr)) {
            LOG_ERROR("LMIntent", "WriteFile failed");
            return IntentType::Unknown;
        }
        
        // Read response
        std::string response = readLineFromBridge(5000);  // 5s timeout
        
        if (response.empty()) {
            LOG_ERROR("LMIntent", "No response from bridge (timeout)");
            return IntentType::Unknown;
        }
        
        // Parse response
        auto j = nlohmann::json::parse(response);
        
        if (j.value("status", "") != "ok") {
            LOG_ERROR("LMIntent", "Bridge error: " + j.value("message", "unknown"));
            return IntentType::Unknown;
        }
        
        auto result = j.value("result", nlohmann::json::object());
        std::string intent = result.value("intent", "unknown");
        
        IntentType type;
        if (intent == "command") {
            type = IntentType::Command;
        } else if (intent == "banter") {
            type = IntentType::Banter;
        } else {
            type = IntentType::Unknown;
        }
        
        LOG_DEBUG("LMIntent", "Classified '" + line + "' as " + intentTypeToString(type));
        return type;
        
    } catch (const std::exception& e) {
        LOG_ERROR("LMIntent", std::string("Classification failed: ") + e.what());
        return IntentType::Unknown;
    }
#else
    // Fallback: use existing AI backend
    extern std::future<std::string> callAIAsync(const std::string& prompt);
    
    std::string prompt = "Classify as 'command' or 'banter': " + line;
    auto future = callAIAsync(prompt);
    std::string response = future.get();
    
    if (response.find("command") != std::string::npos) {
        return IntentType::Command;
    } else if (response.find("banter") != std::string::npos) {
        return IntentType::Banter;
    }
    
    return IntentType::Unknown;
#endif
}

} // namespace GRIM
