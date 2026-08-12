#include "cosyvoice_provider.hpp"

#include "logger.hpp"

#include <chrono>
#include <filesystem>
#include <mutex>
#include <random>
#include <thread>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#ifdef _WIN32
    #include <windows.h>
#endif

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace Voice {
namespace {

std::string randomFileStem(std::size_t length) {
    static constexpr char characters[] =
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    static thread_local std::mt19937 generator{std::random_device{}()};
    static thread_local std::uniform_int_distribution<std::size_t> distribution(
        0, sizeof(characters) - 2);

    std::string result;
    result.reserve(length);
    for (std::size_t i = 0; i < length; ++i) {
        result.push_back(characters[distribution(generator)]);
    }
    return result;
}

#ifdef _WIN32
std::string quoteArgument(const std::string& value) {
    std::string result = "\"";
    for (char character : value) {
        if (character == '"') {
            result += "\\\"";
        } else {
            result += character;
        }
    }
    result += '"';
    return result;
}
#endif

class CosyVoiceProvider final : public ITTSProvider {
public:
    explicit CosyVoiceProvider(CosyVoiceProviderConfig config)
        : config_(std::move(config)) {}

    ~CosyVoiceProvider() override {
        shutdown();
    }

    const char* providerId() const noexcept override {
        return "fun-cosyvoice3";
    }

    bool initialize() override;
    void shutdown() noexcept override;

    TTSProviderState state() const noexcept override {
        return state_;
    }

    TTSProviderCapabilities capabilities() const override {
        TTSProviderCapabilities result;
        result.supports_voice_selection = true;
        result.supports_voice_cloning = true;
        result.supports_language_selection = true;
        result.languages = {
            "zh", "en", "ja", "ko", "de", "es", "fr", "it", "ru"
        };
        return result;
    }

    TTSSynthesisResult synthesize(
        const TTSSynthesisRequest& request) override;

private:
#ifdef _WIN32
    bool launchBridge();
    bool writeRequest(const json& request);
    json readResponse(int timeout_ms);
    void readStderr();
    void closeHandles() noexcept;

    HANDLE child_stdin_ = nullptr;
    HANDLE child_stdout_ = nullptr;
    HANDLE child_stderr_ = nullptr;
    PROCESS_INFORMATION process_{};
    std::thread stderr_thread_;
#endif

    CosyVoiceProviderConfig config_;
    TTSProviderState state_ = TTSProviderState::Stopped;
    std::mutex synthesis_mutex_;
};

bool CosyVoiceProvider::initialize() {
    if (state_ == TTSProviderState::Ready) {
        return true;
    }

    state_ = TTSProviderState::Starting;

#ifndef _WIN32
    LOG_ERROR("Voice/CosyVoice", "The local CosyVoice bridge is currently implemented for Windows");
    state_ = TTSProviderState::Failed;
    return false;
#else
    const struct RequiredPath {
        const char* name;
        const std::string* value;
        bool directory;
    } required_paths[] = {
        {"Python executable", &config_.python_executable, false},
        {"CosyVoice repository", &config_.repository_path, true},
        {"CosyVoice model", &config_.model_path, true},
        {"CosyVoice bridge", &config_.bridge_script, false},
    };

    for (const RequiredPath& required : required_paths) {
        const fs::path path(*required.value);
        const bool exists = required.directory
            ? fs::is_directory(path)
            : fs::is_regular_file(path);
        if (!exists) {
            LOG_ERROR(
                "Voice/CosyVoice",
                std::string(required.name) + " is not available locally: " + path.string());
            state_ = TTSProviderState::Failed;
            return false;
        }
    }

    std::error_code directory_error;
    fs::create_directories(config_.output_directory, directory_error);
    if (directory_error) {
        LOG_ERROR(
            "Voice/CosyVoice",
            "Unable to create output directory: " + directory_error.message());
        state_ = TTSProviderState::Failed;
        return false;
    }

    if (!launchBridge()) {
        state_ = TTSProviderState::Failed;
        return false;
    }

    const json response = readResponse(config_.startup_timeout_ms);
    if (response.value("status", "") != "ready") {
        LOG_ERROR(
            "Voice/CosyVoice",
            "Bridge failed to become ready: " +
                response.value("message", std::string("no handshake received")));
        closeHandles();
        state_ = TTSProviderState::Failed;
        return false;
    }

    state_ = TTSProviderState::Ready;
    LOG_PHASE("Fun-CosyVoice 3 bridge ready", true);
    return true;
#endif
}

void CosyVoiceProvider::shutdown() noexcept {
#ifdef _WIN32
    try {
        if (child_stdin_) {
            writeRequest({{"command", "exit"}});
        }
    } catch (...) {
        LOG_ERROR("Voice/CosyVoice", "Unable to send bridge shutdown request");
    }

    if (process_.hProcess) {
        WaitForSingleObject(process_.hProcess, 2000);
    }
    closeHandles();
#endif
    state_ = TTSProviderState::Stopped;
}

TTSSynthesisResult CosyVoiceProvider::synthesize(
    const TTSSynthesisRequest& request) {
    if (state_ != TTSProviderState::Ready) {
        return {false, "", "ERR_TTS_PROVIDER_NOT_READY", "Fun-CosyVoice 3 is not ready"};
    }
    if (request.text.empty()) {
        return {false, "", "ERR_TTS_EMPTY_TEXT", "TTS request text is empty"};
    }

#ifndef _WIN32
    return {false, "", "ERR_TTS_PLATFORM_UNSUPPORTED", "CosyVoice bridge is unavailable"};
#else
    std::lock_guard<std::mutex> lock(synthesis_mutex_);

    const fs::path output_path =
        fs::path(config_.output_directory) / (randomFileStem(32) + ".wav");
    json bridge_request = {
        {"command", "speak"},
        {"text", request.text},
        {"speaker", request.voice_id},
        {"language", request.language},
        {"reference_audio", config_.reference_audio_path},
        {"reference_text", config_.reference_text},
        {"out", output_path.string()},
    };

    if (!writeRequest(bridge_request)) {
        state_ = TTSProviderState::Failed;
        return {false, "", "ERR_TTS_BRIDGE_WRITE", "Unable to write to CosyVoice bridge"};
    }

    const json response = readResponse(config_.synthesis_timeout_ms);
    if (response.value("status", "") != "ok") {
        return {
            false,
            "",
            response.value("error_code", std::string("ERR_TTS_SYNTHESIS_FAILED")),
            response.value("message", std::string("CosyVoice did not produce audio"))
        };
    }

    const std::string audio_path = response.value("file", std::string{});
    if (audio_path.empty() || !fs::is_regular_file(audio_path)) {
        return {
            false,
            "",
            "ERR_TTS_OUTPUT_MISSING",
            "CosyVoice reported success without a readable audio file"
        };
    }

    return {true, audio_path, "", ""};
#endif
}

#ifdef _WIN32
bool CosyVoiceProvider::launchBridge() {
    SECURITY_ATTRIBUTES attributes{};
    attributes.nLength = sizeof(attributes);
    attributes.bInheritHandle = TRUE;

    HANDLE child_stdout_write = nullptr;
    HANDLE child_stdin_read = nullptr;
    HANDLE child_stderr_write = nullptr;

    if (!CreatePipe(&child_stdout_, &child_stdout_write, &attributes, 0)
        || !SetHandleInformation(child_stdout_, HANDLE_FLAG_INHERIT, 0)
        || !CreatePipe(&child_stdin_read, &child_stdin_, &attributes, 0)
        || !SetHandleInformation(child_stdin_, HANDLE_FLAG_INHERIT, 0)
        || !CreatePipe(&child_stderr_, &child_stderr_write, &attributes, 0)
        || !SetHandleInformation(child_stderr_, HANDLE_FLAG_INHERIT, 0)) {
        LOG_ERROR("Voice/CosyVoice", "Unable to create CosyVoice bridge pipes");
        if (child_stdout_write) CloseHandle(child_stdout_write);
        if (child_stdin_read) CloseHandle(child_stdin_read);
        if (child_stderr_write) CloseHandle(child_stderr_write);
        closeHandles();
        return false;
    }

    STARTUPINFOA startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = child_stdin_read;
    startup.hStdOutput = child_stdout_write;
    startup.hStdError = child_stderr_write;

    std::string command =
        quoteArgument(config_.python_executable) + " " +
        quoteArgument(config_.bridge_script) + " --persistent --repo " +
        quoteArgument(config_.repository_path) + " --model " +
        quoteArgument(config_.model_path);
    std::vector<char> mutable_command(command.begin(), command.end());
    mutable_command.push_back('\0');

    ZeroMemory(&process_, sizeof(process_));
    const BOOL created = CreateProcessA(
        nullptr,
        mutable_command.data(),
        nullptr,
        nullptr,
        TRUE,
        CREATE_NO_WINDOW,
        nullptr,
        config_.repository_path.c_str(),
        &startup,
        &process_);

    CloseHandle(child_stdout_write);
    CloseHandle(child_stdin_read);
    CloseHandle(child_stderr_write);

    if (!created) {
        LOG_ERROR(
            "Voice/CosyVoice",
            "CreateProcessA failed: " + std::to_string(GetLastError()));
        closeHandles();
        return false;
    }

    try {
        stderr_thread_ = std::thread(&CosyVoiceProvider::readStderr, this);
    } catch (const std::exception& error) {
        LOG_ERROR(
            "Voice/CosyVoice",
            std::string("Unable to start bridge stderr reader: ") + error.what());
        TerminateProcess(process_.hProcess, 1);
        closeHandles();
        return false;
    }
    return true;
}

bool CosyVoiceProvider::writeRequest(const json& request) {
    if (!child_stdin_) {
        return false;
    }

    const std::string line = request.dump() + "\n";
    DWORD written = 0;
    return WriteFile(
               child_stdin_,
               line.data(),
               static_cast<DWORD>(line.size()),
               &written,
               nullptr)
        && written == line.size();
}

json CosyVoiceProvider::readResponse(int timeout_ms) {
    if (!child_stdout_) {
        return json::object();
    }

    const auto deadline = std::chrono::steady_clock::now()
        + std::chrono::milliseconds(timeout_ms);
    std::string line;

    while (std::chrono::steady_clock::now() < deadline) {
        DWORD available = 0;
        if (!PeekNamedPipe(child_stdout_, nullptr, 0, nullptr, &available, nullptr)) {
            return json::object();
        }
        if (available == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        char character = '\0';
        DWORD read = 0;
        if (!ReadFile(child_stdout_, &character, 1, &read, nullptr) || read == 0) {
            return json::object();
        }
        if (character == '\r') {
            continue;
        }
        if (character != '\n') {
            line.push_back(character);
            continue;
        }
        if (line.empty()) {
            continue;
        }

        try {
            return json::parse(line);
        } catch (const json::parse_error&) {
            LOG_DEBUG("Voice/CosyVoice", "Skipped non-JSON bridge output: " + line);
            line.clear();
        }
    }

    return json::object();
}

void CosyVoiceProvider::readStderr() {
    std::string line;
    while (child_stderr_) {
        DWORD available = 0;
        if (!PeekNamedPipe(child_stderr_, nullptr, 0, nullptr, &available, nullptr)) {
            break;
        }
        if (available == 0) {
            DWORD exit_code = 0;
            if (process_.hProcess
                && GetExitCodeProcess(process_.hProcess, &exit_code)
                && exit_code != STILL_ACTIVE) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        char character = '\0';
        DWORD read = 0;
        if (!ReadFile(child_stderr_, &character, 1, &read, nullptr) || read == 0) {
            break;
        }
        if (character == '\r') {
            continue;
        }
        if (character == '\n') {
            if (!line.empty()) {
                LOG_DEBUG("Voice/CosyVoice/Stderr", line);
                line.clear();
            }
            continue;
        }
        line.push_back(character);
    }
    if (!line.empty()) {
        LOG_DEBUG("Voice/CosyVoice/Stderr", line);
    }
}

void CosyVoiceProvider::closeHandles() noexcept {
    if (child_stdin_) {
        CloseHandle(child_stdin_);
        child_stdin_ = nullptr;
    }
    if (child_stdout_) {
        CloseHandle(child_stdout_);
        child_stdout_ = nullptr;
    }
    if (child_stderr_) {
        CloseHandle(child_stderr_);
        child_stderr_ = nullptr;
    }
    if (stderr_thread_.joinable()) {
        try {
            stderr_thread_.join();
        } catch (...) {
            LOG_ERROR("Voice/CosyVoice", "Unable to join bridge stderr reader");
        }
    }
    if (process_.hThread) {
        CloseHandle(process_.hThread);
        process_.hThread = nullptr;
    }
    if (process_.hProcess) {
        CloseHandle(process_.hProcess);
        process_.hProcess = nullptr;
    }
}
#endif

} // namespace

std::unique_ptr<ITTSProvider> createCosyVoiceProvider(
    CosyVoiceProviderConfig config) {
    return std::make_unique<CosyVoiceProvider>(std::move(config));
}

} // namespace Voice
