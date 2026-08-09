//======================================================//
//  GRIM-text HTTP Server
//  Ollama-compatible HTTP bridge for the trainer-owned
//  Phase1/Phase2 inference runtime.
//
//  Public endpoint: http://127.0.0.1:11435
//  Worker process:  train_gpu --inference
//======================================================//

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace {

struct BridgeOptions {
    int public_port = 11435;
    int worker_port = 11436;
    std::filesystem::path train_gpu_path;
};

int parsePort(const std::string& value, const char* option_name) {
    std::size_t parsed = 0;
    int port = 0;
    try {
        port = std::stoi(value, &parsed);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("grim_text_server: ") + option_name +
                                 " requires an integer port, got '" + value + "'");
    }
    if (parsed != value.size() || port <= 0 || port > 65535) {
        throw std::runtime_error(std::string("grim_text_server: ") + option_name +
                                 " must be in 1..65535, got '" + value + "'");
    }
    return port;
}

BridgeOptions parseBridgeOptions(int argc, char** argv) {
    BridgeOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto requireValue = [&](const char* option_name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("grim_text_server: ") +
                                         option_name + " requires a value");
            }
            return argv[++i];
        };

        if (arg == "--public-port") {
            options.public_port = parsePort(requireValue("--public-port"), "--public-port");
        } else if (arg == "--worker-port") {
            options.worker_port = parsePort(requireValue("--worker-port"), "--worker-port");
        } else if (arg == "--train-gpu") {
            options.train_gpu_path = requireValue("--train-gpu");
        } else {
            throw std::runtime_error("grim_text_server: unknown argument '" + arg + "'");
        }
    }

    if (options.public_port == options.worker_port) {
        throw std::runtime_error(
            "grim_text_server: --public-port and --worker-port must be different");
    }
    return options;
}

std::filesystem::path currentExecutablePath() {
#ifdef _WIN32
    std::vector<wchar_t> buffer(32768);
    const DWORD count = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (count == 0 || count >= buffer.size()) {
        throw std::runtime_error("grim_text_server: failed to resolve the server executable path");
    }
    return std::filesystem::weakly_canonical(std::filesystem::path(buffer.data()));
#else
    std::vector<char> buffer(4096);
    const ssize_t count = readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
    if (count < 0) {
        throw std::runtime_error("grim_text_server: failed to resolve the server executable path");
    }
    buffer[static_cast<size_t>(count)] = '\0';
    return std::filesystem::weakly_canonical(std::filesystem::path(buffer.data()));
#endif
}

std::string pathListForError(const std::vector<std::filesystem::path>& paths) {
    std::ostringstream oss;
    for (const auto& path : paths) {
        oss << "\n  - " << path.string();
    }
    return oss.str();
}

std::filesystem::path resolveTrainGpuPath(const std::filesystem::path& explicit_path) {
    if (!explicit_path.empty()) {
        if (!std::filesystem::exists(explicit_path)) {
            throw std::runtime_error("grim_text_server: --train-gpu path does not exist: " +
                                     explicit_path.string());
        }
        return std::filesystem::weakly_canonical(explicit_path);
    }

    const auto server_exe = currentExecutablePath();
    const auto server_dir = server_exe.parent_path();
    const auto cwd = std::filesystem::current_path();

    std::vector<std::filesystem::path> candidates;
#ifdef _WIN32
    candidates.push_back(server_dir / "train_gpu.exe");
    candidates.push_back(cwd / "resources" / "models" / "GRIM-text" / "training" / "build" / "Release" / "train_gpu.exe");
    candidates.push_back(cwd / "resources" / "models" / "GRIM-text" / "training" / "build" / "Debug" / "train_gpu.exe");
    candidates.push_back(cwd / "resources" / "models" / "GRIM-text" / "training" / "TrainingLoop" / "build" / "Release" / "train_gpu.exe");
    candidates.push_back(cwd / "resources" / "models" / "GRIM-text" / "training" / "TrainingLoop" / "build" / "Debug" / "train_gpu.exe");
#else
    candidates.push_back(server_dir / "train_gpu");
    candidates.push_back(cwd / "resources" / "models" / "GRIM-text" / "training" / "build" / "Release" / "train_gpu");
    candidates.push_back(cwd / "resources" / "models" / "GRIM-text" / "training" / "build" / "Debug" / "train_gpu");
    candidates.push_back(cwd / "resources" / "models" / "GRIM-text" / "training" / "TrainingLoop" / "build" / "train_gpu");
#endif

    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return std::filesystem::weakly_canonical(candidate);
        }
    }

    throw std::runtime_error("grim_text_server: could not locate train_gpu executable. Checked:" +
                             pathListForError(candidates) +
                             "\nPass --train-gpu <path> to make the process boundary explicit.");
}

#ifdef _WIN32
std::wstring utf8ToWide(const std::string& value) {
    if (value.empty()) {
        return std::wstring();
    }
    const int count = MultiByteToWideChar(CP_UTF8, 0, value.c_str(), -1, nullptr, 0);
    if (count <= 0) {
        throw std::runtime_error("grim_text_server: failed to convert process command to UTF-16");
    }
    std::wstring wide(static_cast<size_t>(count - 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, value.c_str(), -1, wide.data(), count);
    return wide;
}
#endif

class TrainGpuProcess {
public:
    TrainGpuProcess() = default;
    TrainGpuProcess(const TrainGpuProcess&) = delete;
    TrainGpuProcess& operator=(const TrainGpuProcess&) = delete;

    ~TrainGpuProcess() {
        stop();
    }

    void start(const std::filesystem::path& train_gpu_path, int worker_port) {
#ifdef _WIN32
        if (process_info_.hProcess || job_handle_) {
            throw std::runtime_error("grim_text_server: train_gpu worker process is already started");
        }

        HANDLE job_handle = CreateJobObjectW(nullptr, nullptr);
        if (!job_handle) {
            throw std::runtime_error("grim_text_server: CreateJobObjectW failed for train_gpu worker "
                                     "(GetLastError=" + std::to_string(GetLastError()) + ")");
        }

        JOBOBJECT_EXTENDED_LIMIT_INFORMATION job_info{};
        job_info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (!SetInformationJobObject(
                job_handle,
                JobObjectExtendedLimitInformation,
                &job_info,
                sizeof(job_info))) {
            const DWORD error = GetLastError();
            CloseHandle(job_handle);
            throw std::runtime_error("grim_text_server: SetInformationJobObject failed for train_gpu worker "
                                     "(GetLastError=" + std::to_string(error) + ")");
        }

        STARTUPINFOW startup_info{};
        startup_info.cb = sizeof(startup_info);
        PROCESS_INFORMATION process_info{};

        std::string command = "\"" + train_gpu_path.string() + "\" --inference --inference-worker-port " +
                              std::to_string(worker_port);
        std::wstring wide_command = utf8ToWide(command);
        std::vector<wchar_t> mutable_command(wide_command.begin(), wide_command.end());
        mutable_command.push_back(L'\0');

        const BOOL created = CreateProcessW(
            nullptr,
            mutable_command.data(),
            nullptr,
            nullptr,
            FALSE,
            CREATE_SUSPENDED,
            nullptr,
            nullptr,
            &startup_info,
            &process_info);

        if (!created) {
            const DWORD error = GetLastError();
            CloseHandle(job_handle);
            throw std::runtime_error("grim_text_server: CreateProcessW failed for " + train_gpu_path.string() +
                                     " (GetLastError=" + std::to_string(error) + ")");
        }

        if (!AssignProcessToJobObject(job_handle, process_info.hProcess)) {
            const DWORD error = GetLastError();
            TerminateProcess(process_info.hProcess, 1);
            WaitForSingleObject(process_info.hProcess, 5000);
            CloseHandle(process_info.hThread);
            CloseHandle(process_info.hProcess);
            CloseHandle(job_handle);
            throw std::runtime_error("grim_text_server: AssignProcessToJobObject failed for train_gpu worker "
                                     "(GetLastError=" + std::to_string(error) + ")");
        }

        if (ResumeThread(process_info.hThread) == static_cast<DWORD>(-1)) {
            const DWORD error = GetLastError();
            TerminateProcess(process_info.hProcess, 1);
            WaitForSingleObject(process_info.hProcess, 5000);
            CloseHandle(process_info.hThread);
            CloseHandle(process_info.hProcess);
            CloseHandle(job_handle);
            throw std::runtime_error("grim_text_server: ResumeThread failed for train_gpu worker "
                                     "(GetLastError=" + std::to_string(error) + ")");
        }

        process_info_ = process_info;
        job_handle_ = job_handle;
#else
        const pid_t pid = fork();
        if (pid < 0) {
            throw std::runtime_error("grim_text_server: fork failed while launching train_gpu");
        }
        if (pid == 0) {
            const std::string worker_port_text = std::to_string(worker_port);
            execl(train_gpu_path.c_str(), train_gpu_path.c_str(), "--inference",
                  "--inference-worker-port", worker_port_text.c_str(), static_cast<char*>(nullptr));
            _exit(127);
        }
        child_pid_ = pid;
#endif
    }

    bool isRunning() const {
#ifdef _WIN32
        if (!process_info_.hProcess) {
            return false;
        }
        DWORD exit_code = 0;
        if (!GetExitCodeProcess(process_info_.hProcess, &exit_code)) {
            return false;
        }
        return exit_code == STILL_ACTIVE;
#else
        if (child_pid_ <= 0) {
            return false;
        }
        int status = 0;
        const pid_t result = waitpid(child_pid_, &status, WNOHANG);
        return result == 0;
#endif
    }

    void stop() {
#ifdef _WIN32
        if (process_info_.hProcess) {
            DWORD exit_code = 0;
            if (GetExitCodeProcess(process_info_.hProcess, &exit_code) && exit_code == STILL_ACTIVE) {
                TerminateProcess(process_info_.hProcess, 0);
                WaitForSingleObject(process_info_.hProcess, 5000);
            }
            CloseHandle(process_info_.hProcess);
            process_info_.hProcess = nullptr;
        }
        if (process_info_.hThread) {
            CloseHandle(process_info_.hThread);
            process_info_.hThread = nullptr;
        }
        if (job_handle_) {
            CloseHandle(job_handle_);
            job_handle_ = nullptr;
        }
#else
        if (child_pid_ > 0) {
            if (isRunning()) {
                kill(child_pid_, SIGTERM);
                waitpid(child_pid_, nullptr, 0);
            }
            child_pid_ = -1;
        }
#endif
    }

private:
#ifdef _WIN32
    PROCESS_INFORMATION process_info_{};
    HANDLE job_handle_ = nullptr;
#else
    pid_t child_pid_ = -1;
#endif
};

void waitForWorkerReady(const TrainGpuProcess& worker, int worker_port) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(10);
    while (std::chrono::steady_clock::now() < deadline) {
        if (!worker.isRunning()) {
            throw std::runtime_error("grim_text_server: train_gpu exited before inference worker became ready");
        }

        httplib::Client client("127.0.0.1", worker_port);
        client.set_connection_timeout(0, 200000);
        client.set_read_timeout(1, 0);
        auto response = client.Get("/internal/status");
        if (response && response->status == 200) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    throw std::runtime_error("grim_text_server: timed out waiting for train_gpu inference worker readiness");
} 

void forwardToWorker(
    int worker_port,
    const char* worker_path,
    const httplib::Request& req,
    httplib::Response& res)
{
    httplib::Client client("127.0.0.1", worker_port);
    client.set_connection_timeout(2, 0);
    client.set_read_timeout(600, 0);
    auto worker_response = client.Post(worker_path, req.body, "application/json");
    if (!worker_response) {
        res.status = 502;
        res.set_content(json({{"error", "train_gpu inference worker did not return a response"}}).dump(),
                        "application/json");
        return;
    }

    res.status = worker_response->status;
    std::string content_type = worker_response->get_header_value("Content-Type");
    if (content_type.empty()) {
        content_type = "application/json";
    }
    res.set_content(worker_response->body, content_type.c_str());
}

void forwardStatusFromWorker(int worker_port, httplib::Response& res) {
    httplib::Client client("127.0.0.1", worker_port);
    client.set_connection_timeout(2, 0);
    client.set_read_timeout(5, 0);
    auto worker_response = client.Get("/internal/status");
    if (!worker_response) {
        res.status = 502;
        res.set_content(json({{"error", "train_gpu inference worker status unavailable"}}).dump(),
                        "application/json");
        return;
    }

    res.status = worker_response->status;
    res.set_content(worker_response->body, "application/json");
}

} // namespace

int main(int argc, char** argv)
{
#ifdef _WIN32
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);
#endif

    try {
        const BridgeOptions options = parseBridgeOptions(argc, argv);
        const auto train_gpu_path = resolveTrainGpuPath(options.train_gpu_path);

        std::cout << "========================================\n";
        std::cout << "  GRIM-text HTTP Bridge v1.0.0\n";
        std::cout << "  Ollama-compatible API\n";
        std::cout << "========================================\n";
        std::cout << "[GRIM-text] Launching train_gpu inference owner: " << train_gpu_path.string() << "\n";
        std::cout << "[GRIM-text] Public port: " << options.public_port << "\n";
        std::cout << "[GRIM-text] Internal worker port: " << options.worker_port << "\n";
        std::cout << "[GRIM-text] Model/vocabulary config owner: train_gpu -> ai_config.json\n";

        TrainGpuProcess worker;
        worker.start(train_gpu_path, options.worker_port);
        waitForWorkerReady(worker, options.worker_port);
        std::cout << "[GRIM-text] ✓ train_gpu inference worker ready\n";

        httplib::Server svr;

        svr.Get("/", [&](const httplib::Request&, httplib::Response& res) {
            json response = { 
                {"status", "ok"},
                {"model", "grim-text"},
                {"version", "1.0.0"},
                {"runtime_owner", "train_gpu"}
            };
            res.set_content(response.dump(), "application/json");
        });

        svr.Get("/api/tags", [&](const httplib::Request&, httplib::Response& res) {
            json response = {
                {"models", json::array({
                    {{"name", "grim-text"}, {"modified_at", "2025-11-05T00:00:00Z"},
                     {"size", 1024 * 1024 * 768}, {"digest", "grim-text-v1"}}
                })}
            };
            res.set_content(response.dump(), "application/json");
        });

        svr.Get("/api/status", [&](const httplib::Request&, httplib::Response& res) {
            forwardStatusFromWorker(options.worker_port, res);
        });

        svr.Post("/api/generate", [&](const httplib::Request& req, httplib::Response& res) {
            forwardToWorker(options.worker_port, "/internal/generate", req, res);
        });

        svr.Post("/api/chat", [&](const httplib::Request& req, httplib::Response& res) {
            forwardToWorker(options.worker_port, "/internal/chat", req, res);
        });

        std::cout << "[GRIM-text] Starting HTTP bridge on http://127.0.0.1:"
                  << options.public_port << "\n";
        std::cout << "[GRIM-text] Press Ctrl+C to stop.\n";

        svr.listen("127.0.0.1", options.public_port);
        worker.stop();
    } catch (const std::exception& e) {
        std::cerr << "[GRIM-text] ERROR: " << e.what() << "\n";
#ifdef _WIN32
        WSACleanup();
#endif
        return 1;
    }

#ifdef _WIN32
    WSACleanup();
#endif
    return 0;
}
 