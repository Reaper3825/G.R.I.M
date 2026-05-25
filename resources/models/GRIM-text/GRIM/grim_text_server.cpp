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

struct ServerOptions {
    int public_port = 11435;
    int worker_port = 11436;
    std::filesystem::path train_gpu_path;
};

std::filesystem::path currentExecutablePath(char** argv) {
    if (!argv || !argv[0] || std::string(argv[0]).empty()) {
        throw std::runtime_error("grim_text_server: argv[0] is empty; cannot locate train_gpu.exe");
    }
    std::filesystem::path exe_path(argv[0]);
    if (exe_path.is_relative()) {
        exe_path = std::filesystem::current_path() / exe_path;
    }
    return std::filesystem::weakly_canonical(exe_path);
}

std::string pathListForError(const std::vector<std::filesystem::path>& paths) {
    std::ostringstream oss;
    for (const auto& path : paths) {
        oss << "\n  - " << path.string();
    }
    return oss.str();
}

std::filesystem::path resolveTrainGpuPath(char** argv, const std::filesystem::path& explicit_path) {
    if (!explicit_path.empty()) {
        if (!std::filesystem::exists(explicit_path)) {
            throw std::runtime_error("grim_text_server: --train-gpu path does not exist: " +
                                     explicit_path.string());
        }
        return std::filesystem::weakly_canonical(explicit_path);
    }

    const auto server_exe = currentExecutablePath(argv);
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

ServerOptions parseOptions(int argc, char** argv) {
    ServerOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config") {
            throw std::runtime_error("grim_text_server: --config is not supported; train_gpu owns startup config loading");
        }
        if (arg == "--port") {
            if (i + 1 >= argc) {
                throw std::runtime_error("grim_text_server: --port requires a value");
            }
            options.public_port = std::stoi(argv[++i]);
            continue;
        }
        if (arg == "--worker-port") {
            if (i + 1 >= argc) {
                throw std::runtime_error("grim_text_server: --worker-port requires a value");
            }
            options.worker_port = std::stoi(argv[++i]);
            continue;
        }
        if (arg == "--train-gpu") {
            if (i + 1 >= argc) {
                throw std::runtime_error("grim_text_server: --train-gpu requires a value");
            }
            options.train_gpu_path = std::filesystem::path(argv[++i]);
            continue;
        }
        throw std::runtime_error("grim_text_server: unsupported argument: " + arg);
    }
    if (options.public_port <= 0 || options.public_port > 65535) {
        throw std::runtime_error("grim_text_server: --port must be in 1..65535");
    }
    if (options.worker_port <= 0 || options.worker_port > 65535) {
        throw std::runtime_error("grim_text_server: --worker-port must be in 1..65535");
    }
    if (options.public_port == options.worker_port) {
        throw std::runtime_error("grim_text_server: --port and --worker-port must be different");
    }
    return options;
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
            0,
            nullptr,
            nullptr,
            &startup_info,
            &process_info);

        if (!created) {
            throw std::runtime_error("grim_text_server: CreateProcessW failed for " + train_gpu_path.string() +
                                     " (GetLastError=" + std::to_string(GetLastError()) + ")");
        }

        process_info_ = process_info;
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

} // namespace

int main(int argc, char** argv)
{
#ifdef _WIN32
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);
#endif

    try {
        const ServerOptions options = parseOptions(argc, argv);
        const auto train_gpu_path = resolveTrainGpuPath(argv, options.train_gpu_path);

        std::cout << "========================================\n";
        std::cout << "  GRIM-text HTTP Bridge v1.0.0\n";
        std::cout << "  Ollama-compatible API\n";
        std::cout << "========================================\n";
        std::cout << "[GRIM-text] Launching train_gpu inference owner: " << train_gpu_path.string() << "\n";
        std::cout << "[GRIM-text] Internal worker port: " << options.worker_port << "\n";

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

        svr.Post("/api/generate", [&](const httplib::Request& req, httplib::Response& res) {
            forwardToWorker(options.worker_port, "/internal/generate", req, res);
        });

        svr.Post("/api/chat", [&](const httplib::Request& req, httplib::Response& res) {
            forwardToWorker(options.worker_port, "/internal/chat", req, res);
        });

        std::cout << "[GRIM-text] Starting HTTP bridge on http://127.0.0.1:" << options.public_port << "\n";
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
