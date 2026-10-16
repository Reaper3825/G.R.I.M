//======================================================//
//  GRIM-text HTTP Server
//  Ollama-compatible HTTP bridge for MMO-managed model
//  workers. Bridge liveness is independent of model readiness.
//
//  Public endpoint: http://127.0.0.1:11435
//  Worker lifecycle: bridge-owned child; MMO owns the bridge
//======================================================//

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#pragma comment(lib, "ws2_32.lib")
#endif

#include <iostream>
#include <chrono>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include "../../../../core/grim_text_server_api.hpp"
#include <httplib.h>
#include <nlohmann/json.hpp>
#include "grim_text_worker_process.hpp"

using json = nlohmann::json;

namespace {

struct BridgeOptions {
    int public_port = GRIM::ServerAPI::kDefaultPublicPort;
    int worker_port = GRIM::ServerAPI::kDefaultWorkerPort;
    std::filesystem::path worker_executable;
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
        } else if (arg == "--worker-exe") {
            options.worker_executable = requireValue("--worker-exe");
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

bool workerReady(int worker_port) {
    httplib::Client client(GRIM::ServerAPI::kLoopbackHost, worker_port);
    client.set_connection_timeout(0, 200000);
    client.set_read_timeout(1, 0);
    auto response = client.Get(GRIM::ServerAPI::Worker::kStatus);
    return response && response->status == 200;
}

class WorkerRuntime {
public:
    explicit WorkerRuntime(const BridgeOptions& options) : port_(options.worker_port) {
        // All existing launchers use the GRIM root as cwd. Also support a cwd
        // below it, without changing train_gpu's canonical config discovery.
        auto root = std::filesystem::current_path();
        while (!std::filesystem::is_regular_file(root / "ai_config.json")) {
            const auto parent = root.parent_path();
            if (parent == root) throw std::runtime_error("Cannot locate GRIM root (ai_config.json)");
            root = parent;
        }
        std::filesystem::current_path(root);
        auto executable = options.worker_executable;
        if (executable.empty()) {
#ifdef _WIN32
            executable = "resources/models/GRIM-text/training/TrainingLoop/build/Release/train_gpu.exe";
#else
            executable = "resources/models/GRIM-text/training/TrainingLoop/build/train_gpu";
#endif
        }
        executable = std::filesystem::absolute(executable);
        log_path_ = root / "logs/grim_text_worker.log";
        if (workerReady(port_)) {
            throw std::runtime_error("Inference worker port is already serving a worker: " + std::to_string(port_));
        }
        std::cout << "[GRIM-text] Starting worker: " << executable.string()
                  << " --inference --inference-worker-port " << port_ << std::endl;
        std::cout << "[GRIM-text] Worker log: " << log_path_.string() << std::endl;
        process_.start(executable, port_, log_path_);
    }

    std::string state() {
        if (!running()) return "failed";
        return workerReady(port_) ? "ready" : "loading";
    }

    void waitUntilReady() {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
        while (running()) {
            if (workerReady(port_)) return;
            if (std::chrono::steady_clock::now() >= deadline) {
                throw std::runtime_error("Inference worker is still loading after 120 seconds. See " + log_path_.string());
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        throw std::runtime_error("Inference worker exited. See " + log_path_.string());
    }

private:
    bool running() {
        std::lock_guard<std::mutex> lock(mutex_);
        return process_.running();
    }
    int port_;
    std::filesystem::path log_path_;
    std::mutex mutex_;
    GrimTextWorkerProcess process_;
};

void forwardToWorker(
    WorkerRuntime& worker,
    int worker_port,
    const char* worker_path,
    const httplib::Request& req,
    httplib::Response& res)
{
    try {
        worker.waitUntilReady();
    } catch (const std::exception& e) {
        res.status = 503;
        res.set_content(json({{"error", "router_unavailable"}, {"message", e.what()}}).dump(),
                        GRIM::ServerAPI::kJsonContentType);
        return;
    }
    httplib::Client client(GRIM::ServerAPI::kLoopbackHost, worker_port);
    client.set_connection_timeout(2, 0);
    client.set_read_timeout(std::string(worker_path)==GRIM::ServerAPI::Worker::kInspect?3600:600, 0);
    auto worker_response = client.Post(worker_path, req.body, GRIM::ServerAPI::kJsonContentType);
    if (!worker_response) {
        res.status = 503;
        res.set_content(json({
                            {"error", "router_unavailable"},
                            {"message", "The MMO router model is not loaded"}
                        }).dump(),
                        GRIM::ServerAPI::kJsonContentType);
        return;
    }

    res.status = worker_response->status;
    std::string content_type = worker_response->get_header_value("Content-Type");
    if (content_type.empty()) {
        content_type = GRIM::ServerAPI::kJsonContentType;
    }
    res.set_content(worker_response->body, content_type.c_str());
}

void forwardStatusFromWorker(WorkerRuntime& worker, int worker_port, httplib::Response& res) {
    httplib::Client client(GRIM::ServerAPI::kLoopbackHost, worker_port);
    client.set_connection_timeout(0, 200000);
    client.set_read_timeout(1, 0);
    auto worker_response = client.Get(GRIM::ServerAPI::Worker::kStatus);

    json response = {
        {"status", "ok"},
        {"service", GRIM::ServerAPI::kServiceName},
        {"router_status", worker.state()}
    };
    if (worker_response && worker_response->status == 200) {
        response["router_status"] = "ready";
        auto worker_status = json::parse(worker_response->body, nullptr, false);
        if (!worker_status.is_discarded()) {
            response["worker"] = std::move(worker_status);
        }
    }

    res.status = 200;
    res.set_content(response.dump(), GRIM::ServerAPI::kJsonContentType);
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

        std::cout << "========================================\n";
        std::cout << "  GRIM-text HTTP Bridge v1.0.0\n";
        std::cout << "  Ollama-compatible API\n";
        std::cout << "========================================\n";
        std::cout << "[GRIM-text] Public port: " << options.public_port << "\n";
        std::cout << "[GRIM-text] Internal worker port: " << options.worker_port << "\n";
        std::cout << "[GRIM-text] Worker lifecycle owner: HTTP bridge (MMO-managed)\n";

        httplib::Server svr;
        // Reserve the public port before spawning a GPU worker. Duplicate bridge
        // launches must fail without creating an orphan or loading a second model.
        if (!svr.bind_to_port(GRIM::ServerAPI::kLoopbackHost, options.public_port)) {
            throw std::runtime_error("Cannot bind public bridge port " + std::to_string(options.public_port));
        }
        WorkerRuntime worker(options);

        // Bridge liveness does not depend on a loaded model or its execution lock.
        svr.Get(GRIM::ServerAPI::Public::kHealth, [](const httplib::Request&, httplib::Response& res) {
            res.set_content(json({{"status", "ok"}, {"service", GRIM::ServerAPI::kServiceName}}).dump(),
                            GRIM::ServerAPI::kJsonContentType);
        });

        svr.Post(GRIM::ServerAPI::Public::kTokenizerRun, [&](const httplib::Request& req, httplib::Response& res) {
            forwardToWorker(worker, options.worker_port, GRIM::ServerAPI::Worker::kTokenizerRun, req, res);
        });

        svr.Post(GRIM::ServerAPI::Public::kTokenizerEncode, [&](const httplib::Request& req, httplib::Response& res) {
            forwardToWorker(worker, options.worker_port, GRIM::ServerAPI::Worker::kTokenizerEncode, req, res);
        });

        svr.Get(GRIM::ServerAPI::Public::kRoot, [&](const httplib::Request&, httplib::Response& res) {
            json response = { 
                {"status", "ok"},
                {"service", GRIM::ServerAPI::kServiceName},
                {"version", "1.0.0"},
                {"router_status", worker.state()},
                {"runtime_owner", "http_bridge"}
            };
            res.set_content(response.dump(), GRIM::ServerAPI::kJsonContentType);
        });

        svr.Get(GRIM::ServerAPI::Public::kTags, [&](const httplib::Request&, httplib::Response& res) {
            json models = json::array();
            if (workerReady(options.worker_port)) {
                models.push_back({
                    {"name", "grim-text-router"},
                    {"status", "ready"}
                });
            }
            json response = {{"models", std::move(models)}};
            res.set_content(response.dump(), GRIM::ServerAPI::kJsonContentType);
        });

        svr.Get(GRIM::ServerAPI::Public::kStatus, [&](const httplib::Request&, httplib::Response& res) {
            forwardStatusFromWorker(worker, options.worker_port, res);
        });

        svr.Post(GRIM::ServerAPI::Public::kInspect, [&](const httplib::Request& req, httplib::Response& res) {
            forwardToWorker(worker, options.worker_port, GRIM::ServerAPI::Worker::kInspect, req, res);
        });

        svr.Post(GRIM::ServerAPI::Public::kGenerate, [&](const httplib::Request& req, httplib::Response& res) {
            forwardToWorker(worker, options.worker_port, GRIM::ServerAPI::Worker::kGenerate, req, res);
        });

        svr.Post(GRIM::ServerAPI::Public::kChat, [&](const httplib::Request& req, httplib::Response& res) {
            forwardToWorker(worker, options.worker_port, GRIM::ServerAPI::Worker::kChat, req, res);
        });

        std::cout << "[GRIM-text] Starting HTTP bridge on http://127.0.0.1:"
                  << options.public_port << "\n";
        std::cout << "[GRIM-text] Press Ctrl+C to stop.\n";

        if (!svr.listen_after_bind()) throw std::runtime_error("HTTP bridge listener failed");
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
