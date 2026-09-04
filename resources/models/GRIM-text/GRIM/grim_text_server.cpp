//======================================================//
//  GRIM-text HTTP Server
//  Ollama-compatible HTTP bridge for MMO-managed model
//  workers. Bridge liveness is independent of model readiness.
//
//  Public endpoint: http://127.0.0.1:11435
//  Worker lifecycle: owned by the MMO model loader
//======================================================//

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#pragma comment(lib, "ws2_32.lib")
#endif

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#include <httplib.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace {

struct BridgeOptions {
    int public_port = 11435;
    int worker_port = 11436;
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
    httplib::Client client("127.0.0.1", worker_port);
    client.set_connection_timeout(0, 200000);
    client.set_read_timeout(1, 0);
    auto response = client.Get("/internal/status");
    return response && response->status == 200;
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
        res.status = 503;
        res.set_content(json({
                            {"error", "router_unavailable"},
                            {"message", "The MMO router model is not loaded"}
                        }).dump(),
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
    client.set_connection_timeout(0, 200000);
    client.set_read_timeout(1, 0);
    auto worker_response = client.Get("/internal/status");

    json response = {
        {"status", "ok"},
        {"service", "grim_text_server"},
        {"router_status", "unloaded"}
    };
    if (worker_response && worker_response->status == 200) {
        response["router_status"] = "ready";
        auto worker_status = json::parse(worker_response->body, nullptr, false);
        if (!worker_status.is_discarded()) {
            response["worker"] = std::move(worker_status);
        }
    }

    res.status = 200;
    res.set_content(response.dump(), "application/json");
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
        std::cout << "[GRIM-text] Model lifecycle owner: MMO model loader\n";
        std::cout << "[GRIM-text] Router state: unloaded\n";

        httplib::Server svr;

        svr.Get("/", [&](const httplib::Request&, httplib::Response& res) {
            json response = { 
                {"status", "ok"},
                {"service", "grim_text_server"},
                {"version", "1.0.0"},
                {"router_status", workerReady(options.worker_port) ? "ready" : "unloaded"},
                {"runtime_owner", "mmo_model_loader"}
            };
            res.set_content(response.dump(), "application/json");
        });

        svr.Get("/api/tags", [&](const httplib::Request&, httplib::Response& res) {
            json models = json::array();
            if (workerReady(options.worker_port)) {
                models.push_back({
                    {"name", "grim-text-router"},
                    {"status", "ready"}
                });
            }
            json response = {{"models", std::move(models)}};
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
