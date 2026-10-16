// Host-only lifecycle/transport test. The child is this test executable, never
// train_gpu; no checkpoint, CUDA context, or training loop is loaded.
#define main grim_text_bridge_main
#include "../resources/models/GRIM-text/GRIM/grim_text_server.cpp"
#undef main

#include <fstream>

namespace fs = std::filesystem;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

int unusedPort() {
    httplib::Server probe;
    const int port = probe.bind_to_any_port("127.0.0.1");
    require(port > 0, "Cannot reserve test port");
    std::thread listener([&] { probe.listen_after_bind(); });
    probe.wait_until_ready();
    probe.stop();
    listener.join();
    return port;
}

int main(int argc, char** argv) {
#ifdef _WIN32
    WSADATA data;
    WSAStartup(MAKEWORD(2, 2), &data);
#endif
    if (argc > 1 && std::string(argv[1]) == "--inference") {
        if (fs::path(argv[0]).stem().string() == "worker_fail") return 17;
        if (argc != 4 || std::string(argv[2]) != "--inference-worker-port") return 18;
        if (!fs::is_regular_file("ai_config.json")) return 19;
        std::cout << "fake worker launched in config root" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        httplib::Server worker;
        worker.Get(GRIM::ServerAPI::Worker::kStatus, [](const auto&, auto& res) {
            res.set_content("{\"status\":\"ready\"}", "application/json");
        });
        worker.Post(GRIM::ServerAPI::Worker::kChat, [](const auto& req, auto& res) {
            res.set_content(req.body, "application/json");
        });
        return worker.listen("127.0.0.1", std::stoi(argv[3])) ? 0 : 20;
    }
    const auto original_cwd = fs::current_path();
    const auto executable = fs::absolute(argv[0]);
    const auto root = original_cwd / ".codex_tmp/bridge worker test";
    try {
        fs::create_directories(root / "nested");
        std::ofstream(root / "ai_config.json") << "{}";
        fs::current_path(root / "nested");
        BridgeOptions options;
        options.worker_port = unusedPort();
        options.worker_executable = executable;
        {
            WorkerRuntime worker(options);
            require(fs::current_path() == root, "Worker cwd did not resolve to config root");
            httplib::Request req;
            req.body = "{\"prompt\":\"test\"}";
            httplib::Response res;
            forwardToWorker(worker, options.worker_port, GRIM::ServerAPI::Worker::kChat, req, res);
            require(res.status == 200 && res.body == req.body, "Bridge did not forward request");
            require(worker.state() == "ready", "Request did not wait for worker readiness");
            // Another bridge must not take ownership of an existing worker.
            bool rejected = false;
            try { WorkerRuntime duplicate(options); } catch (const std::exception&) { rejected = true; }
            require(rejected, "Duplicate worker was allowed");
        }
        require(!workerReady(options.worker_port), "Owned worker survived bridge cleanup");
        options.worker_executable = root / ("worker_fail" + executable.extension().string());
        fs::copy_file(executable, options.worker_executable, fs::copy_options::overwrite_existing);
        {
            WorkerRuntime failed(options);
            httplib::Request req;
            httplib::Response res;
            forwardToWorker(failed, options.worker_port, GRIM::ServerAPI::Worker::kChat, req, res);
            require(res.status == 503, "Worker failure did not return 503");
            require(res.body.find("worker exited") != std::string::npos, "Missing worker failure diagnosis");
            require(res.body.find("grim_text_worker.log") != std::string::npos, "Missing worker log location");
            require(failed.state() == "failed", "Exited worker status was not failed");
        }
        options.worker_executable = root / "missing_worker.exe";
        bool missing_rejected = false;
        try { WorkerRuntime missing(options); } catch (const std::exception&) { missing_rejected = true; }
        require(missing_rejected, "Missing executable was allowed");
        fs::current_path(original_cwd);
        std::cout << "PASS: worker arguments, config cwd, readiness, forwarding, ownership, exit diagnostics, missing executable\n";
        return 0;
    } catch (const std::exception& e) {
        fs::current_path(original_cwd);
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
