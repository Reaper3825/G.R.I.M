// Host-only transport test; does not start GRIM or a model worker.
#include "control/tokenizer_client.hpp"
#include <atomic>
#include <future>
#include <iostream>
#include <thread>
#include <stdexcept>

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    using json = nlohmann::json;
    httplib::Server server;
    std::atomic<int> mode{0};
    server.Get("/health", [&](const httplib::Request&, httplib::Response& response) {
        response.set_content(json({{"service", mode == 1 ? "other_server" : "grim_text_server"}}).dump(), "application/json");
    });
    server.Post("/api/tokenizer/run", [&](const httplib::Request& request, httplib::Response& response) {
        if (json::parse(request.body) != json::object()) {
            response.status = 400;
            response.set_content("{\"error\":\"Validation requires an object\"}", "application/json");
            return;
        }
        response.set_content(json({{"status", "success"}, {"total_vocab_size", 1000},
            {"numeric_vocab_size", 46}, {"validation_tests_passed", 6}, {"validation_tests_total", 6}}).dump(), "application/json");
    });
    const std::string text = "quotes \" and \\ and\nUTF-8: \xC3\xA9";
    server.Post("/api/tokenizer/encode", [&](const httplib::Request& request, httplib::Response& response) {
        const auto body = json::parse(request.body);
        if (mode == 2 || mode == 3) {
            response.status = mode == 2 ? 409 : 503;
            response.set_content(json({{"error", "unavailable"}, {"message", "Load a model or retry"}}).dump(), "application/json");
        } else if (mode == 4) {
            response.set_content("not json", "application/json");
        } else if (mode == 5) {
            response.status = 422;
            response.set_content("{\"status\":\"error\",\"error\":\"validation failed\"}", "application/json");
        } else {
            response.set_content(json({{"status", "success"}, {"input_text", body.at("text")},
                {"decoded_text", body.at("text")}, {"token_count", 1},
                {"tokens", json::array({{{"id", 319}, {"piece", body.at("text")}, {"type", "unigram"}}})}}).dump(), "application/json");
        }
    });
    const int port = server.bind_to_any_port("127.0.0.1");
    require(port > 0, "bind failed");
    auto listener = std::async(std::launch::async, [&] { return server.listen_after_bind(); });
    while (!server.is_running()) std::this_thread::yield();
    try {
        GRIMText::TokenizerClient client("http://127.0.0.1:" + std::to_string(port));
        require(client.isServerRunning(), "bridge health failed");
        mode = 1;
        require(!client.isServerRunning(), "accepted a different service");
        mode = 0;
        const auto validation = client.runTokenizer();
        require(validation.success && validation.numeric_vocab_size == 46 && validation.validation_tests_passed == 6,
                "validation request/result contract failed");
        const auto encoded = client.encodeText(text);
        require(encoded.success && encoded.input_text == text && encoded.decoded_text == text &&
                encoded.tokens.size() == 1 && encoded.tokens[0].piece == text, "encode text was corrupted");
        for (int failure_mode : {2, 3, 4, 5}) {
            mode = failure_mode;
            const auto failed = client.encodeText(text);
            require(!failed.success && !failed.error.empty(), "lost server/JSON failure");
        }
        require(!client.encodeText("").success, "accepted empty text");
        server.stop();
        listener.get();
        require(!client.isServerRunning(), "reported stopped bridge online");
        require(!client.runTokenizer().success, "reported connection failure as success");
    } catch (...) {
        server.stop();
        if (listener.valid()) listener.get();
        throw;
    }
    std::cout << "Tokenizer transport checks passed\n";
}
