#pragma once
// JSON diagnostics transport through the GRIM-text public bridge.
// Call from background work; this client owns no server lifecycle or model state.
#include "../core/grim_text_server_api.hpp"
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <utility>

namespace GRIMText {
class TokenizerClient {
public:
    explicit TokenizerClient(std::string url) : url_(std::move(url)) {}
    bool isServerRunning() const {
        try {
            httplib::Client client(url_);
            client.set_connection_timeout(1, 0);
            client.set_read_timeout(1, 0);
            const auto response = client.Get(GRIM::ServerAPI::Public::kHealth);
            if (!response || response->status != 200) return false;
            const auto status = nlohmann::json::parse(response->body);
            return status.value("service", std::string{}) == GRIM::ServerAPI::kServiceName;
        } catch (...) { return false; }
    }
    struct TokenizerResult {
        bool success = false;
        int total_vocab_size = 0;
        int unigram_vocab_size = 0;
        int byte_vocab_size = 0;
        int atom_vocab_size = 0;
        int numeric_vocab_size = 0;
        int newline_vocab_size = 0;
        int special_token_count = 0;
        int pad_id = 0;
        int unk_id = 0;
        int bos_id = 0;
        int eos_id = 0;
        std::string vocab_path;
        int validation_tests_passed = 0;
        int validation_tests_total = 0;
        double validation_time_ms = 0.0;
        std::string error;
        std::string phase;
        std::vector<std::string> failures;
    };
    
    TokenizerResult runTokenizer() {
        TokenizerResult result;
        try {
            httplib::Client connection(url_);
            connection.set_connection_timeout(2, 0);
            auto* client = &connection;
            client->set_read_timeout(60);  // Tokenizer may take a while
            
            const auto body = nlohmann::json::object();
            
            std::string bodyStr = body.dump();
            auto res = client->Post(GRIM::ServerAPI::Public::kTokenizerRun, bodyStr, GRIM::ServerAPI::kJsonContentType);
            
            if (!res) {
                result.error = "Connection failed";
                result.phase = "connection";
                return result;
            }
            
            auto j = nlohmann::json::parse(res->body);
            if (res->status != 200 && res->status != 422) {
                result.error = j.value("message", j.value("error", std::string("Tokenizer request failed")));
                return result;
            }
            std::string status = j.value("status", "");
            result.success = (status == "success");
            result.total_vocab_size = j.value("total_vocab_size", 0);
            result.unigram_vocab_size = j.value("unigram_vocab_size", 0);
            result.byte_vocab_size = j.value("byte_vocab_size", 0);
            result.atom_vocab_size = j.value("atom_vocab_size", 0);
            result.numeric_vocab_size = j.value("numeric_vocab_size", 0);
            result.newline_vocab_size = j.value("newline_vocab_size", 0);
            result.special_token_count = j.value("special_token_count", 0);
            result.pad_id = j.value("pad_id", 0);
            result.unk_id = j.value("unk_id", 0);
            result.bos_id = j.value("bos_id", 0);
            result.eos_id = j.value("eos_id", 0);
            result.vocab_path = j.value("vocab_path", "");
            result.validation_tests_passed = j.value("validation_tests_passed", 0);
            result.validation_tests_total = j.value("validation_tests_total", 0);
            result.validation_time_ms = j.value("validation_time_ms", 0.0);
            result.error = j.value("error", "");
            result.phase = j.value("phase", "");
            
            if (j.contains("failures")) {
                for (const auto& f : j["failures"]) {
                    result.failures.push_back(f.get<std::string>());
                }
            }
            
            return result;
        } catch (const std::exception& e) {
            result.error = e.what();
            result.phase = "client";
            return result;
        }
    }
    
    // Encode text using tokenizer
    struct EncodeToken {
        int id = 0;
        std::string piece;
        std::string type;  // "special", "byte", "atom", "unigram"
    };

    struct EncodeResult {
        bool success = false;
        int token_count = 0;
        std::string input_text;
        std::string decoded_text;
        double encode_time_ms = 0.0;
        int total_vocab_size = 0;
        std::string error;
        std::vector<EncodeToken> tokens;
    };

    EncodeResult encodeText(const std::string& text) {
        EncodeResult result;
        if (text.empty()) {
            result.error = "Text is empty";
            return result;
        }
        try {
            httplib::Client connection(url_);
            connection.set_connection_timeout(2, 0);
            auto* client = &connection;
            client->set_read_timeout(30);

            nlohmann::json body;
            body["text"] = text;

            std::string bodyStr = body.dump();
            auto res = client->Post(GRIM::ServerAPI::Public::kTokenizerEncode, bodyStr, GRIM::ServerAPI::kJsonContentType);

            if (!res) {
                result.error = "Connection failed";
                return result;
            }

            auto j = nlohmann::json::parse(res->body);
            if (res->status != 200 && res->status != 422) {
                result.error = j.value("message", j.value("error", std::string("Tokenizer request failed")));
                return result;
            }
            std::string status = j.value("status", "");
            result.success = (status == "success");
            result.token_count = j.value("token_count", 0);
            result.input_text = j.value("input_text", "");
            result.decoded_text = j.value("decoded_text", "");
            result.encode_time_ms = j.value("encode_time_ms", 0.0);
            result.total_vocab_size = j.value("total_vocab_size", 0);
            result.error = j.value("error", "");

            if (j.contains("tokens") && j["tokens"].is_array()) {
                for (const auto& tok : j["tokens"]) {
                    EncodeToken et;
                    et.id = tok.value("id", 0);
                    et.piece = tok.value("piece", "");
                    et.type = tok.value("type", "");
                    result.tokens.push_back(std::move(et));
                }
            }

            return result;
        } catch (const std::exception& e) {
            result.error = e.what();
            return result;
        }
    }

private:
    std::string url_;
};
} // namespace GRIMText
