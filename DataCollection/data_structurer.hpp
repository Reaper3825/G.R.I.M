//======================================================//
//  GRIM Data Structurer
//  Converts raw prose into structured Q/A training data
//  using an LLM backend (Ollama / GRIM submodel).
//
//  Pipeline step: runs after preprocessing, before cache write.
//  Sends cleaned text to the LLM and parses structured output.
//======================================================//

#pragma once

#include <string>
#include <vector>
#include <functional>
#include <future>
#include <mutex>
#include <atomic>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

namespace GRIM {
namespace DataCollection {

//======================================================//
//  Configuration
//======================================================//

struct DataStructuringConfig {
    bool enabled = true;
    std::string mode = "qa";            // "qa", "conversation", "instruct", "raw"
    std::string ollama_model;           // Dedicated model — MUST be set if enabled
    std::string ollama_url = "http://127.0.0.1:11434";
    int parallel_requests = 4;
    int timeout_ms = 60000;
    int max_input_chars = 3000;
    bool skip_already_structured = true;
    int max_entries_per_run = 0;        // 0 = unlimited

    static DataStructuringConfig fromJson(const nlohmann::json& j) {
        DataStructuringConfig c;
        c.enabled = j.value("enabled", true);
        c.mode = j.value("mode", "qa");
        c.ollama_model = j.value("ollama_model", "");
        c.parallel_requests = j.value("parallel_requests", 4);
        c.timeout_ms = j.value("timeout_ms", 60000);
        c.max_input_chars = j.value("max_input_chars", 3000);
        c.skip_already_structured = j.value("skip_already_structured", true);
        c.max_entries_per_run = j.value("max_entries_per_run", 0);
        return c;
    }
};

//======================================================//
//  Prompt Templates
//======================================================//

namespace prompts {

inline std::string qaSystemPrompt() {
    return R"(You are a training data formatter. Your job is to read a passage of text and generate question-and-answer pairs from it.

Rules:
1. Generate an appropriate number of Q/A pairs based on the content density (1-5 pairs).
2. Questions should be specific, clear, and answerable from the text.
3. Answers should be thorough but concise, using information from the text.
4. Cover the key topics and facts in the passage.
5. Output ONLY a JSON array — no markdown, no explanation, no preamble.

Output format (strict JSON array):
[{"question": "...", "answer": "..."}, {"question": "...", "answer": "..."}])";
}

inline std::string conversationSystemPrompt() {
    return R"(You are a training data formatter. Your job is to read a passage of text and generate a natural conversation about it between a human and an assistant.

Rules:
1. The human asks questions or makes comments about the content.
2. The assistant provides helpful, accurate responses using the text.
3. Generate 2-4 turns of conversation.
4. Output ONLY a JSON array — no markdown, no explanation, no preamble.

Output format (strict JSON array):
[{"role": "human", "content": "..."}, {"role": "assistant", "content": "..."}, ...])";
}

inline std::string instructSystemPrompt() {
    return R"(You are a training data formatter. Your job is to read a passage of text and generate an instruction-response pair from it.

Rules:
1. Create an instruction that a user might give to an AI that would require the knowledge in the passage.
2. Create a response that uses the passage content to fulfill the instruction.
3. Output ONLY a JSON object — no markdown, no explanation, no preamble.

Output format (strict JSON object):
{"instruction": "...", "response": "..."})";
}

inline std::string getSystemPrompt(const std::string& mode) {
    if (mode == "qa") return qaSystemPrompt();
    if (mode == "conversation") return conversationSystemPrompt();
    if (mode == "instruct") return instructSystemPrompt();
    return qaSystemPrompt();
}

} // namespace prompts

//======================================================//
//  CURL Helpers
//======================================================//

namespace detail {

struct CurlResponse {
    std::string body;
    long status_code = 0;
    std::string error;
};

inline size_t curlWriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t realsize = size * nmemb;
    auto* str = static_cast<std::string*>(userp);
    const size_t MAX_SIZE = 2 * 1024 * 1024; // 2MB limit for LLM responses
    if (str->size() + realsize > MAX_SIZE) return 0;
    str->append(static_cast<char*>(contents), realsize);
    return realsize;
}

inline CurlResponse curlPost(const std::string& url,
                              const std::string& json_body,
                              int timeout_ms) {
    CurlResponse resp;
    CURL* curl = curl_easy_init();
    if (!curl) {
        resp.error = "Failed to initialize CURL";
        return resp;
    }

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_body.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(json_body.size()));
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curlWriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp.body);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, static_cast<long>(timeout_ms));
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        resp.error = curl_easy_strerror(res);
    } else {
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &resp.status_code);
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return resp;
}

inline CurlResponse curlGet(const std::string& url, int timeout_ms = 5000) {
    CurlResponse resp;
    CURL* curl = curl_easy_init();
    if (!curl) {
        resp.error = "Failed to initialize CURL";
        return resp;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curlWriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp.body);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, static_cast<long>(timeout_ms));
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        resp.error = curl_easy_strerror(res);
    } else {
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &resp.status_code);
    }

    curl_easy_cleanup(curl);
    return resp;
}

} // namespace detail

//======================================================//
//  DataStructurer
//======================================================//

class DataStructurer {
public:
    explicit DataStructurer(const DataStructuringConfig& config)
        : config_(config) {}

    // Check if Ollama is reachable
    bool checkOllamaHealth() const {
        std::string url = config_.ollama_url + "/api/tags";
        auto resp = detail::curlGet(url, 5000);
        if (!resp.error.empty() || resp.status_code != 200) {
            std::cerr << "[DataStructurer] Ollama health check failed: "
                      << (resp.error.empty() ? "HTTP " + std::to_string(resp.status_code) : resp.error) << "\n";
            return false;
        }
        return true;
    }

    // Check if text is already in structured format (Q/A, conversation, etc.)
    static bool isAlreadyStructured(const std::string& text) {
        if (text.size() < 4) return false;
        // Check common structured prefixes
        if (text.compare(0, 2, "Q:") == 0) return true;
        if (text.compare(0, 6, "Human:") == 0) return true;
        if (text.compare(0, 5, "User:") == 0) return true;
        if (text.compare(0, 12, "Instruction:") == 0) return true;
        if (text.compare(0, 10, "### Human:") == 0) return true;
        if (text.compare(0, 8, "<|user|>") == 0) return true;
        return false;
    }

    // Structure a single text entry into Q/A formatted strings.
    // Returns one string per Q/A pair: "Q: ...\n\nA: ..."
    // On failure, returns empty vector (caller should keep original).
    std::vector<std::string> structureEntry(const std::string& content) const {
        // Truncate input if too long
        std::string input = content;
        if (static_cast<int>(input.size()) > config_.max_input_chars) {
            input = input.substr(0, config_.max_input_chars);
            // Try to break at last sentence boundary
            size_t last_period = input.rfind(". ");
            if (last_period != std::string::npos && last_period > input.size() / 2) {
                input = input.substr(0, last_period + 1);
            }
        }

        std::string system_prompt = prompts::getSystemPrompt(config_.mode);

        // Build Ollama /api/chat request
        nlohmann::json body;
        body["model"] = config_.ollama_model;
        body["messages"] = nlohmann::json::array({
            {{"role", "system"}, {"content", system_prompt}},
            {{"role", "user"}, {"content", "Generate structured training data from this text:\n\n" + input}}
        });
        body["stream"] = false;
        body["keep_alive"] = "30m";
        body["options"] = {
            {"temperature", 0.3},   // Low temp for consistent formatting
            {"num_predict", 2048},
            {"top_p", 0.9}
        };

        std::string endpoint = config_.ollama_url + "/api/chat";
        auto resp = detail::curlPost(endpoint, body.dump(), config_.timeout_ms);

        if (!resp.error.empty() || resp.status_code != 200) {
            std::cerr << "[DataStructurer] LLM call failed: "
                      << (resp.error.empty() ? "HTTP " + std::to_string(resp.status_code) : resp.error) << "\n";
            return {};
        }

        // Parse Ollama response
        auto j = nlohmann::json::parse(resp.body, nullptr, false);
        if (j.is_discarded() || !j.contains("message") || !j["message"].contains("content")) {
            std::cerr << "[DataStructurer] Invalid Ollama response format\n";
            return {};
        }

        std::string llm_output = j["message"]["content"].get<std::string>();
        return parseLLMOutput(llm_output, config_.mode);
    }

    // Process a batch of texts in parallel.
    // Returns structured replacements indexed by position in the input vector.
    // structured_results[i] = replacement strings for texts[i].
    // Empty vector at position i means structuring failed for that entry.
    struct BatchResult {
        std::vector<std::vector<std::string>> structured;
        size_t succeeded = 0;
        size_t failed = 0;
        size_t skipped = 0;
    };

    BatchResult structureBatch(
        const std::vector<std::string>& texts,
        std::function<void(size_t done, size_t total)> progress_cb = nullptr) const {

        BatchResult result;
        result.structured.resize(texts.size());

        std::atomic<size_t> next_idx{0};
        std::atomic<size_t> done_count{0};
        std::mutex result_mutex;

        auto worker = [&]() {
            while (true) {
                size_t idx = next_idx.fetch_add(1);
                if (idx >= texts.size()) break;

                const auto& text = texts[idx];

                // Skip already-structured entries
                if (config_.skip_already_structured && isAlreadyStructured(text)) {
                    std::lock_guard<std::mutex> lock(result_mutex);
                    result.structured[idx] = {text}; // Keep as-is
                    result.skipped++;
                    done_count++;
                    if (progress_cb) progress_cb(done_count.load(), texts.size());
                    continue;
                }

                // Call LLM with one retry on failure
                auto pairs = structureEntry(text);
                if (pairs.empty()) {
                    // Retry once
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                    pairs = structureEntry(text);
                }

                {
                    std::lock_guard<std::mutex> lock(result_mutex);
                    if (pairs.empty()) {
                        // Keep original on failure
                        result.structured[idx] = {text};
                        result.failed++;
                    } else {
                        result.structured[idx] = std::move(pairs);
                        result.succeeded++;
                    }
                }

                done_count++;
                if (progress_cb) progress_cb(done_count.load(), texts.size());
            }
        };

        int num_workers = std::min(config_.parallel_requests, static_cast<int>(texts.size()));
        num_workers = std::max(num_workers, 1);

        std::vector<std::future<void>> futures;
        for (int i = 0; i < num_workers; ++i) {
            futures.push_back(std::async(std::launch::async, worker));
        }
        for (auto& f : futures) {
            f.get();
        }

        return result;
    }

private:
    DataStructuringConfig config_;

    // Extract JSON from LLM output that may contain markdown fences or preamble
    static std::string extractJson(const std::string& raw) {
        // Try to find JSON array or object in the output
        // Look for first [ or { and matching ] or }
        size_t start = std::string::npos;
        char open_char = 0;
        char close_char = 0;

        for (size_t i = 0; i < raw.size(); ++i) {
            if (raw[i] == '[') {
                start = i;
                open_char = '[';
                close_char = ']';
                break;
            }
            if (raw[i] == '{') {
                start = i;
                open_char = '{';
                close_char = '}';
                break;
            }
        }

        if (start == std::string::npos) return raw;

        // Find matching close by counting nesting depth
        int depth = 0;
        bool in_string = false;
        bool escaped = false;
        for (size_t i = start; i < raw.size(); ++i) {
            char c = raw[i];
            if (in_string) {
                if (escaped) {
                    escaped = false;
                } else if (c == '\\') {
                    escaped = true;
                } else if (c == '"') {
                    in_string = false;
                }
            } else {
                if (c == '"') {
                    in_string = true;
                } else if (c == open_char) {
                    depth++;
                } else if (c == close_char) {
                    depth--;
                    if (depth == 0) {
                        return raw.substr(start, i - start + 1);
                    }
                }
            }
        }

        // Couldn't find balanced close — return from start to end
        return raw.substr(start);
    }

    // Parse LLM output into formatted Q/A strings
    static std::vector<std::string> parseLLMOutput(const std::string& raw_output, const std::string& mode) {
        std::string json_str = extractJson(raw_output);
        auto parsed = nlohmann::json::parse(json_str, nullptr, false);
        if (parsed.is_discarded()) return {};

        std::vector<std::string> results;

        if (mode == "qa") {
            // Expect: [{"question":"...","answer":"..."},...]
            if (parsed.is_array()) {
                for (const auto& pair : parsed) {
                    if (pair.contains("question") && pair.contains("answer")) {
                        std::string q = pair["question"].get<std::string>();
                        std::string a = pair["answer"].get<std::string>();
                        if (!q.empty() && !a.empty()) {
                            results.push_back("Q: " + q + "\n\nA: " + a);
                        }
                    }
                }
            } else if (parsed.contains("question") && parsed.contains("answer")) {
                // Single object instead of array
                std::string q = parsed["question"].get<std::string>();
                std::string a = parsed["answer"].get<std::string>();
                if (!q.empty() && !a.empty()) {
                    results.push_back("Q: " + q + "\n\nA: " + a);
                }
            }
        } else if (mode == "conversation") {
            // Expect: [{"role":"human","content":"..."},{"role":"assistant","content":"..."},...]
            if (parsed.is_array() && !parsed.empty()) {
                std::string convo;
                for (const auto& turn : parsed) {
                    if (!turn.contains("role") || !turn.contains("content")) continue;
                    std::string role = turn["role"].get<std::string>();
                    std::string text = turn["content"].get<std::string>();
                    if (!convo.empty()) convo += "\n\n";
                    if (role == "human" || role == "user") {
                        convo += "Human: " + text;
                    } else if (role == "assistant" || role == "bot") {
                        convo += "Assistant: " + text;
                    } else {
                        convo += role + ": " + text;
                    }
                }
                if (!convo.empty()) results.push_back(convo);
            }
        } else if (mode == "instruct") {
            // Expect: {"instruction":"...","response":"..."}
            if (parsed.contains("instruction") && parsed.contains("response")) {
                std::string inst = parsed["instruction"].get<std::string>();
                std::string resp = parsed["response"].get<std::string>();
                if (!inst.empty() && !resp.empty()) {
                    results.push_back("Q: " + inst + "\n\nA: " + resp);
                }
            }
        }

        return results;
    }
};

} // namespace DataCollection
} // namespace GRIM
