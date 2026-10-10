#pragma once

// Shared HTTP contract for the GRIM-text bridge and its model worker.
// This is separate from the DLL function-table ABI in plugin_api.hpp.
// The bridge owns public HTTP routing; the worker owns model/tokenizer state.
// Runtime configuration may override these default ports.
#include <string>

namespace GRIM::ServerAPI {

inline constexpr int kDefaultPublicPort = 11435;
inline constexpr int kDefaultWorkerPort = 11436;
inline constexpr char kLoopbackHost[] = "127.0.0.1";
inline constexpr char kServiceName[] = "grim_text_server";
inline constexpr char kJsonContentType[] = "application/json";

inline std::string defaultPublicURL() {
    return std::string("http://") + kLoopbackHost + ":" + std::to_string(kDefaultPublicPort);
}

namespace Public {
inline constexpr char kRoot[] = "/";                  // GET bridge info
inline constexpr char kHealth[] = "/health";          // GET bridge liveness
inline constexpr char kTags[] = "/api/tags";          // GET loaded model list
inline constexpr char kStatus[] = "/api/status";      // GET worker status
inline constexpr char kTokenizerRun[] = "/api/tokenizer/run";       // POST {}
inline constexpr char kTokenizerEncode[] = "/api/tokenizer/encode"; // POST {"text": ...}
inline constexpr char kInspect[] = "/api/inspect";    // POST lens inspection request
inline constexpr char kGenerate[] = "/api/generate";  // POST Ollama-compatible prompt
inline constexpr char kChat[] = "/api/chat";          // POST Ollama-compatible messages
} // namespace Public

// Private loopback routes used by the bridge to reach the model worker.
namespace Worker {
inline constexpr char kStatus[] = "/internal/status";
inline constexpr char kTokenizerRun[] = "/internal/tokenizer/run";
inline constexpr char kTokenizerEncode[] = "/internal/tokenizer/encode";
inline constexpr char kInspect[] = "/internal/inspect";
inline constexpr char kGenerate[] = "/internal/generate";
inline constexpr char kChat[] = "/internal/chat";
} // namespace Worker

} // namespace GRIM::ServerAPI
