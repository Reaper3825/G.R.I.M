// ToolTrainingParser — Phase 1.0
//
// Two responsibilities:
//   1. Robust model output parsing: brace-depth-aware JSON extraction
//      replaces the naive first-'{'-last-'}' approach in ai_interpret()
//   2. Training exemplar collection: captures every model→action
//      interaction (success, failure, gap, conversation) as a JSONL
//      record for LoRA fine-tuning
//
// CorrectionTupleCollector captures *rejections*.
// ToolTrainingParser captures the *full spectrum* including successes,
// so the fine-tuning dataset has both positive and negative signal.
//
// Thread-safe: collector serialized under mutex.
//======================================================//
#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace GRIM::MMO {

// =========================================================
// ParsedModelOutput — result of robust JSON extraction
// =========================================================
struct ParsedModelOutput {
    bool        valid = false;        // JSON parsed successfully
    std::string intent;               // "command", "conversation", "question"
    std::string suggested_command;    // for intent=="command"
    std::string response_text;        // for intent=="conversation"/"question"

    // Raw data for diagnostics
    std::string extracted_json;       // the JSON substring we extracted
    std::string raw_reply;            // full model reply (for training record)

    // Parsed JSON object (available when valid==true)
    nlohmann::json parsed;
};

// =========================================================
// ToolTrainingExample — one interaction record
// =========================================================
struct ToolTrainingExample {
    // Identity
    std::string session_id;
    std::string turn_id;
    uint64_t    timestamp = 0;

    // Input
    std::string raw_user_input;       // what the user said
    std::string available_tools_hash; // hash of tool prompt (detects registry changes)

    // Model output
    std::string raw_model_output;     // full model reply
    std::string extracted_json;       // JSON substring extracted
    std::string parsed_intent;        // "command", "conversation", "question", "parse_failure"
    std::string parsed_tool_id;       // resolved command name (empty if conversation)
    std::string parsed_args;          // resolved arguments

    // Outcome
    enum class Outcome : uint8_t {
        Success         = 0,   // command dispatched and succeeded
        Failure         = 1,   // command dispatched but failed
        GapDetected     = 2,   // tool not in registry → ToolGapPlanner
        PolicyDenied    = 3,   // ActionPolicyRegistry blocked
        PolicyVerify    = 4,   // policy required user verification
        Conversation    = 5,   // no command — conversational reply
        ParseFailure    = 6,   // could not extract valid JSON from model output
        ShadowSuppressed = 7   // shadow mode — would have executed
    };
    Outcome outcome = Outcome::ParseFailure;

    // Confidence signals
    float router_confidence = 0.0f;
    float intent_confidence = 0.0f;
    float entity_confidence = 0.0f;
};

// =========================================================
// ToolTrainingParser
//
// Usage:
//   auto& parser = ToolTrainingParser::instance();
//
//   // 1. Parse model output (replaces naive JSON extraction)
//   auto parsed = parser.parseModelOutput(raw_reply);
//
//   // 2. After outcome is known, record the example
//   ToolTrainingExample ex;
//   ex.raw_user_input = input;
//   ex.raw_model_output = raw_reply;
//   ex.outcome = ToolTrainingExample::Outcome::Success;
//   parser.record(std::move(ex));
//
//   // 3. At shutdown, flush to disk
//   parser.flush("tool_training_examples.jsonl");
// =========================================================
class ToolTrainingParser {
public:
    static ToolTrainingParser& instance();

    // ─── Parsing ──────────────────────────────────────────

    // Extract and parse JSON from raw model output.
    // Uses brace-depth counting to handle nested objects and
    // models that wrap JSON in markdown/commentary.
    ParsedModelOutput parseModelOutput(const std::string& raw_reply) const;

    // ─── Recording ────────────────────────────────────────

    // Record a completed interaction as a training example.
    void record(ToolTrainingExample example);

    // ─── Export ───────────────────────────────────────────

    // Flush all pending examples to JSONL file (append mode).
    // Returns number of examples written.
    size_t flush(const std::string& output_path);

    // Get count of unflushed examples.
    size_t pendingCount() const;

    // Clear all pending examples.
    void clear();

private:
    ToolTrainingParser() = default;

    // Find the outermost balanced JSON object in the string.
    // Returns {start, end} (inclusive) or nullopt if none found.
    static std::optional<std::pair<size_t, size_t>>
    findBalancedJson(const std::string& text);

    mutable std::mutex mutex_;
    std::vector<ToolTrainingExample> pending_;
};

} // namespace GRIM::MMO
