// ToolTrainingParser.cpp — robust model output parsing and training exemplar collection.
//======================================================//

#include "ToolTrainingParser.hpp"

#include <fstream>
#include <ctime>
#include <stdexcept>

namespace GRIM::MMO {

// ─── Singleton ────────────────────────────────────────────

ToolTrainingParser& ToolTrainingParser::instance() {
    static ToolTrainingParser inst;
    return inst;
}

// ─── findBalancedJson ─────────────────────────────────────
//
// Walk the string tracking brace depth and string literals.
// Returns the span of the first balanced top-level { ... }.
// Handles:
//   - Nested objects/arrays
//   - String literals with escaped quotes
//   - Models that prefix JSON with markdown/commentary

std::optional<std::pair<size_t, size_t>>
ToolTrainingParser::findBalancedJson(const std::string& text) {
    size_t start = std::string::npos;
    int    depth = 0;
    bool   in_string = false;
    bool   escape = false;

    for (size_t i = 0; i < text.size(); ++i) {
        char c = text[i];

        if (escape) {
            escape = false;
            continue;
        }

        if (c == '\\' && in_string) {
            escape = true;
            continue;
        }

        if (c == '"') {
            in_string = !in_string;
            continue;
        }

        if (in_string) continue;

        if (c == '{') {
            if (depth == 0) start = i;
            ++depth;
        } else if (c == '}') {
            if (depth <= 0) continue; // stray closing brace
            --depth;
            if (depth == 0 && start != std::string::npos) {
                return std::make_pair(start, i);
            }
        }
    }

    return std::nullopt;
}

// ─── parseModelOutput ─────────────────────────────────────

ParsedModelOutput ToolTrainingParser::parseModelOutput(const std::string& raw_reply) const {
    ParsedModelOutput result;
    result.raw_reply = raw_reply;

    if (raw_reply.empty()) {
        return result;
    }

    // Find balanced JSON object
    auto span = findBalancedJson(raw_reply);
    if (!span) {
        return result;
    }

    result.extracted_json = raw_reply.substr(
        span->first, span->second - span->first + 1);

    // Parse
    nlohmann::json j = nlohmann::json::parse(result.extracted_json, nullptr, false);
    if (j.is_discarded() || !j.is_object()) {
        return result;
    }

    result.valid  = true;
    result.parsed = std::move(j);
    result.intent = result.parsed.value("intent", "conversation");

    if (result.intent == "command") {
        result.suggested_command = result.parsed.value("suggested_command", "");
    } else {
        result.response_text = result.parsed.value("response", "");
    }

    return result;
}

// ─── record ───────────────────────────────────────────────

void ToolTrainingParser::record(ToolTrainingExample example) {
    if (example.timestamp == 0) {
        example.timestamp = static_cast<uint64_t>(std::time(nullptr));
    }

    std::lock_guard<std::mutex> lock(mutex_);
    pending_.push_back(std::move(example));
}

// ─── Serialization ────────────────────────────────────────

static const char* outcomeToString(ToolTrainingExample::Outcome o) {
    switch (o) {
        case ToolTrainingExample::Outcome::Success:          return "success";
        case ToolTrainingExample::Outcome::Failure:          return "failure";
        case ToolTrainingExample::Outcome::GapDetected:      return "gap_detected";
        case ToolTrainingExample::Outcome::PolicyDenied:     return "policy_denied";
        case ToolTrainingExample::Outcome::PolicyVerify:     return "policy_verify";
        case ToolTrainingExample::Outcome::Conversation:     return "conversation";
        case ToolTrainingExample::Outcome::ParseFailure:     return "parse_failure";
        case ToolTrainingExample::Outcome::ShadowSuppressed: return "shadow_suppressed";
    }
    return "unknown";
}

static nlohmann::json exampleToJson(const ToolTrainingExample& ex) {
    nlohmann::json j;
    j["session_id"] = ex.session_id;
    j["turn_id"]    = ex.turn_id;
    j["timestamp"]  = ex.timestamp;

    j["input"] = {
        {"raw_user_input",       ex.raw_user_input},
        {"available_tools_hash", ex.available_tools_hash}
    };

    j["model_output"] = {
        {"raw",            ex.raw_model_output},
        {"extracted_json", ex.extracted_json},
        {"parsed_intent",  ex.parsed_intent},
        {"parsed_tool_id", ex.parsed_tool_id},
        {"parsed_args",    ex.parsed_args}
    };

    j["outcome"] = outcomeToString(ex.outcome);

    j["confidence"] = {
        {"router",  ex.router_confidence},
        {"intent",  ex.intent_confidence},
        {"entity",  ex.entity_confidence}
    };

    return j;
}

// ─── flush ────────────────────────────────────────────────

size_t ToolTrainingParser::flush(const std::string& output_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pending_.empty()) return 0;

    std::ofstream out(output_path, std::ios::app);
    if (!out.is_open()) {
        throw std::runtime_error(
            "ToolTrainingParser::flush — cannot open: " + output_path);
    }

    size_t count = 0;
    for (const auto& ex : pending_) {
        out << exampleToJson(ex).dump(-1) << '\n';
        ++count;
    }

    pending_.clear();
    return count;
}

// ─── Accessors ────────────────────────────────────────────

size_t ToolTrainingParser::pendingCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pending_.size();
}

void ToolTrainingParser::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_.clear();
}

} // namespace GRIM::MMO
