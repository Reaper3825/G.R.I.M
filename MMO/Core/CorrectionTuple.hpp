// CorrectionTuple — typed artifact for user corrections.
//
// When the user rejects a proposed action and provides a correction,
// we capture the full context as a training example:
//   - What the model proposed (rejected)
//   - What the user intended (accepted)
//   - The NLP/context state at the time
//
// These tuples are exported in JSONL format for LoRA fine-tuning,
// teaching the router to avoid rejected actions and prefer corrections.
//
// Flow:
//   ActionEpisode.user_rejected = true
//     → CorrectionTupleCollector::collect(session_id)
//       → writes to correction_tuples.jsonl
//
// Thread-safe: collector serialized under mutex.
//======================================================//
#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace GRIM::MMO {

// =========================================================
// CorrectionTuple — one training example from a user correction
// =========================================================
struct CorrectionTuple {
    // Identity
    std::string session_id;
    std::string turn_id;
    uint64_t    timestamp = 0;

    // Rejected proposal
    struct {
        std::string tool_id;
        std::string proposed_args;
        float       confidence = 0.0f;
        float       risk       = 0.0f;
    } rejected;

    // User's correction
    struct {
        std::string corrected_text;     // raw user correction input
        std::string resolved_tool_id;   // what tool the correction maps to (if resolved)
        std::string resolved_args;      // parsed args from correction (if resolved)
    } correction;

    // Context at the time of the error
    struct {
        std::string raw_input;           // original user input that triggered the proposal
        std::string nlp_summary;         // NLP annotation summary
        std::vector<std::string> router_tags;
        std::vector<std::string> subject_tags;
        std::string selected_route;      // which sub-model was consulted
        std::string mood;
    } context;

    // Training signal
    enum class Signal : uint8_t {
        NegativeOnly = 0,   // no correction provided, just rejection
        Corrected    = 1,   // user provided a correction
        Confirmed    = 2    // correction was executed successfully
    };
    Signal signal = Signal::NegativeOnly;
};

// =========================================================
// CorrectionTupleCollector — builds and stores tuples
//
// Usage:
//   auto& collector = CorrectionTupleCollector::instance();
//   collector.collect(session_id, episode, context_snapshot);
//   collector.flush("corrections/correction_tuples.jsonl");
//
// Tuples accumulate in memory and are flushed to disk
// on demand (e.g. at checkpoint or shutdown).
// =========================================================
class CorrectionTupleCollector {
public:
    static CorrectionTupleCollector& instance();

    // Build a CorrectionTuple from an ActionEpisode that was rejected.
    // Caller provides the NLP/routing context at the time.
    void collect(const std::string& session_id,
                 const std::string& turn_id,
                 const std::string& raw_input,
                 const std::string& rejected_tool_id,
                 const std::string& rejected_args,
                 float              rejected_confidence,
                 float              rejected_risk,
                 const std::string& correction_text,
                 const std::string& resolved_tool_id,
                 const std::string& nlp_summary,
                 const std::vector<std::string>& router_tags,
                 const std::vector<std::string>& subject_tags,
                 const std::string& selected_route,
                 const std::string& mood);

    // Mark the latest tuple for a session as confirmed (correction executed successfully).
    void markConfirmed(const std::string& session_id, const std::string& turn_id);

    // Flush all pending tuples to a JSONL file (append mode).
    // Returns number of tuples written.
    size_t flush(const std::string& output_path);

    // Get count of unflushed tuples.
    size_t pendingCount() const;

    // Clear all pending tuples (e.g. after flush).
    void clear();

private:
    CorrectionTupleCollector() = default;

    mutable std::mutex mutex_;
    std::vector<CorrectionTuple> pending_;
};

} // namespace GRIM::MMO
