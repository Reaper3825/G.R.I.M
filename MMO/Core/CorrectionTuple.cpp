// CorrectionTuple.cpp — collection and JSONL export of correction tuples.
//======================================================//

#include "CorrectionTuple.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <ctime>

namespace GRIM::MMO {

// ─── Singleton ────────────────────────────────────────────

CorrectionTupleCollector& CorrectionTupleCollector::instance() {
    static CorrectionTupleCollector inst;
    return inst;
}

// ─── Collect ──────────────────────────────────────────────

void CorrectionTupleCollector::collect(
    const std::string& session_id,
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
    const std::string& mood)
{
    CorrectionTuple tuple;
    tuple.session_id = session_id;
    tuple.turn_id    = turn_id;
    tuple.timestamp  = static_cast<uint64_t>(std::time(nullptr));

    tuple.rejected.tool_id        = rejected_tool_id;
    tuple.rejected.proposed_args  = rejected_args;
    tuple.rejected.confidence     = rejected_confidence;
    tuple.rejected.risk           = rejected_risk;

    tuple.correction.corrected_text   = correction_text;
    tuple.correction.resolved_tool_id = resolved_tool_id;

    tuple.context.raw_input      = raw_input;
    tuple.context.nlp_summary    = nlp_summary;
    tuple.context.router_tags    = router_tags;
    tuple.context.subject_tags   = subject_tags;
    tuple.context.selected_route = selected_route;
    tuple.context.mood           = mood;

    tuple.signal = correction_text.empty()
        ? CorrectionTuple::Signal::NegativeOnly
        : CorrectionTuple::Signal::Corrected;

    std::lock_guard<std::mutex> lock(mutex_);
    pending_.push_back(std::move(tuple));
}

// ─── Mark confirmed ──────────────────────────────────────

void CorrectionTupleCollector::markConfirmed(
    const std::string& session_id,
    const std::string& turn_id)
{
    std::lock_guard<std::mutex> lock(mutex_);
    // Walk backwards to find the most recent matching tuple
    for (auto it = pending_.rbegin(); it != pending_.rend(); ++it) {
        if (it->session_id == session_id && it->turn_id == turn_id) {
            it->signal = CorrectionTuple::Signal::Confirmed;
            return;
        }
    }
}

// ─── Flush to JSONL ──────────────────────────────────────

static nlohmann::json tupleToJson(const CorrectionTuple& t) {
    nlohmann::json j;
    j["session_id"] = t.session_id;
    j["turn_id"]    = t.turn_id;
    j["timestamp"]  = t.timestamp;

    j["rejected"] = {
        {"tool_id",  t.rejected.tool_id},
        {"args",     t.rejected.proposed_args},
        {"confidence", t.rejected.confidence},
        {"risk",     t.rejected.risk}
    };

    j["correction"] = {
        {"corrected_text",   t.correction.corrected_text},
        {"resolved_tool_id", t.correction.resolved_tool_id},
        {"resolved_args",    t.correction.resolved_args}
    };

    j["context"] = {
        {"raw_input",      t.context.raw_input},
        {"nlp_summary",    t.context.nlp_summary},
        {"router_tags",    t.context.router_tags},
        {"subject_tags",   t.context.subject_tags},
        {"selected_route", t.context.selected_route},
        {"mood",           t.context.mood}
    };

    const char* signal_str = "negative_only";
    if (t.signal == CorrectionTuple::Signal::Corrected) signal_str = "corrected";
    if (t.signal == CorrectionTuple::Signal::Confirmed) signal_str = "confirmed";
    j["signal"] = signal_str;

    return j;
}

size_t CorrectionTupleCollector::flush(const std::string& output_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pending_.empty()) return 0;

    std::ofstream out(output_path, std::ios::app);
    if (!out.is_open()) {
        throw std::runtime_error(
            "CorrectionTupleCollector::flush — cannot open: " + output_path);
    }

    size_t count = 0;
    for (const auto& tuple : pending_) {
        out << tupleToJson(tuple).dump(-1) << '\n';
        ++count;
    }

    pending_.clear();
    return count;
}

// ─── Accessors ────────────────────────────────────────────

size_t CorrectionTupleCollector::pendingCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pending_.size();
}

void CorrectionTupleCollector::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_.clear();
}

} // namespace GRIM::MMO
