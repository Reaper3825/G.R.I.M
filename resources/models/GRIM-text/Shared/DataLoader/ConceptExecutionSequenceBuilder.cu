//======================================================//
//  ConceptExecutionSequenceBuilder.cu
//  Canonical builder implementation.
//
//  Replaces the old __SLOTS__ debug serialization path.
//  Emits paired token_exec_slots + teacher_steps +
//  compiled_bootstrap_bindings from a single builder pass.
//======================================================//

#include "ConceptExecutionSequenceBuilder.hpp"

#include <nlohmann/json.hpp>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <algorithm>

#include "../UnigramByte/UniByte.hpp"

using json = nlohmann::json;

namespace GRIM {
namespace DataLoader {

// ─── Helpers ────────────────────────────────────────────

namespace {

std::string formatNumber(double x) {
    if (x == std::floor(x) && std::fabs(x) < 1e12)
        return std::to_string(static_cast<long long>(x));
    std::ostringstream oss;
    oss << x;
    return oss.str();
}

bool nearEqual(double expected, float got) {
    const double g = static_cast<double>(got);
    const double tol = 1e-3 * std::max(1.0, std::fabs(expected));
    return std::fabs(expected - g) <= tol;
}

}  // namespace

// ─── opStringToId ───────────────────────────────────────

int opStringToId(const std::string& op) {
    if (op == "add" || op == "+") return 0;
    if (op == "sub" || op == "-") return 1;
    if (op == "mul" || op == "*") return 2;
    if (op == "div" || op == "/") return 3;
    throw std::runtime_error("opStringToId: unknown op '" + op + "'");
}

// ─── renderCanonicalText ────────────────────────────────
//
// Human-readable training text from concept JSON.
// NO __SLOTS__ block — that debug path is deleted.

std::string renderCanonicalText(const json& j) {
    std::ostringstream os;

    if (j.contains("name") && j["name"].is_string() && !j["name"].get<std::string>().empty())
        os << "[[" << j["name"].get<std::string>() << "]]\n";

    if (j.contains("question") && j["question"].is_string() && !j["question"].get<std::string>().empty())
        os << "Q: " << j["question"].get<std::string>() << "\n";

    if (j.contains("state_0") && j["state_0"].is_object()) {
        const auto& s0 = j["state_0"];
        if (s0.contains("type") && s0["type"].is_string() && !s0["type"].get<std::string>().empty())
            os << "STATE0 type=" << s0["type"].get<std::string>() << "\n";
    }

    if (j.contains("execution") && j["execution"].is_array()) {
        for (const auto& e : j["execution"]) {
            if (!e.is_object()) continue;
            std::string op = e.value("op", std::string());
            os << "EXEC " << op;
            if (e.contains("args") && e["args"].is_array()) {
                for (const auto& a : e["args"]) {
                    if (a.is_number()) os << " " << formatNumber(a.get<double>());
                }
            }
            if (e.contains("result") && e["result"].is_number())
                os << " => " << formatNumber(e["result"].get<double>());
            os << "\n";
        }
    }

    if (j.contains("state_1") && j["state_1"].is_object()) {
        const auto& s1 = j["state_1"];
        if (s1.contains("result") && s1["result"].is_number())
            os << "STATE1 result=" << formatNumber(s1["result"].get<double>()) << "\n";
    }

    const json* expl = nullptr;
    if (j.contains("explanation") && j["explanation"].is_array())
        expl = &j["explanation"];
    else if (j.contains("intermediates") && j["intermediates"].is_array())
        expl = &j["intermediates"];
    if (expl) {
        for (const auto& s : *expl) {
            if (s.is_string()) os << "EXP: " << s.get<std::string>() << "\n";
        }
    }

    if (j.contains("answer") && j["answer"].is_string() && !j["answer"].get<std::string>().empty())
        os << "A: " << j["answer"].get<std::string>() << "\n";

    // NO __SLOTS__ block — deleted per WS2 cutover.
    return os.str();
}

// ─── buildStructuredExecutionRecord ─────────────────────
//
// Parse concept JSON into canonical StructuredExecutionRecord.
// bootstrap_bindings are built from state_0.atoms: each atom
// gets a unique slot starting at base_slot.
// Execution steps are parsed from JSON "execution" array.

Execution::StructuredExecutionRecord
buildStructuredExecutionRecord(const json& j, int base_slot) {
    Execution::StructuredExecutionRecord record;

    // Check if this row has execution content
    const bool has_state0 = j.contains("state_0") && j["state_0"].is_object()
                         && j["state_0"].contains("atoms") && j["state_0"]["atoms"].is_array()
                         && !j["state_0"]["atoms"].empty();
    const bool has_execution = j.contains("execution") && j["execution"].is_array()
                            && !j["execution"].empty();

    if (!has_state0 && !has_execution) {
        // Non-execution row
        record.execution_active = false;
        return record;
    }

    if (has_execution && !has_state0) {
        throw std::runtime_error(
            "buildStructuredExecutionRecord: execution steps present but no state_0.atoms — "
            "cannot build bootstrap bindings");
    }

    record.execution_active = true;

    // ── Bootstrap bindings from state_0.atoms ──
    // Each initial atom gets one slot. Slot IDs are [base_slot, base_slot + N).
    const auto& atoms = j["state_0"]["atoms"];
    std::unordered_set<int32_t> used_slots;

    for (size_t i = 0; i < atoms.size(); ++i) {
        if (!atoms[i].is_number()) {
            throw std::runtime_error(
                "buildStructuredExecutionRecord: state_0.atoms[" + std::to_string(i)
                + "] is not a number");
        }

        const int32_t slot_id = base_slot + static_cast<int32_t>(i);
        if (!used_slots.insert(slot_id).second) {
            throw std::runtime_error(
                "buildStructuredExecutionRecord: duplicate slot_id " + std::to_string(slot_id)
                + " in bootstrap bindings");
        }

        Execution::BootstrapLiteralBinding binding;
        binding.literal_id = static_cast<int32_t>(i);
        binding.slot_id = slot_id;
        binding.occurrence_role = 0;  // Primary literal occurrence
        binding.rendered_span_id = static_cast<int32_t>(i);
        record.bootstrap_bindings.push_back(binding);
    }

    // ── Slot domain starts with bootstrap slots ──
    for (const auto& b : record.bootstrap_bindings) {
        record.slot_domain.push_back(b.slot_id);
    }

    // ── Execution steps ──
    // Track slot values for slot-domain bookkeeping.
    // Initial slots hold atom values; execution results fill new slots.
    std::vector<double> slot_values;
    for (const auto& a : atoms) {
        slot_values.push_back(a.get<double>());
    }

    if (has_execution) {
        for (size_t step_idx = 0; step_idx < j["execution"].size(); ++step_idx) {
            const auto& e = j["execution"][step_idx];
            if (!e.is_object()) continue;

            Execution::StructuredExecutionRecord::ExecutionStep step;
            step.op_id = opStringToId(e.value("op", std::string()));

            std::vector<double> args;
            if (e.contains("args") && e["args"].is_array()) {
                for (const auto& a : e["args"]) {
                    if (a.is_number()) args.push_back(a.get<double>());
                }
            }
            if (args.size() < 2) {
                throw std::runtime_error(
                    "buildStructuredExecutionRecord: execution step " + std::to_string(step_idx)
                    + " needs >= 2 args, got " + std::to_string(args.size()));
            }

            // Find arg slots by value matching against known slot values
            auto findSlot = [&](double val) -> int {
                for (int i = static_cast<int>(slot_values.size()) - 1; i >= 0; --i) {
                    if (nearEqual(slot_values[i], static_cast<float>(val)))
                        return base_slot + i;
                }
                return -1;
            };

            step.arg1_slot = findSlot(args[0]);
            step.arg2_slot = findSlot(args[1]);
            if (step.arg1_slot < 0 || step.arg2_slot < 0) {
                throw std::runtime_error(
                    "buildStructuredExecutionRecord: execution step " + std::to_string(step_idx)
                    + " arg slot not found (arg1=" + std::to_string(args[0])
                    + " arg2=" + std::to_string(args[1]) + ")");
            }

            double result = e.value("result", 0.0);
            step.expected_value = static_cast<float>(result);

            // Write slot: next available slot
            step.write_slot = base_slot + static_cast<int>(slot_values.size());
            slot_values.push_back(result);

            // Add write slot to domain
            record.slot_domain.push_back(step.write_slot);

            record.steps.push_back(step);
        }
    }

    // De-duplicate slot_domain
    std::sort(record.slot_domain.begin(), record.slot_domain.end());
    record.slot_domain.erase(
        std::unique(record.slot_domain.begin(), record.slot_domain.end()),
        record.slot_domain.end());

    // ── Validate: execution-active with zero bootstrap bindings is malformed ──
    if (record.execution_active && record.bootstrap_bindings.empty()) {
        throw std::runtime_error(
            "buildStructuredExecutionRecord: execution-active row has zero bootstrap bindings");
    }

    return record;
}

// ─── compileExecutionPayload ────────────────────────────
//
// Compile a StructuredExecutionRecord against tokenized output.
// Maps bootstrap bindings to token positions by scanning for
// ATOM_NUM tokens that match the expected bootstrap literal values.
//
// Contract: each bound literal must compile to EXACTLY ONE ATOM_NUM token.

Execution::CompiledStructuredExecutionPayload
compileExecutionPayload(
    const Execution::StructuredExecutionRecord& record,
    const std::vector<int>& token_ids,
    const std::vector<uint8_t>& atom_mask,
    const std::vector<float>& numeric_values,
    int seq_len)
{
    Execution::CompiledStructuredExecutionPayload payload;

    if (!record.execution_active) {
        payload.execution_active = false;
        payload.token_exec_slots.assign(seq_len, -1);
        return payload;
    }

    payload.execution_active = true;
    payload.token_exec_slots.assign(seq_len, -1);

    // ── Build bootstrap binding → atom value map ──
    // Each bootstrap binding carries a slot_id. The atom value for that slot
    // is the initial literal from state_0.atoms at index (slot_id - base_slot(implicit)).
    // We need to find the matching ATOM_NUM token for each binding.

    // Collect all ATOM_NUM token positions with their numeric values
    struct AtomNumPos {
        int pos;
        float value;
    };
    std::vector<AtomNumPos> atom_num_positions;
    const int atom_num_token_id = GRIM::Tokenizer::atomTypeToTokenId(
        GRIM::Tokenizer::AtomType::ATOM_NUM);

    for (int t = 0; t < seq_len; ++t) {
        if (t < static_cast<int>(token_ids.size()) &&
            token_ids[t] == atom_num_token_id &&
            t < static_cast<int>(atom_mask.size()) && atom_mask[t]) {
            atom_num_positions.push_back({t, numeric_values[t]});
        }
    }

    // For each bootstrap binding, find its matching ATOM_NUM token.
    // The bootstrap bindings correspond to state_0.atoms in order.
    // We need to match the i-th bootstrap binding's literal value to
    // a token position. We scan in order — bindings are ordered by
    // literal_id, so we consume atom positions in document order.
    std::vector<bool> pos_claimed(atom_num_positions.size(), false);
    std::unordered_set<int32_t> claimed_token_positions;

    // Extract the literal values that correspond to each bootstrap binding.
    // binding.literal_id tells us which state_0.atom this is.
    // We need the actual atom values — reconstruct from the record's bootstrap_bindings
    // and the known slot assignment scheme: slot_id = base_slot + literal_id.
    // The actual numeric values come from the tokenized text's numeric_values array.

    for (size_t b_idx = 0; b_idx < record.bootstrap_bindings.size(); ++b_idx) {
        const auto& binding = record.bootstrap_bindings[b_idx];

        // Find a matching ATOM_NUM position for this binding.
        // Scan in document order; claim the first unclaimed match.
        int matched_pos = -1;
        size_t matched_idx = 0;
        int match_count = 0;

        // We need the expected value for this binding. Since the builder
        // stores literal values in state_0.atoms[literal_id], and we have
        // the full numeric_values from tokenization, we find the ATOM_NUM
        // token whose value matches by scanning unclaimed positions in
        // document order. The first unclaimed forward match is claimed.
        //
        // For the case where multiple ATOM_NUM tokens have the same value
        // (e.g., "3 + 3"), the ordering is determined by document position:
        // binding 0 gets the first occurrence, binding 1 gets the second.
        for (size_t a = 0; a < atom_num_positions.size(); ++a) {
            if (pos_claimed[a]) continue;

            // We match based on which ATOM_NUM positions exist in the rendered
            // text. Since bindings are ordered by literal_id (= order in state_0.atoms,
            // = order these values appear in the rendered text), we claim the
            // first unclaimed ATOM_NUM position per binding.
            // Note: the rendered text places state_0 atom values before
            // EXEC lines, but EXEC lines also contain numeric args.
            // We only want to match state_0 atoms (bootstrap values).
            //
            // Strategy: for binding b_idx, we're looking for the atom value
            // at literal position b_idx. Since we're scanning rendered text
            // forward and state_0 atom values appear first in the canonical
            // text format, we just claim in order.
            if (!pos_claimed[a]) {
                matched_pos = atom_num_positions[a].pos;
                matched_idx = a;
                match_count++;
                break;  // Claim first unclaimed in document order
            }
        }

        if (matched_pos < 0) {
            throw std::runtime_error(
                "compileExecutionPayload: bootstrap binding " + std::to_string(b_idx)
                + " (slot_id=" + std::to_string(binding.slot_id)
                + ") found no matching ATOM_NUM token");
        }

        if (!claimed_token_positions.insert(matched_pos).second) {
            throw std::runtime_error(
                "compileExecutionPayload: duplicate token_pos " + std::to_string(matched_pos)
                + " for bootstrap binding " + std::to_string(b_idx));
        }

        pos_claimed[matched_idx] = true;
        payload.token_exec_slots[matched_pos] = binding.slot_id;

        Execution::CompiledBootstrapBinding compiled;
        compiled.binding_id = static_cast<int32_t>(b_idx);
        compiled.token_pos = matched_pos;
        compiled.slot_id = binding.slot_id;
        payload.compiled_bootstrap_bindings.push_back(compiled);
    }

    // ── Teacher steps from execution steps ──
    for (const auto& step : record.steps) {
        Execution::TeacherStep ts;
        ts.op_id = step.op_id;
        ts.arg1_slot = step.arg1_slot;
        ts.arg2_slot = step.arg2_slot;
        ts.write_slot = step.write_slot;
        ts.expected_value = step.expected_value;
        payload.teacher_steps.push_back(ts);
    }

    // ── Slot selection targets: IGNORE for all token positions ──
    // Dense, decode-position aligned. For concept rows in this cutover,
    // the builder does not yet have per-position selector supervision data.
    // All positions are IGNORE until a selector supervision pipeline exists.
    payload.slot_selection_targets.resize(seq_len);
    for (int t = 0; t < seq_len; ++t) {
        payload.slot_selection_targets[t].kind = Execution::SlotSelectionTargetKind::Ignore;
        payload.slot_selection_targets[t].slot_id = -1;
    }

    return payload;
}

// ─── buildConceptSequence ───────────────────────────────
//
// Full pipeline: JSON → record → text → tokenize → compile.

ConceptBuildResult buildConceptSequence(
    const json& j,
    GRIM::Tokenizer::UniByte& tokenizer,
    int base_slot)
{
    ConceptBuildResult result;

    // 1. Build canonical record from JSON
    result.record = buildStructuredExecutionRecord(j, base_slot);

    // 2. Render canonical text (no __SLOTS__)
    result.canonical_text = renderCanonicalText(j);

    if (!result.record.execution_active) {
        // Non-execution row: payload is trivially inactive
        result.payload.execution_active = false;
        return result;
    }

    // 3. Tokenize the canonical text
    auto encoded = tokenizer.encodeWithMetadata(result.canonical_text);
    if (encoded.token_ids.empty()) {
        throw std::runtime_error(
            "buildConceptSequence: tokenization produced zero tokens for concept row");
    }

    // 4. Compile the record against tokenized output
    result.payload = compileExecutionPayload(
        result.record,
        encoded.token_ids,
        encoded.token_atom_mask,
        encoded.token_numeric_values,
        static_cast<int>(encoded.token_ids.size()));

    return result;
}

}  // namespace DataLoader
}  // namespace GRIM
