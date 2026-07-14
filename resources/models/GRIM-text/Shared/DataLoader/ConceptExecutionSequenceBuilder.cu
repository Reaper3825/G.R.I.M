//======================================================//
//  ConceptExecutionSequenceBuilder.cu
//  Canonical builder implementation.
//
//  Replaces the old __SLOTS__ debug serialization path.
//  Emits paired token_exec_slots + teacher_steps +
//  compiled_bootstrap_bindings from a single builder pass.
//
//  Provenance model:
//    renderWithSpans() → RenderedLiteralSpan per bootstrap literal
//    tokenizeWithMetadata() → StructuralSpan per detected atom
//    compileExecutionPayload() → match by content_offset intersection
//                                 (not document-order claiming)
//  Runtime proofs:
//    - Coordinate alignment: atom content bytes == rendered literal bytes
//    - Render-order: literal_spans monotonic + binding_index == index
//    - Arg slot identity: canonical via "arg_slots" when available,
//      value-based + result validation as fallback
//======================================================//

#include "ConceptExecutionSequenceBuilder.hpp"

#include <nlohmann/json.hpp>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <algorithm>

#include "../UnigramByte/Detectors/StructuralSpan.hpp"
#include "../UnigramByte/UniByte.hpp"

using json = nlohmann::json;

namespace GRIM {
namespace DataLoader {

// ─── Helpers ────────────────────────────────────────────

namespace {

Execution::ExecutionGateTarget parseExecutionGateTarget(const json& j, bool execution_active) {
    if (execution_active) {
        return Execution::ExecutionGateTarget::EXECUTE;
    }

    if (!j.contains("execution_gate_target")) {
        return Execution::ExecutionGateTarget::IGNORE;
    }
    if (!j["execution_gate_target"].is_string()) {
        throw std::runtime_error(
            "parseExecutionGateTarget: execution_gate_target must be a string: "
            "\"noop\", \"execute\", or \"ignore\"");
    }

    std::string value = j["execution_gate_target"].get<std::string>();
    std::transform(value.begin(), value.end(), value.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (value == "noop") return Execution::ExecutionGateTarget::NOOP;
    if (value == "ignore") return Execution::ExecutionGateTarget::IGNORE;
    if (value == "execute") {
        throw std::runtime_error(
            "parseExecutionGateTarget: execution_gate_target=execute requires an authored execution program");
    }
    throw std::runtime_error(
        "parseExecutionGateTarget: unknown execution_gate_target=\"" + value + "\"");
}

std::string formatNumber(double x) {
    if (x == std::floor(x) && std::fabs(x) < 1e12)
        return std::to_string(static_cast<long long>(x));
    std::ostringstream oss;
    oss << x;
    return oss.str();
}

// Compute an arithmetic operation for result validation.
double computeOp(int op_id, double a, double b) {
    switch (op_id) {
        case 0: return a + b;
        case 1: return a - b;
        case 2: return a * b;
        case 3:
            if (b == 0.0)
                throw std::runtime_error("computeOp: division by zero");
            return a / b;
        default:
            throw std::runtime_error("computeOp: unknown op_id " + std::to_string(op_id));
    }
}

// Tight relative tolerance for result validation.
// Values originate from the same JSON source so should be nearly identical;
// the only divergence is from floating-point computation (e.g., division).
bool resultMatches(double computed, double expected) {
    if (computed == expected) return true;
    const double denom = std::max(1.0, std::fabs(expected));
    return std::fabs(computed - expected) <= 1e-9 * denom;
}

// ─── compileExecutionPayload ────────────────────────────
//
// File-local: only called from buildConceptSequence().
//
// Maps bootstrap bindings to token positions via character-offset
// intersection between RenderedLiteralSpan (from renderer) and
// StructuralSpan::content_offset (from atom detector).
//
// Contract: each rendered literal span must intersect EXACTLY ONE
// ATOM_NUM token's StructuralSpan. Zero matches or multiple matches
// are structural violations (throw).

Execution::CompiledStructuredExecutionPayload
compileExecutionPayload(
    const Execution::StructuredExecutionRecord& record,
    const std::vector<RenderedLiteralSpan>& literal_spans,
    const std::string& rendered_text,
    const std::vector<int>& token_ids,
    const std::vector<const Tokenizer::StructuralSpan*>& token_to_span,
    int seq_len,
    int execution_prompt_end_pos,
    int execution_prompt_length)
{
    Execution::CompiledStructuredExecutionPayload payload;
    payload.execution_gate_target = record.execution_gate_target;
    payload.execution_prompt_end_pos = execution_prompt_end_pos;
    payload.execution_prompt_length = execution_prompt_length;

    if (!record.execution_active) {
        payload.execution_active = false;
        payload.token_exec_slots.assign(seq_len, -1);
        return payload;
    }

    payload.execution_active = true;
    payload.token_exec_slots.assign(seq_len, -1);

    const int atom_int_token_id = GRIM::Tokenizer::atomTypeToTokenId(
        GRIM::Tokenizer::AtomType::ATOM_INT);
    const int atom_float_token_id = GRIM::Tokenizer::atomTypeToTokenId(
        GRIM::Tokenizer::AtomType::ATOM_FLOAT);

    if (literal_spans.size() != record.bootstrap_bindings.size()) {
        throw std::runtime_error(
            "compileExecutionPayload: literal_spans.size()=" + std::to_string(literal_spans.size())
            + " != bootstrap_bindings.size()=" + std::to_string(record.bootstrap_bindings.size()));
    }

    std::unordered_set<int32_t> claimed_token_positions;

    for (size_t b_idx = 0; b_idx < record.bootstrap_bindings.size(); ++b_idx) {
        const auto& binding = record.bootstrap_bindings[b_idx];
        const auto& lit = literal_spans[b_idx];

        // Find the ATOM_NUM token whose StructuralSpan content_offset
        // falls within this rendered literal span.
        int matched_pos = -1;
        int match_count = 0;

        for (int t = 0; t < seq_len; ++t) {
            if (token_ids[t] != atom_int_token_id && token_ids[t] != atom_float_token_id) continue;
            if (!token_to_span[t]) continue;

            const auto* span = token_to_span[t];
            const size_t atom_start = static_cast<size_t>(span->content_offset);
            const size_t atom_end   = atom_start + static_cast<size_t>(span->content_length);

            // Exact containment: the atom's content range must fall within
            // the rendered literal span's byte range.
            if (atom_start >= lit.byte_start && atom_end <= lit.byte_end) {
                // ── Coordinate-system alignment proof ──
                // Verify that the bytes the atom detector found match the bytes
                // the renderer placed. This proves both coordinate systems refer
                // to the same string and the offsets are mutually consistent.
                std::string_view atom_content(
                    span->buffer_ptr + span->content_offset,
                    span->content_length);
                std::string_view rendered_literal(
                    rendered_text.data() + lit.byte_start,
                    lit.byte_end - lit.byte_start);
                if (atom_content != rendered_literal) {
                    throw std::runtime_error(
                        "compileExecutionPayload: coordinate mismatch at binding "
                        + std::to_string(b_idx) + ": atom detector content=\""
                        + std::string(atom_content) + "\" vs rendered literal=\""
                        + std::string(rendered_literal) + "\" — byte spaces diverged");
                }

                matched_pos = t;
                match_count++;
            }
        }

        if (match_count == 0) {
            throw std::runtime_error(
                "compileExecutionPayload: bootstrap binding " + std::to_string(b_idx)
                + " (slot_id=" + std::to_string(binding.slot_id)
                + ") rendered span [" + std::to_string(lit.byte_start)
                + "," + std::to_string(lit.byte_end)
                + ") matched zero numeric atom tokens");
        }
        if (match_count > 1) {
            throw std::runtime_error(
                "compileExecutionPayload: bootstrap binding " + std::to_string(b_idx)
                + " (slot_id=" + std::to_string(binding.slot_id)
                + ") rendered span [" + std::to_string(lit.byte_start)
                + "," + std::to_string(lit.byte_end)
                + ") matched " + std::to_string(match_count) + " numeric atom tokens (must be exactly 1)");
        }

        if (!claimed_token_positions.insert(matched_pos).second) {
            throw std::runtime_error(
                "compileExecutionPayload: token_pos " + std::to_string(matched_pos)
                + " claimed by multiple bootstrap bindings");
        }

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

    return payload;
}

}  // namespace (anonymous)

// ─── opStringToId ───────────────────────────────────────

int opStringToId(const std::string& op) {
    std::string lower;
    lower.reserve(op.size());
    for (char c : op) lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (lower == "add" || lower == "+") return 0;
    if (lower == "sub" || lower == "-") return 1;
    if (lower == "mul" || lower == "*") return 2;
    if (lower == "div" || lower == "/") return 3;
    throw std::runtime_error("opStringToId: unknown op '" + op + "'");
}

// ─── renderWithSpans ────────────────────────────────────
//
// Canonical text rendering with bound-span provenance.
//
// STATE0 section renders bootstrap literal values explicitly so
// the atom detector sees them. Character offsets are tracked via
// ostringstream::tellp() and recorded in literal_spans.
//
// The rendered literal span for binding i is the byte range
// [byte_start, byte_end) of the formatted number in the output.
// This is later intersected with StructuralSpan::content_offset
// from the atom detector to locate the unique ATOM_NUM token.

CanonicalRenderResult renderWithSpans(const json& j) {
    CanonicalRenderResult result;
    std::ostringstream os;

    // NOTE: "name" is intentionally NOT rendered into canonical text.
    // It is metadata only (filtering, UI, concept-block identity).
    // Tokenizing it leaks IDs / labels into the training signal.

    if (j.contains("question") && j["question"].is_string() && !j["question"].get<std::string>().empty()) {
        os << "Q: " << j["question"].get<std::string>() << "\n";
        result.execution_prompt_byte_end = static_cast<size_t>(os.tellp());
    }

    if (j.contains("state_0") && j["state_0"].is_object()) {
        const auto& s0 = j["state_0"];
        os << "STATE0";

        if (s0.contains("type") && s0["type"].is_string() && !s0["type"].get<std::string>().empty())
            os << " type=" << s0["type"].get<std::string>();

        // Render bootstrap literal values with explicit span tracking.
        // Each atom value appears in the text at a known byte offset;
        // this is the ONLY place bootstrap literals are rendered.
        if (s0.contains("atoms") && s0["atoms"].is_array()) {
            int binding_idx = 0;
            for (const auto& a : s0["atoms"]) {
                if (!a.is_number()) continue;
                os << " ";
                const size_t byte_start = static_cast<size_t>(os.tellp());
                os << formatNumber(a.get<double>());
                const size_t byte_end = static_cast<size_t>(os.tellp());
                result.literal_spans.push_back({binding_idx, byte_start, byte_end});
                binding_idx++;
            }
        }

        os << "\n";
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
        os << "A: " << j["answer"].get<std::string>() << "\n";
    }

    result.text = os.str();
    return result;
}

// ─── renderCanonicalText ────────────────────────────────
//
// Convenience wrapper: returns plain text for vocab corpus building.

std::string renderCanonicalText(const json& j) {
    return renderWithSpans(j).text;
}

// ─── renderPlainText ────────────────────────────────────
//
// Returns raw text content from concept block fields WITHOUT
// the canonical Q:/STATE0/EXP:/EXEC/A: prefixes. Used for
// pretraining (PT) curriculums that should be tokenized as
// natural text, not structured concept format.

std::string renderPlainText(const json& j, bool format_as_concept) {
    if (format_as_concept)
        return renderCanonicalText(j);

    std::ostringstream os;

    if (j.contains("question") && j["question"].is_string() &&
        !j["question"].get<std::string>().empty())
        os << j["question"].get<std::string>() << "\n";

    const json* expl = nullptr;
    if (j.contains("explanation") && j["explanation"].is_array())
        expl = &j["explanation"];
    else if (j.contains("intermediates") && j["intermediates"].is_array())
        expl = &j["intermediates"];
    if (expl) {
        for (const auto& s : *expl) {
            if (s.is_string())
                os << s.get<std::string>() << "\n";
        }
    }

    if (j.contains("answer") && j["answer"].is_string() &&
        !j["answer"].get<std::string>().empty())
        os << j["answer"].get<std::string>() << "\n";

    return os.str();
}

// ─── buildStructuredExecutionRecord ─────────────────────
//
// Parse concept JSON into canonical StructuredExecutionRecord.
// Bootstrap bindings from state_0.atoms; execution steps from
// JSON "execution" array.
//
// Arg slot resolution:
//   Each execution step MUST carry "arg_slots": [i, j] with explicit
//   slot indices. The builder validates that op(slot_values[i], slot_values[j])
//   matches the expected result. Value-based fallback resolution is deleted
//   per Rule 20 — it produced ambiguities for duplicate/coincident values.

Execution::StructuredExecutionRecord
buildStructuredExecutionRecord(const json& j, int base_slot) {
    Execution::StructuredExecutionRecord record;

    const bool has_state0 = j.contains("state_0") && j["state_0"].is_object()
                         && j["state_0"].contains("atoms") && j["state_0"]["atoms"].is_array()
                         && !j["state_0"]["atoms"].empty();
    const bool has_execution = j.contains("execution") && j["execution"].is_array()
                            && !j["execution"].empty();

    if (!has_state0 && !has_execution) {
        record.execution_active = false;
        record.execution_gate_target = parseExecutionGateTarget(j, false);
        return record;
    }

    if (has_execution && !has_state0) {
        throw std::runtime_error(
            "buildStructuredExecutionRecord: execution steps present but no state_0.atoms — "
            "cannot build bootstrap bindings");
    }

    record.execution_active = true;
    record.execution_gate_target = Execution::ExecutionGateTarget::EXECUTE;

    // ── Bootstrap bindings from state_0.atoms ──
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
        binding.occurrence_role = 0;
        binding.rendered_span_id = static_cast<int32_t>(i);
        record.bootstrap_bindings.push_back(binding);
    }

    for (const auto& b : record.bootstrap_bindings) {
        record.slot_domain.push_back(b.slot_id);
    }

    // ── Execution steps with result-validated arg resolution ──
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

            const double expected_result = e.value("result", 0.0);

            int resolved_1 = -1, resolved_2 = -1;

            // ── arg_slots is REQUIRED — no value-based fallback (Rule 20) ──
            if (!e.contains("arg_slots") || !e["arg_slots"].is_array()
                || e["arg_slots"].size() < 2) {
                throw std::runtime_error(
                    "buildStructuredExecutionRecord: step " + std::to_string(step_idx)
                    + " missing required \"arg_slots\" array — value-based resolution "
                    "is deleted (Rule 20). Add \"arg_slots\": [i, j] to the concept JSON.");
            }

            resolved_1 = e["arg_slots"][0].get<int>();
            resolved_2 = e["arg_slots"][1].get<int>();

            if (resolved_1 < 0 || resolved_1 >= static_cast<int>(slot_values.size())) {
                throw std::runtime_error(
                    "buildStructuredExecutionRecord: step " + std::to_string(step_idx)
                    + " arg_slots[0]=" + std::to_string(resolved_1)
                    + " out of range [0, " + std::to_string(slot_values.size()) + ")");
            }
            if (resolved_2 < 0 || resolved_2 >= static_cast<int>(slot_values.size())) {
                throw std::runtime_error(
                    "buildStructuredExecutionRecord: step " + std::to_string(step_idx)
                    + " arg_slots[1]=" + std::to_string(resolved_2)
                    + " out of range [0, " + std::to_string(slot_values.size()) + ")");
            }

            // Validate: canonical slot refs must produce the expected result.
            const double computed = computeOp(step.op_id,
                slot_values[resolved_1], slot_values[resolved_2]);
            if (!resultMatches(computed, expected_result)) {
                throw std::runtime_error(
                    "buildStructuredExecutionRecord: step " + std::to_string(step_idx)
                    + " arg_slots [" + std::to_string(resolved_1) + ","
                    + std::to_string(resolved_2) + "] produce "
                    + std::to_string(computed) + " but expected "
                    + std::to_string(expected_result));
            }

            step.arg1_slot = base_slot + resolved_1;
            step.arg2_slot = base_slot + resolved_2;
            step.expected_value = static_cast<float>(expected_result);

            step.write_slot = base_slot + static_cast<int>(slot_values.size());
            slot_values.push_back(expected_result);

            record.slot_domain.push_back(step.write_slot);
            record.steps.push_back(step);
        }
    }

    // De-duplicate slot_domain
    std::sort(record.slot_domain.begin(), record.slot_domain.end());
    record.slot_domain.erase(
        std::unique(record.slot_domain.begin(), record.slot_domain.end()),
        record.slot_domain.end());

    if (record.execution_active && record.bootstrap_bindings.empty()) {
        throw std::runtime_error(
            "buildStructuredExecutionRecord: execution-active row has zero bootstrap bindings");
    }

    return record;
}

// ─── buildConceptSequence ───────────────────────────────
//
// Full pipeline: JSON → record → render with spans → tokenize
// → build token-to-span correlation → compile via offset intersection.

ConceptBuildResult buildConceptSequence(
    const json& j,
    GRIM::Tokenizer::UniByte& tokenizer,
    int base_slot)
{
    ConceptBuildResult result;

    // 1. Build canonical record from JSON (with result-validated arg resolution)
    result.record = buildStructuredExecutionRecord(j, base_slot);

    // 2. Render canonical text with bound-span provenance
    auto render = renderWithSpans(j);
    result.canonical_text = render.text;

    // ── Render-order structural invariant ──
    // Verify literal_spans are monotonically ordered and index-aligned.
    // A renderer bug or reordering would silently break provenance
    // pairing with bootstrap_bindings (which pair by index).
    for (size_t i = 0; i < render.literal_spans.size(); ++i) {
        const auto& span = render.literal_spans[i];
        if (span.binding_index != static_cast<int>(i)) {
            throw std::runtime_error(
                "buildConceptSequence: literal_spans[" + std::to_string(i)
                + "].binding_index=" + std::to_string(span.binding_index)
                + " (expected " + std::to_string(i)
                + ") — renderer emitted bindings out of order");
        }
        if (span.byte_start >= span.byte_end) {
            throw std::runtime_error(
                "buildConceptSequence: literal_spans[" + std::to_string(i)
                + "] has empty range [" + std::to_string(span.byte_start)
                + "," + std::to_string(span.byte_end) + ")");
        }
        if (i > 0 && span.byte_start < render.literal_spans[i - 1].byte_end) {
            throw std::runtime_error(
                "buildConceptSequence: literal_spans[" + std::to_string(i)
                + "].byte_start=" + std::to_string(span.byte_start)
                + " overlaps previous span ending at "
                + std::to_string(render.literal_spans[i - 1].byte_end)
                + " — renderer produced overlapping or non-monotonic spans");
        }
    }

    int execution_prompt_length = 0;
    int execution_prompt_end_pos = -1;
    if (render.execution_prompt_byte_end > 0) {
        const std::string rendered_prompt =
            result.canonical_text.substr(0, render.execution_prompt_byte_end);
        auto prompt_encoded = tokenizer.tokenizeWithMetadata(rendered_prompt);
        execution_prompt_length = static_cast<int>(prompt_encoded.token_ids.size());
        execution_prompt_end_pos = execution_prompt_length - 1;
    }
    if (result.record.execution_gate_target != Execution::ExecutionGateTarget::IGNORE
        && execution_prompt_length <= 0) {
        throw std::runtime_error(
            "buildConceptSequence: supervised execution gate target requires a non-empty prompt");
    }

    if (!result.record.execution_active) {
        result.payload.execution_active = false;
        result.payload.execution_gate_target = result.record.execution_gate_target;
        result.payload.execution_prompt_length = execution_prompt_length;
        result.payload.execution_prompt_end_pos = execution_prompt_end_pos;
        return result;
    }

    // 3. Tokenize the canonical text
    auto encoded = tokenizer.tokenizeWithMetadata(result.canonical_text);
    if (encoded.token_ids.empty()) {
        throw std::runtime_error(
            "buildConceptSequence: tokenization produced zero tokens for concept row");
    }

    const int seq_len = static_cast<int>(encoded.token_ids.size());
    if (execution_prompt_length > seq_len) {
        throw std::runtime_error(
            "buildConceptSequence: prompt token count exceeds full sequence length");
    }

    // 4. Build token-position → StructuralSpan correlation.
    //    The encoder pushes to result.atoms in the same order as atom tokens
    //    appear in the output. Walking token_atom_mask and atoms in parallel
    //    yields a per-token span pointer.
    std::vector<const Tokenizer::StructuralSpan*> token_to_span(seq_len, nullptr);
    size_t atom_span_idx = 0;
    for (int t = 0; t < seq_len; ++t) {
        if (encoded.token_atom_mask[t]) {
            if (atom_span_idx >= encoded.atoms.size()) {
                throw std::runtime_error(
                    "buildConceptSequence: atom token at position " + std::to_string(t)
                    + " has no corresponding StructuralSpan (atoms.size()="
                    + std::to_string(encoded.atoms.size()) + ")");
            }
            token_to_span[t] = &encoded.atoms[atom_span_idx];
            atom_span_idx++;
        }
    }
    if (atom_span_idx != encoded.atoms.size()) {
        throw std::runtime_error(
            "buildConceptSequence: " + std::to_string(encoded.atoms.size())
            + " StructuralSpans but only " + std::to_string(atom_span_idx)
            + " atom tokens found");
    }

    // 5. Compile the record against tokenized output using span provenance
    result.payload = compileExecutionPayload(
        result.record,
        render.literal_spans,
        result.canonical_text,
        encoded.token_ids,
        token_to_span,
        seq_len,
        execution_prompt_end_pos,
        execution_prompt_length);

    return result;
}

}  // namespace DataLoader
}  // namespace GRIM
