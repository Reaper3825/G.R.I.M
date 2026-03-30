//======================================================//
//  ConceptExecutionSequenceBuilder.hpp
//  Canonical builder: concept JSON → StructuredExecutionRecord
//  → compiled payload → TrainingSequence execution fields.
//
//  This is the ONLY builder that emits execution-active
//  concept rows into TrainingSequence. The old __SLOTS__
//  tail block, tail-number slot recovery, and the superseded
//  slot/teacher-step helper paths are deleted.
//
//  Owns: concept JSON parsing into canonical types, canonical
//  text rendering with bound-span provenance, bootstrap binding
//  compilation via character-offset intersection with tokenizer
//  StructuralSpan data, paired emission of token_exec_slots
//  + teacher_steps + compiled_bootstrap_bindings from one pass.
//
//  Must NOT own: batch padding, GPU execution, GRMT IO,
//  validation logic, selector layer weights.
//
//  Provenance guarantees:
//    - Each bootstrap literal maps to ONE rendered character span
//    - Each rendered span maps to EXACTLY ONE ATOM_NUM token via
//      StructuralSpan::content_offset intersection (not doc-order)
//    - Coordinate-system alignment is PROVEN at runtime: atom
//      detector content bytes are compared to rendered literal
//      bytes after each match (crashes on divergence)
//    - Render-order invariant: literal_spans are verified
//      monotonically ordered with binding_index == array index
//      (catches any renderer reordering that would break pairing)
//    - Arg slot resolution: canonical when JSON carries "arg_slots"
//      (explicit slot indices); value-based with result validation
//      as fallback when JSON carries values only
//======================================================//

#pragma once

#include <string>
#include <vector>
#include <cstdint>

#include "../Execution/ExecutionMetadata.hpp"

// Forward-declare nlohmann::json to avoid header dependency here.
// Callers that need the full type include <nlohmann/json.hpp> themselves.
namespace nlohmann { class json; }

namespace GRIM {

namespace Tokenizer { class UniByte; }

namespace DataLoader {

// ─── Provenance types ───────────────────────────────────

// Byte-range provenance for one bootstrap literal rendered into the
// canonical text. Populated by renderWithSpans(); consumed by the
// compilation step to locate the unique ATOM_NUM token for each binding.
struct RenderedLiteralSpan {
    int binding_index;     // Index in bootstrap_bindings
    size_t byte_start;     // Byte offset in rendered text (inclusive)
    size_t byte_end;       // Byte offset in rendered text (exclusive)
};

// Render result carrying both the canonical text and the character-offset
// provenance for every bootstrap literal placed in that text.
struct CanonicalRenderResult {
    std::string text;
    std::vector<RenderedLiteralSpan> literal_spans;
};

// Result of building one concept row.
// Contains the canonical rendered text and the compiled execution payload.
struct ConceptBuildResult {
    std::string canonical_text;
    Execution::StructuredExecutionRecord record;
    Execution::CompiledStructuredExecutionPayload payload;
};

// ─── Public API ─────────────────────────────────────────

// Parse concept JSON into a StructuredExecutionRecord.
// Arg slot resolution uses exact double equality with explicit candidate
// enumeration and result validation. Throws on:
//   - execution-active with zero bootstrap bindings
//   - duplicate slot_id initialization
//   - execution step with < 2 args
//   - unknown op string
//   - ambiguous arg slot resolution (multiple valid combinations)
//   - arg slot resolution produces wrong result
Execution::StructuredExecutionRecord
buildStructuredExecutionRecord(const nlohmann::json& j, int base_slot);

// Render canonical training text with bound-span provenance.
// STATE0 section renders bootstrap literal values explicitly so
// the tokenizer can detect them; character offsets are tracked
// in literal_spans for downstream compilation.
CanonicalRenderResult renderWithSpans(const nlohmann::json& j);

// Render canonical training text (plain text, no provenance).
// Convenience wrapper around renderWithSpans() for vocab corpus building.
std::string renderCanonicalText(const nlohmann::json& j);

// Full pipeline: parse JSON → build record → render text → tokenize → compile.
// Compilation maps bootstrap bindings to ATOM_NUM token positions via
// character-offset intersection between RenderedLiteralSpan and
// StructuralSpan::content_offset (not document-order claiming).
// Returns ConceptBuildResult with canonical text and compiled payload.
// Throws on any structural violation.
ConceptBuildResult buildConceptSequence(
    const nlohmann::json& j,
    GRIM::Tokenizer::UniByte& tokenizer,
    int base_slot);

// Convert op string to numeric op_id.
// Throws on unknown op string.
int opStringToId(const std::string& op);

}  // namespace DataLoader
}  // namespace GRIM
