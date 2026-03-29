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
//  text rendering (no __SLOTS__), bootstrap binding compilation
//  against tokenized output, paired emission of token_exec_slots
//  + teacher_steps + compiled_bootstrap_bindings from one pass.
//
//  Must NOT own: batch padding, GPU execution, GRMT IO,
//  validation logic, selector layer weights.
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

// Result of building one concept row.
// Contains the canonical rendered text and the compiled execution payload.
struct ConceptBuildResult {
    std::string canonical_text;
    Execution::StructuredExecutionRecord record;
    Execution::CompiledStructuredExecutionPayload payload;
};

// ─── Public API ─────────────────────────────────────────

// Parse concept JSON into a StructuredExecutionRecord.
// Throws on malformed JSON or structural violations:
//   - execution-active with zero bootstrap bindings
//   - duplicate slot_id initialization
//   - execution step with < 2 args
//   - unknown op string
Execution::StructuredExecutionRecord
buildStructuredExecutionRecord(const nlohmann::json& j, int base_slot);

// Render canonical training text from concept JSON.
// NO __SLOTS__ block. Human-readable Q/A/EXEC format only.
std::string renderCanonicalText(const nlohmann::json& j);

// Compile a StructuredExecutionRecord against tokenized output.
// Populates token_exec_slots, compiled_bootstrap_bindings, teacher_steps,
// and slot_selection_targets from the same builder pass.
//
// Requires each bound literal span to compile to exactly one ATOM_NUM token.
// Rejects duplicate bootstrap slot_id or duplicate compiled token_pos.
// Rejects execution-active rows with zero bootstrap bindings.
//
// Parameters:
//   record      - canonical structured record (from buildStructuredExecutionRecord)
//   token_ids   - tokenized output (from tokenizer.encodeWithMetadata)
//   atom_mask   - per-token atom mask (from tokenizer)
//   numeric_values - per-token numeric values (from tokenizer)
//   seq_len     - number of tokens
//
// Returns compiled payload with all fields populated.
Execution::CompiledStructuredExecutionPayload
compileExecutionPayload(
    const Execution::StructuredExecutionRecord& record,
    const std::vector<int>& token_ids,
    const std::vector<uint8_t>& atom_mask,
    const std::vector<float>& numeric_values,
    int seq_len);

// Full pipeline: parse JSON → build record → render text → tokenize → compile.
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
