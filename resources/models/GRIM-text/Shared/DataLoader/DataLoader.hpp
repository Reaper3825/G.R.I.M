#pragma once

#include "../HyperParameters/HyperparameterGroupings.hpp"

namespace GRIM {

// Prepares GRIM-text training data from curriculum concept blocks.
// - Uses `TokenizerHP` for all tokenizer settings and resolved artifact paths.
// - If a fresh GRMT already exists and `TokenizerHP::force_rebuild_vocab` is false, it is left as-is.
// - Writes GRMT sequences with per-token numeric side-channel data.
// - Loads `concept_blocks.jsonl` via ConceptExecutionSequenceBuilder:
//   canonical structured execution records with paired bootstrap/teacher payloads.
// - Optionally filtered by `curriculum_manifest.json` (concept_block_ids).
//
// Returns true on success. Throws on fatal errors (missing concept blocks).
bool PrepareTrainingDataFromCache(const HyperParameters::TokenizerHP& tokenizer_hp);

} // namespace GRIM

