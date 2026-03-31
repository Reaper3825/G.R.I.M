#pragma once

#include <string>

namespace GRIM {

namespace Config { struct GrimTextPaths; }

// Prepares GRIM-text training data from curriculum concept blocks.
// - Uses `GrimTextPaths` (ai_config.json) for all paths.
// - If a fresh GRMT already exists and `force_rebuild` is false, it is left as-is.
// - Writes GRMT sequences with per-token numeric side-channel data.
// - Loads `concept_blocks.jsonl` via ConceptExecutionSequenceBuilder:
//   canonical structured execution records with paired bootstrap/teacher payloads.
// - Optionally filtered by `curriculum_manifest.json` (concept_block_ids).
// - On success, updates `out_training_data_path` (and optionally `out_vocab_path`).
//
// Returns true on success. Throws on fatal errors (missing concept blocks).
bool PrepareTrainingDataFromCache(
	const Config::GrimTextPaths& paths,
	std::string& out_training_data_path,
	std::string& out_vocab_path,
	bool force_rebuild = false);

} // namespace GRIM

