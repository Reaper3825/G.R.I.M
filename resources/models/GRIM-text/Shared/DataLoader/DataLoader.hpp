#pragma once

#include <string>

namespace GRIM {

namespace Config { struct GrimTextPaths; }

// Prepares GRIM-text training data from cached checkpoints/verified data.
// - Uses `GrimTextPaths` (ai_config.json) for all paths.
// - If a fresh GRMT already exists and `force_rebuild` is false, it is left as-is.
// - Writes GRMT sequences with per-token numeric side-channel data.
// - On success, updates `out_training_data_path` (and optionally `out_vocab_path`)
//   and can clear consumed cache directories when `clear_cache` is true.
//
// Returns true on success, false on non-fatal issues (e.g. no cache present).
bool PrepareTrainingDataFromCache(
	const Config::GrimTextPaths& paths,
	std::string& out_training_data_path,
	std::string& out_vocab_path,
	bool force_rebuild = false,
	bool clear_cache = true);

} // namespace GRIM

