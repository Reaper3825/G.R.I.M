#pragma once

#include <memory>

#include "../HyperParameters/HyperparameterGroupings.hpp"

class TrainingLogger;
namespace GRIM { namespace Tokenizer { class UniByte; } }

namespace GRIM {

// Prepares GRIM-text training data from curriculum concept blocks.
// - Writes `TokenizerHP::output_data_path`; training separately consumes
//   `TokenizerHP::data_path`, so multiple GRMTs may share one vocab.
// - A valid existing output is cached. A missing/mismatched output is rebuilt
//   with the shared vocab unless `force_rebuild_vocab=true` explicitly allows
//   tokenizer retraining.
// - Writes GRMT sequences with per-token numeric side-channel data.
// - Loads `concept_blocks.fb` via ConceptExecutionSequenceBuilder:
//   canonical structured execution records with paired bootstrap/teacher payloads.
// - Optionally filtered by `curriculum_manifest.json` (concept_block_ids).
//
// Returns true on success. Throws on fatal errors (missing concept blocks).
bool PrepareTrainingDataFromCache(const HyperParameters::TokenizerHP& tokenizer_hp);

} // namespace GRIM

namespace GRIMText::Training {

struct TrainingContext;
struct MemorySnapshot;

void syncRuntimeVocabSizeFromActualOrThrow(
	GRIM::Config::AiConfigSnapshot& config,
	std::uint32_t actual_vocab_size,
	const char* caller);

std::unique_ptr<GRIM::Tokenizer::UniByte> LoadInferenceTokenizer(
	const GRIM::Config::AiConfigSnapshot& config,
	::TrainingLogger& logger);

void LoadTrainingData(TrainingContext& ctx, const MemorySnapshot& startup_memory_snapshot);

} // namespace GRIMText::Training

