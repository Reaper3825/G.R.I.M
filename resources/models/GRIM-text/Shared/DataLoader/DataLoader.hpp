#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include "../Curriculum/CurriculumMetadata.hpp"
#include "../HyperParameters/HyperparameterGroupings.hpp"

class TrainingLogger;
namespace GRIM { namespace Tokenizer { class UniByte; } }

namespace GRIM {

CurriculumMetadata LoadCurriculumMetadataFromRegistry(
    const std::filesystem::path& directory,
    const std::string& curriculum_name);

// Prepares GRIM-text training data from curriculum concept blocks.
// - Writes `TokenizerHP::output_data_path`; training separately consumes
//   `TokenizerHP::data_path`, so multiple GRMTs may share one vocab.
// - A valid existing output is cached. A missing/mismatched output is rebuilt
//   with the shared vocab unless `force_rebuild_vocab=true` explicitly allows
//   tokenizer retraining.
// - Writes GRMT sequences with per-token numeric side-channel data.
// - Loads `concept_blocks.fb`, renders canonical learning text, and calls
//   UniByte's tokenization entry point once per sequence.
// - Structured execution compilation is intentionally stubbed until state
//   atoms bind directly from AtomTable entries to execution slots.
// - Resolves curriculum_registry.json: curriculum course_ids -> courses -> concept_block_ids.
//
// Returns true on success. Throws on fatal errors (missing concept blocks).
bool PrepareTrainingDataFromCache(const HyperParameters::TokenizerHP& tokenizer_hp);

} // namespace GRIM

namespace GRIMText::Training {

struct TrainingContext;

void syncRuntimeVocabSizeFromActualOrThrow(
	GRIM::Config::AiConfigSnapshot& config,
	std::uint32_t actual_vocab_size,
	const char* caller);

std::unique_ptr<GRIM::Tokenizer::UniByte> LoadInferenceTokenizer(
	const GRIM::Config::AiConfigSnapshot& config,
	::TrainingLogger& logger);

void LoadTrainingData(TrainingContext& ctx);

} // namespace GRIMText::Training

