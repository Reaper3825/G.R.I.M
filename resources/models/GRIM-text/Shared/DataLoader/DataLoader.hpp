#pragma once

#include <cstdint>
#include <memory>

#include "../Curriculum/CurriculumMetadata.hpp"
#include "../HyperParameters/HyperparameterGroupings.hpp"

class TrainingLogger;
namespace GRIM { namespace Tokenizer { class UniByte; } }

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

