#include "OutputUnigramPrior.hpp"

#include "../Phase1_Startup.hpp"
#include "../../../Shared/HyperParameters/HyperParameters_GPU.hpp"
#include "../../../Shared/LogRecorder/LogRecorder.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace GRIMText::Training {

void authorOutputUnigramPrior(TrainingContext& ctx, std::uint32_t vocab_size) {
	const bool enabled = GRIM::HyperParameters::snapshotTrainingConfigField<bool>(
		ctx.config,
		"lm_head_unigram_bias");
	if (!enabled) {
		ctx.data.output_unigram_prior = {};
		return;
	}
	if (ctx.model_parameter_source_plan == ModelParameterSourcePlan::UNRESOLVED) {
		throw std::runtime_error(
			"authorOutputUnigramPrior: parameter source is unresolved; CheckpointPlanReady must run before training-data initialization");
	}
	if (ctx.model_parameter_source_plan == ModelParameterSourcePlan::CHECKPOINT_RESTORE) {
		ctx.data.output_unigram_prior = {};
		GRIM::Logging::EmitModuleInfo(
			GRIM::Logging::ModuleId::Training,
			"[DataLoader] lm_head_unigram_bias: skipped corpus prior authoring because checkpoint restore owns the bias value",
			0);
		return;
	}

	if (vocab_size == 0) {
		throw std::runtime_error("authorOutputUnigramPrior: vocab_size must be positive");
	}

	std::vector<double> counts(static_cast<std::size_t>(vocab_size), 0.0);
	double total_targets = 0.0;
	for (const auto& seq : ctx.data.train_seqs) {
		for (int target : seq.targets) {
			if (target >= 0 && target < static_cast<int>(vocab_size)) {
				counts[static_cast<std::size_t>(target)] += 1.0;
				total_targets += 1.0;
			}
		}
	}
	if (total_targets <= 0.0) {
		throw std::runtime_error(
			"authorOutputUnigramPrior: no valid training targets to estimate p(v)");
	}

	constexpr double kSmoothing = 1.0;
	const double denom = total_targets + kSmoothing * static_cast<double>(vocab_size);
	auto& prior = ctx.data.output_unigram_prior;
	prior.log_bias.assign(static_cast<std::size_t>(vocab_size), 0.0f);
	prior.vocab_size = vocab_size;
	prior.seen_tokens = 0;
	prior.total_targets = static_cast<std::uint64_t>(total_targets);
	for (std::uint32_t token = 0; token < vocab_size; ++token) {
		const double count = counts[static_cast<std::size_t>(token)];
		const double p = (count + kSmoothing) / denom;
		prior.log_bias[static_cast<std::size_t>(token)] = static_cast<float>(std::log(p));
		if (count > 0.0) {
			++prior.seen_tokens;
		}
	}

	GRIM::Logging::EmitModuleInfo(
		GRIM::Logging::ModuleId::Training,
		"[DataLoader] lm_head_unigram_bias: authored output unigram prior | vocab=" +
			std::to_string(prior.vocab_size) + " seen_tokens=" + std::to_string(prior.seen_tokens) +
			" total_targets=" + std::to_string(prior.total_targets),
		0);
}

} // namespace GRIMText::Training
