#include "DataLoader.hpp"

#include "CurriculumRegistry.hpp"
#include "../GRMT/GrmtFormat.hpp"
#include "../LogRecorder/LogRecorder.hpp"
#include "../UnigramByte/TokenLayout.hpp"
#include "../UnigramByte/UniByte.hpp"
#include "../TokenizerArtifacts/GrmtCorpusIO.hpp"
#include "../TokenizerArtifacts/TokenizerArtifactBundle.hpp"
#include "../HyperParameters/HyperParameters_GPU.hpp"
#include "../HyperParameters/HyperparameterGroupings.hpp"
#include "../../training/Phases/Startup/SlidingWindow.hpp"
#include "../../training/Phases/Startup/OutputUnigramPrior.hpp"
#include "../../training/Phases/ConfigDump.hpp"
#include "../../training/Phases/Phase1_Startup.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace GRIMText::Training {

namespace Internal {

void validateStartupPaths(
	const GRIM::HyperParameters::TokenizerHP& tokenizer_hp,
	const GRIM::HyperParameters::PathsHP& paths_hp)
{
	if (!fs::exists(tokenizer_hp.vocab_path)) {
		throw std::runtime_error("Vocabulary file does not exist: " + tokenizer_hp.vocab_path);
	}
	if (!fs::exists(tokenizer_hp.data_path)) {
		throw std::runtime_error("Training data file does not exist: " + tokenizer_hp.data_path);
	}
	if (paths_hp.output_model_path.empty()) {
		throw std::runtime_error("Output model path not configured");
	}
	if (paths_hp.checkpoint_dir.empty()) {
		throw std::runtime_error("Checkpoint directory not configured");
	}
	if (paths_hp.log_dir.empty()) {
		throw std::runtime_error("Log directory not configured");
	}

	auto model_parent = fs::path(paths_hp.output_model_path).parent_path();
	if (!model_parent.empty()) {
		fs::create_directories(model_parent);
	}
	fs::create_directories(paths_hp.checkpoint_dir);
	fs::create_directories(paths_hp.log_dir);
}

namespace {

using GrmtSequence = GRIM::TokenizerArtifacts::GrmtSequence;
using ProgressCallback = std::function<void(const std::string&)>;

struct LoadedTrainingCorpus {
	std::vector<GrmtSequence> sequences;
	std::uint32_t vocab_size = 0;
};

void emitProgress(const ProgressCallback& progress, const std::string& message)
{
	if (progress) {
		progress(message);
	}
}

LoadedTrainingCorpus readGrmtCorpusWithProgressOrThrow(
	const std::string& path,
	const ProgressCallback& progress)
{
	GRIM::TokenizerArtifacts::GrmtCorpusReader reader(path);
	const GRIM::GRMT::Header header = reader.header();

	std::ostringstream header_msg;
	header_msg << "[Data] GRMT header loaded: sequences=" << header.num_sequences
	           << " vocab_size=" << header.vocab_size
	           << " version=" << header.version;
	emitProgress(progress, header_msg.str());

	LoadedTrainingCorpus corpus;
	corpus.vocab_size = header.vocab_size;
	corpus.sequences.reserve(header.num_sequences);

	const std::uint32_t progress_stride =
		(header.num_sequences >= 20u)
			? std::max<std::uint32_t>(1u, header.num_sequences / 10u)
			: header.num_sequences;
	std::uint32_t next_progress = progress_stride;
	const auto load_started_at = std::chrono::steady_clock::now();

	GrmtSequence sequence;
	while (reader.readNext(sequence)) {
		corpus.sequences.push_back(std::move(sequence));
		sequence = GrmtSequence{};

		const std::uint32_t loaded = reader.sequencesRead();
		if (header.num_sequences > 0 &&
			(loaded == header.num_sequences ||
			 (progress_stride > 0 && loaded >= next_progress))) {
			std::ostringstream progress_msg;
			const double elapsed_seconds = std::chrono::duration<double>(
				std::chrono::steady_clock::now() - load_started_at).count();
			if (!(elapsed_seconds > 0.0)) {
				throw std::runtime_error("[DataLoader] GRMT load timer did not advance");
			}
			const double sequences_per_second =
				static_cast<double>(loaded) / elapsed_seconds;
			const double remaining_seconds =
				static_cast<double>(header.num_sequences - loaded) / sequences_per_second;
			progress_msg << "[Data] GRMT load progress: "
			             << loaded << "/" << header.num_sequences
			             << " sequences ("
			             << std::fixed << std::setprecision(1)
			             << (100.0 * static_cast<double>(loaded) /
			                 static_cast<double>(header.num_sequences))
			             << "%) | " << sequences_per_second << " seq/s"
			             << " | ETA " << remaining_seconds << "s";
			emitProgress(progress, progress_msg.str());
			if (loaded < header.num_sequences && progress_stride > 0) {
				next_progress = std::min(header.num_sequences, loaded + progress_stride);
			}
		}
	}

	if (corpus.sequences.size() != header.num_sequences) {
		throw std::runtime_error("[DataLoader] GRMT loaded sequence count mismatch: loaded=" +
			std::to_string(corpus.sequences.size()) +
			" header=" + std::to_string(header.num_sequences));
	}

	std::cout << "[DataLoader] GRMT version " << header.version << std::endl;
	std::cout << "[DataLoader] Sequences: " << header.num_sequences << std::endl;
	std::cout << "[DataLoader] Vocab size: " << header.vocab_size << std::endl;

	return corpus;
}

void sanitizeNumericSideChannels(std::vector<GrmtSequence>& sequences)
{
	size_t nonfinite_total = 0;
	size_t nonfinite_sequences = 0;

	for (auto& seq : sequences) {
		const std::uint32_t seq_len = static_cast<std::uint32_t>(seq.token_ids.size());
		if (seq.token_numeric_values.size() != seq.token_ids.size()) {
			throw std::runtime_error("[DataLoader] token_numeric_values length mismatch during GRMT side-channel validation");
		}
		if (seq.token_atom_mask.size() != seq.token_ids.size()) {
			throw std::runtime_error("[DataLoader] token_atom_mask length mismatch during GRMT side-channel validation");
		}

		size_t seq_nonfinite = 0;
		for (std::uint32_t j = 0; j < seq_len; ++j) {
			if (seq.token_atom_mask[j] && !std::isfinite(seq.token_numeric_values[j])) {
				seq.token_numeric_values[j] = 0.0f;
				++seq_nonfinite;
			}
		}
		if (seq_nonfinite > 0) {
			nonfinite_total += seq_nonfinite;
			nonfinite_sequences++;
		}
	}

	if (nonfinite_total > 0) {
		std::cerr << "[DataLoader] Sanitized " << nonfinite_total
		          << " non-finite numeric values across " << nonfinite_sequences
		          << " sequences (values zeroed)" << std::endl;
	}
}

void logAtomSideChannelDiagnostics(const std::vector<GrmtSequence>& sequences)
{
	size_t total_tokens_loaded = 0;
	size_t atom_tokens_total = 0;
	size_t atom_sequences = 0;
	std::unordered_map<int, size_t> atom_type_counts;
	size_t atom_entries_total = 0;

	for (const auto& seq : sequences) {
		total_tokens_loaded += seq.token_ids.size();
		bool seq_has_atoms = false;
		for (size_t j = 0; j < seq.token_ids.size(); ++j) {
			if (j < seq.token_atom_mask.size() && seq.token_atom_mask[j]) {
				atom_tokens_total++;
				seq_has_atoms = true;
				atom_type_counts[seq.token_ids[j]]++;
			}
			if (j < seq.atom_entry_ids.size() &&
				seq.atom_entry_ids[j] != GRIM::Tokenizer::kAtomEntryNone) {
				atom_entries_total++;
			}
		}
		if (seq_has_atoms) {
			atom_sequences++;
		}
	}

	std::cerr << "[DataLoader] Atom side-channel stats:" << std::endl
	          << "  Total tokens: " << total_tokens_loaded << std::endl
	          << "  Atom tokens: " << atom_tokens_total
	          << " (" << (total_tokens_loaded > 0
	              ? (100.0 * atom_tokens_total / total_tokens_loaded) : 0.0)
	          << "% of tokens)" << std::endl
	          << "  Sequences with atoms: " << atom_sequences
	          << "/" << sequences.size() << std::endl
	          << "  AtomTable entries reconstructed: " << atom_entries_total << std::endl;
	if (!atom_type_counts.empty()) {
		std::cerr << "  Atom type breakdown:" << std::endl;
		for (const auto& [tid, count] : atom_type_counts) {
			auto type = GRIM::Tokenizer::tokenIdToAtomType(tid);
			std::cerr << "    " << GRIM::Tokenizer::atomTypeName(type)
			          << " (token " << tid << "): " << count << std::endl;
		}
	}
	if (atom_tokens_total == 0) {
		std::cerr << "[DataLoader] WARNING: Zero atom tokens in GRMT! "
		          << "Atom detection may not have been enabled during encoding. "
		          << "Delete .grmt files and regenerate with tokenizer_enable_atom_reasoning=true"
		          << std::endl;
	}
}

} // namespace

SequenceData buildPhase1SequenceData(
	const GRIM::HyperParameters::TokenizerHP& tokenizer_hp,
	const GRIM::HyperParameters::DataLoadingHP& data_hp,
	::TrainingLogger& logger)
{
	const int max_seq_len = tokenizer_hp.max_seq_len;
	if (max_seq_len <= 0) {
		throw std::runtime_error(
			"buildPhase1SequenceData: TokenizerHP.max_seq_len must be configured before sliding-window data loading (got " +
			std::to_string(max_seq_len) + ")");
	}

	SequenceData data;

	logger.log("[Data] Loading GRMT corpus from " + tokenizer_hp.data_path + "...");
	auto progress_logger = [&logger](const std::string& message) {
		logger.log(message);
	};
	auto corpus = readGrmtCorpusWithProgressOrThrow(
		tokenizer_hp.data_path,
		progress_logger);
	emitProgress(progress_logger, "[Data] GRMT deserialization complete; validating side channels...");
	sanitizeNumericSideChannels(corpus.sequences);
	logAtomSideChannelDiagnostics(corpus.sequences);
	{
		std::ostringstream done_msg;
		done_msg << "[Data] GRMT validation complete: loaded_sequences=" << corpus.sequences.size()
		         << " vocab_size=" << corpus.vocab_size;
		emitProgress(progress_logger, done_msg.str());
	}

	data.vocab_size = corpus.vocab_size;
	const std::size_t raw_sequence_count = corpus.sequences.size();

	logger.log("[Data] Loaded raw GRMT sequences: count=" +
			   std::to_string(raw_sequence_count) +
			   " vocab_size=" + std::to_string(data.vocab_size));

	std::size_t val_size = corpus.sequences.size() / 10;
	data.val_seqs.assign(
		std::make_move_iterator(corpus.sequences.begin()),
		std::make_move_iterator(corpus.sequences.begin() + static_cast<std::ptrdiff_t>(val_size)));
	data.train_seqs.assign(
		std::make_move_iterator(corpus.sequences.begin() + static_cast<std::ptrdiff_t>(val_size)),
		std::make_move_iterator(corpus.sequences.end()));
	logger.log("[Data] Train/val split ready: train_sequences=" +
			   std::to_string(data.train_seqs.size()) +
			   " val_sequences=" + std::to_string(data.val_seqs.size()) +
			   " holdout_ratio=10%");
	const std::size_t pre_window_train_count = data.train_seqs.size();

	logger.log("[Data] Applying sliding windows to train split...");
	applySlidingWindows(data.train_seqs, "train",
						data_hp.training_stage,
						data_hp.supervised_fields,
						data_hp.unsupervised_fields,
						max_seq_len, data_hp.sliding_window_stride, data_hp.min_seq_valid_tokens,
						tokenizer_hp.add_bos, tokenizer_hp.add_eos, logger);
	logger.log("[Data] Train split post-window sequence count=" +
			   std::to_string(data.train_seqs.size()));
	if (data.train_seqs.empty()) {
		throw std::runtime_error(
			"buildPhase1SequenceData: train split became empty during sliding-window/filter processing "
			"(raw_corpus_sequences=" + std::to_string(raw_sequence_count) +
			", pre_window_train_sequences=" + std::to_string(pre_window_train_count) +
			", max_seq_len=" + std::to_string(max_seq_len) +
			", sliding_window_stride=" + std::to_string(data_hp.sliding_window_stride) +
			", min_seq_valid_tokens=" + std::to_string(data_hp.min_seq_valid_tokens) +
			"). Inspect the preceding [FILTER] log; lower min_seq_valid_tokens if valid short rows are being removed.");
	}

	logger.log("[Data] Applying sliding windows to validation split...");
	applySlidingWindows(data.val_seqs, "val",
						data_hp.training_stage,
						data_hp.supervised_fields,
						data_hp.unsupervised_fields,
						max_seq_len, data_hp.sliding_window_stride, data_hp.min_seq_valid_tokens,
						tokenizer_hp.add_bos, tokenizer_hp.add_eos, logger);
	logger.log("[Data] Validation split post-window sequence count=" +
			   std::to_string(data.val_seqs.size()));

	logger.log("[Data] Materializing train/val sequence views for batching...");
	data.train_views.reserve(data.train_seqs.size());
	data.train_seq_lengths.reserve(data.train_seqs.size());
	for (std::size_t i = 0; i < data.train_seqs.size(); ++i) {
		data.train_views.push_back(&data.train_seqs[i]);
		const uint32_t len = static_cast<uint32_t>(data.train_seqs[i].token_ids.size());
		data.train_seq_lengths.push_back(len);
	}

	data.val_views.reserve(data.val_seqs.size());
	data.val_seq_lengths.reserve(data.val_seqs.size());
	for (std::size_t i = 0; i < data.val_seqs.size(); ++i) {
		data.val_views.push_back(&data.val_seqs[i]);
		const uint32_t len = static_cast<uint32_t>(data.val_seqs[i].token_ids.size());
		data.val_seq_lengths.push_back(len);
	}

	logger.log("[Data] Sequence views ready: train_views=" +
			   std::to_string(data.train_views.size()) +
			   " val_views=" + std::to_string(data.val_views.size()));

	return data;
}

} // namespace Internal

namespace {

std::uint32_t requireActualVocabSizeOrThrow(const SequenceData& data)
{
	if (data.vocab_size == 0) {
		throw std::runtime_error("FATAL: training data missing vocab_size; regenerate GRMT with tokenizer.vocabSize()");
	}
	if (data.train_views.size() != data.train_seqs.size()) {
		throw std::runtime_error("FATAL: train view count does not match train sequence count (views=" +
								 std::to_string(data.train_views.size()) +
								 " seqs=" + std::to_string(data.train_seqs.size()) + ")");
	}
	if (data.val_views.size() != data.val_seqs.size()) {
		throw std::runtime_error("FATAL: val view count does not match val sequence count (views=" +
								 std::to_string(data.val_views.size()) +
								 " seqs=" + std::to_string(data.val_seqs.size()) + ")");
	}

	return data.vocab_size;
}

} // namespace

void syncRuntimeVocabSizeFromActualOrThrow(
	GRIM::Config::AiConfigSnapshot& config,
	std::uint32_t actual_vocab_size,
	const char* caller)
{
	if (actual_vocab_size < static_cast<std::uint32_t>(GRIM::Tokenizer::UNIGRAM_VOCAB_OFFSET)) {
		throw std::runtime_error(std::string(caller) +
			": actual_vocab_size must include special+byte+atom ranges (>= " +
			std::to_string(GRIM::Tokenizer::UNIGRAM_VOCAB_OFFSET) + "), got " +
			std::to_string(actual_vocab_size));
	}
	if (actual_vocab_size > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
		throw std::runtime_error(std::string(caller) +
			": actual_vocab_size=" + std::to_string(actual_vocab_size) +
			" exceeds int capacity for AiConfigSnapshot training.config.vocab_size");
	}
	GRIM::HyperParameters::setSnapshotRuntimeVocabSize(
		config,
		static_cast<int>(actual_vocab_size),
		caller);
}

std::unique_ptr<GRIM::Tokenizer::UniByte> LoadInferenceTokenizer(
	const GRIM::Config::AiConfigSnapshot& config,
	::TrainingLogger& logger)
{
	const auto tokenizer_hp = GRIM::HyperParameters::tokenizerHP(config);

	logger.log("Loading shared tokenizer vocabulary...");
	auto tokenizer = std::make_unique<GRIM::Tokenizer::UniByte>(tokenizer_hp);
	(void)GRIM::TokenizerArtifacts::loadSharedTokenizerVocabulary(tokenizer_hp, *tokenizer);
	logger.log("Initializing tokenizer CUDA Viterbi runtime...");
	if (!tokenizer->initGPU()) {
		throw std::runtime_error("LoadInferenceTokenizer: UniByte::initGPU() returned false after artifact load");
	}

	return tokenizer;
}

void LoadTrainingData(TrainingContext& ctx) {
	using GRIM::Logging::EmitModuleInfo;
	using GRIM::Logging::ModuleId;

	const auto tokenizer_hp = GRIM::HyperParameters::tokenizerHP(ctx.config);
	const auto paths_hp = GRIM::HyperParameters::pathsHP(ctx.config);
	const auto data_hp = GRIM::HyperParameters::dataLoadingHP(ctx.config);
	if (tokenizer_hp.training_curriculum.empty()) {
		throw std::runtime_error(
			"LoadTrainingData: training_curriculum is empty; select the curriculum represented by grim_text_training_data");
	}
	ctx.current_curriculum_metadata = GRIM::LoadCurriculumMetadataFromRegistry(
		fs::path(tokenizer_hp.data_path).parent_path(),
		tokenizer_hp.training_curriculum);
	const std::string configured_training_stage =
		GRIM::HyperParameters::trainingStageToJsonString(data_hp.training_stage);
	if (ctx.current_curriculum_metadata->training_stage != configured_training_stage) {
		throw std::runtime_error(
			"LoadTrainingData: curriculum training_stage='" +
			ctx.current_curriculum_metadata->training_stage +
			"' does not match configured training_stage='" +
			configured_training_stage + "'");
	}
	EmitModuleInfo(
		ModuleId::Training,
		"[Phase1] Training corpus target | curriculum=" + tokenizer_hp.training_curriculum +
			" | data=" + tokenizer_hp.data_path +
			" | shared_vocab=" + tokenizer_hp.vocab_path,
		0);

 	EmitModuleInfo(ModuleId::Training, "[Phase1] Validating paths...", 0);
 	Internal::validateStartupPaths(tokenizer_hp, paths_hp);
 	EmitModuleInfo(ModuleId::Training, "[Phase1] ✓ All paths validated", 0);

	ctx.data = Internal::buildPhase1SequenceData(
		tokenizer_hp,
		data_hp,
		*ctx.logging.logger);

	const std::uint32_t actual_vocab_size = requireActualVocabSizeOrThrow(ctx.data);
	(void)GRIM::Tokenizer::tokenLayoutFromActualVocabOrThrow(
		actual_vocab_size,
		"LoadTrainingData");
	syncRuntimeVocabSizeFromActualOrThrow(ctx.config, actual_vocab_size, "LoadTrainingData");
	authorOutputUnigramPrior(ctx, actual_vocab_size);

	GRIM::HyperParameters::DerivationContext hp_ctx;
	const auto runtime_hp =
		GRIM::HyperParameters::trainingRuntimeControlHP(ctx.config);
	hp_ctx.train_sequence_count = static_cast<int>(ctx.data.train_seqs.size());
	hp_ctx.validation_interval = runtime_hp.validation_interval;
	ctx.derived_schedule = GRIM::HyperParameters::computeDerivedSchedule(
		ctx.config, hp_ctx);

	GRIMText::Training::DataStatsSnapshot data_stats;
	data_stats.data_path = tokenizer_hp.data_path;
	data_stats.vocab_path = tokenizer_hp.vocab_path;
	data_stats.actual_vocab_size = actual_vocab_size;
	data_stats.train_sequence_count = ctx.data.train_seqs.size();
	data_stats.val_sequence_count = ctx.data.val_seqs.size();

	dumpAllHyperparameters(
		ctx.config,
		&ctx.derived_schedule,
		&data_stats,
		[](const std::string& msg) { GRIM::Logging::EmitModuleInfo(GRIM::Logging::ModuleId::Training, msg, 0); });
}
} // namespace GRIMText::Training
