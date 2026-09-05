#include "DataLoader.hpp"

#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <atomic>
#include <mutex>
#include <optional>
#include <cstdlib>
#include <memory>
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <functional>
#include <iterator>
#include <unordered_map>
#include <cstdint>
#include <exception>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <system_error>
#include <chrono>

#include <nlohmann/json.hpp>
#include "../../../../../DataCollection/concept_block_generated.h"
#include "../../../../../DataCollection/concept_block_canonical.hpp"
#include "../ConceptBlock/ConceptBlockSpans.hpp"
#include "../Goal/Goal.hpp"
#include "../Curriculum/CurriculumMetadata.hpp"
#include "../GRMT/GrmtFormat.hpp"
#include "../LogRecorder/LogRecorder.hpp"
#include "../UnigramByte/TokenLayout.hpp"
#include "../UnigramByte/Unigram.hpp"
#include "../../Shared/UnigramByte/UniByte.hpp"
#include "../../Shared/TokenizerArtifacts/GrmtCorpusIO.hpp"
#include "../../Shared/TokenizerArtifacts/TokenizerArtifactBundle.hpp"
#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"  // also pulls in control/ai_config_paths.hpp transitively (correct order)
#include "../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../Batching/BatchPayload.hpp"
#include "../../training/Phases/Startup/SlidingWindow.hpp"
#include "../../training/Phases/ConfigDump.hpp"
#include "../../training/Phases/Phase1_Startup.hpp"

namespace fs = std::filesystem;

namespace GRIM {

// ─── Concept blocks corpus loading ──────────────────────────────────────────
//
// Loads concept_blocks.fb (with a legacy JSONL fallback during rollout) and
// returns canonical JSON objects. Canonical learning text is rendered here
// and sent directly through UniByte's public tokenization entry point.
//
namespace {

using json = nlohmann::json;

// Curriculum registry decoding. Runtime selection retains only the shared
// metadata object and its unified ConceptBlock membership set.
void addCurriculumMembership(const json& source, CurriculumMetadata& metadata) {
	if (!source.contains("concept_block_ids") || !source["concept_block_ids"].is_array()) {
		throw std::runtime_error(
			"[DataLoader] FATAL: curriculum '" + metadata.name +
			"' is missing the concept_block_ids array");
	}
	for (const auto& id : source["concept_block_ids"]) {
		if (!id.is_string() || id.get_ref<const std::string&>().empty()) {
			throw std::runtime_error(
				"[DataLoader] FATAL: curriculum '" + metadata.name +
				"' contains an invalid concept_block_id");
		}
		metadata.concept_block_ids.insert(id.get<std::string>());
	}
}

void readCurriculumMetadata(const json& source,
	                        CurriculumMetadata& metadata,
	                        const std::string& expected_name) {
	metadata.id = source.value("id", std::string{});
	metadata.name = source.value("name", std::string{});
	metadata.training_stage = source.value("training_stage", std::string{});
	if (metadata.id.empty() || metadata.name.empty() || metadata.name != expected_name) {
		throw std::runtime_error(
			"[DataLoader] FATAL: curriculum registry entry has invalid identity metadata");
	}
	if (metadata.training_stage != "pt" && metadata.training_stage != "sft" &&
		metadata.training_stage != "dpo" && metadata.training_stage != "rlhf") {
		throw std::runtime_error(
			"[DataLoader] FATAL: curriculum '" + metadata.name +
			"' has invalid training_stage '" + metadata.training_stage + "'");
	}
	addCurriculumMembership(source, metadata);
}

// curriculum_registry.json is the sole source of curriculum metadata.
CurriculumMetadata loadCurriculumMetadata(const fs::path& dir, const std::string& curriculum_name) {
	if (curriculum_name.empty()) {
		throw std::runtime_error(
			"[DataLoader] FATAL: curriculum name is required for registry lookup");
	}

	const fs::path registry_path = dir / "curriculum_registry.json";
	std::ifstream registry_input(registry_path);
	if (!registry_input.is_open()) {
		throw std::runtime_error(
			"[DataLoader] FATAL: cannot open curriculum registry: " + registry_path.string());
	}

	json registry;
	try {
		registry = json::parse(registry_input);
	} catch (const json::exception& error) {
		throw std::runtime_error(
			"[DataLoader] FATAL: failed to parse " + registry_path.string() +
			": " + error.what());
	}
	if (!registry.contains("curriculums") || !registry["curriculums"].is_array()) {
		throw std::runtime_error(
			"[DataLoader] FATAL: curriculum registry is missing the curriculums array");
	}

	for (const auto& source : registry["curriculums"]) {
		if (!source.is_object() || source.value("name", std::string{}) != curriculum_name) continue;
		CurriculumMetadata metadata;
		readCurriculumMetadata(source, metadata, curriculum_name);
		if (metadata.concept_block_ids.empty()) {
			throw std::runtime_error(
				"[DataLoader] FATAL: curriculum '" + curriculum_name +
				"' has empty concept_block_ids");
		}
		std::cout << "[DataLoader] Curriculum '" << metadata.name
		          << "' loaded from registry: " << registry_path.string()
		          << ", curriculum_id=" << metadata.id
		          << ", training_stage=" << metadata.training_stage
		          << ", format_as_concept=" << (metadata.formatAsConcept() ? "true" : "false")
		          << ", concept_block_ids=" << metadata.concept_block_ids.size()
		          << std::endl;
		return metadata;
	}

	throw std::runtime_error(
		"[DataLoader] FATAL: curriculum '" + curriculum_name +
		"' was not found in " + registry_path.string());
}

std::string fbString(const flatbuffers::String* value) {
	return value ? value->str() : std::string{};
}

json conceptBlockFlatBufferToJson(const GRIMConcept::ConceptBlock& source) {
	json j;
	j["id"] = fbString(source.id());
	j["name"] = fbString(source.name());
	j["prompt"] = fbString(source.prompt());
	j["answer"] = fbString(source.answer());
	j["raw"] = fbString(source.raw());
	j["format_type"] = fbString(source.format_type());
	j["source_sequence_id"] = fbString(source.source_sequence_id());
	j["timestamp"] = source.timestamp();
	j["intermediate_count"] = source.intermediate_count();
	j["knowns"] = json::array();
	if (const auto* values = source.knowns()) {
		for (const auto* value : *values) j["knowns"].push_back(fbString(value));
	}
	j["unknowns"] = json::array();
	if (const auto* values = source.unknowns()) {
		for (const auto* value : *values) j["unknowns"].push_back(fbString(value));
	}
	if (const auto* goal = source.goal()) {
		json goal_json{{"target_state", fbString(goal->target_state())}};
		goal_json["success_criteria"] = json::array();
		if (const auto* criteria = goal->success_criteria()) {
			for (const auto* entry : *criteria) {
				if (!entry) continue;
				goal_json["success_criteria"].push_back(json{
					{"criterion", fbString(entry->criterion())},
					{"evidence", fbString(entry->evidence())}
				});
			}
		}
		goal_json["constraints"] = json::array();
		if (const auto* constraints = goal->constraints()) {
			for (const auto* constraint : *constraints) {
				goal_json["constraints"].push_back(fbString(constraint));
			}
		}
		j["goal"] = std::move(goal_json);
	}

	j["intermediates"] = json::array();
	if (const auto* values = source.intermediates()) {
		for (const auto* value : *values) j["intermediates"].push_back(fbString(value));
	}
	j["explanation"] = json::array();
	if (const auto* values = source.explanation()) {
		for (const auto* value : *values) j["explanation"].push_back(fbString(value));
	}
	j["step_index"] = json::array();
	if (const auto* values = source.step_index()) {
		for (const auto value : *values) j["step_index"].push_back(value);
	}

	if (const auto* steps = source.execution()) {
		j["execution"] = json::array();
		for (const auto* source_step : *steps) {
			if (!source_step) continue;
			json step;
			step["op"] = fbString(source_step->op());
			step["result"] = source_step->result();
			step["args"] = json::array();
			if (const auto* args = source_step->args()) {
				for (const auto value : *args) step["args"].push_back(value);
			}
			step["arg_slots"] = json::array();
			if (const auto* slots = source_step->arg_slots()) {
				for (const auto value : *slots) step["arg_slots"].push_back(value);
			}
			j["execution"].push_back(std::move(step));
		}
	}

	return j;
}

bool selectedByCurriculum(const std::string& id, const CurriculumMetadata& metadata) {
	return metadata.concept_block_ids.find(id) != metadata.concept_block_ids.end();
}

void loadConceptBlocks(const fs::path& cache_dir,
                       std::vector<json>& out,
                       CurriculumMetadata& out_metadata,
                       const std::string& curriculum_name) {
	const fs::path flatbuffer_path = cache_dir / "concept_blocks.fb";
	const fs::path legacy_path = cache_dir / "concept_blocks.jsonl";
	out_metadata = loadCurriculumMetadata(cache_dir, curriculum_name);

	if (fs::exists(flatbuffer_path)) {
		if (fs::exists(legacy_path)) {
			std::error_code fb_time_error;
			std::error_code legacy_time_error;
			const auto fb_time = fs::last_write_time(flatbuffer_path, fb_time_error);
			const auto legacy_time = fs::last_write_time(legacy_path, legacy_time_error);
			if (!fb_time_error && !legacy_time_error && legacy_time > fb_time) {
				throw std::runtime_error(
					"[DataLoader] FATAL: legacy concept_blocks.jsonl is newer than "
					"concept_blocks.fb. Open the dataset in DataHub to refresh the "
					"FlatBuffer before training.");
			}
		}
		std::ifstream input(flatbuffer_path, std::ios::binary | std::ios::ate);
		if (!input.is_open()) {
			throw std::runtime_error(
				"[DataLoader] FATAL: cannot open " + flatbuffer_path.string());
		}
		const std::streamsize size = input.tellg();
		if (size <= 0) {
			throw std::runtime_error(
				"[DataLoader] FATAL: concept-block FlatBuffer is empty: " +
				flatbuffer_path.string());
		}
		input.seekg(0, std::ios::beg);
		std::vector<uint8_t> buffer(static_cast<size_t>(size));
		if (!input.read(reinterpret_cast<char*>(buffer.data()), size)) {
			throw std::runtime_error(
				"[DataLoader] FATAL: failed to read " + flatbuffer_path.string());
		}

		flatbuffers::Verifier verifier(buffer.data(), buffer.size());
		if (!GRIMConcept::VerifyConceptBlockDatasetBuffer(verifier)) {
			throw std::runtime_error(
				"[DataLoader] FATAL: FlatBuffer verification failed for " +
				flatbuffer_path.string());
		}
		const auto* dataset = GRIMConcept::GetConceptBlockDataset(buffer.data());
		constexpr uint32_t supported_schema_version = 2;
		if (!dataset || dataset->schema_version() > supported_schema_version) {
			throw std::runtime_error(
				"[DataLoader] FATAL: unsupported concept-block schema version " +
				std::to_string(dataset ? dataset->schema_version() : 0));
		}

		size_t total = 0;
		size_t accepted = 0;
		if (const auto* blocks = dataset->blocks()) {
			out.reserve(blocks->size());
			for (const auto* block : *blocks) {
				if (!block) continue;
				++total;
				const std::string id = fbString(block->id());
				if (!selectedByCurriculum(id, out_metadata)) continue;
				out.push_back(conceptBlockFlatBufferToJson(*block));
				++accepted;
			}
		}
		std::cout << "[DataLoader] Loaded " << accepted;
		if (!out_metadata.concept_block_ids.empty()) std::cout << "/" << total;
		std::cout << " concept blocks from " << flatbuffer_path.string() << std::endl;
		return;
	}

	// Rollout fallback only. Once the DataHub migration has produced the FB,
	// a corrupt FB never silently falls back to a stale JSONL dataset.
	std::ifstream input(legacy_path);
	if (!input.is_open()) {
		std::cout << "[DataLoader] No concept_blocks.fb or legacy JSONL at "
		          << cache_dir.string() << "\n";
		return;
	}
	std::cout << "[DataLoader] WARNING: using legacy concept_blocks.jsonl; "
	          << "open the dataset in DataHub to migrate it.\n";

	std::string line;
	size_t total = 0;
	size_t accepted = 0;
	while (std::getline(input, line)) {
		if (line.empty()) continue;
		try {
			auto j = json::parse(line);
			++total;
			if (!selectedByCurriculum(j.value("id", std::string()), out_metadata)) continue;
			out.push_back(std::move(j));
			++accepted;
		} catch (const std::exception& e) {
			std::cerr << "[DataLoader] concept_blocks.jsonl skip line: " << e.what() << "\n";
		}
	}
	std::cout << "[DataLoader] Loaded " << accepted;
	if (!out_metadata.concept_block_ids.empty()) std::cout << "/" << total;
	std::cout << " legacy concept blocks from " << legacy_path.string() << std::endl;
}

}  // namespace

CurriculumMetadata LoadCurriculumMetadataFromRegistry(
	const fs::path& directory,
	const std::string& curriculum_name)
{
	return loadCurriculumMetadata(directory, curriculum_name);
}

bool PrepareTrainingDataFromCache(
	const GRIM::HyperParameters::TokenizerHP& tokenizer_hp) {
	const size_t min_cleaned_text_length = static_cast<size_t>(tokenizer_hp.min_cleaned_text_length);

	if (tokenizer_hp.output_data_path.empty()) {
		std::cerr << "[DataLoader] No tokenizer_output_grmt path configured; skipping cache preparation." << std::endl;
		return false;
	}
	if (tokenizer_hp.vocab_path.empty()) {
		throw std::runtime_error("[DataLoader] vocab path is empty; tokenizer generation requires a shared vocabulary path");
	}
	if (tokenizer_hp.tokenizer_curriculum.empty()) {
		throw std::runtime_error("[DataLoader] tokenizer_curriculum is empty; tokenizer generation requires an explicit curriculum target");
	}

	auto build_hp = tokenizer_hp;
	build_hp.data_path = tokenizer_hp.output_data_path;

	std::cout << "[DataLoader] Atom token range fixed at " << GRIM::Tokenizer::ATOM_VOCAB_SIZE
	          << " type tokens" << std::endl;

	GRIM::Tokenizer::UniByte tokenizer(build_hp);

	const bool output_exists = fs::exists(build_hp.data_path);
	const bool vocab_exists = fs::exists(tokenizer_hp.vocab_path);
	bool reuse_shared_vocab = false;

	// Each tokenizer output is independently cacheable against the shared vocab.
	// A mismatch invalidates this GRMT, not the vocabulary token space.
	if (!tokenizer_hp.force_rebuild_vocab && output_exists && vocab_exists) {
		try {
			const auto manifest = GRIM::TokenizerArtifacts::loadTokenizerArtifactBundle(build_hp, tokenizer);
			std::cout << "[DataLoader] Existing tokenizer output is valid for shared vocab; "
			          << "GRMT sequences=" << manifest.grmt_header.num_sequences
			          << ", vocab_size=" << manifest.grmt_header.vocab_size
			          << ". Skipping cache rebuild." << std::endl;
			return true;
		} catch (const std::exception& e) {
			std::cerr << "[DataLoader] Tokenizer output is stale or mismatched: " << e.what()
			          << "; rebuilding only " << build_hp.data_path
			          << " with the existing shared vocab." << std::endl;
			tokenizer = GRIM::Tokenizer::UniByte(build_hp);
			(void)GRIM::TokenizerArtifacts::loadSharedTokenizerVocabulary(build_hp, tokenizer);
			reuse_shared_vocab = true;
		}
	}
	if (!tokenizer_hp.force_rebuild_vocab && !output_exists && vocab_exists) {
		(void)GRIM::TokenizerArtifacts::loadSharedTokenizerVocabulary(build_hp, tokenizer);
		reuse_shared_vocab = true;
	}
	if (reuse_shared_vocab && !tokenizer.initGPU()) {
		throw std::runtime_error(
			"[DataLoader] failed to initialize CUDA tokenizer runtime after loading shared vocab");
	}

	// Log reason for rebuild
	if (tokenizer_hp.force_rebuild_vocab) {
		std::cout << "[DataLoader] force_rebuild_vocab=true; replacing shared vocab and tokenizer output..." << std::endl;
	} else if (reuse_shared_vocab) {
		std::cout << "[DataLoader] Shared vocab is frozen; generating tokenizer output GRMT only." << std::endl;
	} else if (output_exists && !vocab_exists) {
		std::cout << "[DataLoader] Tokenizer output exists but shared vocab is missing; training vocab and rebuilding output." << std::endl;
	} else {
		std::cout << "[DataLoader] Shared vocab and tokenizer output are missing; building both." << std::endl;
	}

	// Derive the data directory from the configured GRMT path.
	fs::path output_path(build_hp.data_path);
	fs::path cache_dir = output_path.parent_path();

	std::cout << "[DataLoader] Preparing GRMT from concept blocks in: "
			  << cache_dir.string() << std::endl;

	std::vector<nlohmann::json> concept_json_entries;
	CurriculumMetadata curriculum_metadata;
	loadConceptBlocks(cache_dir, concept_json_entries, curriculum_metadata, tokenizer_hp.tokenizer_curriculum);

	// ── Curriculum startup summary ──
	std::cout << "[DataLoader] ═══════════ Curriculum Config ═══════════" << std::endl;
	if (!tokenizer_hp.tokenizer_curriculum.empty()) {
		std::cout << "[DataLoader]   tokenizer curriculum = " << tokenizer_hp.tokenizer_curriculum << std::endl;
	} else {
		std::cout << "[DataLoader]   curriculum        = (NONE — loading ALL blocks unfiltered)" << std::endl;
	}
	std::cout << "[DataLoader]   tokenizer output     = " << build_hp.data_path << std::endl;
	std::cout << "[DataLoader]   training curriculum  = " << tokenizer_hp.training_curriculum << std::endl;
	std::cout << "[DataLoader]   training input       = " << tokenizer_hp.data_path << std::endl;
	std::cout << "[DataLoader]   curriculum id    = " << curriculum_metadata.id << std::endl;
	std::cout << "[DataLoader]   curriculum name  = " << curriculum_metadata.name << std::endl;
	std::cout << "[DataLoader]   training stage   = " << curriculum_metadata.training_stage << std::endl;
	std::cout << "[DataLoader]   format concept   = " << (curriculum_metadata.formatAsConcept() ? "true" : "false") << std::endl;
	std::cout << "[DataLoader]   selected blocks  = " << curriculum_metadata.concept_block_ids.size() << std::endl;
	std::cout << "[DataLoader]   min_text_length   = " << min_cleaned_text_length << std::endl;
	std::cout << "[DataLoader]   loaded blocks     = " << concept_json_entries.size() << std::endl;
	std::cout << "[DataLoader] ═══════════════════════════════════════" << std::endl;
	if (!tokenizer_hp.current_model_training.empty()) {
		std::cout << "[DataLoader] Training model: " << tokenizer_hp.current_model_training << std::endl;
	}

	if (concept_json_entries.empty()) {
		std::cerr << "[DataLoader] FATAL: No concept-block entries found in "
				  << cache_dir.string()
				  << "; all training data must come from curriculum concept blocks."
				  << std::endl;
		throw std::runtime_error(
			"DataLoader: concept_blocks.fb is required but empty or missing");
	}

	// No train/val/test split here — Phase1_Startup owns that decision.  
	// DataLoader writes ALL sequences to a single GRMT file.

	if (!reuse_shared_vocab) {
		std::cout << "[DataLoader] Training new tokenizer pieces from concept blocks (target: "
		          << tokenizer_hp.target_vocab_size << " learned pieces)..." << std::endl;
		std::vector<std::string> vocab_corpus;
		vocab_corpus.reserve(concept_json_entries.size());
		for (const auto& cj : concept_json_entries) {
			bool is_raw_text = cj.value("format_type", std::string{}) == "raw" ||
			                   !curriculum_metadata.formatAsConcept();
			if (is_raw_text)
				vocab_corpus.push_back(GRIM::ConceptCanonical::renderPlainText(cj));
			else
				vocab_corpus.push_back(GRIM::ConceptCanonical::render(cj).text);
		}
		if (!tokenizer.unigramLM().trainFromCorpus(vocab_corpus, build_hp)) {
			throw std::runtime_error("[DataLoader] tokenizer training returned false; refusing to encode GRMT without a finalized tokenizer runtime state");
		}
		tokenizer.unigramLM().requireRuntimeReadyForLastTraining("DataLoader::PrepareTrainingDataFromCache");
		const auto& tokenizer_runtime_report = tokenizer.lastTrainingRuntimeReport();
		std::cout << "[DataLoader] Tokenizer runtime finalized for corpus encoding: required_viterbi_workspace_length="
		          << tokenizer_runtime_report.required_viterbi_workspace_length
		          << ", final_piece_count=" << tokenizer_runtime_report.final_piece_count
		          << ", trie_generation=" << tokenizer_runtime_report.finalized_trie_generation
		          << std::endl;
	} else {
		std::cout << "[DataLoader] Reusing shared tokenizer token space (vocab_size="
		          << tokenizer.vocabSize()
		          << "); vocabulary training is disabled for this output." << std::endl;
	}

	using TokenizedSequence = GRIM::TokenizerArtifacts::GrmtSequence;

	// BOS/EOS are NOT added here — Phase1_Startup owns boundary token
	// insertion (add_bos, add_eos config flags) and target fixup for them.

	auto materialize_sequence = [](GRIM::Tokenizer::UniByteResult result)
		-> std::optional<TokenizedSequence> {
		if (result.token_ids.empty()) {
			return std::nullopt;
		}

		TokenizedSequence seq;
		seq.token_ids = std::move(result.token_ids);
		seq.token_numeric_values = std::move(result.token_numeric_values);
		seq.token_atom_flags = std::move(result.token_atom_flags);
		seq.token_atom_mask = std::move(result.token_atom_mask);
		seq.atom_table = std::move(result.atom_table);
		seq.atom_entry_ids = std::move(result.atom_entry_ids);
		seq.local_atom_table = std::move(result.local_atom_table);
		seq.token_local_atom_indices = std::move(result.token_local_atom_indices);
		if (seq.token_numeric_values.size() != seq.token_ids.size() ||
			seq.token_atom_flags.size() != seq.token_ids.size() ||
			seq.token_atom_mask.size() != seq.token_ids.size() ||
			seq.atom_entry_ids.size() != seq.token_ids.size() ||
			seq.token_local_atom_indices.size() != seq.token_ids.size()) {
			throw std::runtime_error("[DataLoader] Token/side-channel length mismatch");
		}

		const size_t seq_len = seq.token_ids.size();
		seq.targets.resize(seq_len, -1);
		for (size_t j = 0; j + 1 < seq_len; ++j) {
			seq.targets[j] = seq.token_ids[j + 1];
		}
		seq.token_exec_slot_indices.assign(seq_len, -1);
		return seq;
	};

	auto render_boundaries = [](const GRIM::ConceptCanonical::RenderResult& rendered) {
		std::vector<size_t> boundaries;
		auto add_span = [&boundaries](const GRIM::ConceptCanonical::LogicalByteSpan& span) {
			if (!span.present) return;
			boundaries.push_back(span.begin);
			boundaries.push_back(span.end);
		};
		add_span(rendered.target_state);
		add_span(rendered.criteria);
		for (const auto& entry : rendered.success_criteria) {
			add_span(entry.criterion);
			add_span(entry.evidence);
		}
		for (const auto& constraint : rendered.constraints) {
			add_span(constraint);
		}
		for (const auto& known : rendered.knowns) {
			add_span(known);
		}
		for (const auto& unknown : rendered.unknowns) {
			add_span(unknown);
		}
		add_span(rendered.answer);
		if (rendered.prompt_byte_end > rendered.prompt_byte_begin) {
			boundaries.push_back(rendered.prompt_byte_begin);
			boundaries.push_back(rendered.prompt_byte_end);
		}
		std::sort(boundaries.begin(), boundaries.end());
		boundaries.erase(
			std::unique(boundaries.begin(), boundaries.end()),
			boundaries.end());
		return boundaries;
	};

	auto boundary_token_position = [](
		const std::vector<size_t>& boundaries,
		const std::vector<size_t>& token_counts,
		size_t byte_position,
		const std::string& field_name) -> size_t {
		const auto found = std::lower_bound(
			boundaries.begin(), boundaries.end(), byte_position);
		if (found == boundaries.end() || *found != byte_position) {
			throw std::runtime_error(
				"[DataLoader] missing logical boundary for " + field_name);
		}
		const size_t index = static_cast<size_t>(found - boundaries.begin());
		if (index >= token_counts.size()) {
			throw std::runtime_error(
				"[DataLoader] token boundary count mismatch for " + field_name);
		}
		return token_counts[index];
	};

	auto token_span = [&boundary_token_position](
		const GRIM::ConceptCanonical::LogicalByteSpan& byte_span,
		const std::vector<size_t>& boundaries,
		const std::vector<size_t>& token_counts,
		const std::string& field_name) -> GRIM::GoalTokenSpan {
		if (!byte_span.present) {
			throw std::runtime_error(
				"[DataLoader] missing logical span for " + field_name);
		}
		const size_t begin = boundary_token_position(
			boundaries, token_counts, byte_span.begin, field_name + ".begin");
		const size_t end = boundary_token_position(
			boundaries, token_counts, byte_span.end, field_name + ".end");
		if (end <= begin || end > static_cast<size_t>(std::numeric_limits<std::int32_t>::max())) {
			throw std::runtime_error(
				"[DataLoader] invalid token span for " + field_name);
		}
		return GRIM::GoalTokenSpan{
			static_cast<std::int32_t>(begin),
			static_cast<std::int32_t>(end)};
	};

	auto assign_prompt_span = [&boundary_token_position](
		TokenizedSequence& sequence,
		const GRIM::ConceptCanonical::RenderResult& rendered,
		const std::vector<size_t>& boundaries,
		const std::vector<size_t>& token_counts) {
		if (rendered.prompt_byte_end <= rendered.prompt_byte_begin) {
			sequence.prompt_length = 0;
			sequence.prompt_end_pos = -1;
			return;
		}
		const size_t begin = boundary_token_position(
			boundaries, token_counts, rendered.prompt_byte_begin,
			"prompt.begin");
		const size_t end = boundary_token_position(
			boundaries, token_counts, rendered.prompt_byte_end,
			"prompt.end");
		if (end <= begin ||
		    end > static_cast<size_t>(std::numeric_limits<std::int32_t>::max())) {
			throw std::runtime_error("[DataLoader] invalid logical prompt span");
		}
		sequence.prompt_length = static_cast<std::int32_t>(end - begin);
		sequence.prompt_end_pos = static_cast<std::int32_t>(end - 1);
	};

	auto span_token_ids = [](const TokenizedSequence& sequence,
	                         const GRIM::GoalTokenSpan& span,
	                         const std::string& field_name) {
		if (!span.valid() ||
		    static_cast<size_t>(span.end) > sequence.token_ids.size()) {
			throw std::runtime_error(
				"[DataLoader] token span is outside sequence for " + field_name);
		}
		return std::vector<std::int32_t>(
			sequence.token_ids.begin() + span.begin,
			sequence.token_ids.begin() + span.end);
	};

	auto apply_sft_answer_contract = [&token_span](
		TokenizedSequence& sequence,
		const GRIM::ConceptCanonical::RenderResult& rendered,
		const std::vector<size_t>& boundaries,
		const std::vector<size_t>& token_counts) {
		const GRIM::GoalTokenSpan answer_span = token_span(
			rendered.answer, boundaries, token_counts, "answer");
		if (rendered.prompt_byte_end <= rendered.prompt_byte_begin) {
			throw std::runtime_error(
				"[DataLoader] SFT row requires a non-empty authored prompt");
		}
		if (answer_span.begin <= 0) {
			throw std::runtime_error(
				"[DataLoader] SFT answer must follow a non-empty functional prompt");
		}

		// SFT prompt geometry is functional, not a literal <prompt> span: every
		// model-visible token before the answer is pinned context. This includes
		// knowns, unknowns, goal decomposition, and explanation/intermediates.
		sequence.prompt_length = answer_span.begin;
		sequence.prompt_end_pos = answer_span.begin - 1;
		for (size_t row = 0; row + 1 < sequence.targets.size(); ++row) {
			const size_t target_position = row + 1;
			if (target_position < static_cast<size_t>(answer_span.begin) ||
			    target_position >= static_cast<size_t>(answer_span.end)) {
				sequence.targets[row] = -1;
			}
		}
	};

	auto materialize_goal = [&token_span, &span_token_ids](
		const json& concept,
		const GRIM::ConceptCanonical::RenderResult& rendered,
		const std::vector<size_t>& boundaries,
		const std::vector<size_t>& token_counts,
		const TokenizedSequence& sequence) -> std::shared_ptr<const GRIM::Goal> {
		if (!concept.contains("goal") || !concept["goal"].is_object()) {
			return nullptr;
		}

		const json& source_goal = concept["goal"];
		auto goal = std::make_shared<GRIM::Goal>();
		const std::string target_state =
			source_goal.value("target_state", std::string{});
		if (!target_state.empty()) {
			GRIM::TargetState target;
			target.span = token_span(
				rendered.target_state, boundaries, token_counts,
				"goal.target_state");
			target.token_ids = span_token_ids(
				sequence, target.span, "goal.target_state");
			goal->target_state = std::move(target);
		}

		if (source_goal.contains("success_criteria")) {
			if (!source_goal["success_criteria"].is_array()) {
				throw std::runtime_error(
					"[DataLoader] goal.success_criteria must be an array");
			}

			const auto& source_entries = source_goal["success_criteria"];
			if (source_entries.size() != rendered.success_criteria.size()) {
				throw std::runtime_error(
					"[DataLoader] rendered success-criteria count mismatch");
			}

			GRIM::SuccessCriteria success_criteria;
			if (!source_entries.empty()) {
				success_criteria.span = token_span(
					rendered.criteria, boundaries, token_counts,
					"goal.success_criteria");
			}
			for (std::size_t index = 0;
			     index < source_entries.size();
			     ++index) {
				const json& source_entry = source_entries[index];
				if (!source_entry.is_object()) {
					throw std::runtime_error(
						"[DataLoader] goal.success_criteria[" +
						std::to_string(index) + "] must be an object");
				}

				GRIM::SuccessCriterion entry;
				const std::string prefix =
					"goal.success_criteria[" + std::to_string(index) + "]";
				entry.criterion_span = token_span(
					rendered.success_criteria[index].criterion,
					boundaries, token_counts, prefix + ".criterion");
				entry.token_ids = span_token_ids(
					sequence, entry.criterion_span, prefix + ".criterion");
				if (rendered.success_criteria[index].evidence.present) {
					entry.evidence_span = token_span(
						rendered.success_criteria[index].evidence,
						boundaries, token_counts, prefix + ".evidence");
					entry.evidence_token_ids = span_token_ids(
						sequence, entry.evidence_span, prefix + ".evidence");
				}
				success_criteria.entries.push_back(std::move(entry));
			}
			if (!success_criteria.entries.empty()) {
				goal->success_criteria = std::move(success_criteria);
			}
		}

		if (source_goal.contains("constraints")) {
			if (!source_goal["constraints"].is_array()) {
				throw std::runtime_error(
					"[DataLoader] goal.constraints must be an array");
			}

			const auto& source_entries = source_goal["constraints"];
			if (source_entries.size() != rendered.constraints.size()) {
				throw std::runtime_error(
					"[DataLoader] rendered constraint count mismatch");
			}

			GRIM::Constraints constraints;
			for (std::size_t index = 0;
			     index < source_entries.size();
			     ++index) {
				if (!source_entries[index].is_string()) {
					throw std::runtime_error(
						"[DataLoader] goal.constraints[" +
						std::to_string(index) + "] must be a string");
				}

				GRIM::Constraint entry;
				const std::string prefix =
					"goal.constraints[" + std::to_string(index) + "]";
				entry.constraint_span = token_span(
					rendered.constraints[index],
					boundaries, token_counts, prefix);
				entry.token_ids = span_token_ids(
					sequence, entry.constraint_span, prefix);
				constraints.entries.push_back(std::move(entry));
			}
			if (!constraints.entries.empty()) {
				goal->constraints = std::move(constraints);
			}
		}

		if (!goal->target_state.has_value() &&
		    !goal->success_criteria.has_value() &&
		    !goal->constraints.has_value()) {
			return nullptr;
		}
		return goal;
	};

	auto materialize_concept_block_spans = [&token_span, &span_token_ids](
		const json& concept,
		const GRIM::ConceptCanonical::RenderResult& rendered,
		const std::vector<size_t>& boundaries,
		const std::vector<size_t>& token_counts,
		const TokenizedSequence& sequence)
		-> std::shared_ptr<const GRIM::ConceptBlockSpans> {
		auto spans = std::make_shared<GRIM::ConceptBlockSpans>();
		auto materialize_entries = [&](
			const char* field,
			const std::vector<GRIM::ConceptCanonical::LogicalByteSpan>& rendered_entries,
			std::vector<GRIM::ConceptBlockSpanEntry>& destination) {
			if (!concept.contains(field)) {
				if (!rendered_entries.empty()) {
					throw std::runtime_error(
						std::string("[DataLoader] rendered ") + field +
						" exist without source entries");
				}
				return;
			}
			if (!concept[field].is_array()) {
				throw std::runtime_error(
					std::string("[DataLoader] ") + field + " must be an array");
			}
			const auto& source_entries = concept[field];
			if (source_entries.size() != rendered_entries.size()) {
				throw std::runtime_error(
					std::string("[DataLoader] rendered ") + field +
					" count mismatch");
			}
			destination.reserve(source_entries.size());
			for (std::size_t index = 0; index < source_entries.size(); ++index) {
				if (!source_entries[index].is_string()) {
					throw std::runtime_error(
						std::string("[DataLoader] ") + field + "[" +
						std::to_string(index) + "] must be a string");
				}
				const std::string prefix =
					std::string(field) + "[" + std::to_string(index) + "]";
				GRIM::ConceptBlockSpanEntry entry;
				entry.span = token_span(
					rendered_entries[index], boundaries, token_counts, prefix);
				entry.token_ids = span_token_ids(sequence, entry.span, prefix);
				destination.push_back(std::move(entry));
			}
		};

		materialize_entries("knowns", rendered.knowns, spans->knowns);
		materialize_entries("unknowns", rendered.unknowns, spans->unknowns);
		if (spans->empty()) {
			return nullptr;
		}
		std::shared_ptr<const GRIM::ConceptBlockSpans> immutable_spans =
			std::move(spans);
		return immutable_spans;
	};

	std::cout << "[DataLoader] Encoding " << concept_json_entries.size()
	          << " concept sequences..." << std::endl << std::flush;
	std::vector<TokenizedSequence> all_tokens;
	all_tokens.reserve(concept_json_entries.size());
	size_t raw_text_count = 0;
	size_t concept_build_failures = 0;
	size_t selected_entries_skipped = 0;  // short text / encoder returned nullopt
	bool warned_execution_bridge_stub = false;
	for (const auto& cj : concept_json_entries) {
		try {
			bool is_raw_text = cj.value("format_type", std::string{}) == "raw" ||
			                   !curriculum_metadata.formatAsConcept();

			if (is_raw_text) {
				// ── Pretraining path: plain text, no execution payload ──
				auto rendered = GRIM::ConceptCanonical::renderPlainTextWithPromptBoundary(cj);
				if (rendered.text.size() < min_cleaned_text_length) { ++selected_entries_skipped; continue; }

				const auto boundaries = render_boundaries(rendered);
				std::vector<size_t> token_counts;
				auto seq = materialize_sequence(tokenizer.tokenizeWithMetadata(
					rendered.text, boundaries, &token_counts));
				if (!seq) { ++selected_entries_skipped; continue; }
				seq->execution_active = false;
				seq->execution_gate_target = GRIM::Execution::ExecutionGateTarget::UNSUPERVISED;
				if (curriculum_metadata.training_stage == "sft") {
					apply_sft_answer_contract(
						*seq, rendered, boundaries, token_counts);
				} else {
					assign_prompt_span(*seq, rendered, boundaries, token_counts);
				}
				all_tokens.push_back(std::move(*seq));
				++raw_text_count;
				continue;
			}

			// STATE0 and EXEC are internal structure, not training text. This is
			// the only atom-tokenization call for the concept encoding path.
			auto rendered = GRIM::ConceptCanonical::render(cj);
			if (rendered.text.size() < min_cleaned_text_length) {
				++selected_entries_skipped;
				continue;
			}

			const auto boundaries = render_boundaries(rendered);
			std::vector<size_t> token_counts;
			auto encoded = tokenizer.tokenizeWithMetadata(
				rendered.text, boundaries, &token_counts);
			auto seq = materialize_sequence(std::move(encoded));
			if (!seq) { ++selected_entries_skipped; continue; }

			seq->execution_active = false;
			seq->execution_gate_target =
				GRIM::Execution::ExecutionGateTarget::UNSUPERVISED;
			if (curriculum_metadata.training_stage == "sft") {
				apply_sft_answer_contract(
					*seq, rendered, boundaries, token_counts);
			} else {
				assign_prompt_span(*seq, rendered, boundaries, token_counts);
			}
			seq->concept_block_spans = materialize_concept_block_spans(
				cj, rendered, boundaries, token_counts, *seq);
			seq->goal = materialize_goal(
				cj, rendered, boundaries, token_counts, *seq);

			const bool has_internal_execution_state =
				(cj.contains("state_0") && cj["state_0"].is_object()) ||
				(cj.contains("execution") && cj["execution"].is_array() &&
				 !cj["execution"].empty());
			if (has_internal_execution_state && !warned_execution_bridge_stub) {
				std::cerr
					<< "[DataLoader] WARNING: execution supervision is stubbed: "
					<< "STATE0/EXEC are not rendered as tokens, and the AtomTable-entry "
					<< "to execution-slot compiler is not implemented yet. Affected rows "
					<< "remain execution-unsupervised."
					<< std::endl;
				warned_execution_bridge_stub = true;
			}
			all_tokens.push_back(std::move(*seq));
		} catch (const std::exception& e) {
			++concept_build_failures;
			std::cerr << "[DataLoader] concept build failed: " << e.what() << "\n";
		}
	}
	if (raw_text_count > 0) {
		std::cout << "[DataLoader] Encoded " << raw_text_count << " Raw + "
		          << (all_tokens.size() - raw_text_count) << " structured sequences" << std::endl;
	}

	// Refuse to write a zero-sequence GRMT — every selected entry failed or was
	// skipped, so there is nothing to train on. Better to fail here than to
	// return true and have Phase1 silently load an empty dataset.
	if (all_tokens.empty()) {
		std::cerr << "[DataLoader] FATAL: no sequences produced from "
		          << concept_json_entries.size() << " selected entries ("
		          << concept_build_failures << " build failures). "
		          << "Cannot write a zero-sequence GRMT." << std::endl;
		return false;
	}

	// For a filtered (named) curriculum, every selected entry was hand-picked
	// by config; an unexpected build failure on any of them is a data/config
	// bug, not noise to be swallowed. Fail loud so it gets fixed at the
	// source instead of producing a quietly-degraded GRMT.
	//
	// Concept build failures are ALWAYS fatal (Rule 20): a thrown exception
	// during concept assembly means a malformed source row, and silently
	// dropping it produces a corpus that diverges from what the user shipped.
	// If a future workflow genuinely needs lenient ingestion, gate it behind
	// an explicit dirty-corpus mode — do not regress this default.
	if (concept_build_failures > 0) {
		std::cerr << "[DataLoader] FATAL: " << concept_build_failures
		          << " concept build failure(s) during encode. Refusing to "
		          << "produce a partial GRMT (Rule 20: no silent drops)."
		          << std::endl;
		return false;
	}

	// Silent skips (short text, empty encoder output) are only fatal under a
	// filtered curriculum, where the curriculum names exactly the entries it
	// expects to train on — dropping any of them silently is a partial GRMT.
	if (!curriculum_metadata.concept_block_ids.empty() && selected_entries_skipped > 0) {
		std::cerr << "[DataLoader] FATAL: " << selected_entries_skipped
		          << " silently-skipped selected entry/entries under a filtered "
		          << "curriculum. Refusing to produce a partial GRMT."
		          << std::endl;
		return false;
	}

	// Write single GRMT file — Phase1_Startup handles train/val splitting
	fs::create_directories(cache_dir);
	fs::path train_grmt = output_path;

	// Log sequence statistics + atom diagnostics
	size_t total_tokens = 0;
	size_t encode_atom_tokens = 0;
	size_t encode_atom_sequences = 0;
	size_t encode_atom_entries = 0;
	for (const auto& seq : all_tokens) {
		total_tokens += seq.token_ids.size();
		bool seq_has_atoms = false;
		for (size_t j = 0; j < seq.token_ids.size(); ++j) {
			if (j < seq.token_atom_mask.size() && seq.token_atom_mask[j]) {
				encode_atom_tokens++;
				seq_has_atoms = true;
			}
			if (j < seq.atom_entry_ids.size() &&
				seq.atom_entry_ids[j] != GRIM::Tokenizer::kAtomEntryNone) {
				encode_atom_entries++;
			}
		}
		if (seq_has_atoms) encode_atom_sequences++;
	}
	std::cout << "[DataLoader] " << all_tokens.size() << " sequences, "
			  << total_tokens << " total tokens" << std::endl;
	std::cout << "[DataLoader] Atom encoding stats: "
			  << encode_atom_tokens << " atom tokens ("
			  << (total_tokens > 0 ? (100.0 * encode_atom_tokens / total_tokens) : 0.0)
			  << "%), " << encode_atom_sequences << "/" << all_tokens.size()
			  << " sequences with atoms, " << encode_atom_entries
			  << " AtomTable entries" << std::endl;
	if (encode_atom_tokens == 0) {
		std::cerr << "[DataLoader] WARNING: Zero atoms detected during encoding! "
				  << "Check scratch_block_reasoning.enabled in ai_config.json" << std::endl;
	}

	GRIM::TokenizerArtifacts::TokenizerBundleSaveReport save_report;
	try {
		if (reuse_shared_vocab) {
			save_report = GRIM::TokenizerArtifacts::saveGrmtForSharedTokenizerVocabulary(
				build_hp, tokenizer, all_tokens);
		} else {
			save_report = GRIM::TokenizerArtifacts::saveTokenizerArtifactBundle(
				build_hp, tokenizer, all_tokens);
		}
	} catch (const std::exception& e) {
		std::cerr << "[DataLoader] FATAL: failed to save tokenizer output: "
		          << e.what() << std::endl;
		return false;
	}
	if (save_report.grmt.dropped_targetless_sequences > 0) {
		std::cerr << "[DataLoader] Dropped "
		          << save_report.grmt.dropped_targetless_sequences
		          << " sequences with 0 valid targets" << std::endl;
	}
	if (!reuse_shared_vocab && tokenizer_hp.save_text_vocab) {
		std::cout << "[DataLoader] Also saved human-readable .txt vocab" << std::endl;
	}

	std::cout << "[DataLoader] Saved tokenizer output:" << std::endl
	          << "  Vocab: " << tokenizer_hp.vocab_path << std::endl
	          << "  GRMT:  " << train_grmt.string() << std::endl
	          << "  Written sequences: " << save_report.grmt.written_sequences << std::endl
	          << "  Vocab size: " << save_report.manifest.grmt_header.vocab_size << std::endl;

	return true;
}

} // namespace GRIM

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
		          << "Delete .grmt files and regenerate with scratch_block_reasoning.enabled=true"
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

