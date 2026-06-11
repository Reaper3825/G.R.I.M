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
#include <unordered_set>
#include <unordered_map>
#include <cstdint>
#include <exception>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>
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
#include "ConceptExecutionSequenceBuilder.hpp"
#include "../../training/Phases/Startup/SlidingWindow.hpp"
#include "../../training/Phases/ConfigDump.hpp"
#include "../../training/Phases/Phase1_Startup.hpp"

namespace fs = std::filesystem;

namespace GRIM {

// ─── Concept blocks corpus loading ──────────────────────────────────────────
//
// Loads concept_blocks.jsonl and returns parsed JSON objects.
// The canonical builder (ConceptExecutionSequenceBuilder) handles all
// structured execution record building, text rendering, and payload
// compilation — no __SLOTS__ debug path.
//
namespace {

using json = nlohmann::json;

// Curriculum filter returned by loadCurriculumFilter().
// concept_ids: blocks that get canonical Q:/STATE0/EXP:/EXEC/A: formatting.
// plaintext_ids: blocks treated as raw text (pretraining mode).
// When has_filter is false, all entries are included as concept blocks.
struct CurriculumFilter {
	std::unordered_set<std::string> concept_ids;
	std::unordered_set<std::string> plaintext_ids;
	bool has_filter = false;
	bool format_as_concept = true;  // curriculum-level flag; false → all blocks render as plain text
};

// Load curriculum filter from curriculum_registry.json by name lookup.
// Falls back to {curriculum_name}.json or curriculum_manifest.json if registry
// doesn't contain the curriculum. THROWS when a named curriculum is specified
// but cannot be found anywhere (Rule 20: no silent fallbacks).
CurriculumFilter loadCurriculumFilter(const fs::path& dir, const std::string& curriculum_name) {
	CurriculumFilter filter;

	if (curriculum_name.empty()) {
		// No curriculum specified — try legacy curriculum_manifest.json
		fs::path manifest = dir / "curriculum_manifest.json";
		if (!fs::exists(manifest)) {
			std::cout << "[DataLoader] No curriculum specified; loading all blocks unfiltered." << std::endl;
			return filter;
		}
		std::ifstream in(manifest);
		if (!in.is_open()) {
			throw std::runtime_error(
				"[DataLoader] FATAL: cannot open curriculum manifest: " + manifest.string());
		}
		json j = json::parse(in);
		if (j.contains("concept_block_ids") && j["concept_block_ids"].is_array()) {
			for (const auto& id : j["concept_block_ids"])
				if (id.is_string()) filter.concept_ids.insert(id.get<std::string>());
		}
		if (j.contains("plaintext_block_ids") && j["plaintext_block_ids"].is_array()) {
			for (const auto& id : j["plaintext_block_ids"])
				if (id.is_string()) filter.plaintext_ids.insert(id.get<std::string>());
		}
		if (j.contains("format_as_concept") && j["format_as_concept"].is_boolean())
			filter.format_as_concept = j["format_as_concept"].get<bool>();
		filter.has_filter = !filter.concept_ids.empty() || !filter.plaintext_ids.empty();
		std::cout << "[DataLoader] Legacy manifest loaded: " << manifest.string() << std::endl;
		return filter;
	}

	// ── Primary path: look up curriculum by name in curriculum_registry.json ──
	fs::path registry_path = dir / "curriculum_registry.json";
	bool found_in_registry = false;

	if (fs::exists(registry_path)) {
		std::ifstream reg_in(registry_path);
		if (reg_in.is_open()) {
			try {
				json reg = json::parse(reg_in);
				if (reg.contains("curriculums") && reg["curriculums"].is_array()) {
					for (const auto& curr : reg["curriculums"]) {
						if (!curr.contains("name") || !curr["name"].is_string()) continue;
						if (curr["name"].get<std::string>() != curriculum_name) continue;

						// Found it — extract block IDs and flags
						found_in_registry = true;
						if (curr.contains("format_as_concept") && curr["format_as_concept"].is_boolean())
							filter.format_as_concept = curr["format_as_concept"].get<bool>();

						if (curr.contains("concept_block_ids") && curr["concept_block_ids"].is_array()) {
							for (const auto& id : curr["concept_block_ids"]) {
								if (id.is_string()) {
									if (filter.format_as_concept)
										filter.concept_ids.insert(id.get<std::string>());
									else
										filter.plaintext_ids.insert(id.get<std::string>());
								}
							}
						}
						// Mixed PT/concept curriculums: registry must honor plaintext_block_ids
						// with the same semantics as the per-curriculum manifest path.
						if (curr.contains("plaintext_block_ids") && curr["plaintext_block_ids"].is_array()) {
							for (const auto& id : curr["plaintext_block_ids"]) {
								if (id.is_string())
									filter.plaintext_ids.insert(id.get<std::string>());
							}
						}
						std::cout << "[DataLoader] Curriculum '" << curriculum_name
						          << "' loaded from registry: " << registry_path.string() << std::endl;
						break;
					}
				}
			} catch (const json::exception& e) {
				std::cerr << "[DataLoader] WARNING: failed to parse curriculum_registry.json: "
				          << e.what() << std::endl;
			}
		}
	}

	// ── Fallback: per-curriculum manifest file {name}.json ──
	if (!found_in_registry) {
		fs::path manifest = dir / (curriculum_name + ".json");
		if (!fs::exists(manifest)) {
			throw std::runtime_error(
				"[DataLoader] FATAL: curriculum '" + curriculum_name
				+ "' not found in curriculum_registry.json and no manifest at: "
				+ manifest.string());
		}
		std::ifstream in(manifest);
		if (!in.is_open()) {
			throw std::runtime_error(
				"[DataLoader] FATAL: cannot open curriculum manifest: " + manifest.string());
		}
		try {
			json j = json::parse(in);
			if (j.contains("concept_block_ids") && j["concept_block_ids"].is_array()) {
				for (const auto& id : j["concept_block_ids"])
					if (id.is_string()) filter.concept_ids.insert(id.get<std::string>());
			}
			if (j.contains("plaintext_block_ids") && j["plaintext_block_ids"].is_array()) {
				for (const auto& id : j["plaintext_block_ids"])
					if (id.is_string()) filter.plaintext_ids.insert(id.get<std::string>());
			}
			if (j.contains("format_as_concept") && j["format_as_concept"].is_boolean())
				filter.format_as_concept = j["format_as_concept"].get<bool>();

			if (!filter.format_as_concept && !filter.concept_ids.empty()) {
				for (const auto& id : filter.concept_ids)
					filter.plaintext_ids.insert(id);
				filter.concept_ids.clear();
			}
			std::cout << "[DataLoader] Curriculum '" << curriculum_name
			          << "' loaded from manifest: " << manifest.string() << std::endl;
		} catch (const json::exception& e) {
			throw std::runtime_error(
				"[DataLoader] FATAL: failed to parse " + manifest.string() + ": " + e.what());
		}
	}

	filter.has_filter = !filter.concept_ids.empty() || !filter.plaintext_ids.empty();

	// For a NAMED curriculum, an empty ID set is always a configuration error.
	// Without this, an empty/missing/typo'd ID list silently expands to the
	// full corpus because loadConceptBlocksJson() only filters when has_filter
	// is true. Rule 20: fail loud rather than train on the wrong data.
	if (!curriculum_name.empty() && !filter.has_filter) {
		throw std::runtime_error(
			"[DataLoader] FATAL: curriculum '" + curriculum_name +
			"' resolved to an empty filter (concept_block_ids and plaintext_block_ids "
			"are both empty/missing). Refusing to silently train on the entire corpus. "
			"Fix the curriculum definition or remove the curriculum name to opt in to "
			"full-corpus training.");
	}

	// Also force has_filter=true for any named curriculum so that the loader
	// applies an explicit (possibly all-rejecting) filter rather than falling
	// through to the unfiltered branch.
	if (!curriculum_name.empty()) {
		filter.has_filter = true;
	}

	std::cout << "[DataLoader]   format_as_concept=" << (filter.format_as_concept ? "true" : "false")
	          << ", concept_ids=" << filter.concept_ids.size()
	          << ", plaintext_ids=" << filter.plaintext_ids.size()
	          << ", has_filter=" << (filter.has_filter ? "true" : "false")
	          << std::endl;
	return filter;
}

void loadConceptBlocksJson(const fs::path& cache_dir,
                           std::vector<json>& out,
                           CurriculumFilter& out_filter,
                           const std::string& curriculum_name = "") {
	fs::path p = cache_dir / "concept_blocks.jsonl";
	std::ifstream in(p);
	if (!in.is_open()) {
		std::cout << "[DataLoader] No concept_blocks.jsonl at " << p.string()
		          << " (optional)\n";
		return;
	}

	// Load optional curriculum filter.
	out_filter = loadCurriculumFilter(cache_dir, curriculum_name);

	std::string line;
	size_t total = 0;
	size_t accepted = 0;
	while (std::getline(in, line)) {
		if (line.empty()) continue;
		try {
			auto j = json::parse(line);
			++total;
			if (out_filter.has_filter) {
				std::string id = j.value("id", std::string());
				if (out_filter.concept_ids.find(id) == out_filter.concept_ids.end() &&
				    out_filter.plaintext_ids.find(id) == out_filter.plaintext_ids.end())
					continue;
			}
			out.push_back(std::move(j));
			++accepted;
		} catch (const std::exception& e) {
			std::cerr << "[DataLoader] concept_blocks.jsonl skip line: " << e.what() << "\n";
		}
	}
	if (out_filter.has_filter) {
		std::cout << "[DataLoader] Loaded " << accepted << "/" << total
		          << " concept blocks (filtered by curriculum manifest) from "
		          << p.string() << std::endl;
	} else {
		std::cout << "[DataLoader] Loaded " << accepted
		          << " concept block entries from " << p.string() << std::endl;
	}
}

}  // namespace

bool PrepareTrainingDataFromCache(
	const GRIM::HyperParameters::TokenizerHP& tokenizer_hp) {
	const size_t min_cleaned_text_length = static_cast<size_t>(tokenizer_hp.min_cleaned_text_length);

	if (tokenizer_hp.data_path.empty()) {
		std::cerr << "[DataLoader] No training_data path configured; skipping cache preparation." << std::endl;
		return false;
	}
	if (tokenizer_hp.vocab_path.empty()) {
		throw std::runtime_error("[DataLoader] vocab path is empty; tokenizer artifacts must be saved/loaded as vocab+GRMT pair");
	}

	std::cout << "[DataLoader] Atom token range fixed at " << GRIM::Tokenizer::ATOM_VOCAB_SIZE
	          << " type tokens" << std::endl;

	GRIM::Tokenizer::UniByte tokenizer(tokenizer_hp);

	const bool training_exists = fs::exists(tokenizer_hp.data_path);
	const bool vocab_exists = fs::exists(tokenizer_hp.vocab_path);
	bool artifact_pair_invalid = false;

	// Vocab + GRMT are one artifact pair. A valid cache must load the vocab and
	// validate the GRMT header vocab against the tokenizer's live token space.
	if (!tokenizer_hp.force_rebuild_vocab && training_exists && vocab_exists) {
		try {
			const auto manifest = GRIM::TokenizerArtifacts::loadTokenizerArtifactBundle(tokenizer_hp, tokenizer);
			std::cout << "[DataLoader] Existing tokenizer artifact bundle is valid; "
			          << "GRMT sequences=" << manifest.grmt_header.num_sequences
			          << ", vocab_size=" << manifest.grmt_header.vocab_size
			          << ". Skipping cache rebuild." << std::endl;
			return true;
		} catch (const std::exception& e) {
			std::cerr << "[DataLoader] Tokenizer artifact bundle invalid: " << e.what()
			          << "; rebuilding vocab+GRMT together." << std::endl;
			artifact_pair_invalid = true;
			tokenizer = GRIM::Tokenizer::UniByte(tokenizer_hp);
		}
	}

	// Log reason for rebuild
	if (tokenizer_hp.force_rebuild_vocab) {
		std::cout << "[DataLoader] force_rebuild_vocab=true, rebuilding tokenizer artifact bundle..." << std::endl;
	} else if (artifact_pair_invalid) {
		std::cout << "[DataLoader] Rebuilding due to invalid tokenizer artifact bundle..." << std::endl;
	} else if (training_exists && !vocab_exists) {
		std::cout << "[DataLoader] GRMT present but vocab missing; rebuilding both artifacts." << std::endl;
	} else if (!training_exists && vocab_exists) {
		std::cout << "[DataLoader] Vocab present but GRMT missing; rebuilding both artifacts." << std::endl;
	} else {
		std::cout << "[DataLoader] Tokenizer artifact bundle missing; building from cache..." << std::endl;
	}

	// Derive the data directory from the configured GRMT path.
	fs::path training_path(tokenizer_hp.data_path);
	fs::path cache_dir = training_path.parent_path();

	std::cout << "[DataLoader] Preparing GRMT from concept blocks in: "
			  << cache_dir.string() << std::endl;

	std::vector<nlohmann::json> concept_json_entries;
	CurriculumFilter curriculum_filter;
	loadConceptBlocksJson(cache_dir, concept_json_entries, curriculum_filter, tokenizer_hp.current_curriculum);

	// ── Curriculum startup summary ──
	std::cout << "[DataLoader] ═══════════ Curriculum Config ═══════════" << std::endl;
	if (!tokenizer_hp.current_curriculum.empty()) {
		std::cout << "[DataLoader]   curriculum        = " << tokenizer_hp.current_curriculum << std::endl;
	} else {
		std::cout << "[DataLoader]   curriculum        = (NONE — loading ALL blocks unfiltered)" << std::endl;
	}
	std::cout << "[DataLoader]   format_as_concept = " << (curriculum_filter.format_as_concept ? "true" : "false") << std::endl;
	std::cout << "[DataLoader]   has_filter        = " << (curriculum_filter.has_filter ? "true" : "false") << std::endl;
	std::cout << "[DataLoader]   concept_ids       = " << curriculum_filter.concept_ids.size() << std::endl;
	std::cout << "[DataLoader]   plaintext_ids     = " << curriculum_filter.plaintext_ids.size() << std::endl;
	std::cout << "[DataLoader]   min_text_length   = " << min_cleaned_text_length << std::endl;
	std::cout << "[DataLoader]   loaded blocks     = " << concept_json_entries.size() << std::endl;
	std::cout << "[DataLoader] ═══════════════════════════════════════" << std::endl;
	if (!tokenizer_hp.current_model_training.empty()) {
		std::cout << "[DataLoader] Training model: " << tokenizer_hp.current_model_training << std::endl;
	}

	if (concept_json_entries.empty()) {
		std::cerr << "[DataLoader] FATAL: No concept_blocks.jsonl entries found in "
				  << cache_dir.string()
				  << "; all training data must come from curriculum concept blocks."
				  << std::endl;
		throw std::runtime_error(
			"DataLoader: concept_blocks.jsonl is required but empty or missing");
	}

	// No train/val/test split here — Phase1_Startup owns that decision.  
	// DataLoader writes ALL sequences to a single GRMT file.

	std::cout << "[DataLoader] Training new tokenizer pieces from concept blocks (target: " 
			  << tokenizer_hp.target_vocab_size << " learned pieces)..." << std::endl;
	std::vector<std::string> vocab_corpus;
	vocab_corpus.reserve(concept_json_entries.size());
	for (const auto& cj : concept_json_entries) {
		std::string entry_id = cj.value("id", std::string());
		bool is_plaintext = !curriculum_filter.format_as_concept ||
		                    (curriculum_filter.has_filter &&
		                     curriculum_filter.plaintext_ids.count(entry_id) > 0);
		if (is_plaintext)
			vocab_corpus.push_back(GRIM::DataLoader::renderPlainText(cj, false));
		else
			vocab_corpus.push_back(GRIM::DataLoader::renderCanonicalText(cj));
	}
	if (!tokenizer.unigramLM().trainFromCorpus(vocab_corpus, tokenizer_hp)) {
		throw std::runtime_error("[DataLoader] tokenizer training returned false; refusing to encode GRMT without a finalized tokenizer runtime state");
	}
	tokenizer.unigramLM().requireRuntimeReadyForLastTraining("DataLoader::PrepareTrainingDataFromCache");
	const auto& tokenizer_runtime_report = tokenizer.lastTrainingRuntimeReport();
	std::cout << "[DataLoader] Tokenizer runtime finalized for corpus encoding: required_viterbi_workspace_length="
	          << tokenizer_runtime_report.required_viterbi_workspace_length
	          << ", final_piece_count=" << tokenizer_runtime_report.final_piece_count
	          << ", trie_generation=" << tokenizer_runtime_report.finalized_trie_generation
	          << std::endl;

	using TokenizedSequence = GRIM::TokenizerArtifacts::GrmtSequence;

	// BOS/EOS are NOT added here — Phase1_Startup owns boundary token
	// insertion (add_bos, add_eos config flags) and target fixup for them.

	auto build_sequence = [&](const std::string& text) -> std::optional<TokenizedSequence> {
		auto result = tokenizer.tokenizeWithMetadata(text);
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
		if (seq.token_numeric_values.size() != seq.token_ids.size() ||
			seq.token_atom_flags.size() != seq.token_ids.size() ||
			seq.token_atom_mask.size() != seq.token_ids.size() ||
			seq.atom_entry_ids.size() != seq.token_ids.size()) {
			throw std::runtime_error("[DataLoader] Token/side-channel length mismatch");
		}

		const size_t seq_len = seq.token_ids.size();
		seq.targets.resize(seq_len, -1);
		for (size_t j = 0; j + 1 < seq_len; ++j) {
			seq.targets[j] = seq.token_ids[j + 1];
		}
		seq.token_exec_slots.assign(seq_len, -1);
		return seq;
	};

	std::cout << "[DataLoader] Encoding " << concept_json_entries.size()
	          << " concept sequences..." << std::endl << std::flush;
	std::vector<TokenizedSequence> all_tokens;
	all_tokens.reserve(concept_json_entries.size());
	int concept_exec_base_slot = 0;
	if (const char* ev = std::getenv("GRIM_CONCEPT_EXEC_BASE_SLOT")) {
		try {
			concept_exec_base_slot = std::stoi(ev);
		} catch (const std::exception& e) {
			throw std::runtime_error(
				"[DataLoader] GRIM_CONCEPT_EXEC_BASE_SLOT is not a valid integer: " +
				std::string(ev) + " (" + e.what() + ")");
		}
	}
	const int expected_exec_steps = tokenizer_hp.execution_block_num_steps;
	size_t plaintext_count = 0;
	size_t concept_build_failures = 0;
	size_t selected_entries_skipped = 0;  // short text / encoder returned nullopt
	for (const auto& cj : concept_json_entries) {
		try {
			std::string entry_id = cj.value("id", std::string());
			bool is_plaintext = !curriculum_filter.format_as_concept ||
			                    (curriculum_filter.has_filter &&
			                     curriculum_filter.plaintext_ids.count(entry_id) > 0);

			if (is_plaintext) {
				// ── Pretraining path: plain text, no execution payload ──
				std::string text = GRIM::DataLoader::renderPlainText(cj, false);
				if (text.size() < min_cleaned_text_length) { ++selected_entries_skipped; continue; }

				auto seq = build_sequence(text);
				if (!seq) { ++selected_entries_skipped; continue; }
				seq->execution_active = false;
				all_tokens.push_back(std::move(*seq));
				++plaintext_count;
				continue;
			}

			// ── Concept path: canonical formatting + execution payload ──
			auto built = GRIM::DataLoader::buildConceptSequence(cj, tokenizer, concept_exec_base_slot);
			if (built.canonical_text.size() < min_cleaned_text_length) { ++selected_entries_skipped; continue; }

			if (built.payload.execution_active) {
				const int actual_steps = static_cast<int>(built.payload.teacher_steps.size());
				if (actual_steps == 0) {
					throw std::runtime_error(
						"DataLoader: execution-active concept entry has 0 teacher_steps "
						"— cannot pad from nothing");
				}
				if (actual_steps > expected_exec_steps) {
					std::string exec_entry_id = "(unknown)";
					if (cj.contains("id") && cj["id"].is_string())
						exec_entry_id = cj["id"].get<std::string>();
					else if (cj.contains("name") && cj["name"].is_string())
						exec_entry_id = cj["name"].get<std::string>();
					throw std::runtime_error(
						"DataLoader: execution-active concept entry \"" + exec_entry_id
						+ "\" has teacher_steps=" + std::to_string(actual_steps)
						+ " > execution_block_num_steps=" + std::to_string(expected_exec_steps)
						+ " — truncation would lose computation; fix data or increase config num_steps");
				}
				// Padding deferred to buildBatchPayload where step_mask is constructed.
				// GRMT stores original step count; batch builder pads + masks.
			}

			auto seq = build_sequence(built.canonical_text);
			if (!seq) { ++selected_entries_skipped; continue; }
			seq->execution_active = built.payload.execution_active;
			if (built.payload.execution_active) {
				seq->token_exec_slots = std::move(built.payload.token_exec_slots);
				seq->compiled_bootstrap_bindings = std::move(built.payload.compiled_bootstrap_bindings);
				seq->teacher_steps = std::move(built.payload.teacher_steps);
			}
			all_tokens.push_back(std::move(*seq));
		} catch (const std::exception& e) {
			++concept_build_failures;
			std::cerr << "[DataLoader] concept build failed: " << e.what() << "\n";
		}
	}
	if (plaintext_count > 0) {
		std::cout << "[DataLoader] Encoded " << plaintext_count << " plaintext (PT) + "
		          << (all_tokens.size() - plaintext_count) << " concept sequences" << std::endl;
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
	if (curriculum_filter.has_filter && selected_entries_skipped > 0) {
		std::cerr << "[DataLoader] FATAL: " << selected_entries_skipped
		          << " silently-skipped selected entry/entries under a filtered "
		          << "curriculum. Refusing to produce a partial GRMT."
		          << std::endl;
		return false;
	}

	// Write single GRMT file — Phase1_Startup handles train/val splitting
	fs::create_directories(cache_dir);
	fs::path train_grmt = training_path;

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
		save_report = GRIM::TokenizerArtifacts::saveTokenizerArtifactBundle(tokenizer_hp, tokenizer, all_tokens);
	} catch (const std::exception& e) {
		std::cerr << "[DataLoader] FATAL: failed to save tokenizer artifact bundle: "
		          << e.what() << std::endl;
		return false;
	}
	if (save_report.grmt.dropped_targetless_sequences > 0) {
		std::cerr << "[DataLoader] Dropped "
		          << save_report.grmt.dropped_targetless_sequences
		          << " sequences with 0 valid targets" << std::endl;
	}
	if (tokenizer_hp.save_text_vocab) {
		std::cout << "[DataLoader] Also saved human-readable .txt vocab" << std::endl;
	}

	std::cout << "[DataLoader] Saved tokenizer artifact bundle:" << std::endl
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

struct NumberEncoderValidationStats {
	size_t numeric_atom_tokens = 0;
	size_t unique_numeric_entries = 0;
	size_t sequences_with_numeric_atoms = 0;
};

struct LoadedTrainingCorpus {
	std::vector<GrmtSequence> sequences;
	std::uint32_t vocab_size = 0;
	NumberEncoderValidationStats number_encoder_stats;
};

NumberEncoderValidationStats validateNumberEncoderSequenceCompatibilityOrThrow(
	const GrmtSequence& seq,
	size_t seq_idx,
	const GRIM::HyperParameters::NumberEncoderConstructionHP& number_encoder_hp)
{
	NumberEncoderValidationStats stats;
	if (!number_encoder_hp.enabled) {
		return stats;
	}
	if (number_encoder_hp.max_digit_slots <= 0) {
		throw std::runtime_error(
			"LoadTrainingData: NumberEncoder is enabled but max_digit_slots=" +
			std::to_string(number_encoder_hp.max_digit_slots) + " is not positive");
	}
	if (number_encoder_hp.max_abs_pow10 <= 0) {
		throw std::runtime_error(
			"LoadTrainingData: NumberEncoder is enabled but max_abs_pow10=" +
			std::to_string(number_encoder_hp.max_abs_pow10) + " is not positive");
	}
	if (seq.atom_entry_ids.size() != seq.token_ids.size()) {
		throw std::runtime_error(
			"LoadTrainingData: atom_entry_ids length mismatch during NumberEncoder validation");
	}

	std::unordered_set<uint32_t> validated_entry_ids;
	for (size_t token_pos = 0; token_pos < seq.token_ids.size(); ++token_pos) {
		const int token_id = seq.token_ids[token_pos];
		if (token_id < GRIM::Tokenizer::ATOM_TOKEN_OFFSET ||
			token_id >= GRIM::Tokenizer::UNIGRAM_VOCAB_OFFSET) {
			continue;
		}

		const auto atom_type = GRIM::Tokenizer::tokenIdToAtomType(token_id);
		if (!GRIM::Tokenizer::isNumericAtom(atom_type)) {
			continue;
		}

		if (stats.numeric_atom_tokens == 0) {
			stats.sequences_with_numeric_atoms = 1;
		}
		++stats.numeric_atom_tokens;

		if (!seq.atom_table) {
			throw std::runtime_error(
				"LoadTrainingData: sequence_index=" + std::to_string(seq_idx) +
				" token_pos=" + std::to_string(token_pos) +
				" has numeric atom token_id=" + std::to_string(token_id) +
				" but no AtomTable");
		}

		const uint32_t entry_id = seq.atom_entry_ids[token_pos];
		if (entry_id == GRIM::Tokenizer::kAtomEntryNone) {
			throw std::runtime_error(
				"LoadTrainingData: sequence_index=" + std::to_string(seq_idx) +
				" token_pos=" + std::to_string(token_pos) +
				" numeric atom token_id=" + std::to_string(token_id) +
				" carries kAtomEntryNone");
		}

		if (!validated_entry_ids.insert(entry_id).second) {
			continue;
		}
		++stats.unique_numeric_entries;

		const auto entry = seq.atom_table->getAtom(entry_id);
		if (!entry.has_value()) {
			throw std::runtime_error(
				"LoadTrainingData: sequence_index=" + std::to_string(seq_idx) +
				" token_pos=" + std::to_string(token_pos) +
				" atom_entry_id=" + std::to_string(entry_id) +
				" is not retrievable from its AtomTable");
		}

		std::ostringstream caller;
		caller << "LoadTrainingData: sequence_index=" << seq_idx
		       << " token_pos=" << token_pos;
		GRIM::Tokenizer::validateNumberEncoderAtomMetadataOrThrow(
			*entry,
			entry_id,
			number_encoder_hp.max_digit_slots,
			number_encoder_hp.max_abs_pow10,
			caller.str().c_str());
	}

	return stats;
}

void emitProgress(const ProgressCallback& progress, const std::string& message)
{
	if (progress) {
		progress(message);
	}
}

LoadedTrainingCorpus readGrmtCorpusWithProgressOrThrow(
	const std::string& path,
	const GRIM::HyperParameters::NumberEncoderConstructionHP& number_encoder_hp,
	const ProgressCallback& progress)
{
	GRIM::TokenizerArtifacts::GrmtCorpusReader reader(path, number_encoder_hp.max_digit_slots);
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

	GrmtSequence sequence;
	while (reader.readNext(sequence)) {
		const auto sequence_number_encoder_stats = validateNumberEncoderSequenceCompatibilityOrThrow(
			sequence,
			corpus.sequences.size(),
			number_encoder_hp);
		if (sequence_number_encoder_stats.numeric_atom_tokens > 0) {
			if (corpus.number_encoder_stats.numeric_atom_tokens == 0) {
				emitProgress(progress,
					"[Data] NumberEncoder validation active during AtomTable reconstruction...");
			}
			corpus.number_encoder_stats.numeric_atom_tokens += sequence_number_encoder_stats.numeric_atom_tokens;
			corpus.number_encoder_stats.unique_numeric_entries += sequence_number_encoder_stats.unique_numeric_entries;
			corpus.number_encoder_stats.sequences_with_numeric_atoms += sequence_number_encoder_stats.sequences_with_numeric_atoms;
		}
		corpus.sequences.push_back(std::move(sequence));
		sequence = GrmtSequence{};

		const std::uint32_t loaded = reader.sequencesRead();
		if (header.num_sequences > 0 &&
			(loaded == header.num_sequences ||
			 (progress_stride > 0 && loaded >= next_progress))) {
			std::ostringstream progress_msg;
			progress_msg << "[Data] GRMT load progress: "
			             << loaded << "/" << header.num_sequences
			             << " sequences ("
			             << std::fixed << std::setprecision(1)
			             << (100.0 * static_cast<double>(loaded) /
			                 static_cast<double>(header.num_sequences))
			             << "%)";
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
	const GRIM::HyperParameters::NumberEncoderConstructionHP& number_encoder_hp,
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
		number_encoder_hp,
		progress_logger);
	emitProgress(progress_logger, "[Data] GRMT deserialization complete; validating side channels...");
	sanitizeNumericSideChannels(corpus.sequences);
	logAtomSideChannelDiagnostics(corpus.sequences);
	if (number_encoder_hp.enabled) {
		std::ostringstream number_encoder_msg;
		number_encoder_msg << "[Data] NumberEncoder validation complete: validated "
		                  << corpus.number_encoder_stats.numeric_atom_tokens
		                  << " numeric atom tokens across "
		                  << corpus.number_encoder_stats.sequences_with_numeric_atoms
		                  << " sequences (unique atom entries="
		                  << corpus.number_encoder_stats.unique_numeric_entries
		                  << ", digit_slots=" << number_encoder_hp.max_digit_slots
		                  << ", max_abs_pow10=" << number_encoder_hp.max_abs_pow10
		                  << ")";
		emitProgress(progress_logger, number_encoder_msg.str());
	}
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

	logger.log("[Data] Applying sliding windows to train split...");
	applySlidingWindows(data.train_seqs, "train",
						max_seq_len, data_hp.sliding_window_stride, data_hp.min_seq_valid_tokens,
						tokenizer_hp.add_bos, tokenizer_hp.add_eos, logger);
	logger.log("[Data] Train split post-window sequence count=" +
			   std::to_string(data.train_seqs.size()));

	logger.log("[Data] Applying sliding windows to validation split...");
	applySlidingWindows(data.val_seqs, "val",
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
	const auto paths_hp = GRIM::HyperParameters::pathsHP(config);
	Internal::validateStartupPaths(tokenizer_hp, paths_hp);

	logger.log("Loading tokenizer artifact bundle...");
	auto tokenizer = std::make_unique<GRIM::Tokenizer::UniByte>(tokenizer_hp);
	(void)GRIM::TokenizerArtifacts::loadTokenizerArtifactBundle(tokenizer_hp, *tokenizer);
	logger.log("Initializing tokenizer CUDA Viterbi runtime...");
	if (!tokenizer->initGPU()) {
		throw std::runtime_error("LoadInferenceTokenizer: UniByte::initGPU() returned false after artifact load");
	}

	return tokenizer;
}

void LoadTrainingData(TrainingContext& ctx, const MemorySnapshot& startup_memory_snapshot) {
	using GRIM::Logging::EmitModuleInfo;
	using GRIM::Logging::ModuleId;

	const auto tokenizer_hp = GRIM::HyperParameters::tokenizerHP(ctx.config);
	const auto number_encoder_hp = GRIM::HyperParameters::numberEncoderConstructionHP(ctx.config);
	const auto paths_hp = GRIM::HyperParameters::pathsHP(ctx.config);
	const auto data_hp = GRIM::HyperParameters::dataLoadingHP(ctx.config);

 	EmitModuleInfo(ModuleId::Training, "[Phase1] Validating paths...", 0);
 	Internal::validateStartupPaths(tokenizer_hp, paths_hp);
 	EmitModuleInfo(ModuleId::Training, "[Phase1] ✓ All paths validated", 0);

	ctx.data = Internal::buildPhase1SequenceData(
		tokenizer_hp,
		number_encoder_hp,
		data_hp,
		*ctx.logging.logger);

	const std::uint32_t actual_vocab_size = requireActualVocabSizeOrThrow(ctx.data);
	(void)GRIM::Tokenizer::tokenLayoutFromActualVocabOrThrow(
		actual_vocab_size,
		"LoadTrainingData");
	syncRuntimeVocabSizeFromActualOrThrow(ctx.config, actual_vocab_size, "LoadTrainingData");

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
	data_stats.memory_device = startup_memory_snapshot.device;
	data_stats.memory_device_name = startup_memory_snapshot.device_name;
	data_stats.memory_total_bytes = startup_memory_snapshot.total_bytes;
	data_stats.memory_free_bytes = startup_memory_snapshot.free_bytes;

	dumpAllHyperparameters(
		ctx.config,
		&ctx.derived_schedule,
		&data_stats,
		[](const std::string& msg) { GRIM::Logging::EmitModuleInfo(GRIM::Logging::ModuleId::Training, msg, 0); });
}
} // namespace GRIMText::Training

