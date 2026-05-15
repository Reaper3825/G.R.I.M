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
#include <stdexcept>
#include <unordered_set>
#include <cstdint>
#include <exception>
#include <cmath>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>
#include "../GRMT/GrmtFormat.hpp"
#include "../../Shared/UnigramByte/UniByte.hpp"
#include "../../Shared/TokenizerArtifacts/TokenizerArtifactBundle.hpp"
#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"  // also pulls in control/ai_config_paths.hpp transitively (correct order)
#include "../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../Batching/BatchPayload.hpp"
#include "ConceptExecutionSequenceBuilder.hpp"

namespace fs = std::filesystem;

namespace GRIM {

using GRIM::Config::GrimTextPaths;

// Minimum cleaned text length to include in training data.
// Shorter texts lack sufficient context for meaningful next-token prediction.
constexpr size_t kMinCleanedTextLength = 20;

// ─── Concept blocks corpus loading ──────────────────────────────────────────
//
// Loads concept_blocks.jsonl and returns parsed JSON objects.
// If curriculum_manifest.json exists alongside the JSONL, only entries
// whose "id" appears in the manifest's concept_block_ids are kept.
// When no manifest is present, all entries are loaded (backward compat).
//
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
	const GrimTextPaths& paths,
	const GRIM::HyperParameters::TokenizerHP& tokenizer_hp,
	std::string& out_training_data_path,
	std::string& out_vocab_path,
	bool force_rebuild) {

	// Resolve primary paths from config
	if (!paths.training_data.empty()) {
		out_training_data_path = paths.training_data;
	}
	if (!paths.vocab.empty()) {
		out_vocab_path = paths.vocab;
	}

	if (out_training_data_path.empty()) {
		std::cerr << "[DataLoader] No training_data path configured; skipping cache preparation." << std::endl;
		return false;
	}
	if (out_vocab_path.empty()) {
		throw std::runtime_error("[DataLoader] vocab path is empty; tokenizer artifacts must be saved/loaded as vocab+GRMT pair");
	}

	GRIM::Tokenizer::configureTokenLayout(GRIM::Tokenizer::kAtomTypeCount);
	std::cout << "[DataLoader] Atom token range fixed at " << GRIM::Tokenizer::ATOM_VOCAB_SIZE
	          << " type tokens" << std::endl;

	GRIM::Tokenizer::UniByte tokenizer(tokenizer_hp);
	GRIM::TokenizerArtifacts::TokenizerArtifactBundle artifacts({out_training_data_path, out_vocab_path});

	const bool training_exists = fs::exists(out_training_data_path);
	const bool vocab_exists = !out_vocab_path.empty() && fs::exists(out_vocab_path);
	bool artifact_pair_invalid = false;

	// Vocab + GRMT are one artifact pair. A valid cache must load the vocab and
	// validate the GRMT header vocab against the tokenizer's live token space.
	if (!force_rebuild && training_exists && vocab_exists) {
		try {
			const auto manifest = artifacts.load(tokenizer);
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
	if (force_rebuild) {
		std::cout << "[DataLoader] force_rebuild=true, rebuilding tokenizer artifact bundle..." << std::endl;
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
	fs::path training_path(out_training_data_path);
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
	tokenizer.train(vocab_corpus);

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
				if (text.size() < kMinCleanedTextLength) { ++selected_entries_skipped; continue; }

				auto seq = build_sequence(text);
				if (!seq) { ++selected_entries_skipped; continue; }
				seq->execution_active = false;
				all_tokens.push_back(std::move(*seq));
				++plaintext_count;
				continue;
			}

			// ── Concept path: canonical formatting + execution payload ──
			auto built = GRIM::DataLoader::buildConceptSequence(cj, tokenizer, concept_exec_base_slot);
			if (built.canonical_text.size() < kMinCleanedTextLength) { ++selected_entries_skipped; continue; }

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
				seq->slot_selection_targets = std::move(built.payload.slot_selection_targets);
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

	GRIM::TokenizerArtifacts::TokenizerBundleSaveOptions save_options;
	save_options.save_text_vocab = tokenizer_hp.save_text_vocab;
	save_options.vocab_score_multiplier = tokenizer_hp.vocab_score_multiplier;
	save_options.grmt.reject_dropped_sequences = curriculum_filter.has_filter;

	GRIM::TokenizerArtifacts::TokenizerBundleSaveReport save_report;
	try {
		save_report = artifacts.save(tokenizer, all_tokens, save_options);
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
	if (save_options.save_text_vocab) {
		std::cout << "[DataLoader] Also saved human-readable .txt vocab" << std::endl;
	}

	std::cout << "[DataLoader] Saved tokenizer artifact bundle:" << std::endl
	          << "  Vocab: " << out_vocab_path << std::endl
	          << "  GRMT:  " << train_grmt.string() << std::endl
	          << "  Written sequences: " << save_report.grmt.written_sequences << std::endl
	          << "  Vocab size: " << save_report.manifest.grmt_header.vocab_size << std::endl;

	return true;
}

} // namespace GRIM

