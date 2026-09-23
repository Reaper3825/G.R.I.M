#include "ConceptBlockGrmtCompiler.hpp"

#include "ConceptBlockCorpusReader.hpp"
#include "../../../../../DataCollection/concept_block_canonical.hpp"
#include "../ConceptBlock/NamedConceptSpans.hpp"
#include "../GRMT/GrmtFormat.hpp"
#include "../UnigramByte/TokenLayout.hpp"
#include "../UnigramByte/UniByte.hpp"
#include "../TokenizerArtifacts/GrmtCorpusIO.hpp"
#include "../TokenizerArtifacts/TokenizerArtifactBundle.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace GRIM {

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

	auto definitions = tokenizer_hp.concept_spans;
	GRIM::validateConceptSpanDefinitions(definitions);
	const auto layout = std::make_shared<const std::string>(GRIM::conceptSpanLayout(definitions));
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
				vocab_corpus.push_back(GRIM::ConceptCanonical::render(cj, definitions).text);
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

	GRIM::resolveConceptSpanDelimiters(definitions, [&](const std::string& text) {
        const auto id = tokenizer.unigramLM().getPieceId(text);
        const auto* piece = tokenizer.unigramLM().getPiece(id);
        return piece && piece->text == text ? id : -1;
    });


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


            GRIM::ConceptCanonical::RenderResult rendered;
            if (is_raw_text) rendered.text = GRIM::ConceptCanonical::renderPlainText(cj);
            else rendered = GRIM::ConceptCanonical::render(cj, definitions);
            if (rendered.text.size() < min_cleaned_text_length) {
                ++selected_entries_skipped;
                continue;
            }
            auto seq = GRIM::encodeConceptBlockRender(rendered, layout,
                [&](const auto& content, const auto& boundaries, auto* counts) {
                    return tokenizer.tokenizeWithMetadata(content, boundaries, counts);
                });
            if (!seq) { ++selected_entries_skipped; continue; }
            if (is_raw_text) ++raw_text_count;

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
			seq->concept_block_id = cj.at("id").get<std::string>();
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
				  << "Check tokenizer_enable_atom_reasoning in model_config.json" << std::endl;
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
