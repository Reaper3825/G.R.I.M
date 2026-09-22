#include "ConceptBlockGrmtCompiler.hpp"

#include "ConceptBlockCorpusReader.hpp"
#include "../../../../../DataCollection/concept_block_canonical.hpp"
#include "../ConceptBlock/ConceptBlockSpans.hpp"
#include "../ConceptBlock/NamedConceptSpans.hpp"
#include "../Goal/Goal.hpp"
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
		for (const auto& named_span : rendered.named_spans) {
			add_span(named_span.span);
		}
		add_span(rendered.target_state);
		add_span(rendered.criteria);
		for (const auto& entry : rendered.success_criteria) {
			add_span(entry.criterion);
			add_span(entry.evidence);
		}
		add_span(rendered.constraints_span);
		for (const auto& constraint : rendered.constraints) {
			add_span(constraint);
		}
		for (const auto& known : rendered.knowns) {
			add_span(known);
		}
		for (const auto& unknown : rendered.unknowns) {
			add_span(unknown);
		}
		add_span(rendered.reasoning);
		add_span(rendered.determine);
		add_span(rendered.define);
		add_span(rendered.execute);
		add_span(rendered.update);
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
			if (!source_entries.empty()) {
				constraints.span = token_span(
					rendered.constraints_span, boundaries, token_counts,
					"goal.constraints");
			}
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
		auto materialize_optional = [&](
			const char* field,
			const GRIM::ConceptCanonical::LogicalByteSpan& rendered_span,
			std::optional<GRIM::ConceptBlockSpanEntry>& destination) {
			if (!rendered_span.present) return;
			GRIM::ConceptBlockSpanEntry entry;
			entry.span = token_span(
				rendered_span, boundaries, token_counts, field);
			entry.token_ids = span_token_ids(sequence, entry.span, field);
			destination = std::move(entry);
		};
		materialize_optional("reasoning", rendered.reasoning, spans->reasoning);
		materialize_optional("determine", rendered.determine, spans->determine);
		materialize_optional("define", rendered.define, spans->define);
		materialize_optional("execute", rendered.execute, spans->execute);
		materialize_optional("update", rendered.update, spans->update);
		if (spans->empty()) {
			return nullptr;
		}
		std::shared_ptr<const GRIM::ConceptBlockSpans> immutable_spans =
			std::move(spans);
		return immutable_spans;
	};

	auto materialize_named_concept_spans = [&token_span, &span_token_ids](
		const GRIM::ConceptCanonical::RenderResult& rendered,
		const std::vector<size_t>& boundaries,
		const std::vector<size_t>& token_counts,
		const TokenizedSequence& sequence)
		-> std::shared_ptr<const GRIM::NamedConceptSpans> {
		auto spans = std::make_shared<GRIM::NamedConceptSpans>();
		spans->entries.reserve(rendered.named_spans.size());
		for (const auto& rendered_span : rendered.named_spans) {
			GRIM::NamedConceptSpan entry;
			entry.name = rendered_span.name;
			entry.span = token_span(
				rendered_span.span, boundaries, token_counts, entry.name);
			entry.token_ids = span_token_ids(sequence, entry.span, entry.name);
			spans->entries.push_back(std::move(entry));
		}
		if (spans->empty()) return nullptr;
		std::shared_ptr<const GRIM::NamedConceptSpans> immutable_spans =
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
				assign_prompt_span(*seq, rendered, boundaries, token_counts);
				seq->named_concept_spans = materialize_named_concept_spans(
					rendered, boundaries, token_counts, *seq);
				if (rendered.answer.present) {
					seq->answer_span = token_span(
						rendered.answer, boundaries, token_counts, "answer");
				}
				seq->concept_block_id = cj.at("id").get<std::string>();
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
			assign_prompt_span(*seq, rendered, boundaries, token_counts);
			seq->named_concept_spans = materialize_named_concept_spans(
				rendered, boundaries, token_counts, *seq);
			if (rendered.answer.present) {
				seq->answer_span = token_span(
					rendered.answer, boundaries, token_counts, "answer");
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
