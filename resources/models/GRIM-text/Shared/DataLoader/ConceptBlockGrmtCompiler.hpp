#pragma once

#include "../HyperParameters/HyperparameterGroupings.hpp"
#include "../../../../../DataCollection/concept_block_canonical.hpp"
#include "../TokenizerArtifacts/GrmtSequence.hpp"
#include <algorithm>
#include <limits>
#include <optional>

namespace GRIM {

// Encode content once and interleave cached delimiter IDs, preserving all
// atom side channels. The callback is UniByte::tokenizeWithMetadata in production;
// keeping this boundary injectable permits CPU-only layout/serialization tests.
template<class Tokenize>
inline std::optional<TokenizerArtifacts::GrmtSequence> encodeConceptBlockRender(
    const ConceptCanonical::RenderResult& rendered,
    const std::shared_ptr<const std::string>& layout,
    Tokenize&& tokenize) {
    using TokenizedSequence = TokenizerArtifacts::GrmtSequence;
	auto materialize_sequence = [](auto result)
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

        std::vector<size_t> boundaries{0, rendered.text.size()};
        for (const auto& e : rendered.named_spans) {
            boundaries.push_back(e.span.begin);
            boundaries.push_back(e.span.end);
        }
        for (const auto& d : rendered.delimiters) {
            boundaries.push_back(d.begin);
            boundaries.push_back(d.end);
            if (d.token_id < 0) throw std::runtime_error("Unresolved concept delimiter");
        }
        std::sort(boundaries.begin(), boundaries.end());
        boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());
        std::string content;
        size_t cursor = 0;
        for (const auto& d : rendered.delimiters) {
            content.append(rendered.text, cursor, d.begin - cursor);
            cursor = d.end;
        }
        content.append(rendered.text, cursor, rendered.text.size() - cursor);
        const auto content_position = [&](size_t pos) {
            size_t removed = 0;
            for (const auto& d : rendered.delimiters) {
                if (d.end <= pos) removed += d.end - d.begin;
                else break;
            }
            return pos - removed;
        };
        std::vector<size_t> content_boundaries;
        for (const auto pos : boundaries) content_boundaries.push_back(content_position(pos));
        std::sort(content_boundaries.begin(), content_boundaries.end());
        content_boundaries.erase(std::unique(content_boundaries.begin(), content_boundaries.end()),
                                 content_boundaries.end());
        std::vector<size_t> counts;
        auto encoded = tokenize(content, content_boundaries, &counts);
        const auto content_token_position = [&](size_t byte) {
            const auto i = std::lower_bound(content_boundaries.begin(), content_boundaries.end(), byte);
            if (i == content_boundaries.end() || *i != byte)
                throw std::runtime_error("Missing concept content boundary");
            return counts.at(static_cast<size_t>(i - content_boundaries.begin()));
        };
        // Inserting in reverse preserves each original content position.
        for (auto i = rendered.delimiters.rbegin(); i != rendered.delimiters.rend(); ++i) {
            const auto pos = static_cast<ptrdiff_t>(content_token_position(content_position(i->begin)));
            encoded.token_ids.insert(encoded.token_ids.begin() + pos, i->token_id);
            encoded.is_byte_fallback.insert(encoded.is_byte_fallback.begin() + pos, false);
            encoded.token_numeric_values.insert(encoded.token_numeric_values.begin() + pos, 0.0f);
            encoded.token_atom_flags.insert(encoded.token_atom_flags.begin() + pos, 0);
            encoded.token_atom_mask.insert(encoded.token_atom_mask.begin() + pos, 0);
            encoded.atom_entry_ids.insert(encoded.atom_entry_ids.begin() + pos, GRIM::Tokenizer::kAtomEntryNone);
            encoded.token_local_atom_indices.insert(encoded.token_local_atom_indices.begin() + pos,
                                                    GRIM::Tokenizer::kLocalAtomIndexNone);
        }
        auto sequence = materialize_sequence(std::move(encoded));
        if (!sequence) return std::nullopt;
        sequence->concept_span_layout = layout;
        auto spans = std::make_shared<GRIM::NamedConceptSpans>();
        const auto token_position = [&](size_t pos) {
            size_t inserted = 0;
            for (const auto& d : rendered.delimiters) if (d.end <= pos) ++inserted;
            const auto result = content_token_position(content_position(pos)) + inserted;
            if (result > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
                throw std::runtime_error("Concept token position exceeds int32");
            return static_cast<int32_t>(result);
        };
        for (const auto& e : rendered.named_spans) {
            spans->entries.push_back({e.name, {token_position(e.span.begin), token_position(e.span.end)},
                                      e.parent_entry_index, e.child_entry_indices});
        }
        if (!spans->empty()) {
            GRIM::validateNamedConceptSpans(*spans, sequence->token_ids.size());
            sequence->named_concept_spans = std::move(spans);
        }
        return sequence;
}

 // Compiles the selected ConceptBlock curriculum into a GRMT tokenizer artifact.
// Existing compatible artifacts are reused according to TokenizerHP.
bool PrepareTrainingDataFromCache(
    const HyperParameters::TokenizerHP& tokenizer_hp);

} // namespace GRIM
