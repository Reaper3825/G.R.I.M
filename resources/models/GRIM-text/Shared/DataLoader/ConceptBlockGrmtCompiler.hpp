#pragma once
#include "../HyperParameters/HyperparameterGroupings.hpp"
#include "../ConceptBlock/SpanTextEncoder.hpp"
#include "../../../../../DataCollection/concept_block_canonical.hpp"
#include "../TokenizerArtifacts/GrmtSequence.hpp"
#include <optional>

namespace GRIM {
template<class Tokenize>
inline std::optional<TokenizerArtifacts::GrmtSequence> encodeConceptBlockRender(
    const ConceptCanonical::RenderResult& rendered,
    const std::shared_ptr<const std::string>& layout, Tokenize&& tokenize) {
    auto encoded = SpanText::encodeSpanText(rendered, std::forward<Tokenize>(tokenize));
    auto& result = encoded.tokens;
    if (result.token_ids.empty()) return std::nullopt;
    TokenizerArtifacts::GrmtSequence seq;
    seq.token_ids = std::move(result.token_ids);
    seq.token_numeric_values = std::move(result.token_numeric_values);
    seq.token_atom_flags = std::move(result.token_atom_flags);
    seq.token_atom_mask = std::move(result.token_atom_mask);
    seq.atom_table = std::move(result.atom_table);
    seq.atom_entry_ids = std::move(result.atom_entry_ids);
    seq.local_atom_table = std::move(result.local_atom_table);
    seq.token_local_atom_indices = std::move(result.token_local_atom_indices);
    seq.concept_span_layout = layout;
    if (!encoded.spans->empty()) seq.named_concept_spans = std::move(encoded.spans);
    seq.targets.assign(seq.token_ids.size(), -1);
    for (size_t j = 0; j + 1 < seq.token_ids.size(); ++j) seq.targets[j] = seq.token_ids[j + 1];
    seq.token_exec_slot_indices.assign(seq.token_ids.size(), -1);
    return seq;
}

bool PrepareTrainingDataFromCache(const HyperParameters::TokenizerHP& tokenizer_hp);
} // namespace GRIM
