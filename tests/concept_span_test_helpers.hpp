#pragma once
#include "../DataCollection/concept_block_canonical.hpp"
#include "../resources/models/GRIM-text/Shared/TokenizerArtifacts/GrmtSequence.hpp"
#include "../resources/models/GRIM-text/Shared/UnigramByte/TokenLayout.hpp"
#include <fstream>
#include <cassert>

inline GRIM::NamedConceptSpanDefinitions testSpanDefinitions() {
    std::ifstream file("tests/fixtures/concept_spans.json");
    if (!file) throw std::runtime_error("Run concept span tests from the repository root");
    auto result = nlohmann::json::parse(file).get<GRIM::NamedConceptSpanDefinitions>();
    GRIM::validateConceptSpanDefinitions(result);
    return result;
}

inline GRIM::TokenizerArtifacts::GrmtSequence testSpanSequence(
    const nlohmann::json& source, const GRIM::NamedConceptSpanDefinitions& definitions) {
    const auto rendered = GRIM::ConceptCanonical::render(source, definitions);
    GRIM::TokenizerArtifacts::GrmtSequence row;
    row.concept_block_id = "generic-span-test";
    row.concept_span_layout = std::make_shared<const std::string>(GRIM::conceptSpanLayout(definitions));
    for (unsigned char c : rendered.text) row.token_ids.push_back(GRIM::Tokenizer::BYTE_TOKEN_OFFSET + c);
    const auto n = row.token_ids.size();
    row.targets.assign(n, -1);
    for (size_t i = 1; i < n; ++i) row.targets[i - 1] = row.token_ids[i];
    row.token_numeric_values.assign(n, 0);
    row.token_atom_mask.assign(n, 0);
    row.token_atom_flags.assign(n, 0);
    row.atom_entry_ids.assign(n, GRIM::Tokenizer::kAtomEntryNone);
    row.token_local_atom_indices.assign(n, GRIM::Tokenizer::kLocalAtomIndexNone);
    row.token_exec_slot_indices.assign(n, -1);
    auto spans = std::make_shared<GRIM::NamedConceptSpans>();
    for (const auto& e : rendered.named_spans)
        spans->entries.push_back({e.name, {static_cast<int32_t>(e.span.begin), static_cast<int32_t>(e.span.end)},
                                 e.parent_entry_index, e.child_entry_indices});
    row.named_concept_spans = spans;
    return row;
}

template<class F> inline void expectSpanError(F&& call) {
    bool rejected = false;
    try { call(); } catch (const std::exception&) { rejected = true; }
    assert(rejected);
}
