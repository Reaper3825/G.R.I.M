#include "concept_span_test_helpers.hpp"
#include "../resources/models/GRIM-text/Shared/DataLoader/ConceptBlockGrmtCompiler.hpp"
#include "../resources/models/GRIM-text/Shared/TokenizerArtifacts/GrmtCorpusIO.hpp"
#include "../resources/models/GRIM-text/Shared/UnigramByte/UniByte.hpp"
#include <iostream>
#include <unordered_map>

int main() try {
    auto definitions = testSpanDefinitions();
    std::unordered_map<int, std::string> pieces;
    int next_id = 1000;
    GRIM::resolveConceptSpanDelimiters(definitions, [&](const std::string& text) {
        pieces[next_id] = text;
        return next_id++;
    });
    const nlohmann::json source{
        {"prompt", "Question"}, {"goal", {
            {"success_criteria", {{{"criterion", "Check"}, {"evidence", "Proof"}},
                                  {{"criterion", "Other"}, {"evidence", ""}}}}}},
        {"answer", "Result"}};
    size_t calls = 0;
    const auto tokenize = [&](const std::string& text, const auto& boundaries, auto* counts) {
        ++calls;
        // Configured tags must never reach the content tokenizer.
        assert(text.find("<prompt>") == std::string::npos && text.find("<criterion>") == std::string::npos);
        GRIM::Tokenizer::UniByteResult result;
        for (unsigned char c : text) result.token_ids.push_back(GRIM::Tokenizer::BYTE_TOKEN_OFFSET + c);
        const auto n = result.token_ids.size();
        result.is_byte_fallback.assign(n, true);
        result.token_numeric_values.assign(n, 0);
        result.token_atom_flags.assign(n, 0);
        result.token_atom_mask.assign(n, 0);
        result.atom_entry_ids.assign(n, GRIM::Tokenizer::kAtomEntryNone);
        result.token_local_atom_indices.assign(n, GRIM::Tokenizer::kLocalAtomIndexNone);
        result.atom_table = std::make_shared<GRIM::Tokenizer::AtomTable>();
        result.local_atom_table = std::make_shared<GRIM::Tokenizer::SequenceLocalAtomTable>();
        counts->assign(boundaries.begin(), boundaries.end());
        return result;
    };
    const auto layout = std::make_shared<const std::string>(GRIM::conceptSpanLayout(definitions));
    const auto rendered = GRIM::ConceptCanonical::render(source, definitions);
    auto row = GRIM::encodeConceptBlockRender(rendered, layout, tokenize).value();
    row.concept_block_id = "cached-delimiter-roundtrip";
    assert(calls == 1);
    std::string decoded;
    for (size_t i = 0; i < row.token_ids.size(); ++i) {
        const int id = row.token_ids[i];
        if (pieces.count(id)) {
            decoded += pieces.at(id);
            assert(row.token_atom_mask[i] == 0 && row.atom_entry_ids[i] == GRIM::Tokenizer::kAtomEntryNone);
        } else decoded += static_cast<char>(id - GRIM::Tokenizer::BYTE_TOKEN_OFFSET);
    }
    assert(decoded == rendered.text);
    assert(row.token_ids.front() == definitions.front().open_delimiter_id);
    row.validateForWrite("unit");
    const auto path = std::filesystem::path(".codex-build/concept-spans/roundtrip.grmt");
    GRIM::TokenizerArtifacts::saveGrmtCorpus(path, {row, row}, 2048);
    GRIM::TokenizerArtifacts::GrmtCorpusReader reader(path);
    assert(reader.conceptSpanLayout() == *layout);
    const auto loaded = reader.readAll();
    assert(loaded.header.version == 34 && loaded.sequences.size() == 2);
    assert(loaded.sequences[0].token_ids == row.token_ids);
    assert(loaded.sequences[0].concept_span_layout == loaded.sequences[1].concept_span_layout);
    const auto& tree = *loaded.sequences[0].named_concept_spans;
    assert(tree.count("success_criterion") == 2 && tree.count("evidence") == 1);
    GRIM::validateNamedConceptSpans(tree, row.token_ids.size());
    for (size_t i = 0; i < tree.size(); ++i) {
        const auto& expected = row.named_concept_spans->entries[i];
        assert(tree.entries[i].span.begin == expected.span.begin && tree.entries[i].span.end == expected.span.end);
        assert(tree.entries[i].parent_entry_index == expected.parent_entry_index);
        assert(tree.entries[i].child_entry_indices == expected.child_entry_indices);
    }
    auto corrupt = row;
    auto bad_tree = std::make_shared<GRIM::NamedConceptSpans>(*row.named_concept_spans);
    bad_tree->entries[2].parent_entry_index = 2;
    corrupt.named_concept_spans = bad_tree;
    expectSpanError([&] { corrupt.validateForWrite("cycle"); });
    auto changed = definitions;
    changed.back().name = "new_name";
    assert(GRIM::conceptSpanLayout(changed) != *layout);
    changed = definitions; changed.back().supervision = GRIM::ConceptSpanSupervision::Context;
    assert(GRIM::conceptSpanLayout(changed) == *layout);
    // Binary format remains valid with an arbitrary additional named node.
    auto generic = row;
    generic.concept_span_layout = std::make_shared<const std::string>("new-layout");
    auto new_tree = std::make_shared<GRIM::NamedConceptSpans>(*row.named_concept_spans);
    new_tree->entries.back().name = "previously_unknown_field";
    generic.named_concept_spans = new_tree;
    const auto second = std::filesystem::path(".codex-build/concept-spans/new-field.grmt");
    GRIM::TokenizerArtifacts::saveGrmtCorpus(second, {generic}, 2048);
    assert(GRIM::TokenizerArtifacts::loadGrmtCorpus(second).sequences[0].named_concept_spans->find("previously_unknown_field"));
    std::cout << "cached-ID insertion, tree GRMT roundtrip, shared layout and extensibility passed\n";
} catch (const std::exception& e) { std::cerr << e.what() << "\n"; return 1; }
