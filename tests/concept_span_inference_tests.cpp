#include "../resources/models/GRIM-text/Shared/DataLoader/ConceptBlockGrmtCompiler.hpp"
#include "../resources/models/GRIM-text/Shared/UnigramByte/UniByte.hpp"
#include <cassert>
#include <iostream>
#include <map>

int main() {
    using namespace GRIM;
    NamedConceptSpanDefinition prompt;
    prompt.name = "question"; // configured name need not be 'prompt'
    prompt.source_path = "/prompt";
    prompt.supervision = ConceptSpanSupervision::Context;
    prompt.open_delimiter = "<q>";
    prompt.close_delimiter = "</q>";
    NamedConceptSpanDefinition wrapper;
    wrapper.name = "wrapper";
    wrapper.source_path = "/state";
    wrapper.supervision = ConceptSpanSupervision::Ignore;
    wrapper.open_delimiter = "<state>";
    wrapper.close_delimiter = "</state>";
    NamedConceptSpanDefinition kept;
    kept.name = "kept";
    kept.source_path = "/kept";
    kept.supervision = ConceptSpanSupervision::Context;
    kept.open_delimiter = "<kept>";
    kept.close_delimiter = "</kept>";
    wrapper.children = {kept};
    NamedConceptSpanDefinitions definitions{prompt, wrapper};
    std::map<int, std::string> pieces;
    int next = 1000;
    resolveConceptSpanDelimiters(definitions, [&](const auto& text) {
        pieces[next] = text;
        return next++;
    });
    const nlohmann::json source{{"prompt", "Q"}, {"state", {{"kept", "K"}}}};
    const auto rendered = ConceptCanonical::renderReasoningPromptWithSpans(source, definitions);
    assert(rendered.text == "<q>\nQ\n</q>\n\n<kept>\nK\n</kept>\n\n");
    assert(rendered.named_spans.size() == 3 && rendered.delimiters.size() == 4);
    assert(rendered.named_spans[2].parent_entry_index == 1);
    assert(rendered.named_spans[1].child_entry_indices == std::vector<std::uint32_t>{2});
    auto encoded = encodeConceptBlockRender(rendered, nullptr,
        [](const std::string& content, const auto& boundaries, auto* counts) {
            Tokenizer::UniByteResult result;
            assert(content.find('<') == std::string::npos);
            for (unsigned char c : content) result.token_ids.push_back(Tokenizer::BYTE_TOKEN_OFFSET + c);
            const auto n = result.token_ids.size();
            result.is_byte_fallback.assign(n, true);
            result.token_numeric_values.assign(n, 0);
            result.token_atom_flags.assign(n, 0);
            result.token_atom_mask.assign(n, 0);
            result.atom_entry_ids.assign(n, Tokenizer::kAtomEntryNone);
            result.token_local_atom_indices.assign(n, Tokenizer::kLocalAtomIndexNone);
            counts->assign(boundaries.begin(), boundaries.end());
            return result;
        }).value();
    validateNamedConceptSpans(*encoded.named_concept_spans, encoded.token_ids.size());
    const auto& prompt_span = encoded.named_concept_spans->entries[0].span;
    assert(prompt_span.begin == 0 && prompt_span.end == 7);
    assert(static_cast<std::size_t>(prompt_span.end) < encoded.token_ids.size());
    std::string decoded;
    for (int id : encoded.token_ids)
        decoded += pieces.count(id) ? pieces.at(id) : std::string(1, static_cast<char>(id - Tokenizer::BYTE_TOKEN_OFFSET));
    assert(decoded == rendered.text);
    std::cout << "inference span tests passed\n";
}
