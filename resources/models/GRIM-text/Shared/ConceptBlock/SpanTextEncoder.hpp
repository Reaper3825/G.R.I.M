#pragma once
#include "SpanTextRenderer.hpp"
#include "../UnigramByte/UniByte.hpp"

namespace GRIM::SpanText {
struct EncodedText {
    Tokenizer::UniByteResult tokens;
    std::shared_ptr<const NamedConceptSpans> spans;
};

// Preserve supplied typed atom boundaries and every aligned side channel.
// Delimiter IDs are resolved by the input owner before this operation.
template<class Tokenize>
inline EncodedText encodeSpanText(const RenderResult& rendered, Tokenize&& tokenize) {
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
        ++encoded.unigram_tokens; // Cached section delimiters are exact unigram pieces.
    }
    const auto n = encoded.token_ids.size();
    if (encoded.token_numeric_values.size() != n || encoded.token_atom_flags.size() != n ||
        encoded.token_atom_mask.size() != n || encoded.atom_entry_ids.size() != n ||
        encoded.token_local_atom_indices.size() != n)
        throw std::runtime_error("encodeSpanText: token/side-channel length mismatch");
    auto spans = std::make_shared<GRIM::NamedConceptSpans>();
    const auto token_position = [&](size_t pos) {
        size_t inserted = 0;
        for (const auto& d : rendered.delimiters) if (d.end <= pos) ++inserted;
        const auto result = content_token_position(content_position(pos)) + inserted;
        if (result > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
            throw std::runtime_error("Span token position exceeds int32");
        return static_cast<int32_t>(result);
    };
    for (const auto& e : rendered.named_spans)
        spans->entries.push_back({e.name, {token_position(e.span.begin), token_position(e.span.end)},
                                  e.parent_entry_index, e.child_entry_indices});
    validateNamedConceptSpans(*spans, encoded.token_ids.size());
    return {std::move(encoded), std::move(spans)};
}

} // namespace GRIM::SpanText
