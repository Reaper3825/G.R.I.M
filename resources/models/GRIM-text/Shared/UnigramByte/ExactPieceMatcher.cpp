#include "ExactPieceMatcher.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace GRIM::Tokenizer {
namespace {

constexpr std::string_view kWordBoundaryMarker = "\xE2\x96\x81";

bool startsWith(std::string_view text, std::string_view prefix) {
    return text.size() >= prefix.size() &&
           text.compare(0, prefix.size(), prefix) == 0;
}

bool hasLexicalEndBoundary(std::string_view text, std::size_t end) {
    if (end == text.size()) return true;
    if (end > text.size()) return false;
    const std::string_view suffix = text.substr(end);
    if (startsWith(suffix, kWordBoundaryMarker)) return true;

    const unsigned char next = static_cast<unsigned char>(text[end]);
    if (next >= 0x80) return false;
    return std::isalnum(next) == 0 && next != '_';
}

bool overlaps(const ExactPieceSpan& lhs, const ExactPieceSpan& rhs) {
    return lhs.start < rhs.end && rhs.start < lhs.end;
}

} // namespace

ExactPieceMatcher::Node::Node() {
    children.fill(-1);
}

ExactPieceMatcher::ExactPieceMatcher() {
    trie_.emplace_back();
}

void ExactPieceMatcher::rebuild(
    const std::vector<ExactPieceDefinition>& definitions) {
    trie_.clear();
    trie_.emplace_back();
    definition_count_ = 0;

    for (const ExactPieceDefinition& definition : definitions) {
        if (definition.text.empty() || definition.token_id < 0) {
            throw std::runtime_error(
                "ExactPieceMatcher::rebuild received an invalid definition");
        }
        int node = 0;
        for (const unsigned char byte : definition.text) {
            int& child = trie_[static_cast<std::size_t>(node)].children[byte];
            if (child < 0) {
                child = static_cast<int>(trie_.size());
                trie_.emplace_back();
            }
            node = child;
        }
        Node& terminal = trie_[static_cast<std::size_t>(node)];
        if (terminal.token_id >= 0) {
            throw std::runtime_error(
                "ExactPieceMatcher::rebuild received duplicate text: " +
                definition.text);
        }
        terminal.token_id = definition.token_id;
        terminal.requires_end_boundary =
            startsWith(definition.text, kWordBoundaryMarker);
        ++definition_count_;
    }
}

std::vector<ExactPieceSpan> ExactPieceMatcher::findMatches(
    std::string_view normalized_text,
    const std::vector<ExactPieceSpan>& excluded_spans) const {
    std::vector<ExactPieceSpan> matches;
    if (empty() || normalized_text.empty()) return matches;

    std::size_t previous_end = 0;
    for (const ExactPieceSpan& excluded : excluded_spans) {
        if (excluded.start > excluded.end || excluded.end > normalized_text.size() ||
            excluded.start < previous_end) {
            throw std::runtime_error(
                "ExactPieceMatcher::findMatches requires sorted, non-overlapping in-range exclusions");
        }
        previous_end = excluded.end;
    }

    std::size_t excluded_index = 0;
    std::size_t pos = 0;
    while (pos < normalized_text.size()) {
        while (excluded_index < excluded_spans.size() &&
               excluded_spans[excluded_index].end <= pos) {
            ++excluded_index;
        }
        if (excluded_index < excluded_spans.size() &&
            excluded_spans[excluded_index].start <= pos &&
            pos < excluded_spans[excluded_index].end) {
            pos = excluded_spans[excluded_index].end;
            continue;
        }

        const std::size_t scan_limit =
            excluded_index < excluded_spans.size()
                ? std::min(normalized_text.size(), excluded_spans[excluded_index].start)
                : normalized_text.size();
        int node = 0;
        ExactPieceSpan best{};
        bool found = false;
        for (std::size_t cursor = pos; cursor < scan_limit; ++cursor) {
            const unsigned char byte =
                static_cast<unsigned char>(normalized_text[cursor]);
            node = trie_[static_cast<std::size_t>(node)].children[byte];
            if (node < 0) break;

            const Node& candidate = trie_[static_cast<std::size_t>(node)];
            const std::size_t end = cursor + 1;
            if (candidate.token_id >= 0 &&
                (!candidate.requires_end_boundary ||
                 hasLexicalEndBoundary(normalized_text, end))) {
                best = ExactPieceSpan{pos, end, candidate.token_id};
                found = true;
            }
        }

        if (found) {
            if (!matches.empty() && overlaps(matches.back(), best)) {
                throw std::runtime_error(
                    "ExactPieceMatcher produced overlapping matches");
            }
            matches.push_back(best);
            pos = best.end;
        } else {
            ++pos;
        }
    }
    return matches;
}

} // namespace GRIM::Tokenizer
