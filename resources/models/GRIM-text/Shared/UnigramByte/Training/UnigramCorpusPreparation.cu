// Corpus normalization and atom/manual exclusion geometry.

#include "UnigramCorpusPreparation.hpp"
#include "../TextUtils.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace GRIM {
namespace Tokenizer {

AtomSpanValidationTotals validateOriginalAtomSpansBeforeNormalization(
    const std::vector<std::string>& texts,
    const std::vector<std::vector<AtomSpan>>& atom_spans) {
    if (atom_spans.size() != texts.size()) {
        throw std::runtime_error("UnigramLM::trainFromCorpus: atom_spans.size()=" +
                                 std::to_string(atom_spans.size()) +
                                 " != texts.size()=" + std::to_string(texts.size()));
    }

    AtomSpanValidationTotals totals;
    for (size_t text_idx = 0; text_idx < texts.size(); ++text_idx) {
        const auto& text = texts[text_idx];
        const auto& spans = atom_spans[text_idx];
        size_t previous_end = 0;
        totals.span_count += spans.size();

        for (size_t span_idx = 0; span_idx < spans.size(); ++span_idx) {
            const AtomSpan& span = spans[span_idx];
            if (span.start > span.end) {
                throw std::runtime_error("UnigramLM::trainFromCorpus: atom span start > end before normalization at text_idx=" +
                                         std::to_string(text_idx) + ", span_idx=" + std::to_string(span_idx) +
                                         ", start=" + std::to_string(span.start) +
                                         ", end=" + std::to_string(span.end));
            }
            if (span.end > text.size()) {
                throw std::runtime_error("UnigramLM::trainFromCorpus: atom span end exceeds original text size before normalization at text_idx=" +
                                         std::to_string(text_idx) + ", span_idx=" + std::to_string(span_idx) +
                                         ", end=" + std::to_string(span.end) +
                                         ", text.size()=" + std::to_string(text.size()));
            }
            if (span.start < previous_end) {
                throw std::runtime_error("UnigramLM::trainFromCorpus: atom spans overlap or are unsorted before normalization at text_idx=" +
                                         std::to_string(text_idx) + ", span_idx=" + std::to_string(span_idx) +
                                         ", start=" + std::to_string(span.start) +
                                         ", previous_end=" + std::to_string(previous_end));
            }
            totals.byte_count += span.end - span.start;
            previous_end = span.end;
        }
    }
    return totals;
}

static size_t maxTrainingSegmentLengthForTrainingUnits(
    const std::vector<std::string>& training_units,
    const std::vector<std::vector<AtomSpan>>& atom_spans) {
    if (atom_spans.size() != training_units.size()) {
        throw std::runtime_error("maxTrainingSegmentLengthForTrainingUnits: atom_spans.size()=" +
                                 std::to_string(atom_spans.size()) +
                                 " != training_units.size()=" + std::to_string(training_units.size()));
    }

    size_t max_segment_length = 0;
    auto observeViterbiRange = [&](const std::string& text, size_t start, size_t end) {
        size_t pos = start;
        while (pos < end) {
            const size_t newline = text.find('\n', pos);
            const size_t segment_end = newline == std::string::npos || newline >= end
                ? end
                : newline;
            max_segment_length = std::max(max_segment_length, segment_end - pos);
            if (segment_end == end) break;
            pos = segment_end + 1;
        }
    };
    for (size_t text_idx = 0; text_idx < training_units.size(); ++text_idx) {
        const auto& text = training_units[text_idx];
        const auto& spans = atom_spans[text_idx];
        if (text.empty()) {
            continue;
        }

        if (spans.empty()) {
            observeViterbiRange(text, 0, text.size());
            continue;
        }

        size_t pos = 0;
        for (size_t span_idx = 0; span_idx < spans.size(); ++span_idx) {
            const AtomSpan& span = spans[span_idx];
            if (span.start > span.end) {
                throw std::runtime_error("maxTrainingSegmentLengthForTrainingUnits: span.start > span.end at text_idx=" +
                                         std::to_string(text_idx) + ", span_idx=" + std::to_string(span_idx) +
                                         ", start=" + std::to_string(span.start) +
                                         ", end=" + std::to_string(span.end));
            }
            if (span.end > text.size()) {
                throw std::runtime_error("maxTrainingSegmentLengthForTrainingUnits: span.end exceeds normalized text size at text_idx=" +
                                         std::to_string(text_idx) + ", span_idx=" + std::to_string(span_idx) +
                                         ", end=" + std::to_string(span.end) +
                                         ", text.size()=" + std::to_string(text.size()));
            }
            if (span.start < pos) {
                throw std::runtime_error("maxTrainingSegmentLengthForTrainingUnits: atom spans overlap or are unsorted at text_idx=" +
                                         std::to_string(text_idx) + ", span_idx=" + std::to_string(span_idx) +
                                         ", start=" + std::to_string(span.start) +
                                         ", previous_end=" + std::to_string(pos));
            }
            if (span.start > pos) {
                observeViterbiRange(text, pos, span.start);
            }
            pos = span.end;
        }
        if (pos < text.size()) {
            observeViterbiRange(text, pos, text.size());
        }
    }

    if (max_segment_length == 0) {
        throw std::runtime_error("maxTrainingSegmentLengthForTrainingUnits: corpus has no non-empty normalized training segments");
    }
    return max_segment_length;
}

UnigramPreparedCorpus prepareUnigramTrainingCorpus(
    const std::vector<std::string>& texts,
    const std::vector<std::vector<AtomSpan>>& effective_atom_spans,
    const std::vector<UnigramPiece>& manual_pieces) {
    // SentencePiece-style whitespace normalization
    std::vector<std::string> norm_texts;
    std::vector<std::vector<AtomSpan>> norm_atom_spans;
    norm_texts.reserve(texts.size());
    norm_atom_spans.reserve(texts.size());
    for (size_t i = 0; i < texts.size(); ++i) {
        auto spans_copy = effective_atom_spans[i];
        norm_texts.push_back(normalizeWithSpans(texts[i], spans_copy));
        norm_atom_spans.push_back(std::move(spans_copy));
    }
    std::cout << "[UnigramLM] Applied whitespace normalization (space/tab -> ▁, LF/CR/CRLF -> fixed newline boundary)"
              << std::endl;

    // Manual pieces are exact pre-Viterbi tokens. Detect their occurrences on
    // normalized text and add them to the same training-exclusion geometry used
    // by atoms and fixed numeric tokens. This keeps mining and EM aligned with
    // runtime tokenization.
    std::vector<ExactPieceDefinition> manual_definitions;
    manual_definitions.reserve(manual_pieces.size());
    for (size_t i = 0; i < manual_pieces.size(); ++i) {
        manual_definitions.push_back(ExactPieceDefinition{
            manual_pieces[i].text,
            tokenIdForIndex(static_cast<int>(i))});
    }
    ExactPieceMatcher manual_matcher;
    manual_matcher.rebuild(manual_definitions);

    std::vector<std::vector<AtomSpan>> norm_training_exclusion_spans =
        norm_atom_spans;
    size_t manual_exact_span_count = 0;
    for (size_t text_idx = 0; text_idx < norm_texts.size(); ++text_idx) {
        std::vector<ExactPieceSpan> blocked;
        blocked.reserve(norm_atom_spans[text_idx].size());
        for (const AtomSpan& span : norm_atom_spans[text_idx]) {
            blocked.push_back(ExactPieceSpan{span.start, span.end, -1});
        }
        const auto matches = manual_matcher.findMatches(
            norm_texts[text_idx], blocked);
        auto& exclusions = norm_training_exclusion_spans[text_idx];
        exclusions.reserve(exclusions.size() + matches.size());
        for (const ExactPieceSpan& match : matches) {
            exclusions.push_back(AtomSpan{match.start, match.end});
        }
        std::sort(exclusions.begin(), exclusions.end(),
                  [](const AtomSpan& lhs, const AtomSpan& rhs) {
                      return lhs.start < rhs.start;
                  });
        manual_exact_span_count += matches.size();
    }
    if (manual_exact_span_count > 0) {
        std::cout << "[UnigramLM] Exact manual vocabulary: excluded "
                  << manual_exact_span_count
                  << " matched spans from learned-piece mining and EM"
                  << std::endl;
    }

    const std::vector<std::string>& training_units = norm_texts;
    const size_t training_segment_max_length =
        maxTrainingSegmentLengthForTrainingUnits(
            training_units, norm_training_exclusion_spans);
    std::cout << "[UnigramLM] Longest normalized training segment: "
              << training_segment_max_length << " bytes" << std::endl;
    return UnigramPreparedCorpus{std::move(norm_texts),
        std::move(norm_training_exclusion_spans), training_segment_max_length};
}

} // namespace Tokenizer
} // namespace GRIM
