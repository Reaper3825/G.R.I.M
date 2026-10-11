// Corpus traversal and posterior aggregation for the Unigram EM E-step.

#include "UnigramExpectationStep.hpp"
#include "UnigramForwardBackward.hpp"

#include <utility>

namespace GRIM {
namespace Tokenizer {

UnigramExpectationStepResult runUnigramExpectationStep(
    const UnigramExpectationStepRequest& request) {
    UnigramForwardBackwardLattice lattice(
        request.pieces, request.enable_byte_fallback,
        "UnigramLM::trainFromCorpus forward-backward E-step");
    UnigramForwardBackwardStats stats(request.pieces.size());
    std::unordered_map<int, double> token_counts;

    for (size_t text_idx = 0; text_idx < request.normalized_texts.size(); ++text_idx) {
        const auto& text = request.normalized_texts[text_idx];
        const auto& spans = request.normalized_exclusion_spans[text_idx];
        if (text.empty()) continue;

        auto processSegment = [&](const std::string& segment) {
            size_t pos = 0;
            while (pos < segment.size()) {
                const size_t newline = segment.find('\n', pos);
                const size_t end = newline == std::string::npos ? segment.size() : newline;
                if (end > pos) {
                    lattice.accumulateSegment(
                        segment.substr(pos, end - pos), stats,
                        "UnigramLM::trainFromCorpus forward-backward E-step");
                }
                if (newline == std::string::npos) break;
                pos = newline + 1;
            }
        };

        if (spans.empty()) {
            processSegment(text);
        } else {
            size_t pos = 0;
            for (const auto& span : spans) {
                if (span.start > pos) {
                    processSegment(text.substr(pos, span.start - pos));
                }
                pos = span.end;
            }
            if (pos < text.size()) {
                processSegment(text.substr(pos));
            }
        }
    }

    token_counts.reserve(stats.piece_expected_counts.size());
    for (size_t i = 0; i < stats.piece_expected_counts.size(); ++i) {
        const double expected_count = stats.piece_expected_counts[i];
        if (expected_count == 0.0) continue;
        token_counts[tokenIdForIndex(static_cast<int>(i))] = expected_count;
    }
    return UnigramExpectationStepResult{std::move(token_counts),
                                       stats.expected_learned_piece_tokens,
                                       stats.expected_fixed_penalty_byte_fallback_tokens,
                                       stats.log_likelihood};
}

} // namespace Tokenizer
} // namespace GRIM
