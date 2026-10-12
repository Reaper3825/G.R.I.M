// Corpus preparation, vocabulary-shape matching cache and posterior aggregation.
#include "UnigramExpectationStep.hpp"
#include "UnigramForwardBackward.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace GRIM::Tokenizer {
namespace {
void requireHostControl(const UnigramTrainingSession& session) {
    if (session.address_space != UnigramTrainingAddressSpace::Host ||
        !session.storage || !session.hp || !session.pieces || !session.piece_to_id) {
        throw std::runtime_error("UnigramTrainingSession: missing host owners or unsupported address space");
    }
}
void bindViews(UnigramTrainingSession& session) {
    auto& storage = *session.storage;
    session.corpus_bytes = storage.corpus_bytes.data();
    session.corpus_byte_count = storage.corpus_bytes.size();
    session.segments = storage.segments.data();
    session.segment_count = storage.segments.size();
    session.matches = storage.matches.data();
    session.match_count = storage.matches.size();
    session.match_offsets = storage.match_offsets.data();
    session.piece_scores = storage.piece_scores.data();
    session.expected_counts = storage.expected_counts.data();
    session.piece_count = storage.piece_scores.size();
    session.alpha = storage.alpha.data();
    session.beta = storage.beta.data();
    session.workspace_capacity = storage.alpha.size();
}
}

void prepareUnigramTrainingSessionCorpus(UnigramTrainingSession& session,
    const std::vector<std::string>& texts,
    const std::vector<std::vector<AtomSpan>>& exclusions) {
    requireHostControl(session);
    if (session.segment_count != 0 || !session.storage->corpus_bytes.empty()) {
        throw std::runtime_error("UnigramTrainingSession: corpus is already prepared");
    }
    if (texts.size() != exclusions.size()) {
        throw std::runtime_error("UnigramTrainingSession: corpus/exclusion shape mismatch");
    }
    auto& storage = *session.storage;
    size_t max_length = 0;
    auto appendRange = [&](const std::string& text, size_t begin, size_t end) {
        while (begin < end) {
            const size_t newline = text.find('\n', begin);
            const size_t stop = newline == std::string::npos ? end : std::min(end, newline);
            if (stop > begin) {
                storage.segments.push_back({storage.corpus_bytes.size(), stop - begin});
                storage.corpus_bytes.insert(storage.corpus_bytes.end(),
                    text.data() + begin, text.data() + stop);
                max_length = std::max(max_length, stop - begin);
            }
            if (stop == end) break;
            begin = stop + 1;
        }
    };
    for (size_t i = 0; i < texts.size(); ++i) {
        size_t pos = 0;
        for (const auto& span : exclusions[i]) {
            if (span.start < pos || span.start > span.end || span.end > texts[i].size()) {
                throw std::runtime_error("UnigramTrainingSession: invalid normalized exclusion span");
            }
            appendRange(texts[i], pos, span.start);
            pos = span.end;
        }
        appendRange(texts[i], pos, texts[i].size());
    }
    if (storage.segments.empty() || max_length == std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("UnigramTrainingSession: no eligible segments or workspace overflow");
    }
    storage.alpha.resize(max_length + 1);
    storage.beta.resize(max_length + 1);
    bindViews(session);
}

void refreshUnigramTrainingSessionVocabulary(UnigramTrainingSession& session) {
    requireHostControl(session);
    if (session.segment_count == 0) {
        throw std::runtime_error("UnigramTrainingSession: prepare corpus before vocabulary matches");
    }
    session.matches_generation = 0;
    session.expectation_generation = 0;
    ++session.vocabulary_generation;
    auto& storage = *session.storage;
    storage.matches.clear();
    storage.match_offsets.clear();
    storage.match_offsets.reserve(storage.corpus_bytes.size() + 1);
    UnigramForwardBackwardLattice lattice(*session.pieces, session.enable_byte_fallback,
        "UnigramTrainingSession vocabulary matching");
    for (const auto& segment : storage.segments) {
        lattice.appendSegmentMatches(std::string_view(
            storage.corpus_bytes.data() + segment.byte_offset, segment.byte_length),
            segment.byte_offset, storage);
    }
    storage.match_offsets.push_back(storage.matches.size());
    storage.piece_scores.resize(session.pieces->size());
    storage.expected_counts.assign(session.pieces->size(), 0.0);
    for (size_t i = 0; i < session.pieces->size(); ++i) {
        storage.piece_scores[i] = (*session.pieces)[i].score;
    }
    bindViews(session);
    session.matches_generation = session.vocabulary_generation;
}

void requireUnigramTrainingSessionReady(const UnigramTrainingSession& session,
    const char* caller) {
    requireHostControl(session);
    const auto& s = *session.storage;
    if (!caller || !caller[0] || session.vocabulary_generation == 0 ||
        session.matches_generation != session.vocabulary_generation ||
        session.piece_count != session.pieces->size() ||
        session.corpus_byte_count != s.corpus_bytes.size() ||
        session.segment_count != s.segments.size() || session.segment_count == 0 ||
        session.match_count != s.matches.size() ||
        session.piece_count != s.piece_scores.size() ||
        session.piece_count != s.expected_counts.size() ||
        s.match_offsets.size() != session.corpus_byte_count + 1 ||
        session.workspace_capacity != s.alpha.size() || s.alpha.size() != s.beta.size() ||
        session.corpus_bytes != s.corpus_bytes.data() ||
        session.segments != s.segments.data() || session.matches != s.matches.data() ||
        session.match_offsets != s.match_offsets.data() ||
        session.piece_scores != s.piece_scores.data() ||
        session.expected_counts != s.expected_counts.data() ||
        session.alpha != s.alpha.data() || session.beta != s.beta.data()) {
        throw std::runtime_error(std::string(caller ? caller : "UnigramTrainingSession") +
            ": stale or incomplete training session addresses");
    }
}

void runUnigramExpectationStep(UnigramTrainingSession& session) {
    requireUnigramTrainingSessionReady(session, "Unigram expectation");
    session.expectation_generation = 0;
    session.log_likelihood = 0.0;
    session.expected_learned_piece_tokens = 0.0;
    session.expected_fixed_penalty_byte_fallback_tokens = 0.0;
    std::fill_n(session.expected_counts, session.piece_count, 0.0);
    // Refresh scores only; the matching structure is unchanged by the M-step.
    for (size_t i = 0; i < session.piece_count; ++i) {
        const float score = (*session.pieces)[i].score;
        if (!(*session.pieces)[i].is_user_defined && !std::isfinite(score)) {
            throw std::runtime_error("Unigram expectation: non-finite learned piece score");
        }
        session.piece_scores[i] = score;
    }
    for (size_t i = 0; i < session.segment_count; ++i) {
        accumulateUnigramTrainingSegment(session, session.segments[i], "Unigram expectation");
    }
    session.expectation_generation = session.vocabulary_generation;
}
} // namespace GRIM::Tokenizer
