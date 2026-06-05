//======================================================//
//  UnigramTrainer.cu
//  Training Pipeline for UnigramLM
//
//  Split compilation unit: trainFromCorpus() is declared
//  as a method on Tokenizer::UnigramLM in Unigram.hpp.
//  This file provides the implementation (full private
//  member access, no friend needed).
//
//  Extracted from Unigram.cu during file-separation refactor.
//  Contains: EM training loop, noise filters, subword mining,
//  candidate ranking, dead token pruning + backfill.
//
//  Author: GRIM Team
//  Date: December 2025
//======================================================//

#include "Unigram.hpp"
#include "AtomTable.hpp"
#include "Detectors/DetectorRegistry.hpp"
#include "TextUtils.hpp"
#include "Training/UnigramForwardBackward.hpp"
#include "VocabWriteOp.hpp"
#include "HyperParameters/HyperparameterGroupings.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <queue>
#include <sstream>
#include <thread>
#include <unordered_set>

namespace GRIM {
namespace Tokenizer {

using UnigramSubwordCount = uint64_t;
using UnigramSubwordCountMap = std::unordered_map<std::string, UnigramSubwordCount>;

//======================================================//
//  Character-Level Validator
//  Rejects garbage characters BEFORE they enter the vocab.
//  Applied to single-character seeds in Step 2 of training.
//======================================================//

static bool isValidVocabCharacter(const std::string& ch) {
    if (ch.empty()) return false;

    uint32_t codepoint = 0;
    size_t codepoint_len = 0;
    if (!utf8DecodeAt(ch, 0, &codepoint, &codepoint_len) || codepoint_len != ch.size()) {
        return false;
    }

    if (codepoint >= 0x20 && codepoint <= 0x7E) return true;
    if (codepoint == 0x09 || codepoint == 0x0A || codepoint == 0x0D) return true;
    if (codepoint < 0x80) return false;

    if (codepoint >= 0x0080 && codepoint <= 0x009F) return false;
    if (codepoint == 0x00A0) return false;
    if (codepoint == 0x00AD) return false;
    if (codepoint == 0x034F) return false;
    if (codepoint >= 0x200B && codepoint <= 0x200F) return false;
    if (codepoint >= 0x202A && codepoint <= 0x202E) return false;
    if (codepoint == 0x2060) return false;
    if (codepoint >= 0x2066 && codepoint <= 0x2069) return false;
    if (codepoint == 0xFEFF) return false;
    if (codepoint >= 0xFFF0 && codepoint <= 0xFFFF) return false;
    if (codepoint >= 0xE0000 && codepoint <= 0xE007F) return false;
    if (codepoint >= 0xE000 && codepoint <= 0xF8FF) return false;
    if (codepoint >= 0xF0000 && codepoint <= 0x10FFFF) return false;
    if (codepoint >= 0xD800 && codepoint <= 0xDFFF) return false;

    return true;
}

//======================================================//
//  Structural Dedup Key
//  Canonical form for vocab candidate selection — strips
//  leading/trailing UTF-8 whitespace/boundary format chars.
//  SentencePiece ▁ is semantic word-initial state and must never be stripped.
//======================================================//

static constexpr uint32_t kSpieceUnderlineCodepoint = 0x2581u;

static bool isStructuralDedupTrimCodepoint(uint32_t cp) {
    if (cp == kSpieceUnderlineCodepoint) return false;
    return isStructuralEdgeWhitespace(cp);
}

static std::string structuralDedupKeyForCandidate(const std::string& s) {
    size_t start = 0;
    size_t end = s.size();
    while (start < end) {
        uint32_t cp = 0;
        size_t len = 0;
        if (utf8DecodeAt(s, start, &cp, &len) && isStructuralDedupTrimCodepoint(cp)) {
            start += len;
            continue;
        }
        if (isWhitespaceASCII(static_cast<unsigned char>(s[start]))) {
            ++start;
            continue;
        }
        break;
    }
    while (end > start) {
        if (isWhitespaceASCII(static_cast<unsigned char>(s[end - 1]))) {
            --end;
            continue;
        }
        size_t i = end - 1;
        while (i > start && (static_cast<unsigned char>(s[i]) & 0xC0) == 0x80)
            --i;
        const size_t ch_start = i;
        uint32_t cp = 0;
        size_t len = 0;
        if (!utf8DecodeAt(s, ch_start, &cp, &len) || ch_start + len != end)
            break;
        if (!isStructuralDedupTrimCodepoint(cp)) break;
        end = ch_start;
    }
    if (start >= end) return "";
    return s.substr(start, end - start);
}

//======================================================//
//  Noise Filters
//  Reject repetitive garbage patterns from subword mining.
//======================================================//

static bool hasExcessiveRunLength(const std::string& s) {
    if (s.empty()) return false;

    char prev = s[0];
    int run = 1;
    for (size_t i = 1; i < s.size(); ++i) {
        char c = s[i];
        if (c == prev) {
            run++;
        } else {
            prev = c;
            run = 1;
        }

        unsigned char uc = static_cast<unsigned char>(c);
        const bool is_alpha = (uc >= 'A' && uc <= 'Z') || (uc >= 'a' && uc <= 'z');
        const bool is_digit = (uc >= '0' && uc <= '9');

        if (is_alpha && run > 3) return true;
        if (is_digit && run > 6) return true;
        if (uc < 128 && isPunct(c) && run > 4) return true;
        if (isWhitespaceASCII(uc) && run > 1) return true;
    }

    return false;
}

static bool isRepeatedPatternNoise(const std::string& s) {
    if (s.size() < 6) return false;

    for (unsigned char c : s) {
        if (isWhitespaceASCII(c)) return false;
    }

    const size_t max_pattern_len = std::min<size_t>(4, s.size() / 3);
    for (size_t pattern_len = 1; pattern_len <= max_pattern_len; ++pattern_len) {
        if (s.size() % pattern_len != 0) continue;
        const size_t repeats = s.size() / pattern_len;
        if (repeats < 3) continue;

        bool repeated = true;
        for (size_t i = pattern_len; i < s.size(); ++i) {
            if (s[i] != s[i % pattern_len]) {
                repeated = false;
                break;
            }
        }
        if (repeated) return true;
    }

    return false;
}

static bool isDoubledTokenNoise(const std::string& s) {
    if (s.size() < 8 || (s.size() % 2) != 0) return false;

    for (unsigned char c : s) {
        if (isWhitespaceASCII(c)) return false;
    }

    const size_t half = s.size() / 2;
    if (half < 4) return false;

    for (size_t i = 0; i < half; ++i) {
        if (s[i] != s[i + half]) return false;
    }

    return true;
}

static bool isWordLevelStutter(const std::string& s) {
    size_t i = 0;
    while (i < s.size() && isWhitespaceASCII(static_cast<unsigned char>(s[i]))) ++i;
    if (i >= s.size()) return false;

    size_t first_start = i;
    while (i < s.size() && !isWhitespaceASCII(static_cast<unsigned char>(s[i]))) ++i;
    size_t first_len = i - first_start;
    if (first_len == 0) return false;

    int word_count = 1;
    while (i < s.size()) {
        while (i < s.size() && isWhitespaceASCII(static_cast<unsigned char>(s[i]))) ++i;
        if (i >= s.size()) break;

        size_t word_start = i;
        while (i < s.size() && !isWhitespaceASCII(static_cast<unsigned char>(s[i]))) ++i;
        size_t word_len = i - word_start;

        if (word_len != first_len) return false;
        for (size_t k = 0; k < word_len; ++k) {
            if (s[word_start + k] != s[first_start + k]) return false;
        }

        word_count++;
    }

    return word_count >= 3;
}

static bool isRepetitionNoise(const std::string& s) {
    return hasExcessiveRunLength(s) || isRepeatedPatternNoise(s) ||
           isDoubledTokenNoise(s) || isWordLevelStutter(s);
}

//======================================================//
//  Subword Validity Gate
//======================================================//

static bool isValidSubword(const std::string& s) {
    if (s.empty()) return false;

    if (s.size() == 1 || utf8SequenceLength(static_cast<unsigned char>(s[0])) == s.size()) {
        return isValidVocabCharacter(s);
    }

    if (isRepetitionNoise(s)) return false;

    for (unsigned char c : s) {
        if ((c < 0x20 && c != 0x09 && c != 0x0A && c != 0x0D) || c == 0x7F)
            return false;
    }

    bool all_space = true;
    for (unsigned char c : s) {
        if (!isWhitespaceASCII(c)) { all_space = false; break; }
    }
    if (all_space) return false;

    return true;
}

//======================================================//
//  Mining Worker Configuration
//======================================================//

static unsigned int resolveSubwordMiningWorkerCount(
    bool enable_parallel_subword_mining,
    int configured_workers,
    size_t sentence_count) {
    if (!enable_parallel_subword_mining) return 1;
    if (sentence_count < 1024) return 1;

    unsigned int workers = std::thread::hardware_concurrency();
    if (workers == 0) workers = 4;
    workers = workers > 2 ? workers - 2 : 1;

    if (configured_workers > 0) {
        workers = static_cast<unsigned int>(configured_workers);
    }

    if (const char* env_workers = std::getenv("GRIM_SUBWORD_MINING_WORKERS")) {
        try {
            const int parsed = std::stoi(env_workers);
            if (parsed > 0) {
                workers = static_cast<unsigned int>(parsed);
            }
        } catch (...) {
            // Ignore malformed env override.
        }
    }

    workers = std::min<unsigned int>(workers, static_cast<unsigned int>(sentence_count));
    return std::max(1u, workers);
}

static size_t resolveSubwordMiningChunkSize(unsigned int workers, size_t sentence_count) {
    if (workers <= 1 || sentence_count == 0) return sentence_count;

    size_t chunk = (sentence_count + (workers * 8) - 1) / (workers * 8);
    chunk = std::max<size_t>(64, chunk);
    chunk = std::min<size_t>(4096, chunk);
    return chunk;
}

static void validateTrainFromCorpusParameters(const std::vector<std::string>& texts,
                                              int target_vocab_size,
                                              float character_coverage,
                                              int min_subword_freq,
                                              int subword_mining_workers) {
    if (texts.empty()) {
        throw std::runtime_error("UnigramLM::trainFromCorpus: texts is empty - caller MUST provide at least one training text");
    }
    if (target_vocab_size <= 0) {
        throw std::runtime_error("UnigramLM::trainFromCorpus: target_vocab_size must be > 0, got " +
                                 std::to_string(target_vocab_size));
    }
    if (!std::isfinite(static_cast<double>(character_coverage)) ||
        character_coverage <= 0.0f || character_coverage > 1.0f) {
        throw std::runtime_error("UnigramLM::trainFromCorpus: character_coverage must be finite and in (0, 1], got " +
                                 std::to_string(character_coverage));
    }
    if (min_subword_freq <= 0) {
        throw std::runtime_error("UnigramLM::trainFromCorpus: min_subword_freq must be > 0, got " +
                                 std::to_string(min_subword_freq));
    }
    if (subword_mining_workers < 0) {
        throw std::runtime_error("UnigramLM::trainFromCorpus: subword_mining_workers must be >= 0, got " +
                                 std::to_string(subword_mining_workers));
    }
}

void requireUnigramFinalCleanupLeavesLearnedPiece(size_t piece_count,
                                                  size_t dead_count,
                                                  const char* caller) {
    if (caller == nullptr || caller[0] == '\0') {
        throw std::runtime_error("requireUnigramFinalCleanupLeavesLearnedPiece: caller label is empty at " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    if (piece_count == 0) {
        throw std::runtime_error(std::string(caller) +
                                 ": learned vocab is already empty before final cleanup; caller MUST preserve at least one learned piece");
    }
    if (dead_count > piece_count) {
        throw std::runtime_error(std::string(caller) +
                                 ": dead token count exceeds learned piece count, dead_count=" +
                                 std::to_string(dead_count) + ", piece_count=" + std::to_string(piece_count));
    }
    if (dead_count == piece_count) {
        throw std::runtime_error(std::string(caller) +
                                 ": final dead-token cleanup would delete every learned piece (piece_count=" +
                                 std::to_string(piece_count) +
                                 "); byte fallback dominated all posterior mass, so training MUST fail before Phase-D lattice construction");
    }
}

void requireUnigramAcceptedCandidateSetIsScorable(size_t accepted_count,
                                                  double total_accepted_count,
                                                  const char* caller) {
    if (caller == nullptr || caller[0] == '\0') {
        throw std::runtime_error("requireUnigramAcceptedCandidateSetIsScorable: caller label is empty at " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    if (accepted_count == 0) {
        throw std::runtime_error(std::string(caller) +
                                 ": candidate admission produced zero accepted learned pieces before initial score normalization; frequency/validity/dedup filters eliminated every candidate");
    }
    if (!std::isfinite(total_accepted_count) || total_accepted_count <= 0.0) {
        throw std::runtime_error(std::string(caller) +
                                 ": accepted candidate set is not scorable; accepted_count=" +
                                 std::to_string(accepted_count) +
                                 ", total_accepted_count=" +
                                 std::to_string(total_accepted_count) +
                                 ". Learned candidate counts MUST sum to a positive finite value before initial score normalization");
    }
}

void requireUnigramLearnedPosteriorMassNotByteFallbackDominated(
    double expected_learned_piece_tokens,
    double expected_fixed_penalty_byte_fallback_tokens,
    const char* phase_label,
    const char* caller) {
    constexpr double kMinimumLearnedToByteFallbackPosteriorRatio = 1.0;

    if (phase_label == nullptr || phase_label[0] == '\0') {
        throw std::runtime_error("requireUnigramLearnedPosteriorMassNotByteFallbackDominated: phase label is empty at " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    if (caller == nullptr || caller[0] == '\0') {
        throw std::runtime_error("requireUnigramLearnedPosteriorMassNotByteFallbackDominated: caller label is empty at " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    if (!std::isfinite(expected_learned_piece_tokens) || expected_learned_piece_tokens < 0.0) {
        throw std::runtime_error(std::string(caller) + ": " + phase_label +
                                 " produced invalid expected_learned_piece_tokens=" +
                                 std::to_string(expected_learned_piece_tokens));
    }
    if (!std::isfinite(expected_fixed_penalty_byte_fallback_tokens) ||
        expected_fixed_penalty_byte_fallback_tokens < 0.0) {
        throw std::runtime_error(std::string(caller) + ": " + phase_label +
                                 " produced invalid expected_fixed_penalty_byte_fallback_tokens=" +
                                 std::to_string(expected_fixed_penalty_byte_fallback_tokens));
    }
    if (expected_fixed_penalty_byte_fallback_tokens == 0.0) {
        return;
    }

    const double learned_to_fallback_ratio =
        expected_learned_piece_tokens / expected_fixed_penalty_byte_fallback_tokens;
    if (learned_to_fallback_ratio < kMinimumLearnedToByteFallbackPosteriorRatio) {
        throw std::runtime_error(std::string(caller) + ": " + phase_label +
                                 " byte fallback dominated posterior mass after EM convergence; "
                                 "expected_learned_piece_tokens=" +
                                 std::to_string(expected_learned_piece_tokens) +
                                 ", expected_fixed_penalty_byte_fallback_tokens=" +
                                 std::to_string(expected_fixed_penalty_byte_fallback_tokens) +
                                 ", learned_to_fallback_ratio=" +
                                 std::to_string(learned_to_fallback_ratio) +
                                 ", required_ratio>=1.0. Training MUST fail instead of treating fallback-dominated evidence as a learned-piece distribution");
    }
}

double scoreUnigramShrinkCandidateForPosteriorCompression(const std::string& piece_text,
                                                          double posterior_expected_count,
                                                          const char* caller) {
    if (caller == nullptr || caller[0] == '\0') {
        throw std::runtime_error("scoreUnigramShrinkCandidateForPosteriorCompression: caller label is empty at " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    if (piece_text.empty()) {
        throw std::runtime_error(std::string(caller) +
                                 ": shrink candidate piece text is empty - learned pieces MUST contain at least one byte");
    }
    if (!std::isfinite(posterior_expected_count) || posterior_expected_count < 0.0) {
        throw std::runtime_error(std::string(caller) +
                                 ": posterior_expected_count must be finite and non-negative, got " +
                                 std::to_string(posterior_expected_count));
    }

    const double byte_span = static_cast<double>(piece_text.size());
    const double compression_gain_per_use = std::max(0.0, byte_span - 1.0);
    const double expected_compression_gain = posterior_expected_count * compression_gain_per_use;

    return posterior_expected_count + expected_compression_gain;
}

UnigramSubwordCount addUnigramSubwordCountsForTraining(UnigramSubwordCount current,
                                                       UnigramSubwordCount delta,
                                                       const char* caller) {
    if (caller == nullptr || caller[0] == '\0') {
        throw std::runtime_error("addUnigramSubwordCountsForTraining: caller label is empty at " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    const UnigramSubwordCount max_count = std::numeric_limits<UnigramSubwordCount>::max();
    if (max_count - current < delta) {
        throw std::runtime_error(std::string(caller) +
                                 ": subword candidate count overflow, current=" +
                                 std::to_string(current) + ", delta=" + std::to_string(delta) +
                                 ", max=" + std::to_string(max_count));
    }
    return current + delta;
}

static void incrementUnigramSubwordCountForTraining(UnigramSubwordCountMap& subword_counts,
                                                    const std::string& subword,
                                                    const char* caller) {
    auto [it, inserted] = subword_counts.try_emplace(subword, static_cast<UnigramSubwordCount>(1));
    if (!inserted) {
        it->second = addUnigramSubwordCountsForTraining(
            it->second,
            static_cast<UnigramSubwordCount>(1),
            caller);
    }
}

struct AtomSpanValidationTotals {
    size_t span_count = 0;
    size_t byte_count = 0;
};

static AtomSpanValidationTotals validateOriginalAtomSpansBeforeNormalization(
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
    for (size_t text_idx = 0; text_idx < training_units.size(); ++text_idx) {
        const auto& text = training_units[text_idx];
        const auto& spans = atom_spans[text_idx];
        if (text.empty()) {
            continue;
        }

        if (spans.empty()) {
            max_segment_length = std::max(max_segment_length, text.size());
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
                max_segment_length = std::max(max_segment_length, span.start - pos);
            }
            pos = span.end;
        }
        if (pos < text.size()) {
            max_segment_length = std::max(max_segment_length, text.size() - pos);
        }
    }

    if (max_segment_length == 0) {
        throw std::runtime_error("maxTrainingSegmentLengthForTrainingUnits: corpus has no non-empty normalized training segments");
    }
    return max_segment_length;
}

//======================================================//
//  Deterministic Subword Mining Sample Plan
//  Corpus byte caps must not turn into prefix-only mining: EM trains on
//  full documents, so candidate mining has to sample across full documents.
//======================================================//

struct SubwordMiningSpan {
    size_t start = 0;
    size_t end = 0;
    size_t context_start = 0;
    size_t context_end = 0;
};

struct SubwordMiningPlan {
    std::vector<std::vector<SubwordMiningSpan>> spans_by_text;
    size_t sampled_bytes = 0;
    size_t context_bytes = 0;
    size_t sampled_spans = 0;
    double sampling_ratio = 1.0;
};

static bool isUtf8ContinuationByte(unsigned char c) {
    return (c & 0xC0) == 0x80;
}

static size_t snapBackwardToUtf8Boundary(const std::string& text, size_t pos) {
    if (pos > text.size()) {
        throw std::runtime_error("snapBackwardToUtf8Boundary: pos exceeds text.size(), pos=" +
                                 std::to_string(pos) + ", text.size()=" + std::to_string(text.size()));
    }
    if (pos == text.size()) return pos;
    while (pos > 0 && isUtf8ContinuationByte(static_cast<unsigned char>(text[pos]))) {
        --pos;
    }
    return pos;
}

static size_t snapForwardToUtf8Boundary(const std::string& text, size_t pos) {
    if (pos > text.size()) {
        throw std::runtime_error("snapForwardToUtf8Boundary: pos exceeds text.size(), pos=" +
                                 std::to_string(pos) + ", text.size()=" + std::to_string(text.size()));
    }
    while (pos < text.size() && isUtf8ContinuationByte(static_cast<unsigned char>(text[pos]))) {
        ++pos;
    }
    return pos;
}

static void appendMergedSubwordMiningSpan(std::vector<SubwordMiningSpan>& spans,
                                          size_t start,
                                          size_t end,
                                          size_t context_start,
                                          size_t context_end,
                                          const std::string& context) {
    if (start > end) {
        throw std::runtime_error(context + ": mining span start > end, start=" +
                                 std::to_string(start) + ", end=" + std::to_string(end));
    }
    if (context_start > context_end) {
        throw std::runtime_error(context + ": mining context start > end, context_start=" +
                                 std::to_string(context_start) + ", context_end=" + std::to_string(context_end));
    }
    if (context_start > start || end > context_end) {
        throw std::runtime_error(context + ": mining context does not contain intended sampled span, context_start=" +
                                 std::to_string(context_start) + ", start=" + std::to_string(start) +
                                 ", end=" + std::to_string(end) + ", context_end=" + std::to_string(context_end));
    }
    if (start == end) return;

    if (!spans.empty() && start <= spans.back().end) {
        if (end > spans.back().end) {
            spans.back().end = end;
        }
        spans.back().context_start = std::min(spans.back().context_start, context_start);
        spans.back().context_end = std::max(spans.back().context_end, context_end);
        return;
    }

    spans.push_back(SubwordMiningSpan{start, end, context_start, context_end});
}

static SubwordMiningPlan buildSubwordMiningPlan(
    const std::vector<std::string>& training_units,
    size_t total_sentence_bytes,
    size_t max_subword_mining_bytes,
    bool use_sampling) {
    if (total_sentence_bytes == 0) {
        throw std::runtime_error("buildSubwordMiningPlan: total_sentence_bytes is zero - caller MUST provide non-empty normalized training text");
    }
    if (max_subword_mining_bytes == 0) {
        throw std::runtime_error("buildSubwordMiningPlan: max_subword_mining_bytes is zero - caller MUST provide a positive byte cap");
    }

    SubwordMiningPlan plan;
    plan.spans_by_text.resize(training_units.size());

    if (use_sampling) {
        plan.sampling_ratio = static_cast<double>(max_subword_mining_bytes) /
                              static_cast<double>(total_sentence_bytes);
    }

    constexpr size_t kTargetSpanBytes = 8192;
    const size_t two_span_budget = static_cast<size_t>(MAX_PIECE_LENGTH) * 2;
    const size_t sampled_span_context_overlap = static_cast<size_t>(MAX_PIECE_LENGTH) > 0
        ? static_cast<size_t>(MAX_PIECE_LENGTH) - 1
        : 0;

    for (size_t text_idx = 0; text_idx < training_units.size(); ++text_idx) {
        const std::string& text = training_units[text_idx];
        if (text.empty()) continue;

        std::vector<SubwordMiningSpan>& spans = plan.spans_by_text[text_idx];
        if (!use_sampling) {
            spans.push_back(SubwordMiningSpan{0, text.size(), 0, text.size()});
        } else {
            size_t target_bytes = static_cast<size_t>(std::ceil(text.size() * plan.sampling_ratio));
            target_bytes = std::max<size_t>(1, target_bytes);
            target_bytes = std::min(target_bytes, text.size());

            if (target_bytes == text.size()) {
                spans.push_back(SubwordMiningSpan{0, text.size(), 0, text.size()});
            } else {
                size_t span_count = (target_bytes + kTargetSpanBytes - 1) / kTargetSpanBytes;
                span_count = std::max<size_t>(1, span_count);
                if (target_bytes >= two_span_budget) {
                    span_count = std::max<size_t>(2, span_count);
                }
                span_count = std::min(span_count, target_bytes);

                const size_t span_bytes = (target_bytes + span_count - 1) / span_count;
                if (span_bytes == 0) {
                    throw std::runtime_error("buildSubwordMiningPlan: computed zero-length span for text_idx=" +
                                             std::to_string(text_idx));
                }
                if (span_bytes > text.size()) {
                    throw std::runtime_error("buildSubwordMiningPlan: span_bytes exceeds text.size() for text_idx=" +
                                             std::to_string(text_idx) + ", span_bytes=" + std::to_string(span_bytes) +
                                             ", text.size()=" + std::to_string(text.size()));
                }

                const size_t max_start = text.size() - span_bytes;
                for (size_t span_idx = 0; span_idx < span_count; ++span_idx) {
                    size_t raw_start = 0;
                    if (span_count == 1) {
                        raw_start = max_start / 2;
                    } else if (span_idx + 1 == span_count) {
                        raw_start = max_start;
                    } else {
                        raw_start = (span_idx * max_start + ((span_count - 1) / 2)) / (span_count - 1);
                    }

                    const size_t start = snapBackwardToUtf8Boundary(text, raw_start);
                    size_t raw_end = start + span_bytes;
                    raw_end = std::min(raw_end, text.size());
                    const size_t end = snapForwardToUtf8Boundary(text, raw_end);

                    const size_t raw_context_start = start > sampled_span_context_overlap
                        ? start - sampled_span_context_overlap
                        : 0;
                    const size_t context_start = snapBackwardToUtf8Boundary(text, raw_context_start);
                    const size_t raw_context_end = std::min(
                        text.size(),
                        end + sampled_span_context_overlap);
                    const size_t context_end = snapForwardToUtf8Boundary(text, raw_context_end);
                    appendMergedSubwordMiningSpan(
                        spans,
                        start,
                        end,
                        context_start,
                        context_end,
                        "buildSubwordMiningPlan text_idx=" + std::to_string(text_idx));
                }
            }
        }

        for (const SubwordMiningSpan& span : spans) {
            if (span.end > text.size()) {
                throw std::runtime_error("buildSubwordMiningPlan: span end exceeds text size at text_idx=" +
                                         std::to_string(text_idx) + ", end=" + std::to_string(span.end) +
                                         ", text.size()=" + std::to_string(text.size()));
            }
            if (span.context_end > text.size()) {
                throw std::runtime_error("buildSubwordMiningPlan: span context end exceeds text size at text_idx=" +
                                         std::to_string(text_idx) + ", context_end=" + std::to_string(span.context_end) +
                                         ", text.size()=" + std::to_string(text.size()));
            }
            if (span.context_start > span.start || span.end > span.context_end) {
                throw std::runtime_error("buildSubwordMiningPlan: span context does not contain intended sampled span at text_idx=" +
                                         std::to_string(text_idx));
            }
            plan.sampled_bytes += span.end - span.start;
            plan.context_bytes += span.context_end - span.context_start;
            ++plan.sampled_spans;
        }
    }

    return plan;
}

//======================================================//
//  Subword Mining from Sentences
//======================================================//

static void validateSubwordMiningRange(const std::string& text,
                                       size_t byte_start,
                                       size_t byte_end,
                                       const char* context) {
    if (byte_start > byte_end) {
        throw std::runtime_error(std::string(context) + ": byte_start > byte_end, byte_start=" +
                                 std::to_string(byte_start) + ", byte_end=" + std::to_string(byte_end));
    }
    if (byte_end > text.size()) {
        throw std::runtime_error(std::string(context) + ": byte_end exceeds text.size(), byte_end=" +
                                 std::to_string(byte_end) + ", text.size()=" + std::to_string(text.size()));
    }
    if (byte_start < text.size() && isUtf8ContinuationByte(static_cast<unsigned char>(text[byte_start]))) {
        throw std::runtime_error(std::string(context) + ": byte_start is inside a UTF-8 sequence, byte_start=" +
                                 std::to_string(byte_start));
    }
    if (byte_end < text.size() && isUtf8ContinuationByte(static_cast<unsigned char>(text[byte_end]))) {
        throw std::runtime_error(std::string(context) + ": byte_end is inside a UTF-8 sequence, byte_end=" +
                                 std::to_string(byte_end));
    }
}

static void validateSubwordMiningCountingRange(const std::string& text,
                                               size_t context_start,
                                               size_t context_end,
                                               size_t count_start,
                                               size_t count_end,
                                               const char* context) {
    validateSubwordMiningRange(text, context_start, context_end, context);
    if (count_start > count_end) {
        throw std::runtime_error(std::string(context) + ": count_start > count_end, count_start=" +
                                 std::to_string(count_start) + ", count_end=" + std::to_string(count_end));
    }
    if (context_start > count_start || count_end > context_end) {
        throw std::runtime_error(std::string(context) + ": intended sampled start range is outside mining context, context_start=" +
                                 std::to_string(context_start) + ", count_start=" + std::to_string(count_start) +
                                 ", count_end=" + std::to_string(count_end) + ", context_end=" + std::to_string(context_end));
    }
    if (count_start < text.size() && isUtf8ContinuationByte(static_cast<unsigned char>(text[count_start]))) {
        throw std::runtime_error(std::string(context) + ": count_start is inside a UTF-8 sequence, count_start=" +
                                 std::to_string(count_start));
    }
    if (count_end < text.size() && isUtf8ContinuationByte(static_cast<unsigned char>(text[count_end]))) {
        throw std::runtime_error(std::string(context) + ": count_end is inside a UTF-8 sequence, count_end=" +
                                 std::to_string(count_end));
    }
}

static void mineSubwordsFromSentence(const std::string& text,
                                     size_t max_len,
                                     UnigramSubwordCountMap& subword_counts,
                                     size_t context_start,
                                     size_t context_end,
                                     size_t count_start,
                                     size_t count_end) {
    validateSubwordMiningCountingRange(text, context_start, context_end, count_start, count_end,
                                       "mineSubwordsFromSentence");
    if (context_start == context_end || count_start == count_end) return;

    std::vector<size_t> char_positions;
    char_positions.reserve((context_end - context_start) + 1);
    for (size_t i = context_start; i < context_end; ) {
        char_positions.push_back(i);
        const size_t seq_len = utf8SequenceLength(static_cast<unsigned char>(text[i]));
        if (i + seq_len > context_end) {
            throw std::runtime_error("mineSubwordsFromSentence: UTF-8 sequence crosses mining span end at byte=" +
                                     std::to_string(i) + ", context_end=" + std::to_string(context_end));
        }
        i += seq_len;
    }
    char_positions.push_back(context_end);

    const size_t num_chars = char_positions.size() - 1;
    for (size_t ci = 0; ci < num_chars; ++ci) {
        const size_t byte_start = char_positions[ci];
        if (byte_start < count_start || byte_start >= count_end) continue;
        for (size_t char_count = 1; char_count <= max_len && ci + char_count <= num_chars; ++char_count) {
            const size_t byte_end = char_positions[ci + char_count];
            if (byte_end - byte_start > max_len) break;

            std::string subword = text.substr(byte_start, byte_end - byte_start);
            if (!isValidSubword(subword)) continue;

            // ▁ marks word-initial position only. Reject pieces where ▁
            // appears anywhere after byte 0 — that means the piece crosses
            // a word boundary (e.g. "the▁", "▁the▁", "he▁").
            {
                size_t search_start = 0;
                if (subword.size() >= SPIECE_UNDERLINE_LEN &&
                    subword.compare(0, SPIECE_UNDERLINE_LEN, SPIECE_UNDERLINE) == 0) {
                    search_start = SPIECE_UNDERLINE_LEN; // skip leading ▁
                }
                if (search_start < subword.size() &&
                    subword.find(SPIECE_UNDERLINE, search_start, SPIECE_UNDERLINE_LEN) != std::string::npos) {
                    continue;
                }
            }

            incrementUnigramSubwordCountForTraining(
                subword_counts,
                subword,
                "mineSubwordsFromSentence subword count increment");
        }
    }
}

static void mineSubwordsFromSentence(const std::string& text,
                                     size_t max_len,
                                     UnigramSubwordCountMap& subword_counts,
                                     size_t byte_start,
                                     size_t byte_end) {
    mineSubwordsFromSentence(text, max_len, subword_counts, byte_start, byte_end, byte_start, byte_end);
}

// Atom-aware overload: skip subwords that START inside an atom span.
static void mineSubwordsFromSentence(const std::string& text,
                                     size_t max_len,
                                     const std::vector<AtomSpan>& atom_spans,
                                     UnigramSubwordCountMap& subword_counts,
                                     size_t context_start,
                                     size_t context_end,
                                     size_t count_start,
                                     size_t count_end) {
    validateSubwordMiningCountingRange(text, context_start, context_end, count_start, count_end,
                                       "mineSubwordsFromSentence atom-aware");
    if (context_start == context_end || count_start == count_end) return;
    if (atom_spans.empty()) {
        mineSubwordsFromSentence(text, max_len, subword_counts, context_start, context_end, count_start, count_end);
        return;
    }

    std::vector<size_t> char_positions;
    char_positions.reserve((context_end - context_start) + 1);
    for (size_t i = context_start; i < context_end; ) {
        char_positions.push_back(i);
        const size_t seq_len = utf8SequenceLength(static_cast<unsigned char>(text[i]));
        if (i + seq_len > context_end) {
            throw std::runtime_error("mineSubwordsFromSentence atom-aware: UTF-8 sequence crosses mining span end at byte=" +
                                     std::to_string(i) + ", context_end=" + std::to_string(context_end));
        }
        i += seq_len;
    }
    char_positions.push_back(context_end);

    std::vector<bool> char_in_atom(char_positions.size() - 1, false);
    size_t span_idx = 0;
    for (size_t ci = 0; ci < char_positions.size() - 1; ++ci) {
        const size_t byte_pos = char_positions[ci];
        while (span_idx < atom_spans.size() && atom_spans[span_idx].end <= byte_pos) {
            ++span_idx;
        }
        if (span_idx < atom_spans.size() &&
            byte_pos >= atom_spans[span_idx].start &&
            byte_pos < atom_spans[span_idx].end) {
            char_in_atom[ci] = true;
        }
    }

    const size_t num_chars = char_positions.size() - 1;
    for (size_t ci = 0; ci < num_chars; ++ci) {
        if (char_in_atom[ci]) continue;

        const size_t byte_start = char_positions[ci];
        if (byte_start < count_start || byte_start >= count_end) continue;
        for (size_t char_count = 1; char_count <= max_len && ci + char_count <= num_chars; ++char_count) {
            bool crosses_atom = false;
            for (size_t k = ci + 1; k < ci + char_count; ++k) {
                if (char_in_atom[k]) { crosses_atom = true; break; }
            }
            if (crosses_atom) break;

            const size_t byte_end = char_positions[ci + char_count];
            if (byte_end - byte_start > max_len) break;

            std::string subword = text.substr(byte_start, byte_end - byte_start);
            if (!isValidSubword(subword)) continue;

            // ▁ marks word-initial position only. Reject pieces where ▁
            // appears anywhere after byte 0 — that means the piece crosses
            // a word boundary (e.g. "the▁", "▁the▁", "he▁").
            {
                size_t search_start = 0;
                if (subword.size() >= SPIECE_UNDERLINE_LEN &&
                    subword.compare(0, SPIECE_UNDERLINE_LEN, SPIECE_UNDERLINE) == 0) {
                    search_start = SPIECE_UNDERLINE_LEN; // skip leading ▁
                }
                if (search_start < subword.size() &&
                    subword.find(SPIECE_UNDERLINE, search_start, SPIECE_UNDERLINE_LEN) != std::string::npos) {
                    continue;
                }
            }

            incrementUnigramSubwordCountForTraining(
                subword_counts,
                subword,
                "mineSubwordsFromSentence atom-aware subword count increment");
        }
    }
}

static void mineSubwordsFromSentence(const std::string& text,
                                     size_t max_len,
                                     const std::vector<AtomSpan>& atom_spans,
                                     UnigramSubwordCountMap& subword_counts,
                                     size_t byte_start,
                                     size_t byte_end) {
    mineSubwordsFromSentence(text, max_len, atom_spans, subword_counts, byte_start, byte_end, byte_start, byte_end);
}

namespace {

[[noreturn]] void throwUnparseableDetectedAtomForTraining(const char* detector_name,
                                                          AtomType atom_type,
                                                          size_t start,
                                                          size_t end,
                                                          std::string_view atom_text,
                                                          std::string_view parse_error) {
    throw std::runtime_error(
        std::string("UnigramLM::trainFromCorpus: detector-emitted atom span is not parseable; upstream detector/data pipeline bug: detector='") +
        (detector_name ? std::string(detector_name) : std::string("<unknown>")) +
        "', atom_type=" + atomTypeName(atom_type) +
        ", span=[" + std::to_string(start) + ", " + std::to_string(end) +
        "), text='" + std::string(atom_text) +
        "', parse_error='" + std::string(parse_error) + "'");
}

} // namespace

//======================================================//
//  trainFromCorpus — TokenizerHP Detector-Prepass Entrypoint
//======================================================//

bool UnigramLM::trainFromCorpus(const std::vector<std::string>& texts,
                                 const ::GRIM::HyperParameters::TokenizerHP& tokenizer_hp) {
    Detector::DetectorRegistry detector_registry;
    if (tokenizer_hp.enable_atom_reasoning) {
        detector_registry = Detector::makeDefaultRawTextDetectorRegistry();
        if (detector_registry.empty()) {
            throw std::runtime_error("UnigramLM::trainFromCorpus: detector registry initialized empty while atom reasoning is enabled");
        }
    }

    std::vector<std::vector<AtomSpan>> all_atom_spans;
    all_atom_spans.reserve(texts.size());

    size_t total_atoms = 0;
    const bool detect = tokenizer_hp.enable_atom_reasoning;
    for (const auto& text : texts) {
        std::vector<AtomSpan> spans;
        if (detect) {
            const Detector::RawTextDetectorOptions detector_options(
                tokenizer_hp.detect_numbers,
                true,
                true);
            const auto detections = detector_registry.scan(text, detector_options);
            spans.reserve(detections.size());
            for (const auto& detection : detections) {
                if (!detection.emitsAtom()) {
                    continue;
                }
                const std::string_view atom_text(text.data() + detection.start,
                                                 detection.end - detection.start);
                auto parsed = AtomTable::parseAtom(
                    detection.atom_type,
                    std::string(atom_text));
                if (!parsed.success) {
                    throwUnparseableDetectedAtomForTraining(detection.detector_name,
                                                            detection.atom_type,
                                                            detection.start,
                                                            detection.end,
                                                            atom_text,
                                                            parsed.error_message);
                }
                spans.push_back({detection.start, detection.end});
            }
            total_atoms += spans.size();
        }
        all_atom_spans.push_back(std::move(spans));
    }

    std::cout << "[UnigramLM] Detected " << total_atoms << " atoms across "
              << texts.size() << " texts (will skip during vocab training); "
              << "atom_reasoning=" << (detect ? "on" : "off") << std::endl;

    const bool trained = trainFromCorpus(texts, all_atom_spans,
                                         tokenizer_hp.target_vocab_size,
                                         tokenizer_hp.character_coverage,
                                         tokenizer_hp.min_subword_freq,
                                         tokenizer_hp.prune_during_mining,
                                         tokenizer_hp.enable_parallel_subword_mining,
                                         tokenizer_hp.subword_mining_workers,
                                         tokenizer_hp.subword_mining_max_bytes);
    if (trained) {
        requireRuntimeReadyForLastTraining("UnigramLM::trainFromCorpus(TokenizerHP)");
    }
    return trained;
}

//======================================================//
//  trainFromCorpus — Full Atom-Aware Implementation
//======================================================//

bool UnigramLM::trainFromCorpus(const std::vector<std::string>& texts,
                                 const std::vector<std::vector<AtomSpan>>& atom_spans,
                                 int target_vocab_size,
                                 float character_coverage,
                                 int min_subword_freq,
                                 bool prune_during_mining,
                                 bool enable_parallel_subword_mining,
                                 int subword_mining_workers,
                                 size_t subword_mining_max_bytes) {
    validateTrainFromCorpusParameters(texts, target_vocab_size, character_coverage,
                                      min_subword_freq, subword_mining_workers);

    // An empty atom_spans vector means "no atom spans for any text". Expand it
    // to a per-text empty list so the rest of the pipeline can index uniformly.
    std::vector<std::vector<AtomSpan>> expanded_empty_atom_spans;
    if (atom_spans.empty()) {
        expanded_empty_atom_spans.resize(texts.size());
    }
    const std::vector<std::vector<AtomSpan>>& effective_atom_spans =
        atom_spans.empty() ? expanded_empty_atom_spans : atom_spans;

    const AtomSpanValidationTotals original_atom_totals =
        validateOriginalAtomSpansBeforeNormalization(texts, effective_atom_spans);

    last_training_runtime_report_ = UnigramTrainingRuntimeReport{};

    std::cout << "[UnigramLM] Training vocabulary from " << texts.size() 
              << " texts (target_vocab_size=" << target_vocab_size << ")" << std::endl;
    std::cout << "[UnigramLM] min_subword_freq=" << min_subword_freq 
              << ", prune_during_mining=" << (prune_during_mining ? "true" : "false")
              << ", parallel_subword_mining=" << (enable_parallel_subword_mining ? "true" : "false")
              << ", subword_mining_workers=" << subword_mining_workers << std::endl;

    if (prune_during_mining) {
        std::cout << "[UnigramLM] prune_during_mining=true requested; exact global subword counting is enforced, so mining-time pruning is disabled"
                  << std::endl;
    }

    if (original_atom_totals.span_count > 0) {
        std::cout << "[UnigramLM] Atom-aware training: " << original_atom_totals.span_count
                  << " atom spans (" << (original_atom_totals.byte_count / 1024) << " KB) will be skipped" << std::endl;
    }
    
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
    std::cout << "[UnigramLM] Applied SentencePiece whitespace normalization (space -> ▁)" << std::endl;

    const std::vector<std::string>& training_units = norm_texts;
    const size_t training_segment_max_length =
        maxTrainingSegmentLengthForTrainingUnits(training_units, norm_atom_spans);
    std::cout << "[UnigramLM] Longest normalized training segment: "
              << training_segment_max_length << " bytes" << std::endl;
    const UnigramSubwordCount MIN_SUBWORD_FREQ = static_cast<UnigramSubwordCount>(min_subword_freq);
    
    size_t total_corpus_bytes = 0;
    for (const auto& text : texts) {
        total_corpus_bytes += text.size();
    }
    std::cout << "[UnigramLM] Total corpus size: " << (total_corpus_bytes / (1024*1024)) << " MB" << std::endl;
    
    size_t total_sentence_bytes = 0;
    for (const auto& sent : training_units) {
        total_sentence_bytes += sent.size();
    }
    
    size_t max_subword_mining_bytes = static_cast<size_t>(HyperParameters::UNIGRAM_MAX_SUBWORD_BYTES);
    if (subword_mining_max_bytes > 0) {
        max_subword_mining_bytes = subword_mining_max_bytes;
    }
    const bool use_sampling = total_sentence_bytes > max_subword_mining_bytes;

    std::cout << "[UnigramLM] Subword mining byte cap: "
              << (max_subword_mining_bytes / (1024 * 1024)) << " MB" << std::endl;

    // Byte-proportional deterministic strided sampling: when the corpus exceeds
    // the mining byte budget, every non-empty document still contributes bytes,
    // but those bytes are distributed as spans across the full document rather
    // than taking only prefixes. EM still trains on full documents, so mining
    // must not starve late-document candidate patterns.
    const SubwordMiningPlan mining_plan = buildSubwordMiningPlan(
        training_units, total_sentence_bytes, max_subword_mining_bytes, use_sampling);
    if (use_sampling) {
        std::cout << "[UnigramLM] Byte-proportional strided mining: ratio="
                  << std::fixed << std::setprecision(3) << mining_plan.sampling_ratio
                  << ", spans=" << mining_plan.sampled_spans
                  << ", sampled_starts=" << (mining_plan.sampled_bytes / (1024*1024)) << " MB"
                  << ", context=" << (mining_plan.context_bytes / (1024*1024)) << " MB from all "
                  << training_units.size() << " documents" << std::defaultfloat << std::endl;
    }
    
    // Step 1: Count character frequencies (use ALL normalized texts)
    std::unordered_map<std::string, int> char_counts;
    size_t total_chars = 0;
    size_t atom_chars_skipped = 0;
    
    for (size_t text_idx = 0; text_idx < norm_texts.size(); ++text_idx) {
        const auto& text = norm_texts[text_idx];
        const auto& spans = norm_atom_spans[text_idx];
        size_t span_i = 0;
        
        for (size_t i = 0; i < text.size(); ) {
            const size_t seq_len = utf8SequenceLength(static_cast<unsigned char>(text[i]));
            
            while (span_i < spans.size() && spans[span_i].end <= i) {
                ++span_i;
            }
            if (span_i < spans.size() && i >= spans[span_i].start && i < spans[span_i].end) {
                atom_chars_skipped++;
                i += seq_len;
                continue;
            }
            
            if (i + seq_len <= text.size()) {
                std::string ch = text.substr(i, seq_len);
                char_counts[ch]++;
                total_chars++;
            }
            i += seq_len;
        }
    }
    
    if (atom_chars_skipped > 0) {
        std::cout << "[UnigramLM] Char counting: skipped " << atom_chars_skipped
                  << " characters inside atom spans" << std::endl;
    }
    if (total_chars == 0) {
        throw std::runtime_error("UnigramLM::trainFromCorpus: corpus has zero trainable non-atom characters after normalization");
    }
    
    // Step 2: Build the Step-2 character seed set.
    // These seeds are a transient training diagnostic only. They must not become
    // learned vocab entries, bias candidate admission/dedup, or receive pruning
    // protection. Single-character learned pieces must enter through ordinary
    // mined-candidate admission so byte fallback remains a post-unigram overflow
    // path instead of a seed-policy side effect.
    std::vector<std::pair<std::string, int>> sorted_chars(char_counts.begin(), char_counts.end());
    std::sort(sorted_chars.begin(), sorted_chars.end(),
              [](const auto& a, const auto& b) {
                  if (a.second != b.second) return a.second > b.second;
                  return a.first < b.first;
              });
    
    clearUnigramVocab(
        UnigramVocabWriteTarget{pieces_, piece_to_id_},
        "UnigramLM::trainFromCorpus initial learned-vocab reset");
    
    size_t covered = 0;
    size_t coverage_target = static_cast<size_t>(total_chars * character_coverage);
    const int MIN_CHAR_FREQUENCY = 10;
    
    std::unordered_set<std::string> char_seeds;
    
    int chars_rejected = 0;
    int chars_too_rare = 0;
    for (const auto& [ch, count] : sorted_chars) {
        if (covered >= coverage_target && char_seeds.size() >= 256) break;
        
        if (count < MIN_CHAR_FREQUENCY) {
            chars_too_rare++;
            continue;
        }
        
        if (!isValidVocabCharacter(ch)) {
            chars_rejected++;
            continue;
        }
        
        char_seeds.insert(ch);
        covered += count;
    }

    std::cout << "[UnigramLM] Initial char coverage: " << char_seeds.size()
              << " characters (Step-2 diagnostic seed set only; NOT learned vocab), coverage: "
              << (100.0f * covered / total_chars) << "%";
    if (chars_rejected > 0 || chars_too_rare > 0) {
        std::cout << " (rejected " << chars_rejected << " garbage";
        if (chars_too_rare > 0) {
            std::cout << ", " << chars_too_rare << " too rare (count < " << MIN_CHAR_FREQUENCY << ")";
        }
        std::cout << ")";
    }
    std::cout << std::endl;
    
    // Diagnostic: show coverage-critical chars
    std::cout << "[UnigramLM] Step-2 character seeds (diagnostic only; mined one-character candidates compete normally):"
              << std::endl;
    for (const auto& [ch, count] : sorted_chars) {
        if (!char_seeds.count(ch)) continue;
        std::string display_text;
        for (unsigned char c : ch) {
            if (c >= 32 && c <= 126) display_text += c;
            else if (c == '\t') display_text += "\\t";
            else if (c == '\n') display_text += "\\n";
            else if (c == '\r') display_text += "\\r";
            else {
                char buf[8];
                snprintf(buf, sizeof(buf), "\\x%02X", c);
                display_text += buf;
            }
        }
        std::stringstream hex_bytes;
        hex_bytes << std::hex;
        for (size_t i = 0; i < ch.size(); ++i) {
            if (i > 0) hex_bytes << " ";
            hex_bytes << std::setw(2) << std::setfill('0') << (int)(unsigned char)ch[i];
        }
        std::cout << "  [char-seed] \"" << display_text
                  << "\" (0x" << hex_bytes.str() << ") count=" << std::dec << count << std::endl;
    }
    
    // Step 3: Generate candidate subwords from SENTENCES
    UnigramSubwordCountMap subword_counts;
    subword_counts.reserve(1000000);

    const size_t num_texts_to_process = training_units.size();
    const size_t progress_interval = std::max<size_t>(1, num_texts_to_process / 20);
    const size_t max_len = static_cast<size_t>(MAX_PIECE_LENGTH);

    unsigned int mining_workers = resolveSubwordMiningWorkerCount(
        enable_parallel_subword_mining, subword_mining_workers, num_texts_to_process);

    std::cout << "[UnigramLM] Mining subwords from " << num_texts_to_process
              << " documents (workers=" << mining_workers
              << ", max_len=" << max_len << ")..." << std::endl;
    const auto mining_start = std::chrono::steady_clock::now();

    if (mining_workers <= 1) {
        for (size_t ti = 0; ti < num_texts_to_process; ++ti) {
            const std::string& text = training_units[ti];

            if (ti % progress_interval == 0) {
                std::cout << "[UnigramLM] Subword mining: " << ti << "/" << num_texts_to_process
                          << " (" << (100 * ti / std::max<size_t>(1, num_texts_to_process)) << "%), "
                          << subword_counts.size() << " unique subwords" << std::endl;
            }

            for (const SubwordMiningSpan& span : mining_plan.spans_by_text[ti]) {
                mineSubwordsFromSentence(text, max_len, norm_atom_spans[ti], subword_counts,
                                         span.context_start, span.context_end, span.start, span.end);
            }
        }
        const auto mining_elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - mining_start).count();
        std::cout << "[UnigramLM] Subword mining pass finished in "
                  << mining_elapsed << "s" << std::endl;
    } else {
        const size_t chunk_size = resolveSubwordMiningChunkSize(mining_workers, num_texts_to_process);
        std::cout << "[UnigramLM] Parallel subword mining: workers=" << mining_workers
                  << ", chunk_size=" << chunk_size << std::endl;

        std::vector<UnigramSubwordCountMap> local_counts(mining_workers);
        constexpr size_t kReservePerWorker = 6000000;
        for (auto& map : local_counts) {
            map.reserve(kReservePerWorker);
        }

        std::atomic<size_t> next_index{0};
        std::atomic<size_t> processed_texts{0};
        std::atomic<size_t> next_progress_log{progress_interval};
        std::atomic<bool> abort{false};
        std::exception_ptr first_error = nullptr;
        std::mutex error_mutex;
        std::mutex log_mutex;

        auto worker_fn = [&](unsigned int worker_id) {
            auto& local = local_counts[worker_id];
            while (!abort.load(std::memory_order_relaxed)) {
                const size_t begin = next_index.fetch_add(chunk_size, std::memory_order_relaxed);
                if (begin >= num_texts_to_process) break;
                const size_t end = std::min(begin + chunk_size, num_texts_to_process);

                try {
                    for (size_t ti = begin; ti < end; ++ti) {
                        if (abort.load(std::memory_order_relaxed)) return;
                        const std::string& text = training_units[ti];
                        for (const SubwordMiningSpan& span : mining_plan.spans_by_text[ti]) {
                            mineSubwordsFromSentence(text, max_len, norm_atom_spans[ti], local,
                                                     span.context_start, span.context_end, span.start, span.end);
                        }
                    }

                    const size_t chunk_done = end - begin;
                    const size_t done = processed_texts.fetch_add(chunk_done, std::memory_order_relaxed) + chunk_done;
                    size_t target = next_progress_log.load(std::memory_order_relaxed);
                    while (done >= target && target <= num_texts_to_process) {
                        if (next_progress_log.compare_exchange_weak(
                                target,
                                target + progress_interval,
                                std::memory_order_relaxed,
                                std::memory_order_relaxed)) {
                            const auto now = std::chrono::steady_clock::now();
                            const double elapsed_sec = std::chrono::duration<double>(now - mining_start).count();
                            const double rate = elapsed_sec > 0.0
                                ? static_cast<double>(done) / elapsed_sec
                                : 0.0;
                            std::lock_guard<std::mutex> lock(log_mutex);
                            std::cout << "[UnigramLM] Subword mining: " << done
                                      << "/" << num_texts_to_process
                                      << " (" << (100 * done / std::max<size_t>(1, num_texts_to_process))
                                      << "%), " << static_cast<size_t>(rate) << " sentences/s"
                                      << std::endl;
                            break;
                        }
                    }
                } catch (...) {
                    abort.store(true, std::memory_order_relaxed);
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (!first_error) first_error = std::current_exception();
                    return;
                }
            }
        };

        std::vector<std::thread> pool;
        pool.reserve(mining_workers);
        for (unsigned int w = 0; w < mining_workers; ++w) {
            pool.emplace_back(worker_fn, w);
        }
        for (auto& thread : pool) {
            thread.join();
        }

        if (first_error) {
            std::rethrow_exception(first_error);
        }

        const auto mining_elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - mining_start).count();
        std::cout << "[UnigramLM] Parallel mining pass finished in "
                  << mining_elapsed << "s; merging thread-local counts..." << std::endl;

        // Parallel tree-reduction merge
        const auto merge_start = std::chrono::steady_clock::now();
        std::cout << "[UnigramLM] Merging " << local_counts.size()
                  << " maps via parallel tree-reduction..." << std::endl;

        auto mergeInto = [](UnigramSubwordCountMap& dst,
                            UnigramSubwordCountMap& src) {
            dst.reserve(dst.size() + src.size());
            for (auto& [k, v] : src) {
                auto [it, inserted] = dst.try_emplace(k, v);
                if (!inserted) {
                    it->second = addUnigramSubwordCountsForTraining(
                        it->second,
                        v,
                        "UnigramLM::trainFromCorpus parallel subword count merge");
                }
            }
            src.clear();
            src.rehash(0);
        };

        while (local_counts.size() > 1) {
            const size_t n = local_counts.size();
            const size_t pairs = n / 2;
            std::vector<std::thread> merge_threads;
            merge_threads.reserve(pairs);
            for (size_t pi = 0; pi < pairs; ++pi) {
                merge_threads.emplace_back([&, pi]() {
                    mergeInto(local_counts[pi * 2], local_counts[pi * 2 + 1]);
                });
            }
            for (auto& t : merge_threads) t.join();

            std::vector<UnigramSubwordCountMap> next_round;
            next_round.reserve((n + 1) / 2);
            for (size_t i = 0; i < n; i += 2) {
                next_round.push_back(std::move(local_counts[i]));
            }
            local_counts = std::move(next_round);

            const double round_sec = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - merge_start).count();
            std::cout << "[UnigramLM] Merge round done: " << local_counts.size()
                      << " maps remaining, largest=" << local_counts[0].size()
                      << " entries (" << round_sec << "s)" << std::endl;
        }

        subword_counts = std::move(local_counts[0]);
        local_counts.clear();
        local_counts.shrink_to_fit();

        const double merge_total_sec = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - merge_start).count();
        std::cout << "[UnigramLM] Merge aggregation complete in "
                  << merge_total_sec << "s" << std::endl;
    }
    
    std::cout << "[UnigramLM] Subword mining complete: " << subword_counts.size() << " unique subwords" << std::endl;
    
    // Step 4: Select every data-qualified subword candidate.
    // The target vocab size is the final pruning cap only; it must not cap
    // initial candidate selection or derive any hidden seed-vocab size.
    std::cout << "[UnigramLM] Preparing sortable candidate list (" << subword_counts.size()
              << " total entries)..." << std::endl;
    const auto sort_start = std::chrono::steady_clock::now();

    size_t eligible_candidates = 0;
    for (const auto& [_, count] : subword_counts) {
        if (count >= MIN_SUBWORD_FREQ) {
            ++eligible_candidates;
        }
    }
    std::cout << "[UnigramLM] Frequency filter (min_freq=" << MIN_SUBWORD_FREQ
              << ") keeps " << eligible_candidates << " candidates" << std::endl;

    using SubwordEntry = std::pair<const std::string, UnigramSubwordCount>;
    std::vector<const SubwordEntry*> ranked_subwords;
    ranked_subwords.reserve(eligible_candidates);
    for (const auto& entry : subword_counts) {
        if (entry.second >= MIN_SUBWORD_FREQ) {
            ranked_subwords.push_back(&entry);
        }
    }
    const auto prep_elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - sort_start).count();
    std::cout << "[UnigramLM] Candidate index ready in " << prep_elapsed
              << "s; sorting..." << std::endl;
    std::sort(ranked_subwords.begin(), ranked_subwords.end(),
              [](const SubwordEntry* a, const SubwordEntry* b) {
                  if (a->second != b->second) return a->second > b->second;
                  return a->first < b->first;
              });
    const auto sort_elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - sort_start).count();
    std::cout << "[UnigramLM] Candidate sort complete in "
              << sort_elapsed << "s" << std::endl;
    
    int added = 0;
    int filtered = 0;
    int repetition_filtered = 0;
    int structural_dedup_rejected = 0;

    std::unordered_set<std::string> dedup_keys_seen;
    dedup_keys_seen.reserve(ranked_subwords.size());
    std::cout << "[UnigramLM] Learned candidate admission and structural dedup start from mined subwords only; Step-2 character seeds do not alter learned-vocab selection"
              << std::endl;

    struct AcceptedPiece { std::string text; UnigramSubwordCount count; };
    std::vector<AcceptedPiece> accepted;
    accepted.reserve(ranked_subwords.size());

    for (const SubwordEntry* entry : ranked_subwords) {
        const std::string& subword = entry->first;
        const UnigramSubwordCount count = entry->second;
        if (hasPiece(subword)) continue;

        if (isRepetitionNoise(subword)) {
            repetition_filtered++;
            continue;
        }
        if (!isValidSubword(subword)) {
            filtered++;
            continue;
        }

        std::string dedup_key = structuralDedupKeyForCandidate(subword);
        if (dedup_key.empty()) {
            filtered++;
            continue;
        }
        if (dedup_keys_seen.count(dedup_key)) {
            structural_dedup_rejected++;
            continue;
        }
        dedup_keys_seen.insert(dedup_key);

        // The structural key is comparison-only. The learned piece must retain
        // the exact mined subword text so boundary markers/edge bytes are not
        // silently stripped from the tokenizer vocabulary.
        accepted.push_back({subword, count});
    }

    double total_accepted_count = 0.0;
    for (const auto& ap : accepted) total_accepted_count += ap.count;
    requireUnigramAcceptedCandidateSetIsScorable(
        accepted.size(),
        total_accepted_count,
        "UnigramLM::trainFromCorpus initial candidate score normalization");

    for (const auto& ap : accepted) {
        float score = static_cast<float>(std::log(ap.count / total_accepted_count));
        UnigramPiece piece;
        piece.text = ap.text;
        piece.score = score;
        piece.is_user_defined = false;
        applyUnigramVocabWriteOp(UnigramVocabWriteRequest{
            UnigramVocabWriteTarget{pieces_, piece_to_id_},
            std::move(piece),
            tokenIdForIndex(static_cast<int>(pieces_.size())),
            UnigramVocabWriteMode::AppendOnly,
            "UnigramLM::trainFromCorpus candidate append"});
        added++;
    }
    
    if (filtered > 0) {
        std::cout << "[UnigramLM] Filtered " << filtered << " invalid subword patterns" << std::endl;
    }
    if (repetition_filtered > 0) {
        std::cout << "[UnigramLM] Filtered " << repetition_filtered
                  << " repetition/stutter subword patterns" << std::endl;
    }
    if (structural_dedup_rejected > 0) {
        std::cout << "[UnigramLM] Rejected " << structural_dedup_rejected
                  << " candidates (structural edge-trim dedup only; ▁ preserved)" << std::endl;
    }
    std::cout << "[UnigramLM] Added " << added << " data-selected subwords (min_freq=" << MIN_SUBWORD_FREQ
              << ", final_cap=" << target_vocab_size << "), total candidate vocab: " << pieces_.size() << std::endl;
    if (pieces_.empty()) {
        throw std::runtime_error("UnigramLM::trainFromCorpus: no learned subword candidates survived frequency/validity filters; lower min_subword_freq or provide more corpus text");
    }
    
    // Step 5: Iterative soft EM + Shrinking (SentencePiece-style)
    //
    // Starting with the full data-selected candidate vocab, we iteratively:
    //   1. Run true Unigram forward-backward EM to convergence
    //   2. Rank tokens by posterior expected mass plus expected compression gain
    //   3. Remove the bottom 25% of tokens (lowest posterior mass)
    //   4. Repeat until vocab <= target size
    //
    // This ensures every surviving token genuinely earns its slot by
    // contributing to compression — directly improving entropy, bytes/token,
    // and fertility vs. the old top-K-by-frequency approach.
    constexpr int    EM_MAX_ITERATIONS     = 50;
    constexpr double EM_CONVERGENCE_THRESH = 0.0001;
    constexpr double SMOOTHING             = 0.1;
    constexpr float  SHRINK_KEEP_RATIO     = 0.75f;  // Keep 75% each round

    struct EStepResult {
        std::unordered_map<int, double> learned_token_counts;
        double expected_learned_piece_tokens = 0.0;
        double expected_fixed_penalty_byte_fallback_tokens = 0.0;
        double log_likelihood = 0.0;
    };

    auto runEStep = [&]() -> EStepResult {
        UnigramForwardBackwardLattice lattice(
            pieces_, enable_byte_fallback_, "UnigramLM::trainFromCorpus forward-backward E-step");
        UnigramForwardBackwardStats stats(pieces_.size());
        std::unordered_map<int, double> token_counts;

        for (size_t text_idx = 0; text_idx < norm_texts.size(); ++text_idx) {
            const auto& text = norm_texts[text_idx];
            const auto& spans = norm_atom_spans[text_idx];
            if (text.empty()) continue;

            auto processSegment = [&](const std::string& segment) {
                if (segment.empty()) return;
                lattice.accumulateSegment(
                    segment, stats, "UnigramLM::trainFromCorpus forward-backward E-step");
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
        return EStepResult{std::move(token_counts),
                           stats.expected_learned_piece_tokens,
                           stats.expected_fixed_penalty_byte_fallback_tokens,
                           stats.log_likelihood};
    };

    auto runMStep = [&](const std::unordered_map<int, double>& token_counts, double learned_total_tokens) -> int {
        // Byte fallback remains outside the normalized learned-piece distribution.
        // Its UNKNOWN_SCORE path is a fixed unnormalized per-byte coverage penalty, so the
        // M-step normalizer is learned-piece posterior mass + learned-piece smoothing only.
        double smoothed_total = learned_total_tokens + SMOOTHING * static_cast<double>(pieces_.size());
        int zero_count = 0;
        for (size_t i = 0; i < pieces_.size(); ++i) {
            auto& piece = pieces_[i];
            if (piece.is_user_defined) continue;
            int tid = tokenIdForIndex(static_cast<int>(i));
            double expected_count = 0.0;
            auto count_it = token_counts.find(tid);
            if (count_it != token_counts.end()) {
                expected_count = count_it->second;
            }
            double count = expected_count + SMOOTHING;
            piece.score = static_cast<float>(std::log(count / smoothed_total));
            if (expected_count == 0.0) {
                zero_count++;
            }
        }
        return zero_count;
    };

    struct EMConvergenceResult {
        std::unordered_map<int, double> learned_token_counts;
        int iterations = 0;
        double expected_learned_piece_tokens = 0.0;
        double expected_fixed_penalty_byte_fallback_tokens = 0.0;
        double log_likelihood = 0.0;
    };

    auto requireConvergedPhaseNotFallbackDominated = [](const EMConvergenceResult& result,
                                                        const char* phase_label) {
        requireUnigramLearnedPosteriorMassNotByteFallbackDominated(
            result.expected_learned_piece_tokens,
            result.expected_fixed_penalty_byte_fallback_tokens,
            phase_label,
            "UnigramLM::trainFromCorpus converged forward-backward phase");
    };

    auto runEMToConvergence = [&](const char* phase_label) -> EMConvergenceResult {
        double prev_ll = -1e30;
        EStepResult last_e_step;
        int iter = 0;
        for (; iter < EM_MAX_ITERATIONS; ++iter) {
            EStepResult e_step = runEStep();
            int unused = runMStep(e_step.learned_token_counts, e_step.expected_learned_piece_tokens);

            double relative_change = 1.0;
            if (prev_ll >= -1e20) {
                relative_change = std::abs(
                    (e_step.log_likelihood - prev_ll) /
                    std::min(std::abs(prev_ll), std::abs(e_step.log_likelihood)));
            }

            std::cout << "[UnigramLM] " << phase_label << " iter " << (iter + 1)
                      << ": LL=" << std::fixed << std::setprecision(2) << e_step.log_likelihood
                      << ", expected_learned_piece_tokens=" << e_step.expected_learned_piece_tokens
                      << ", expected_fixed_penalty_byte_fallback_tokens=" << e_step.expected_fixed_penalty_byte_fallback_tokens
                      << ", unused=" << unused
                      << ", delta=" << std::scientific << std::setprecision(4) << relative_change
                      << std::defaultfloat << std::endl;

            last_e_step = std::move(e_step);
            bool converged = (iter > 0 && relative_change < EM_CONVERGENCE_THRESH);
            prev_ll = last_e_step.log_likelihood;
            if (converged) {
                std::cout << "[UnigramLM] " << phase_label << " converged after " << (iter + 1)
                          << " iterations (delta=" << std::scientific << std::setprecision(4)
                          << relative_change << std::defaultfloat << ")" << std::endl;
                ++iter;
                break;
            }
        }
        if (iter == EM_MAX_ITERATIONS) {
            std::cout << "[UnigramLM] " << phase_label << " hit max iterations ("
                      << EM_MAX_ITERATIONS << ") without full convergence" << std::endl;
        }
        return EMConvergenceResult{std::move(last_e_step.learned_token_counts),
                                   iter,
                                   last_e_step.expected_learned_piece_tokens,
                                   last_e_step.expected_fixed_penalty_byte_fallback_tokens,
                                   last_e_step.log_likelihood};
    };

    // ---- Phase A: initial EM to convergence on full candidate vocab ----
    std::cout << "[UnigramLM] Phase-A: forward-backward EM on data-selected candidate vocab (" << pieces_.size()
              << " pieces, final_cap=" << target_vocab_size << ")" << std::endl;
    EMConvergenceResult phase_a_result = runEMToConvergence("Phase-A");
    requireConvergedPhaseNotFallbackDominated(phase_a_result, "Phase-A");
    std::unordered_map<int, double> phase_a_counts = std::move(phase_a_result.learned_token_counts);


    // ---- Phase B: iterative shrinking to target vocab size ----
    // Each round removes the bottom 25% of tokens by posterior expected mass
    // plus expected compression gain. This keeps the actual soft
    // E-step occupancy evidence, while preventing frequent tiny fragments from
    // automatically outranking less frequent pieces that save more bytes.
    // Do not multiply by |score| here: that overprotects rare low-probability
    // pieces even when the forward-backward posterior says they are barely used.
    int shrink_round = 0;
    while (static_cast<int>(pieces_.size()) > target_vocab_size) {
        shrink_round++;

        // How many to keep this round (at least target_vocab_size, and never
        // below the number of user-defined pieces because those are protected).
        int current_size = static_cast<int>(pieces_.size());
        int user_defined_count = 0;
        for (const auto& piece : pieces_) {
            if (piece.is_user_defined) {
                ++user_defined_count;
            }
        }
        int keep_count = std::max(
            std::max(target_vocab_size, user_defined_count),
            static_cast<int>(current_size * SHRINK_KEEP_RATIO)
        );

        // Rank by posterior expected mass plus expected compression gain.
        // Higher value means the current soft segmentation either
        // relies on the piece more often or the piece earns its slot by saving bytes.
        struct TokenValue {
            int index;
            double value;
        };
        std::vector<TokenValue> token_values;
        token_values.reserve(pieces_.size());
        std::unordered_set<int> keep_indices;
        keep_indices.reserve(static_cast<size_t>(keep_count));

        for (size_t i = 0; i < pieces_.size(); ++i) {
            if (pieces_[i].is_user_defined) {
                // User-defined learned pieces are inserted into the protected
                // keep set before posterior-ranked fill slots are considered.
                keep_indices.insert(static_cast<int>(i));
                continue;
            }
            int tid = tokenIdForIndex(static_cast<int>(i));
            double count = 0.0;
            auto count_it = phase_a_counts.find(tid);
            if (count_it != phase_a_counts.end()) {
                count = count_it->second;
            }
            const double value = scoreUnigramShrinkCandidateForPosteriorCompression(
                pieces_[i].text,
                count,
                "UnigramLM::trainFromCorpus shrink ranking");
            token_values.push_back({static_cast<int>(i), value});
        }

        // Sort by value descending — most valuable first
        std::sort(token_values.begin(), token_values.end(),
                  [](const TokenValue& a, const TokenValue& b) {
                      if (a.value != b.value) return a.value > b.value;
                      return a.index < b.index;
                  });

        // Fill remaining slots with the top non-user-defined pieces by posterior
        // mass plus expected compression gain.
        const int fill_slots = std::max(0, keep_count - static_cast<int>(keep_indices.size()));
        for (int k = 0; k < fill_slots && k < static_cast<int>(token_values.size()); ++k) {
            keep_indices.insert(token_values[k].index);
        }

        int removed = current_size - static_cast<int>(keep_indices.size());
        if (removed == 0 && current_size > target_vocab_size) {
            if (token_values.empty()) {
                std::cout << "[UnigramLM] Shrink stopped at " << current_size
                          << " pieces: target_vocab_size=" << target_vocab_size
                          << " cannot be reached without pruning " << user_defined_count
                          << " protected user-defined pieces" << std::endl;
                break;
            }
            throw std::runtime_error("UnigramLM::trainFromCorpus shrink made no progress despite " +
                                     std::to_string(token_values.size()) +
                                     " prunable pieces; keep_count=" + std::to_string(keep_count) +
                                     ", current_size=" + std::to_string(current_size));
        }
        std::cout << "[UnigramLM] Shrink round " << shrink_round
                  << ": " << current_size << " -> " << keep_indices.size()
                  << " tokens (removed " << removed
                  << ", protected_user_defined=" << user_defined_count << ")" << std::endl;

        // Compact pieces_ to survivors only
        std::vector<UnigramPiece> surviving;
        surviving.reserve(keep_indices.size());
        for (size_t i = 0; i < pieces_.size(); ++i) {
            if (keep_indices.count(static_cast<int>(i))) {
                surviving.push_back(std::move(pieces_[i]));
            }
        }
        rewriteUnigramVocab(
            UnigramVocabWriteTarget{pieces_, piece_to_id_},
            std::move(surviving),
            "UnigramLM::trainFromCorpus shrink compaction");

        // Re-converge EM on the smaller vocab
        std::string label = "Shrink-" + std::to_string(shrink_round);
        EMConvergenceResult shrink_result = runEMToConvergence(label.c_str());
        requireConvergedPhaseNotFallbackDominated(shrink_result, label.c_str());
        phase_a_counts = std::move(shrink_result.learned_token_counts);
    }

    if (shrink_round > 0) {
        std::cout << "[UnigramLM] Iterative shrinking complete after " << shrink_round
                  << " rounds. Final vocab: " << pieces_.size() << " pieces" << std::endl;
    }

    // ---- Phase C: final dead-token cleanup ----
    // Remove any tokens that still have zero posterior expected count after convergence
    int pruned = 0;
    {
        std::unordered_set<int> dead_indices;
        for (size_t i = 0; i < pieces_.size(); ++i) {
            if (pieces_[i].is_user_defined) continue;
            int tid = tokenIdForIndex(static_cast<int>(i));
            double count = 0.0;
            auto count_it = phase_a_counts.find(tid);
            if (count_it != phase_a_counts.end()) {
                count = count_it->second;
            }
            if (count == 0.0) {
                dead_indices.insert(static_cast<int>(i));
            }
        }

        if (!dead_indices.empty()) {
            requireUnigramFinalCleanupLeavesLearnedPiece(
                pieces_.size(),
                dead_indices.size(),
                "UnigramLM::trainFromCorpus final dead-token cleanup");

            std::cout << "[UnigramLM] Final cleanup: pruning " << dead_indices.size()
                      << " dead tokens (posterior expected count == 0)" << std::endl;

            std::vector<UnigramPiece> surviving;
            surviving.reserve(pieces_.size() - dead_indices.size());
            for (size_t i = 0; i < pieces_.size(); ++i) {
                if (!dead_indices.count(static_cast<int>(i))) {
                    surviving.push_back(std::move(pieces_[i]));
                }
            }
            pruned = static_cast<int>(dead_indices.size());
            rewriteUnigramVocab(
                UnigramVocabWriteTarget{pieces_, piece_to_id_},
                std::move(surviving),
                "UnigramLM::trainFromCorpus final dead-token cleanup");
        } else {
            std::cout << "[UnigramLM] No dead tokens after shrinking — vocab is clean" << std::endl;
        }
    }

    // ---- Phase D: final reconvergence after cleanup ----
    if (pruned > 0) {
        std::cout << "[UnigramLM] Reconverging after pruning " << pruned
                  << " dead tokens (vocab now " << pieces_.size() << ")" << std::endl;
        EMConvergenceResult phase_d_result = runEMToConvergence("Phase-D");
        requireConvergedPhaseNotFallbackDominated(phase_d_result, "Phase-D");
    }

    // Final trie build with converged scores
    buildTrie();

    if (!initGPUForMaxSequenceLength(training_segment_max_length)) {
        throw std::runtime_error("UnigramLM::trainFromCorpus: final tokenizer runtime upload failed after final buildTrie()");
    }

    last_training_runtime_report_.required_viterbi_workspace_length = training_segment_max_length;
    last_training_runtime_report_.finalized_trie_generation = trie_generation_;
    last_training_runtime_report_.final_piece_count = pieceCount();

    const auto runtime_snapshot = runtimeStateSnapshot();
    std::cout << "[UnigramLM] Final tokenizer runtime state: required_viterbi_workspace_length="
              << last_training_runtime_report_.required_viterbi_workspace_length
              << ", uploaded_workspace_max_length=" << runtime_snapshot.workspace_max_length
              << ", trie_generation=" << runtime_snapshot.live_trie_generation
              << std::endl;
    
    std::cout << "[UnigramLM] Training complete. Final vocab size: " << pieces_.size() << std::endl;
    return true;
}

} // namespace Tokenizer
} // namespace GRIM
