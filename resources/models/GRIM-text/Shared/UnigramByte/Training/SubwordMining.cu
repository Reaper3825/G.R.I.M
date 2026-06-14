//======================================================//
//  SubwordMining.cu
//  Training-only unigram subword candidate mining
//======================================================//

#include "SubwordMining.hpp"

#include "../TextUtils.hpp"
#include "../TokenLayout.hpp"
#include "../Unigram.hpp"
#include "HyperParameters/HyperparameterGroupings.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace GRIM {
namespace Tokenizer {
namespace {

bool isUtf8ContinuationByte(unsigned char c) {
    return (c & 0xC0) == 0x80;
}

bool hasExcessiveRunLength(const std::string& s) {
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

bool isRepeatedPatternNoise(const std::string& s) {
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

bool isDoubledTokenNoise(const std::string& s) {
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

bool isWordLevelStutter(const std::string& s) {
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

bool isValidUnigramVocabCharacterImpl(const std::string& ch) {
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

bool isValidUnigramSubwordImpl(const std::string& s) {
    if (s.empty()) return false;

    if (s.size() == 1 || utf8SequenceLength(static_cast<unsigned char>(s[0])) == s.size()) {
        return isValidUnigramVocabCharacterImpl(s);
    }

    if (isUnigramRepetitionNoise(s)) return false;

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

unsigned int resolveSubwordMiningWorkerCount(
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
        const std::string env_text(env_workers);
        size_t consumed = 0;
        const int parsed = std::stoi(env_text, &consumed);
        if (consumed != env_text.size() || parsed <= 0) {
            throw std::runtime_error("resolveSubwordMiningWorkerCount: GRIM_SUBWORD_MINING_WORKERS must be a positive integer, got '" +
                                     env_text + "'");
        }
        workers = static_cast<unsigned int>(parsed);
    }

    workers = std::min<unsigned int>(workers, static_cast<unsigned int>(sentence_count));
    return std::max(1u, workers);
}

size_t resolveSubwordMiningChunkSize(unsigned int workers, size_t sentence_count) {
    if (workers <= 1 || sentence_count == 0) return sentence_count;

    size_t chunk = (sentence_count + (workers * 8) - 1) / (workers * 8);
    chunk = std::max<size_t>(64, chunk);
    chunk = std::min<size_t>(4096, chunk);
    return chunk;
}

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

size_t snapBackwardToUtf8Boundary(const std::string& text, size_t pos) {
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

size_t snapForwardToUtf8Boundary(const std::string& text, size_t pos) {
    if (pos > text.size()) {
        throw std::runtime_error("snapForwardToUtf8Boundary: pos exceeds text.size(), pos=" +
                                 std::to_string(pos) + ", text.size()=" + std::to_string(text.size()));
    }
    while (pos < text.size() && isUtf8ContinuationByte(static_cast<unsigned char>(text[pos]))) {
        ++pos;
    }
    return pos;
}

void appendMergedSubwordMiningSpan(std::vector<SubwordMiningSpan>& spans,
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

SubwordMiningPlan buildSubwordMiningPlan(
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
    size_t sampled_span_context_overlap = 0;
    if (static_cast<size_t>(MAX_PIECE_LENGTH) > 0) {
        sampled_span_context_overlap = static_cast<size_t>(MAX_PIECE_LENGTH) - 1;
    }

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

                    size_t raw_context_start = 0;
                    if (start > sampled_span_context_overlap) {
                        raw_context_start = start - sampled_span_context_overlap;
                    }
                    const size_t context_start = snapBackwardToUtf8Boundary(text, raw_context_start);
                    const size_t raw_context_end = std::min(text.size(), end + sampled_span_context_overlap);
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

void validateSubwordMiningRange(const std::string& text,
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

void validateSubwordMiningCountingRange(const std::string& text,
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

void incrementUnigramSubwordCountForTraining(UnigramSubwordCountMap& subword_counts,
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

bool crossesSentencePieceWordBoundary(const std::string& subword) {
    size_t search_start = 0;
    if (subword.size() >= SPIECE_UNDERLINE_LEN &&
        subword.compare(0, SPIECE_UNDERLINE_LEN, SPIECE_UNDERLINE) == 0) {
        search_start = SPIECE_UNDERLINE_LEN;
    }
    return search_start < subword.size() &&
           subword.find(SPIECE_UNDERLINE, search_start, SPIECE_UNDERLINE_LEN) != std::string::npos;
}

std::vector<size_t> collectCharacterPositions(const std::string& text,
                                              size_t context_start,
                                              size_t context_end,
                                              const char* caller) {
    std::vector<size_t> char_positions;
    char_positions.reserve((context_end - context_start) + 1);
    for (size_t i = context_start; i < context_end; ) {
        char_positions.push_back(i);
        const size_t seq_len = utf8SequenceLength(static_cast<unsigned char>(text[i]));
        if (i + seq_len > context_end) {
            throw std::runtime_error(std::string(caller) + ": UTF-8 sequence crosses mining span end at byte=" +
                                     std::to_string(i) + ", context_end=" + std::to_string(context_end));
        }
        i += seq_len;
    }
    char_positions.push_back(context_end);
    return char_positions;
}

void countSubwordsFromCharacterPositions(const std::string& text,
                                         size_t max_len,
                                         const std::vector<size_t>& char_positions,
                                         const std::vector<bool>* char_in_atom,
                                         size_t count_start,
                                         size_t count_end,
                                         UnigramSubwordCountMap& subword_counts,
                                         const char* increment_context) {
    const size_t num_chars = char_positions.size() - 1;
    for (size_t ci = 0; ci < num_chars; ++ci) {
        if (char_in_atom != nullptr && (*char_in_atom)[ci]) continue;

        const size_t byte_start = char_positions[ci];
        if (byte_start < count_start || byte_start >= count_end) continue;
        for (size_t char_count = 1; char_count <= max_len && ci + char_count <= num_chars; ++char_count) {
            if (char_in_atom != nullptr) {
                bool crosses_atom = false;
                for (size_t k = ci + 1; k < ci + char_count; ++k) {
                    if ((*char_in_atom)[k]) { crosses_atom = true; break; }
                }
                if (crosses_atom) break;
            }

            const size_t byte_end = char_positions[ci + char_count];
            if (byte_end - byte_start > max_len) break;

            std::string subword = text.substr(byte_start, byte_end - byte_start);
            if (!isValidUnigramSubwordImpl(subword)) continue;
            if (crossesSentencePieceWordBoundary(subword)) continue;

            incrementUnigramSubwordCountForTraining(
                subword_counts,
                subword,
                increment_context);
        }
    }
}

void mineSubwordsFromSentence(const std::string& text,
                              size_t max_len,
                              UnigramSubwordCountMap& subword_counts,
                              size_t context_start,
                              size_t context_end,
                              size_t count_start,
                              size_t count_end) {
    validateSubwordMiningCountingRange(text, context_start, context_end, count_start, count_end,
                                       "mineSubwordsFromSentence");
    if (context_start == context_end || count_start == count_end) return;

    std::vector<size_t> char_positions = collectCharacterPositions(
        text,
        context_start,
        context_end,
        "mineSubwordsFromSentence");
    countSubwordsFromCharacterPositions(
        text,
        max_len,
        char_positions,
        nullptr,
        count_start,
        count_end,
        subword_counts,
        "mineSubwordsFromSentence subword count increment");
}

void mineSubwordsFromSentence(const std::string& text,
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

    std::vector<size_t> char_positions = collectCharacterPositions(
        text,
        context_start,
        context_end,
        "mineSubwordsFromSentence atom-aware");

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

    countSubwordsFromCharacterPositions(
        text,
        max_len,
        char_positions,
        &char_in_atom,
        count_start,
        count_end,
        subword_counts,
        "mineSubwordsFromSentence atom-aware subword count increment");
}

void mergeUnigramSubwordCounts(UnigramSubwordCountMap& dst,
                               UnigramSubwordCountMap& src) {
    dst.reserve(dst.size() + src.size());
    for (auto& [k, v] : src) {
        auto [it, inserted] = dst.try_emplace(k, v);
        if (!inserted) {
            it->second = addUnigramSubwordCountsForTraining(
                it->second,
                v,
                "mineUnigramSubwordsFromTrainingUnits parallel subword count merge");
        }
    }
    src.clear();
    src.rehash(0);
}

void validateSubwordMiningRequest(const UnigramSubwordMiningRequest& request) {
    if (request.log_prefix == nullptr || request.log_prefix[0] == '\0') {
        throw std::runtime_error("mineUnigramSubwordsFromTrainingUnits: log_prefix is empty - caller MUST provide a diagnostic prefix");
    }
    if (request.training_units.empty()) {
        throw std::runtime_error("mineUnigramSubwordsFromTrainingUnits: training_units is empty - caller MUST provide normalized training text");
    }
    if (request.atom_spans.size() != request.training_units.size()) {
        throw std::runtime_error("mineUnigramSubwordsFromTrainingUnits: atom_spans.size()=" +
                                 std::to_string(request.atom_spans.size()) +
                                 " != training_units.size()=" + std::to_string(request.training_units.size()));
    }
    if (request.configured_worker_count < 0) {
        throw std::runtime_error("mineUnigramSubwordsFromTrainingUnits: configured_worker_count must be >= 0, got " +
                                 std::to_string(request.configured_worker_count));
    }
}

} // namespace

bool isValidUnigramVocabCharacter(const std::string& ch) {
    return isValidUnigramVocabCharacterImpl(ch);
}

bool isUnigramRepetitionNoise(const std::string& s) {
    return hasExcessiveRunLength(s) || isRepeatedPatternNoise(s) ||
           isDoubledTokenNoise(s) || isWordLevelStutter(s);
}

bool isValidUnigramSubword(const std::string& s) {
    return isValidUnigramSubwordImpl(s);
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

UnigramSubwordMiningResult mineUnigramSubwordsFromTrainingUnits(
    const UnigramSubwordMiningRequest& request) {
    validateSubwordMiningRequest(request);

    UnigramSubwordMiningResult result;
    for (const auto& sent : request.training_units) {
        result.total_training_bytes += sent.size();
    }

    result.max_mining_bytes = request.configured_max_mining_bytes;
    if (result.max_mining_bytes == 0) {
        result.max_mining_bytes = result.total_training_bytes;
    }
    result.used_sampling = result.total_training_bytes > result.max_mining_bytes;

    if (request.configured_max_mining_bytes == 0) {
        std::cout << request.log_prefix << " Subword mining byte cap: uncapped (full corpus)" << std::endl;
    } else {
        std::cout << request.log_prefix << " Subword mining byte cap: "
                  << (result.max_mining_bytes / (1024 * 1024)) << " MB" << std::endl;
    }

    const SubwordMiningPlan mining_plan = buildSubwordMiningPlan(
        request.training_units,
        result.total_training_bytes,
        result.max_mining_bytes,
        result.used_sampling);
    result.sampling_ratio = mining_plan.sampling_ratio;
    result.sampled_start_bytes = mining_plan.sampled_bytes;
    result.context_bytes = mining_plan.context_bytes;
    result.sampled_spans = mining_plan.sampled_spans;

    if (result.used_sampling) {
        std::cout << request.log_prefix << " Byte-proportional strided mining: ratio="
                  << std::fixed << std::setprecision(3) << result.sampling_ratio
                  << ", spans=" << result.sampled_spans
                  << ", sampled_starts=" << (result.sampled_start_bytes / (1024*1024)) << " MB"
                  << ", context=" << (result.context_bytes / (1024*1024)) << " MB from all "
                  << request.training_units.size() << " documents" << std::defaultfloat << std::endl;
    }

    result.subword_counts.reserve(1000000);

    const size_t num_texts_to_process = request.training_units.size();
    const size_t progress_interval = std::max<size_t>(1, num_texts_to_process / 20);
    const size_t max_len = static_cast<size_t>(MAX_PIECE_LENGTH);

    result.worker_count = resolveSubwordMiningWorkerCount(
        request.enable_parallel_subword_mining,
        request.configured_worker_count,
        num_texts_to_process);

    std::cout << request.log_prefix << " Mining subwords from " << num_texts_to_process
              << " documents (workers=" << result.worker_count
              << ", max_len=" << max_len << ")..." << std::endl;
    const auto mining_start = std::chrono::steady_clock::now();

    if (result.worker_count <= 1) {
        for (size_t ti = 0; ti < num_texts_to_process; ++ti) {
            const std::string& text = request.training_units[ti];

            if (ti % progress_interval == 0) {
                std::cout << request.log_prefix << " Subword mining: " << ti << "/" << num_texts_to_process
                          << " (" << (100 * ti / std::max<size_t>(1, num_texts_to_process)) << "%), "
                          << result.subword_counts.size() << " unique subwords" << std::endl;
            }

            for (const SubwordMiningSpan& span : mining_plan.spans_by_text[ti]) {
                mineSubwordsFromSentence(text, max_len, request.atom_spans[ti], result.subword_counts,
                                         span.context_start, span.context_end, span.start, span.end);
            }
        }
        const double mining_elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - mining_start).count();
        std::cout << request.log_prefix << " Subword mining pass finished in "
                  << mining_elapsed << "s" << std::endl;
    } else {
        const size_t chunk_size = resolveSubwordMiningChunkSize(result.worker_count, num_texts_to_process);
        std::cout << request.log_prefix << " Parallel subword mining: workers=" << result.worker_count
                  << ", chunk_size=" << chunk_size << std::endl;

        std::vector<UnigramSubwordCountMap> local_counts(result.worker_count);
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
                        const std::string& text = request.training_units[ti];
                        for (const SubwordMiningSpan& span : mining_plan.spans_by_text[ti]) {
                            mineSubwordsFromSentence(text, max_len, request.atom_spans[ti], local,
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
                            double rate = 0.0;
                            if (elapsed_sec > 0.0) {
                                rate = static_cast<double>(done) / elapsed_sec;
                            }
                            std::lock_guard<std::mutex> lock(log_mutex);
                            std::cout << request.log_prefix << " Subword mining: " << done
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
        pool.reserve(result.worker_count);
        for (unsigned int w = 0; w < result.worker_count; ++w) {
            pool.emplace_back(worker_fn, w);
        }
        for (auto& thread : pool) {
            thread.join();
        }

        if (first_error) {
            std::rethrow_exception(first_error);
        }

        const double mining_elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - mining_start).count();
        std::cout << request.log_prefix << " Parallel mining pass finished in "
                  << mining_elapsed << "s; merging thread-local counts..." << std::endl;

        const auto merge_start = std::chrono::steady_clock::now();
        std::cout << request.log_prefix << " Merging " << local_counts.size()
                  << " maps via parallel tree-reduction..." << std::endl;

        while (local_counts.size() > 1) {
            const size_t n = local_counts.size();
            const size_t pairs = n / 2;
            std::vector<std::thread> merge_threads;
            merge_threads.reserve(pairs);
            for (size_t pi = 0; pi < pairs; ++pi) {
                merge_threads.emplace_back([&, pi]() {
                    mergeUnigramSubwordCounts(local_counts[pi * 2], local_counts[pi * 2 + 1]);
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
            std::cout << request.log_prefix << " Merge round done: " << local_counts.size()
                      << " maps remaining, largest=" << local_counts[0].size()
                      << " entries (" << round_sec << "s)" << std::endl;
        }

        result.subword_counts = std::move(local_counts[0]);
        local_counts.clear();
        local_counts.shrink_to_fit();

        const double merge_total_sec = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - merge_start).count();
        std::cout << request.log_prefix << " Merge aggregation complete in "
                  << merge_total_sec << "s" << std::endl;
    }

    std::cout << request.log_prefix << " Subword mining complete: " << result.subword_counts.size()
              << " unique subwords" << std::endl;
    return result;
}

} // namespace Tokenizer
} // namespace GRIM
