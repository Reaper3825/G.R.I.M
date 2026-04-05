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
#include "TextUtils.hpp"
#include "HyperParameters/HyperParameters_GPU.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <thread>
#include <unordered_set>

namespace GRIM {
namespace Tokenizer {

//======================================================//
//  Character-Level Validator
//  Rejects garbage characters BEFORE they enter the vocab.
//  Applied to single-character seeds in Step 2 of training.
//======================================================//

static bool isValidVocabCharacter(const std::string& ch) {
    if (ch.empty()) return false;

    // Single-byte ASCII path
    if (ch.size() == 1) {
        unsigned char c = static_cast<unsigned char>(ch[0]);
        if (c >= 0x20 && c <= 0x7E) return true;
        if (c == 0x09 || c == 0x0A || c == 0x0D) return true;
        return false;
    }

    // Multi-byte UTF-8 path: decode codepoint and check against known garbage ranges
    unsigned char b0 = static_cast<unsigned char>(ch[0]);
    uint32_t codepoint = 0;

    if (ch.size() == 2 && (b0 & 0xE0) == 0xC0) {
        unsigned char b1 = static_cast<unsigned char>(ch[1]);
        codepoint = ((b0 & 0x1F) << 6) | (b1 & 0x3F);
    } else if (ch.size() == 3 && (b0 & 0xF0) == 0xE0) {
        unsigned char b1 = static_cast<unsigned char>(ch[1]);
        unsigned char b2 = static_cast<unsigned char>(ch[2]);
        codepoint = ((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F);
    } else if (ch.size() == 4 && (b0 & 0xF8) == 0xF0) {
        unsigned char b1 = static_cast<unsigned char>(ch[1]);
        unsigned char b2 = static_cast<unsigned char>(ch[2]);
        unsigned char b3 = static_cast<unsigned char>(ch[3]);
        codepoint = ((b0 & 0x07) << 18) | ((b1 & 0x3F) << 12) | ((b2 & 0x3F) << 6) | (b3 & 0x3F);
    } else {
        return false;
    }

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
//======================================================//

static std::string structuralDedupKeyForCandidate(const std::string& s) {
    size_t start = 0;
    size_t end = s.size();
    while (start < end) {
        uint32_t cp = 0;
        size_t len = 0;
        if (utf8DecodeAt(s, start, &cp, &len) && isStructuralEdgeWhitespace(cp)) {
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
        if (!isStructuralEdgeWhitespace(cp)) break;
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
//  Prefix Extension Dedup
//  Rejects candidates that are 1-3 UTF-8 char extensions of
//  an already-accepted piece with the same corpus count.
//  e.g., "an▁also" is rejected if "an▁als" was already
//  accepted with the same count — they co-occur identically.
//======================================================//

static bool isPrefixExtensionDuplicate(
    const std::string& dedup_key,
    int count,
    const std::unordered_set<std::string>& seen_keys,
    const std::unordered_map<std::string, int>& key_to_count)
{
    // Try truncating 1, 2, 3 UTF-8 characters from the end.
    // Walk backwards over continuation bytes to find char boundaries.
    size_t end = dedup_key.size();
    for (int trim = 0; trim < 3 && end > 0; ++trim) {
        // Step back one UTF-8 character
        --end;
        while (end > 0 && (static_cast<unsigned char>(dedup_key[end]) & 0xC0) == 0x80) {
            --end;
        }
        if (end == 0) break;  // Don't dedup against empty prefix

        std::string prefix = dedup_key.substr(0, end);
        if (seen_keys.count(prefix)) {
            auto it = key_to_count.find(prefix);
            if (it != key_to_count.end() && it->second == count) {
                return true;  // Prefix with same count exists — this is a redundant extension
            }
        }
    }
    return false;
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

//======================================================//
//  Subword Mining from Sentences
//======================================================//

static void mineSubwordsFromSentence(const std::string& text,
                                     size_t max_len,
                                     std::unordered_map<std::string, int>& subword_counts) {
    if (text.empty()) return;

    std::vector<size_t> char_positions;
    char_positions.reserve(text.size() + 1);
    for (size_t i = 0; i < text.size(); ) {
        char_positions.push_back(i);
        i += utf8SequenceLength(static_cast<unsigned char>(text[i]));
    }
    char_positions.push_back(text.size());

    const size_t num_chars = char_positions.size() - 1;
    for (size_t ci = 0; ci < num_chars; ++ci) {
        const size_t byte_start = char_positions[ci];
        for (size_t char_count = 2; char_count <= max_len && ci + char_count <= num_chars; ++char_count) {
            const size_t byte_end = char_positions[ci + char_count];
            if (byte_end - byte_start > 64) continue;

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

            subword_counts[subword]++;
        }
    }
}

// Atom-aware overload: skip subwords that START inside an atom span.
static void mineSubwordsFromSentence(const std::string& text,
                                     size_t max_len,
                                     const std::vector<AtomSpan>& atom_spans,
                                     std::unordered_map<std::string, int>& subword_counts) {
    if (text.empty()) return;
    if (atom_spans.empty()) {
        mineSubwordsFromSentence(text, max_len, subword_counts);
        return;
    }

    std::vector<size_t> char_positions;
    char_positions.reserve(text.size() + 1);
    for (size_t i = 0; i < text.size(); ) {
        char_positions.push_back(i);
        i += utf8SequenceLength(static_cast<unsigned char>(text[i]));
    }
    char_positions.push_back(text.size());

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
        for (size_t char_count = 2; char_count <= max_len && ci + char_count <= num_chars; ++char_count) {
            bool crosses_atom = false;
            for (size_t k = ci + 1; k < ci + char_count; ++k) {
                if (char_in_atom[k]) { crosses_atom = true; break; }
            }
            if (crosses_atom) break;

            const size_t byte_end = char_positions[ci + char_count];
            if (byte_end - byte_start > 64) continue;

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

            subword_counts[subword]++;
        }
    }
}

//======================================================//
//  trainFromCorpus — Delegating Overload
//======================================================//

bool UnigramLM::trainFromCorpus(const std::vector<std::string>& texts,
                                 int target_vocab_size,
                                 float character_coverage,
                                 int min_subword_freq,
                                 bool prune_during_mining,
                                 bool enable_parallel_subword_mining,
                                 int subword_mining_workers,
                                 size_t subword_mining_max_bytes) {
    std::vector<std::vector<AtomSpan>> empty_spans(texts.size());
    return trainFromCorpus(texts, empty_spans, target_vocab_size,
                           character_coverage, min_subword_freq,
                           prune_during_mining, enable_parallel_subword_mining,
                           subword_mining_workers, subword_mining_max_bytes);
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
    if (atom_spans.size() != texts.size()) {
        throw std::runtime_error("[UnigramLM] atom_spans.size()=" + std::to_string(atom_spans.size())
                                  + " != texts.size()=" + std::to_string(texts.size()));
    }

    std::cout << "[UnigramLM] Training vocabulary from " << texts.size() 
              << " texts (target_vocab_size=" << target_vocab_size << ")" << std::endl;
    std::cout << "[UnigramLM] min_subword_freq=" << min_subword_freq 
              << ", prune_during_mining=" << (prune_during_mining ? "true" : "false")
              << ", parallel_subword_mining=" << (enable_parallel_subword_mining ? "true" : "false")
              << ", subword_mining_workers=" << subword_mining_workers << std::endl;

    // Count total atoms for logging
    size_t total_atom_spans = 0;
    size_t total_atom_bytes = 0;
    for (const auto& spans : atom_spans) {
        total_atom_spans += spans.size();
        for (const auto& s : spans) {
            total_atom_bytes += (s.end - s.start);
        }
    }
    if (total_atom_spans > 0) {
        std::cout << "[UnigramLM] Atom-aware training: " << total_atom_spans
                  << " atom spans (" << (total_atom_bytes / 1024) << " KB) will be skipped" << std::endl;
    }
    
    // SentencePiece-style whitespace normalization
    std::vector<std::string> norm_texts;
    std::vector<std::vector<AtomSpan>> norm_atom_spans;
    norm_texts.reserve(texts.size());
    norm_atom_spans.reserve(texts.size());
    for (size_t i = 0; i < texts.size(); ++i) {
        auto spans_copy = atom_spans[i];
        norm_texts.push_back(normalizeWithSpans(texts[i], spans_copy));
        norm_atom_spans.push_back(std::move(spans_copy));
    }
    std::cout << "[UnigramLM] Applied SentencePiece whitespace normalization (space -> ▁)" << std::endl;

    const std::vector<std::string>& training_units = norm_texts;
    const int MIN_SUBWORD_FREQ = min_subword_freq;
    
    size_t total_corpus_bytes = 0;
    for (const auto& text : texts) {
        total_corpus_bytes += text.size();
    }
    std::cout << "[UnigramLM] Total corpus size: " << (total_corpus_bytes / (1024*1024)) << " MB" << std::endl;
    
    size_t total_sentence_bytes = 0;
    for (const auto& sent : training_units) {
        total_sentence_bytes += sent.size();
    }
    
    const size_t max_subword_mining_bytes =
        (subword_mining_max_bytes > 0)
            ? subword_mining_max_bytes
            : static_cast<size_t>(HyperParameters::UNIGRAM_MAX_SUBWORD_BYTES);
    const bool use_sampling = total_sentence_bytes > max_subword_mining_bytes;
    std::vector<size_t> sample_indices;

    std::cout << "[UnigramLM] Subword mining byte cap: "
              << (max_subword_mining_bytes / (1024 * 1024)) << " MB" << std::endl;
    
    if (use_sampling) {
        std::mt19937 rng(42);
        std::vector<size_t> all_indices(training_units.size());
        std::iota(all_indices.begin(), all_indices.end(), 0);
        std::shuffle(all_indices.begin(), all_indices.end(), rng);
        
        size_t sampled_bytes = 0;
        for (size_t idx : all_indices) {
            if (sampled_bytes >= max_subword_mining_bytes) break;
            sample_indices.push_back(idx);
            sampled_bytes += training_units[idx].size();
        }
        std::cout << "[UnigramLM] Sampling " << sample_indices.size() << " documents (" 
                  << (sampled_bytes / (1024*1024)) << " MB) for subword mining" << std::endl;
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
    
    // Step 2: Build initial vocabulary (all characters meeting coverage)
    std::vector<std::pair<std::string, int>> sorted_chars(char_counts.begin(), char_counts.end());
    std::sort(sorted_chars.begin(), sorted_chars.end(),
              [](const auto& a, const auto& b) {
                  if (a.second != b.second) return a.second > b.second;
                  return a.first < b.first;
              });
    
    pieces_.clear();
    piece_to_id_.clear();
    
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
              << " characters (byte-layer covered), coverage: "
              << (100.0f * covered / total_chars) << "%";
    if (chars_rejected > 0 || chars_too_rare > 0) {
        std::cout << " (rejected " << chars_rejected << " garbage";
        if (chars_too_rare > 0) {
            std::cout << ", " << chars_too_rare << " too rare (count < " << MIN_CHAR_FREQUENCY << ")";
        }
        std::cout << ")";
    }
    std::cout << std::endl;
    
    // Diagnostic: show byte-covered chars
    std::cout << "[UnigramLM] Byte-covered chars (NOT in unigram vocab, handled by byte layer):" << std::endl;
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
        int byte_id = (int)(unsigned char)ch[0] + BYTE_TOKEN_OFFSET;
        std::cout << "  [byte:" << byte_id << "] \"" << display_text
                  << "\" (0x" << hex_bytes.str() << ") count=" << std::dec << count << std::endl;
    }
    
    // Step 3: Generate candidate subwords from SENTENCES
    std::unordered_map<std::string, int> subword_counts;
    subword_counts.reserve(1000000);

    const size_t num_texts_to_process = use_sampling ? sample_indices.size() : training_units.size();
    const size_t progress_interval = std::max<size_t>(1, num_texts_to_process / 20);
    const size_t max_len = use_sampling
        ? static_cast<size_t>(MAX_PIECE_LENGTH)
        : std::min(static_cast<size_t>(MAX_PIECE_LENGTH), size_t(16));

    auto sentenceForIndex = [&](size_t idx) -> const std::string& {
        return use_sampling ? training_units[sample_indices[idx]] : training_units[idx];
    };

    auto sentenceAtomsForIndex = [&](size_t idx) -> const std::vector<AtomSpan>& {
        return use_sampling ? norm_atom_spans[sample_indices[idx]] : norm_atom_spans[idx];
    };

    unsigned int mining_workers = resolveSubwordMiningWorkerCount(
        enable_parallel_subword_mining, subword_mining_workers, num_texts_to_process);
    if (prune_during_mining && mining_workers > 1) {
        std::cout << "[UnigramLM] prune_during_mining=true; using single-thread mining to preserve pruning semantics"
                  << std::endl;
        mining_workers = 1;
    }

    std::cout << "[UnigramLM] Mining subwords from " << num_texts_to_process
              << " documents (workers=" << mining_workers
              << ", max_len=" << max_len << ")..." << std::endl;
    const auto mining_start = std::chrono::steady_clock::now();

    if (mining_workers <= 1) {
        for (size_t ti = 0; ti < num_texts_to_process; ++ti) {
            const std::string& text = sentenceForIndex(ti);

            if (ti % progress_interval == 0) {
                std::cout << "[UnigramLM] Subword mining: " << ti << "/" << num_texts_to_process
                          << " (" << (100 * ti / std::max<size_t>(1, num_texts_to_process)) << "%), "
                          << subword_counts.size() << " unique subwords" << std::endl;
            }

            mineSubwordsFromSentence(text, max_len, sentenceAtomsForIndex(ti), subword_counts);

            if (prune_during_mining && subword_counts.size() > 50000000) {
                std::cout << "[UnigramLM] Pruning low-frequency subwords to control memory..." << std::endl;
                for (auto it = subword_counts.begin(); it != subword_counts.end(); ) {
                    if (it->second < 3) {
                        it = subword_counts.erase(it);
                    } else {
                        ++it;
                    }
                }
                std::cout << "[UnigramLM] After pruning: " << subword_counts.size() << " subwords" << std::endl;
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

        std::vector<std::unordered_map<std::string, int>> local_counts(mining_workers);
        constexpr size_t kReservePerWorker = 6000000;
        for (auto& map : local_counts) {
            map.reserve(kReservePerWorker);
        }
        constexpr size_t kLocalPruneHighWater = 5000000;
        constexpr int    kLocalPruneMinFreq   = 2;

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
                        mineSubwordsFromSentence(sentenceForIndex(ti), max_len, sentenceAtomsForIndex(ti), local);
                    }
                    if (local.size() > kLocalPruneHighWater) {
                        for (auto it = local.begin(); it != local.end(); ) {
                            if (it->second < kLocalPruneMinFreq)
                                it = local.erase(it);
                            else
                                ++it;
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

        auto mergeInto = [](std::unordered_map<std::string, int>& dst,
                            std::unordered_map<std::string, int>& src) {
            dst.reserve(dst.size() + src.size());
            for (auto& [k, v] : src) {
                auto [it, inserted] = dst.try_emplace(k, v);
                if (!inserted) it->second += v;
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

            std::vector<std::unordered_map<std::string, int>> next_round;
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
    
    // Step 4: Add top-K most frequent subwords up to target_vocab_size
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

    using SubwordEntry = std::pair<const std::string, int>;
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
    int prefix_extension_rejected = 0;
    // Seed with 3x target vocab for iterative shrinking (SentencePiece-style).
    // Starting oversized then pruning by marginal likelihood contribution ensures
    // every surviving token genuinely earns its slot → higher entropy, better compression.
    constexpr int SEED_MULTIPLIER = 3;
    const int seed_vocab_size = target_vocab_size > 0 ? SEED_MULTIPLIER * target_vocab_size : std::numeric_limits<int>::max();
    const int max_to_add = seed_vocab_size;

    std::unordered_set<std::string> dedup_keys_seen;
    std::unordered_map<std::string, int> dedup_key_to_count;
    for (const auto& ch : char_seeds) {
        std::string key = structuralDedupKeyForCandidate(ch);
        if (!key.empty()) {
            dedup_keys_seen.insert(key);
            dedup_key_to_count[key] = 0;  // char seeds have no meaningful count
        }
    }
    std::cout << "[UnigramLM] Pre-seeded " << dedup_keys_seen.size()
              << " structural dedup keys from byte-layer chars" << std::endl;

    struct AcceptedPiece { std::string text; int count; };
    std::vector<AcceptedPiece> accepted;
    accepted.reserve(std::min<size_t>(max_to_add, ranked_subwords.size()));

    for (const SubwordEntry* entry : ranked_subwords) {
        const std::string& subword = entry->first;
        const int count = entry->second;
        if (static_cast<int>(accepted.size()) >= max_to_add) break;
        if (hasPiece(subword)) continue;
        if (char_seeds.count(subword)) continue;

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
        // Reject candidates that are 1-3 char extensions of an already-accepted
        // piece with the same corpus count (e.g., "an▁also" when "an▁als" exists
        // at the same frequency — they co-occur in identical contexts).
        if (isPrefixExtensionDuplicate(dedup_key, count, dedup_keys_seen, dedup_key_to_count)) {
            prefix_extension_rejected++;
            continue;
        }
        dedup_keys_seen.insert(dedup_key);
        dedup_key_to_count[dedup_key] = count;

        accepted.push_back({dedup_key, count});
    }

    double total_accepted_count = 0.0;
    for (const auto& ap : accepted) total_accepted_count += ap.count;
    if (total_accepted_count < 1.0) total_accepted_count = 1.0;

    for (const auto& ap : accepted) {
        float score = static_cast<float>(std::log(ap.count / total_accepted_count));
        addPiece(ap.text, score, false);
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
                  << " candidates (structural edge-trim dedup only)" << std::endl;
    }
    if (prefix_extension_rejected > 0) {
        std::cout << "[UnigramLM] Rejected " << prefix_extension_rejected
                  << " candidates (prefix-extension dedup, same-count near-duplicates)" << std::endl;
    }
    std::cout << "[UnigramLM] Added " << added << " subwords (min_freq=" << MIN_SUBWORD_FREQ 
              << ", target=" << target_vocab_size << "), total vocab: " << pieces_.size() << std::endl;
    
    // Step 5: Iterative EM + Shrinking (SentencePiece-style)
    //
    // Starting with an oversized seed vocab (3x target), we iteratively:
    //   1. Run EM to convergence (estimate token log-probabilities)
    //   2. Compute each token's marginal log-likelihood contribution
    //   3. Remove the bottom 25% of tokens (lowest contribution)
    //   4. Repeat until vocab <= target size
    //
    // This ensures every surviving token genuinely earns its slot by
    // contributing to compression — directly improving entropy, bytes/token,
    // and fertility vs. the old top-K-by-frequency approach.
    constexpr int    EM_MAX_ITERATIONS     = 50;
    constexpr double EM_CONVERGENCE_THRESH = 0.0001;
    constexpr double SMOOTHING             = 0.1;
    constexpr float  SHRINK_KEEP_RATIO     = 0.75f;  // Keep 75% each round

    auto runEStep = [&]() -> std::tuple<std::unordered_map<int, double>, double, double> {
        std::unordered_map<int, double> token_counts;
        double total_tokens = 0.0;
        double log_likelihood = 0.0;

        for (size_t text_idx = 0; text_idx < norm_texts.size(); ++text_idx) {
            const auto& text = norm_texts[text_idx];
            const auto& spans = norm_atom_spans[text_idx];
            if (text.empty()) continue;

            auto processSegment = [&](const std::string& segment) {
                if (segment.empty()) return;
                auto nodes = viterbi(segment);
                log_likelihood += nodes.back().score;
                auto tokens = backtrack(nodes, static_cast<int>(segment.size()));
                for (int token_id : tokens) {
                    token_counts[token_id] += 1.0;
                    total_tokens += 1.0;
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
        return {std::move(token_counts), total_tokens, log_likelihood};
    };

    auto runMStep = [&](const std::unordered_map<int, double>& token_counts, double total_tokens) -> int {
        double smoothed_total = total_tokens + SMOOTHING * static_cast<double>(pieces_.size());
        int zero_count = 0;
        for (size_t i = 0; i < pieces_.size(); ++i) {
            auto& piece = pieces_[i];
            if (piece.is_user_defined) continue;
            int tid = tokenIdForIndex(static_cast<int>(i));
            double count = (token_counts.count(tid) ? token_counts.at(tid) : 0.0) + SMOOTHING;
            piece.score = static_cast<float>(std::log(count / smoothed_total));
            if (!token_counts.count(tid) || token_counts.at(tid) == 0.0) {
                zero_count++;
            }
        }
        return zero_count;
    };

    auto runEMToConvergence = [&](const char* phase_label) -> std::pair<std::unordered_map<int, double>, int> {
        double prev_ll = -1e30;
        std::unordered_map<int, double> last_counts;
        int iter = 0;
        for (; iter < EM_MAX_ITERATIONS; ++iter) {
            buildTrie();
            auto [token_counts, total_tokens, log_likelihood] = runEStep();
            int unused = runMStep(token_counts, total_tokens);

            double relative_change = (prev_ll < -1e20)
                ? 1.0
                : std::abs((log_likelihood - prev_ll) / std::min(std::abs(prev_ll), std::abs(log_likelihood)));

            std::cout << "[UnigramLM] " << phase_label << " iter " << (iter + 1)
                      << ": LL=" << std::fixed << std::setprecision(2) << log_likelihood
                      << ", tokens=" << static_cast<int64_t>(total_tokens)
                      << ", unused=" << unused
                      << ", delta=" << std::scientific << std::setprecision(4) << relative_change
                      << std::defaultfloat << std::endl;

            last_counts = std::move(token_counts);
            bool converged = (iter > 0 && relative_change < EM_CONVERGENCE_THRESH);
            prev_ll = log_likelihood;
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
        return {std::move(last_counts), iter};
    };

    // Helper: rebuild piece_to_id_ from pieces_ after any mutation
    auto rebuildPieceIndex = [&]() {
        piece_to_id_.clear();
        for (size_t i = 0; i < pieces_.size(); ++i) {
            piece_to_id_[pieces_[i].text] = static_cast<int>(i);
        }
    };

    // ---- Phase A: initial EM to convergence on seed vocab ----
    std::cout << "[UnigramLM] Phase-A: EM on seed vocab (" << pieces_.size()
              << " pieces, target=" << target_vocab_size << ")" << std::endl;
    auto [phase_a_counts, phase_a_iters] = runEMToConvergence("Phase-A");

    // ---- Phase B: iterative shrinking to target vocab size ----
    // Each round removes the bottom 25% of tokens by marginal log-likelihood
    // contribution: loss_i = count_i * score_i (always <= 0, closest to 0 = least useful).
    // Tokens that are never or rarely selected by Viterbi, or have very low scores,
    // contribute almost nothing to overall compression and are pruned first.
    int shrink_round = 0;
    while (static_cast<int>(pieces_.size()) > target_vocab_size) {
        shrink_round++;

        // How many to keep this round (at least target_vocab_size)
        int current_size = static_cast<int>(pieces_.size());
        int keep_count = std::max(
            target_vocab_size,
            static_cast<int>(current_size * SHRINK_KEEP_RATIO)
        );

        // Compute marginal log-likelihood contribution for each token
        // loss_i = count_i * score_i (both <= 0 terms, so product >= 0 inverted)
        // We use |count * score| so higher = more valuable
        struct TokenValue {
            int index;
            double value;  // |count * score| — higher means more valuable
        };
        std::vector<TokenValue> token_values;
        token_values.reserve(pieces_.size());

        for (size_t i = 0; i < pieces_.size(); ++i) {
            if (pieces_[i].is_user_defined || pieces_[i].is_special) {
                // Protected tokens get infinite value — never pruned
                token_values.push_back({static_cast<int>(i), std::numeric_limits<double>::max()});
                continue;
            }
            int tid = tokenIdForIndex(static_cast<int>(i));
            double count = phase_a_counts.count(tid) ? phase_a_counts.at(tid) : 0.0;
            double score = static_cast<double>(pieces_[i].score);
            // Marginal LL contribution = count * score (score is negative log-prob)
            // Tokens with count=0 or tiny count*|score| contribute least
            double marginal_ll = count * std::abs(score);
            token_values.push_back({static_cast<int>(i), marginal_ll});
        }

        // Sort by value descending — most valuable first
        std::sort(token_values.begin(), token_values.end(),
                  [](const TokenValue& a, const TokenValue& b) {
                      return a.value > b.value;
                  });

        // Keep the top keep_count tokens
        std::unordered_set<int> keep_indices;
        for (int k = 0; k < keep_count && k < static_cast<int>(token_values.size()); ++k) {
            keep_indices.insert(token_values[k].index);
        }

        int removed = current_size - static_cast<int>(keep_indices.size());
        std::cout << "[UnigramLM] Shrink round " << shrink_round
                  << ": " << current_size << " -> " << keep_indices.size()
                  << " tokens (removed " << removed << ")" << std::endl;

        // Compact pieces_ to survivors only
        std::vector<UnigramPiece> surviving;
        surviving.reserve(keep_indices.size());
        for (size_t i = 0; i < pieces_.size(); ++i) {
            if (keep_indices.count(static_cast<int>(i))) {
                surviving.push_back(std::move(pieces_[i]));
            }
        }
        pieces_ = std::move(surviving);
        rebuildPieceIndex();

        // Re-converge EM on the smaller vocab
        std::string label = "Shrink-" + std::to_string(shrink_round);
        auto [shrink_counts, shrink_iters] = runEMToConvergence(label.c_str());
        phase_a_counts = std::move(shrink_counts);
    }

    if (shrink_round > 0) {
        std::cout << "[UnigramLM] Iterative shrinking complete after " << shrink_round
                  << " rounds. Final vocab: " << pieces_.size() << " pieces" << std::endl;
    }

    // ---- Phase C: final dead-token cleanup ----
    // Remove any tokens that still have zero Viterbi count after convergence
    int pruned = 0;
    {
        std::unordered_set<int> dead_indices;
        for (size_t i = 0; i < pieces_.size(); ++i) {
            if (pieces_[i].is_user_defined) continue;
            int tid = tokenIdForIndex(static_cast<int>(i));
            const double count = phase_a_counts.count(tid) ? phase_a_counts.at(tid) : 0.0;
            if (count < 1.0) {
                dead_indices.insert(static_cast<int>(i));
            }
        }

        if (!dead_indices.empty()) {
            std::cout << "[UnigramLM] Final cleanup: pruning " << dead_indices.size()
                      << " dead tokens (Viterbi count < 1)" << std::endl;

            std::vector<UnigramPiece> surviving;
            surviving.reserve(pieces_.size() - dead_indices.size());
            for (size_t i = 0; i < pieces_.size(); ++i) {
                if (!dead_indices.count(static_cast<int>(i))) {
                    surviving.push_back(std::move(pieces_[i]));
                }
            }
            pieces_ = std::move(surviving);
            pruned = static_cast<int>(dead_indices.size());
            rebuildPieceIndex();
        } else {
            std::cout << "[UnigramLM] No dead tokens after shrinking — vocab is clean" << std::endl;
        }
    }

    // ---- Phase D: final reconvergence after cleanup ----
    if (pruned > 0) {
        std::cout << "[UnigramLM] Reconverging after pruning " << pruned
                  << " dead tokens (vocab now " << pieces_.size() << ")" << std::endl;
        auto [phase_d_counts, phase_d_iters] = runEMToConvergence("Phase-D");
        (void)phase_d_counts;
        (void)phase_d_iters;
    }

    // Final trie build with converged scores
    buildTrie();
    
    std::cout << "[UnigramLM] Training complete. Final vocab size: " << pieces_.size() << std::endl;
    return true;
}

} // namespace Tokenizer
} // namespace GRIM
