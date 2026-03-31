//======================================================//
//  Unigram.cu
//  CUDA implementation of Unigram Language Model tokenizer
//======================================================//

#include "Unigram.hpp"
#include "Byte.hpp"
#include "HyperParameters/HyperParameters_GPU.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
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

// Check if character is punctuation (ASCII subset for speed)
static inline bool isPunct(char c) {
    return (c >= '!' && c <= '/') ||  // !"#$%&'()*+,-./
           (c >= ':' && c <= '@') ||  // :;<=>?@
           (c >= '[' && c <= '`') ||  // [\]^_`
           (c >= '{' && c <= '~');    // {|}~
}

// Punctuation isolation guard for Viterbi.
// Returns true if the character at the given position is a punctuation character
// that should be tokenized in isolation (never merged with adjacent letters/digits).
// Used in both CPU Viterbi and GPU kernel to enforce punctuation boundary splitting.
// Note: space (0x20) is NOT punctuation — it's a normal character that can appear
// in multi-word vocab tokens like "of the".
__host__ __device__ static inline bool isPunctBoundary(unsigned char c) {
    return (c >= '!' && c <= '/') ||  // !"#$%&'()*+,-./
           (c >= ':' && c <= '@') ||  // :;<=>?@
           (c >= '[' && c <= '`') ||  // [\]^_`
           (c >= '{' && c <= '~');    // {|}~
}

static inline bool isWhitespaceASCII(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

static size_t utf8SequenceLength(unsigned char c) {
    if ((c & 0x80) == 0x00) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

// Decode one UTF-8 codepoint at byte index `pos`. Returns false if truncated or ill-formed.
static bool utf8DecodeAt(const std::string& s, size_t pos, uint32_t* out_cp, size_t* out_len) {
    if (pos >= s.size()) return false;
    unsigned char b0 = static_cast<unsigned char>(s[pos]);
    if ((b0 & 0x80) == 0) {
        *out_cp = b0;
        *out_len = 1;
        return true;
    }
    const size_t n = utf8SequenceLength(b0);
    if (n < 2 || pos + n > s.size()) return false;
    for (size_t k = 1; k < n; ++k) {
        if ((static_cast<unsigned char>(s[pos + k]) & 0xC0) != 0x80) return false;
    }
    uint32_t cp = 0;
    if (n == 2) {
        unsigned char b1 = static_cast<unsigned char>(s[pos + 1]);
        cp = ((b0 & 0x1Fu) << 6) | (b1 & 0x3Fu);
        if (cp < 0x80u) return false;
    } else if (n == 3) {
        unsigned char b1 = static_cast<unsigned char>(s[pos + 1]);
        unsigned char b2 = static_cast<unsigned char>(s[pos + 2]);
        cp = ((b0 & 0x0Fu) << 12) | ((b1 & 0x3Fu) << 6) | (b2 & 0x3Fu);
        if (cp < 0x800u) return false;
        if (cp >= 0xD800u && cp <= 0xDFFFu) return false;
    } else if (n == 4) {
        unsigned char b1 = static_cast<unsigned char>(s[pos + 1]);
        unsigned char b2 = static_cast<unsigned char>(s[pos + 2]);
        unsigned char b3 = static_cast<unsigned char>(s[pos + 3]);
        cp = ((b0 & 0x07u) << 18) | ((b1 & 0x3Fu) << 12) | ((b2 & 0x3Fu) << 6) | (b3 & 0x3Fu);
        if (cp < 0x10000u || cp > 0x10FFFFu) return false;
    } else {
        return false;
    }
    *out_cp = cp;
    *out_len = n;
    return true;
}

// Codepoints stripped only at substring edges for structural vocab dedup (training).
// Covers ASCII whitespace, Unicode Zs, line/paragraph separators, ZWSP/bidi at boundaries, BOM.
static bool isStructuralEdgeWhitespace(uint32_t cp) {
    if (cp <= 0x20u) {
        return cp == 0x09u || cp == 0x0Au || cp == 0x0Bu || cp == 0x0Cu || cp == 0x0Du || cp == 0x20u;
    }
    if (cp == 0x85u) return true;   // NEL
    if (cp == 0xA0u) return true;  // NBSP
    if (cp == 0x1680u) return true; // Ogham space mark
    if (cp >= 0x2000u && cp <= 0x200Au) return true; // General punctuation spaces
    if (cp >= 0x200Bu && cp <= 0x200Fu) return true; // ZWSP, ZWNJ, ZWJ, marks
    if (cp == 0x2028u || cp == 0x2029u) return true; // Line / paragraph separator
    if (cp == 0x202Fu) return true; // Narrow no-break space
    if (cp == 0x205Fu) return true; // Medium mathematical space
    if (cp == 0x2060u) return true;  // Word joiner
    if (cp == 0x3000u) return true;  // Ideographic space
    if (cp == 0xFEFFu) return true;  // BOM / ZWNBSP
    return false;
}

//======================================================//
//  SentencePiece-style Whitespace Normalization
//  Replaces spaces with ▁ (U+2581) and prepends ▁ at the start.
//  This makes word-boundary spaces part of vocab pieces ("▁the",
//  "▁and") instead of falling back to byte token 36.
//======================================================//

// U+2581 LOWER ONE EIGHTH BLOCK — the SentencePiece word-boundary marker
static constexpr const char SPIECE_UNDERLINE[] = "\xE2\x96\x81";
static constexpr size_t SPIECE_UNDERLINE_LEN = 3;

// Normalize text for tokenization: prepend ▁, replace all spaces with ▁.
// "Hello World" → "▁Hello▁World"
static std::string normalizeSpaces(const std::string& text) {
    if (text.empty()) return text;

    size_t space_count = 0;
    for (char c : text) {
        if (c == ' ') ++space_count;
    }

    std::string result;
    result.reserve(text.size() + SPIECE_UNDERLINE_LEN + space_count * 2);
    result.append(SPIECE_UNDERLINE, SPIECE_UNDERLINE_LEN);

    for (char c : text) {
        if (c == ' ') {
            result.append(SPIECE_UNDERLINE, SPIECE_UNDERLINE_LEN);
        } else {
            result.push_back(c);
        }
    }
    return result;
}

// Reverse normalization: replace ▁ → space, strip leading space.
// "▁Hello▁World" → "Hello World"
static std::string denormalizeSpaces(const std::string& text) {
    std::string result;
    result.reserve(text.size());

    for (size_t i = 0; i < text.size(); ) {
        if (i + SPIECE_UNDERLINE_LEN <= text.size() &&
            static_cast<unsigned char>(text[i])   == 0xE2 &&
            static_cast<unsigned char>(text[i+1]) == 0x96 &&
            static_cast<unsigned char>(text[i+2]) == 0x81) {
            result.push_back(' ');
            i += SPIECE_UNDERLINE_LEN;
        } else {
            result.push_back(text[i]);
            ++i;
        }
    }

    // Strip leading space (from the prepended ▁)
    if (!result.empty() && result[0] == ' ') {
        result.erase(0, 1);
    }
    return result;
}

// Normalize text and adjust atom span byte offsets to match.
// Each space (1 byte) expands to ▁ (3 bytes), plus 3-byte prepend.
static std::string normalizeWithSpans(const std::string& text,
                                      std::vector<Tokenizer::AtomSpan>& spans) {
    if (text.empty()) return text;

    // Build orig→norm byte offset mapping
    std::vector<size_t> orig_to_norm(text.size() + 1);
    size_t norm_pos = SPIECE_UNDERLINE_LEN; // 3-byte prepend

    for (size_t i = 0; i < text.size(); ++i) {
        orig_to_norm[i] = norm_pos;
        norm_pos += (text[i] == ' ') ? SPIECE_UNDERLINE_LEN : 1;
    }
    orig_to_norm[text.size()] = norm_pos;

    for (auto& span : spans) {
        size_t new_start = (span.start <= text.size()) ? orig_to_norm[span.start] : norm_pos;
        size_t new_end   = (span.end   <= text.size()) ? orig_to_norm[span.end]   : norm_pos;
        span.start = new_start;
        span.end   = new_end;
    }

    return normalizeSpaces(text);
}

// Public static method wrappers (accessible from other TUs via Unigram.hpp)
std::string Tokenizer::UnigramLM::normalizeForTokenization(const std::string& text) {
    return normalizeSpaces(text);
}
std::string Tokenizer::UnigramLM::denormalizeFromTokenization(const std::string& text) {
    return denormalizeSpaces(text);
}

//======================================================//
//  Character-Level Validator
//  Rejects garbage characters BEFORE they enter the vocab.
//  Applied to single-character seeds in Step 2 of training.
//  Without this, control chars, BOM, zero-width spaces, NBSP,
//  soft hyphens etc. get vocab slots and cascade into subwords.
//======================================================//

static bool isValidVocabCharacter(const std::string& ch) {
    if (ch.empty()) return false;

    // Single-byte ASCII path
    if (ch.size() == 1) {
        unsigned char c = static_cast<unsigned char>(ch[0]);
        // Printable ASCII (0x20-0x7E) — space through tilde
        if (c >= 0x20 && c <= 0x7E) return true;
        // Common whitespace: tab (0x09), newline (0x0A), carriage return (0x0D)
        if (c == 0x09 || c == 0x0A || c == 0x0D) return true;
        // Everything else (0x00-0x08, 0x0B-0x0C, 0x0E-0x1F, 0x7F) is control garbage
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
        // Invalid UTF-8 sequence
        return false;
    }

    // Reject known garbage Unicode codepoints:
    // U+00A0  Non-breaking space (HTML &nbsp;)
    // U+00AD  Soft hyphen (invisible layout hint)
    // U+00B7  Middle dot (often OCR/encoding artifact)
    // U+034F  Combining grapheme joiner
    // U+200B  Zero-width space
    // U+200C  Zero-width non-joiner
    // U+200D  Zero-width joiner
    // U+200E  Left-to-right mark
    // U+200F  Right-to-left mark
    // U+202A-202E  Bidi embedding/override controls
    // U+2060  Word joiner
    // U+2066-2069  Bidi isolate controls
    // U+FEFF  BOM / zero-width no-break space
    // U+FFF0-FFFF  Specials block (includes replacement char U+FFFD)
    // U+E0000-E007F  Tags block
    // U+0080-009F  C1 control characters
    if (codepoint >= 0x0080 && codepoint <= 0x009F) return false;  // C1 controls
    if (codepoint == 0x00A0) return false;  // NBSP
    if (codepoint == 0x00AD) return false;  // Soft hyphen
    if (codepoint == 0x034F) return false;  // Combining grapheme joiner
    if (codepoint >= 0x200B && codepoint <= 0x200F) return false;  // Zero-width + bidi marks
    if (codepoint >= 0x202A && codepoint <= 0x202E) return false;  // Bidi controls
    if (codepoint == 0x2060) return false;  // Word joiner
    if (codepoint >= 0x2066 && codepoint <= 0x2069) return false;  // Bidi isolates
    if (codepoint == 0xFEFF) return false;  // BOM
    if (codepoint >= 0xFFF0 && codepoint <= 0xFFFF) return false;  // Specials
    if (codepoint >= 0xE0000 && codepoint <= 0xE007F) return false;  // Tags
    // Reject private use areas (vendor-specific garbage)
    if (codepoint >= 0xE000 && codepoint <= 0xF8FF) return false;    // BMP private use
    if (codepoint >= 0xF0000 && codepoint <= 0x10FFFF) return false; // Supplementary private use
    // Reject surrogates (invalid in UTF-8)
    if (codepoint >= 0xD800 && codepoint <= 0xDFFF) return false;

    // Everything else (Latin Extended, common symbols, CJK, emoji, etc.) is OK
    return true;
}

// Canonical structural form for vocab *candidate selection* during training.
// Strips leading/trailing UTF-8 whitespace / boundary format chars, with an ASCII
// byte fallback when a leading/trailing byte is not the start of well-formed UTF-8
// (avoids getting stuck on messy corpus edges). Interior bytes and case unchanged.
// Callers store this string as the piece text so "this", " this", "this " do not
// occupy multiple trie entries.
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

// Reject long repeated runs (e.g., "aaaaaa", "!!!!!!", "word  word")
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

// Reject repeated whole-pattern tokens like "hahaha", "abcabcabc", "121212"
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

// Reject long doubled tokens like "wordword" or "testtest"
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

// Reject stutter-like tokens where the same word repeats 3+ times ("i i i")
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

// Encoding-safety gate for candidate subwords.
// Rejects control bytes, invalid characters, pure whitespace, and repetition noise.
static bool isValidSubword(const std::string& s) {
    if (s.empty()) return false;

    // Single characters: validate instead of blindly accepting.
    // Without this, control chars and garbage Unicode enter the vocab.
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

static unsigned int resolveSubwordMiningWorkerCount(
    bool enable_parallel_subword_mining,
    int configured_workers,
    size_t sentence_count) {
    if (!enable_parallel_subword_mining) return 1;
    if (sentence_count < 1024) return 1;

    unsigned int workers = std::thread::hardware_concurrency();
    if (workers == 0) workers = 4;
    // Leave 2 logical threads for the OS and IO — use the rest for mining.
    workers = workers > 2 ? workers - 2 : 1;
    // No artificial cap — hardware_concurrency already reflects the physical limit.
    // (i7-12700F = 20 logical threads → 18 mining workers)

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

static void mineSubwordsFromSentence(const std::string& text,
                                     size_t max_len,
                                     std::unordered_map<std::string, int>& subword_counts) {
    if (text.empty()) return;

    // Build UTF-8 character boundaries once per sentence.
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
            if (byte_end - byte_start > 64) continue;  // Skip likely garbage spans

            std::string subword = text.substr(byte_start, byte_end - byte_start);
            if (!isValidSubword(subword)) continue;

            // Cross-boundary fragment rejection: if the subword contains an
            // internal space (spans across a word boundary), it must start
            // and end at word boundaries in the source text.  Otherwise we
            // get fragments like "semi-major ax" mined from "semi-major axis"
            // where "axis" is truncated to "ax".
            if (subword.find(' ') != std::string::npos) {
                bool starts_mid_word = (byte_start > 0 &&
                    !isWhitespaceASCII(static_cast<unsigned char>(text[byte_start])) &&
                    !isWhitespaceASCII(static_cast<unsigned char>(text[byte_start - 1])));
                bool ends_mid_word = (byte_end < text.size() &&
                    !isWhitespaceASCII(static_cast<unsigned char>(text[byte_end - 1])) &&
                    !isWhitespaceASCII(static_cast<unsigned char>(text[byte_end])));
                if (starts_mid_word || ends_mid_word) continue;
            }

            subword_counts[subword]++;
        }
    }
}

// Atom-aware overload: skip subwords that START inside an atom span.
// Atom spans are sorted by start offset (guaranteed by detectStructures).
static void mineSubwordsFromSentence(const std::string& text,
                                     size_t max_len,
                                     const std::vector<Tokenizer::AtomSpan>& atom_spans,
                                     std::unordered_map<std::string, int>& subword_counts) {
    if (text.empty()) return;
    if (atom_spans.empty()) {
        // Fast path: no atoms, use original logic
        mineSubwordsFromSentence(text, max_len, subword_counts);
        return;
    }

    // Build UTF-8 character boundaries once per sentence.
    std::vector<size_t> char_positions;
    char_positions.reserve(text.size() + 1);
    for (size_t i = 0; i < text.size(); ) {
        char_positions.push_back(i);
        i += utf8SequenceLength(static_cast<unsigned char>(text[i]));
    }
    char_positions.push_back(text.size());

    // Build a fast lookup: for each byte position, is it inside an atom span?
    // Use sorted spans with binary search for O(log N) per position.
    // But simpler: precompute a skip flag per character index.
    std::vector<bool> char_in_atom(char_positions.size() - 1, false);
    size_t span_idx = 0;
    for (size_t ci = 0; ci < char_positions.size() - 1; ++ci) {
        const size_t byte_pos = char_positions[ci];
        // Advance span_idx past spans that end before this position
        while (span_idx < atom_spans.size() && atom_spans[span_idx].end <= byte_pos) {
            ++span_idx;
        }
        // Check if current position falls inside the current span
        if (span_idx < atom_spans.size() &&
            byte_pos >= atom_spans[span_idx].start &&
            byte_pos < atom_spans[span_idx].end) {
            char_in_atom[ci] = true;
        }
    }

    const size_t num_chars = char_positions.size() - 1;
    for (size_t ci = 0; ci < num_chars; ++ci) {
        // Skip characters that start inside an atom span
        if (char_in_atom[ci]) continue;

        const size_t byte_start = char_positions[ci];
        for (size_t char_count = 2; char_count <= max_len && ci + char_count <= num_chars; ++char_count) {
            // If any character in this subword range is inside an atom, skip
            bool crosses_atom = false;
            for (size_t k = ci + 1; k < ci + char_count; ++k) {
                if (char_in_atom[k]) { crosses_atom = true; break; }
            }
            if (crosses_atom) break;  // All longer spans will also cross

            const size_t byte_end = char_positions[ci + char_count];
            if (byte_end - byte_start > 64) continue;

            std::string subword = text.substr(byte_start, byte_end - byte_start);
            if (!isValidSubword(subword)) continue;

            // Cross-boundary fragment rejection (same logic as non-atom overload).
            if (subword.find(' ') != std::string::npos) {
                bool starts_mid_word = (byte_start > 0 &&
                    !isWhitespaceASCII(static_cast<unsigned char>(text[byte_start])) &&
                    !isWhitespaceASCII(static_cast<unsigned char>(text[byte_start - 1])));
                bool ends_mid_word = (byte_end < text.size() &&
                    !isWhitespaceASCII(static_cast<unsigned char>(text[byte_end - 1])) &&
                    !isWhitespaceASCII(static_cast<unsigned char>(text[byte_end])));
                if (starts_mid_word || ends_mid_word) continue;
            }

            subword_counts[subword]++;
        }
    }
}

//======================================================//
namespace Tokenizer {

//======================================================//
//  CUDA Kernels
//======================================================//

// Kernel: Sequential Viterbi forward pass
// CRITICAL: Viterbi has O(n) sequential dependency - each position depends on ALL previous.
// Parallel execution would cause data races (reading viterbi_scores before they're computed).
// We parallelize the TRIE SEARCH within each position, not across positions.
__global__ void kernelViterbiForward(
    const char* __restrict__ text,
    size_t length,
    const int* __restrict__ trie_children,    // [num_nodes * 256]
    const int* __restrict__ trie_token_ids,   // [num_nodes]
    const float* __restrict__ trie_scores,    // [num_nodes]
    int num_trie_nodes,
    float* __restrict__ viterbi_scores,       // [length + 1]
    int* __restrict__ viterbi_prev,           // [length + 1]
    int* __restrict__ viterbi_tokens,         // [length + 1]
    bool* __restrict__ needs_fallback,        // [length]
    int unk_id,
    float unk_score,
    bool enable_byte_fallback
) {
    // Single thread processes positions SEQUENTIALLY to maintain Viterbi invariants
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    // Initialize position 0
    viterbi_scores[0] = 0.0f;
    viterbi_prev[0] = -1;
    viterbi_tokens[0] = -1;
    
    // Process each position sequentially (required for correctness)
    for (size_t pos = 1; pos <= length; ++pos) {
        unsigned char cur_byte = static_cast<unsigned char>(text[pos - 1]);
        bool cur_is_punct = isPunctBoundary(cur_byte);
        
        // PUNCTUATION ISOLATION GUARD:
        // If this position's character is punctuation, force it to be a single
        // byte token. Skip trie matching entirely — punctuation is never merged
        // with adjacent letters/digits.
        if (cur_is_punct) {
            // Emit as byte token: BYTE_TOKEN_OFFSET + byte_value
            viterbi_scores[pos] = viterbi_scores[pos - 1] + unk_score;
            viterbi_prev[pos] = static_cast<int>(pos - 1);
            viterbi_tokens[pos] = static_cast<int>(cur_byte) + 4;  // BYTE_TOKEN_OFFSET = 4
            continue;
        }
        
        float best_score = -1e30f;
        int best_prev = -1;
        int best_token = unk_id;
        bool found_match = false;
        
        // Try all possible pieces ending at position `pos`
        // Walk backwards from pos, traversing trie
        int node = 0;  // Start at trie root
        
        for (size_t start = pos; start > 0 && (pos - start) < MAX_PIECE_LENGTH; --start) {
            size_t idx = start - 1;
            unsigned char c = static_cast<unsigned char>(text[idx]);
            
            // PUNCTUATION ISOLATION GUARD:
            // Stop backward walk if we hit a punctuation character —
            // pieces must not span across a punctuation boundary.
            if (isPunctBoundary(c)) break;
            
            // Navigate trie
            int child = trie_children[node * 256 + c];
            if (child < 0) break;  // No path in trie
            
            node = child;
            
            // Check if this node is end of a token
            int token_id = trie_token_ids[node];
            if (token_id >= 0) {
                float piece_score = trie_scores[node];
                // Safe: viterbi_scores[start - 1] was computed in previous iteration
                float candidate_score = viterbi_scores[start - 1] + piece_score;
                
                if (candidate_score > best_score) {
                    best_score = candidate_score;
                    best_prev = static_cast<int>(start - 1);
                    best_token = token_id;
                    found_match = true;
                }
            }
        }
        
        // If no match found, mark for byte fallback
        if (!found_match) {
            best_prev = static_cast<int>(pos - 1);
            best_token = unk_id;
            best_score = viterbi_scores[pos - 1] + unk_score;

            if (enable_byte_fallback && needs_fallback) {
                needs_fallback[pos - 1] = true;
            }
        }
        
        viterbi_scores[pos] = best_score;
        viterbi_prev[pos] = best_prev;
        viterbi_tokens[pos] = best_token;
    }
}

// Kernel: Backtrack Viterbi path
__global__ void kernelViterbiBacktrack(
    size_t length,
    const int* __restrict__ viterbi_prev,
    const int* __restrict__ viterbi_tokens,
    int* __restrict__ output_tokens,
    int* __restrict__ output_count,
    int max_tokens
) {
    // Single thread does backtracking
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    // Count tokens first
    int count = 0;
    int pos = static_cast<int>(length);
    while (pos > 0) {
        count++;
        pos = viterbi_prev[pos];
    }
    
    // Clamp to max_tokens to prevent buffer overflow
    if (count > max_tokens) {
        count = max_tokens;
    }
    
    // Write tokens in reverse
    *output_count = count;
    pos = static_cast<int>(length);
    int write_idx = count - 1;
    while (pos > 0 && write_idx >= 0) {
        output_tokens[write_idx] = viterbi_tokens[pos];
        pos = viterbi_prev[pos];
        write_idx--;
    }
}

// Kernel: Decode tokens to text
__global__ void kernelUnigramDecode(
    const int* __restrict__ token_ids,
    size_t count,
    const char* __restrict__ piece_data,
    const int* __restrict__ piece_offsets,
    const int* __restrict__ piece_lengths,
    int vocab_offset,
    int vocab_size,
    char* __restrict__ output,
    size_t* __restrict__ output_length,
    size_t max_output
) {
    // Single thread for now (could parallelize with scan)
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    size_t out_pos = 0;
    
    for (size_t i = 0; i < count && out_pos < max_output; ++i) {
        int tid = token_ids[i];
        
        // Check if it's a unigram token
        if (tid >= vocab_offset && tid < vocab_offset + vocab_size) {
            int piece_idx = tid - vocab_offset;
            int offset = piece_offsets[piece_idx];
            int len = piece_lengths[piece_idx];
            
            for (int j = 0; j < len && out_pos < max_output; ++j) {
                output[out_pos++] = piece_data[offset + j];
            }
        }
        // Byte tokens handled by caller
    }
    
    *output_length = out_pos;
}

// Kernel: Trie lookup for batch encoding
__global__ void kernelTrieLookup(
    const char* __restrict__ text,
    size_t length,
    size_t start_pos,
    const int* __restrict__ trie_children,
    const int* __restrict__ trie_token_ids,
    const float* __restrict__ trie_scores,
    int* __restrict__ match_token,
    int* __restrict__ match_length,
    float* __restrict__ match_score
) {
    // Single thread traverses trie from start_pos
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    int node = 0;
    int best_token = -1;
    int best_length = 0;
    float best_score = -1e30f;
    
    for (size_t i = start_pos; i < length && (i - start_pos) < MAX_PIECE_LENGTH; ++i) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        int child = trie_children[node * 256 + c];
        
        if (child < 0) break;
        node = child;
        
        int token_id = trie_token_ids[node];
        if (token_id >= 0) {
            float score = trie_scores[node];
            if (score > best_score) {
                best_token = token_id;
                best_length = static_cast<int>(i - start_pos + 1);
                best_score = score;
            }
        }
    }
    
    *match_token = best_token;
    *match_length = best_length;
    *match_score = best_score;
}

//======================================================//
//  UnigramLM Implementation
//======================================================//

UnigramLM::UnigramLM() 
    : gpu_(std::make_unique<GPUData>())
{
    // Start with empty vocabulary - special tokens added by load() or trainFromCorpus()
}

UnigramLM::~UnigramLM() {
    if (gpu_ && gpu_->initialized) {
        cudaFree(gpu_->d_trie_children);
        cudaFree(gpu_->d_trie_token_ids);
        cudaFree(gpu_->d_trie_scores);
        cudaFree(gpu_->d_piece_data);
        cudaFree(gpu_->d_piece_offsets);
        cudaFree(gpu_->d_piece_lengths);
        cudaFree(gpu_->d_viterbi_scores);
        cudaFree(gpu_->d_viterbi_prev);
        cudaFree(gpu_->d_viterbi_tokens);
    }
}

UnigramLM::UnigramLM(UnigramLM&& other) noexcept
    : pieces_(std::move(other.pieces_))
    , piece_to_id_(std::move(other.piece_to_id_))
    , unk_id_(other.unk_id_)
    , pad_id_(other.pad_id_)
    , bos_id_(other.bos_id_)
    , eos_id_(other.eos_id_)
    , enable_byte_fallback_(other.enable_byte_fallback_)
    , trie_(std::move(other.trie_))
    , gpu_(std::move(other.gpu_))
{
}

UnigramLM& UnigramLM::operator=(UnigramLM&& other) noexcept {
    if (this != &other) {
        pieces_ = std::move(other.pieces_);
        piece_to_id_ = std::move(other.piece_to_id_);
        unk_id_ = other.unk_id_;
        pad_id_ = other.pad_id_;
        bos_id_ = other.bos_id_;
        eos_id_ = other.eos_id_;
        enable_byte_fallback_ = other.enable_byte_fallback_;
        trie_ = std::move(other.trie_);
        gpu_ = std::move(other.gpu_);
    }
    return *this;
}

//--------------------------------------------------//
// Vocabulary Management
//--------------------------------------------------//

void UnigramLM::addPiece(const std::string& text, float score, bool is_user_defined) {
    if (piece_to_id_.count(text)) {
        // Update existing piece's score. Token ID is immutable (= UNIGRAM_VOCAB_OFFSET + index).
        int idx = piece_to_id_[text];
        pieces_[idx].score = score;
        if (!pieces_[idx].is_user_defined) {
            pieces_[idx].is_user_defined = is_user_defined;
        }
        return;
    }
    
    UnigramPiece piece;
    piece.text = text;
    piece.score = score;
    // token_id is NOT stored — it's ALWAYS (UNIGRAM_VOCAB_OFFSET + index).
    piece.is_special = (!text.empty() && text.front() == '<' && text.back() == '>');
    piece.is_user_defined = is_user_defined;
    
    piece_to_id_[text] = static_cast<int>(pieces_.size());
    pieces_.push_back(piece);
}

const UnigramPiece* UnigramLM::getPiece(int token_id) const {
    // Special tokens are not in pieces_ — they have absolute IDs 0-3
    // Callers should check isSpecialToken() separately if they need special token info
    int idx = token_id - UNIGRAM_VOCAB_OFFSET;
    if (idx < 0 || idx >= static_cast<int>(pieces_.size())) {
        return nullptr;
    }
    return &pieces_[idx];
}

int UnigramLM::getPieceId(const std::string& text) const {
    // Check special tokens by name
    if (text == "<unk>") return UNK_TOKEN_ID;
    if (text == "<pad>") return PAD_TOKEN_ID;
    if (text == "<s>")   return BOS_TOKEN_ID;
    if (text == "</s>")  return EOS_TOKEN_ID;
    
    auto it = piece_to_id_.find(text);
    if (it == piece_to_id_.end()) {
        return unk_id_;  // UNK_TOKEN_ID = 0 (absolute)
    }
    return UNIGRAM_VOCAB_OFFSET + it->second;
}

bool UnigramLM::hasPiece(const std::string& text) const {
    return piece_to_id_.count(text) > 0;
}

bool UnigramLM::load(const std::string& vocab_path) {
    std::cout << "[UnigramLM] Opening vocab file: " << vocab_path << std::endl << std::flush;
    std::ifstream file(vocab_path);
    if (!file.is_open()) {
        std::cerr << "[UnigramLM] Failed to open vocab file: " << vocab_path << std::endl;
        return false;
    }
    
    std::cout << "[UnigramLM] Clearing existing vocab..." << std::endl << std::flush;
    pieces_.clear();
    piece_to_id_.clear();
    
    // Special tokens are now at absolute IDs 0-3, NOT stored in pieces_.
    // They are handled by getPieceId() directly.
    
    std::cout << "[UnigramLM] Reading vocab pieces..." << std::endl << std::flush;
    std::string line;
    int line_count = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        line_count++;
        if (line_count == 1) {
            std::cout << "[UnigramLM] First line: " << line.substr(0, std::min(size_t(50), line.size())) << std::endl << std::flush;
        }
        
        // Format: <piece>\t<score>
        size_t tab_pos = line.rfind('\t');
        if (tab_pos == std::string::npos) {
            std::cout << "[UnigramLM] WARNING: Line " << line_count << " has no tab, skipping" << std::endl << std::flush;
            continue;
        }
        
        try {
            std::string piece = line.substr(0, tab_pos);
            std::string score_str = line.substr(tab_pos + 1);
            
            // Skip lines with empty or invalid scores
            if (score_str.empty() || score_str == "\t" || score_str == " ") {
                std::cout << "[UnigramLM] WARNING: Line " << line_count << " has empty score, skipping" << std::endl << std::flush;
                continue;
            }
            
            float score = std::stof(score_str);
            
            // Skip special tokens from file — they're handled as absolute IDs 0-3
            if (piece == "<unk>" || piece == "<pad>" || piece == "<s>" || piece == "</s>") {
                continue;
            }
            
            if (line_count == 1) {
                std::cout << "[UnigramLM] First piece: '" << piece.substr(0, std::min(size_t(30), piece.size())) 
                          << "', score: " << score << std::endl << std::flush;
                std::cout << "[UnigramLM] Calling addPiece..." << std::endl << std::flush;
            }
            
            addPiece(piece, score, false);
            
            if (line_count == 1) {
                std::cout << "[UnigramLM] First addPiece succeeded" << std::endl << std::flush;
            }
        } catch (const std::exception& e) {
            std::cout << "[UnigramLM] ERROR parsing line " << line_count << ": " << e.what() << std::endl << std::flush;
            std::cout << "[UnigramLM]   Line content: " << line.substr(0, std::min(size_t(100), line.size())) << std::endl << std::flush;
            std::cout << "[UnigramLM]   Skipping this line and continuing..." << std::endl << std::flush;
            continue;  // Skip bad lines instead of failing
        }
        
        if (line_count % 10000 == 0) {
            std::cout << "[UnigramLM] Loaded " << line_count << " pieces..." << std::endl << std::flush;
        }
    }
    
    std::cout << "[UnigramLM] Building trie..." << std::endl << std::flush;
    buildTrie();
    std::cout << "[UnigramLM] Trie built" << std::endl << std::flush;
    
    std::cout << "[UnigramLM] Loaded " << pieces_.size() << " pieces from " << vocab_path << std::endl;
    return true;
}

bool UnigramLM::loadBinary(const std::string& vocab_path) {
    std::ifstream bin_file(vocab_path, std::ios::binary);
    if (!bin_file.is_open()) {
        std::cerr << "[UnigramLM] Failed to open binary vocab file: " << vocab_path << std::endl;
        return false;
    }
    
    // Read and verify KTMG magic (4 bytes)
    char magic[4];
    bin_file.read(magic, 4);
    if (magic[0] != 'K' || magic[1] != 'T' || magic[2] != 'M' || magic[3] != 'G') {
        throw std::runtime_error("[UnigramLM] Invalid binary vocab magic header - file corrupted or not a KTMG vocab file");
    }
    
    // Read version (2 bytes) - v4 required (SentencePiece ▁ normalization)
    uint16_t version;
    bin_file.read(reinterpret_cast<char*>(&version), 2);
    if (version != 4) {
        throw std::runtime_error("[UnigramLM] Vocab file version " + std::to_string(version) + 
            " is not supported. Required version 4 (SentencePiece normalization). Retrain tokenizer.");
    }
    
    // Skip checksum (4 bytes)
    uint32_t checksum;
    bin_file.read(reinterpret_cast<char*>(&checksum), 4);
    
    // Read config vocab_size (4 bytes) - number of unigram pieces
    uint32_t config_vocab_size;
    bin_file.read(reinterpret_cast<char*>(&config_vocab_size), 4);
    
    // Skip max_length (4 bytes)
    uint32_t max_length;
    bin_file.read(reinterpret_cast<char*>(&max_length), 4);
    
    // Skip flags (3 bytes)
    char flags[3];
    bin_file.read(flags, 3);
    
    // Read total vocab size (4 bytes) - includes bytes + atoms + unigram
    uint32_t total_vocab_size;
    bin_file.read(reinterpret_cast<char*>(&total_vocab_size), 4);
    
    // Clear existing vocab
    pieces_.clear();
    piece_to_id_.clear();
    pieces_.reserve(config_vocab_size);
    
    // Read pieces: length (4 bytes) + text + score (4 bytes float) + token_id (4 bytes)
    std::vector<char> text_buffer;
    text_buffer.reserve(MAX_PIECE_LENGTH);
    
    for (uint32_t i = 0; i < config_vocab_size; ++i) {
        uint32_t len;
        bin_file.read(reinterpret_cast<char*>(&len), 4);
        
        if (len > MAX_PIECE_LENGTH) {
            throw std::runtime_error("[UnigramLM] Invalid piece length " + std::to_string(len) + 
                " at index " + std::to_string(i) + " - vocab file corrupted");
        }
        
        text_buffer.resize(len);
        bin_file.read(text_buffer.data(), len);
        std::string text(text_buffer.data(), len);
        
        float score;
        bin_file.read(reinterpret_cast<char*>(&score), 4);
        
        int token_id;
        bin_file.read(reinterpret_cast<char*>(&token_id), 4);
        
        // Skip special tokens from binary — they're at absolute IDs 0-3 now
        if (text == "<unk>" || text == "<pad>" || text == "<s>" || text == "</s>") {
            continue;
        }
        
        // Validate stored token_id matches position-derived ID.
        // If mismatch, the vocab file was produced by the buggy tokenizer.
        int expected_id = UNIGRAM_VOCAB_OFFSET + static_cast<int>(pieces_.size());
        if (token_id != expected_id) {
            throw std::runtime_error(
                "[UnigramLM] vocab.bin token_id mismatch at piece " + std::to_string(i) +
                " ('" + text.substr(0, 30) + "'): stored=" + std::to_string(token_id) +
                " expected=" + std::to_string(expected_id) +
                ". Retrain tokenizer to fix (old vocab had EM prune/backfill collision bug).");
        }
        
        addPiece(text, score, false);
    }
    
    if (!bin_file) {
        throw std::runtime_error("[UnigramLM] Error reading binary vocab file - unexpected EOF or read error");
    }
    
    buildTrie();
    
    std::cout << "[UnigramLM] Loaded " << pieces_.size() << " pieces from binary: " << vocab_path << std::endl;
    std::cout << "[UnigramLM] Embedding vocab size: " << total_vocab_size
              << " (" << NUM_SPECIAL_TOKENS << " special + " << BYTE_VOCAB_SIZE << " bytes + "
              << ATOM_VOCAB_SIZE << " atom type placeholders + "
              << pieces_.size() << " unigram pieces)" << std::endl;
    return true;
}

bool UnigramLM::save(const std::string& vocab_path, bool save_text_format) const {
    // Primary: Save binary format (.bin)
    // Binary format: KTMG magic + version + checksum + config + vocab_size + pieces
    std::string bin_path = vocab_path;
    size_t dot_pos = bin_path.rfind('.');
    if (dot_pos != std::string::npos) {
        std::string ext = bin_path.substr(dot_pos);
        if (ext != ".bin") {
            bin_path = bin_path.substr(0, dot_pos) + ".bin";
        }
    } else {
        bin_path += ".bin";
    }
    
    std::ofstream bin_file(bin_path, std::ios::binary);
    if (!bin_file.is_open()) {
        std::cerr << "[UnigramLM] Failed to create binary vocab file: " << bin_path << std::endl;
        return false;
    }
    
    // Header: KTMG magic (4 bytes)
    const char magic[4] = {'K', 'T', 'M', 'G'};
    bin_file.write(magic, 4);
    
    // Version (2 bytes) - version 4 (SentencePiece ▁-normalized pieces)
    uint16_t version = 4;
    bin_file.write(reinterpret_cast<const char*>(&version), 2);
    
    // Checksum placeholder (4 bytes) - not used currently
    uint32_t checksum = 0;
    bin_file.write(reinterpret_cast<const char*>(&checksum), 4);
    
    // Config vocab_size (4 bytes) - number of unigram pieces
    uint32_t config_vocab_size = static_cast<uint32_t>(pieces_.size());
    bin_file.write(reinterpret_cast<const char*>(&config_vocab_size), 4);
    
    // Max length (4 bytes)
    uint32_t max_length = MAX_PIECE_LENGTH;
    bin_file.write(reinterpret_cast<const char*>(&max_length), 4);
    
    // 3 bools (3 bytes) - reserved for config flags
    char flags[3] = {0, 0, 0};
    bin_file.write(flags, 3);
    
    // Actual vocab size including special+byte+atom offsets (4 bytes)
    uint32_t total_vocab_size = static_cast<uint32_t>(
        NUM_SPECIAL_TOKENS + BYTE_VOCAB_SIZE + ATOM_VOCAB_SIZE + pieces_.size());
    bin_file.write(reinterpret_cast<const char*>(&total_vocab_size), 4);
    
    // Write pieces: length (4 bytes) + text + score (4 bytes float) + token_id (4 bytes)
    // token_id is position-derived (UNIGRAM_VOCAB_OFFSET + i), written for format compat.
    for (size_t i = 0; i < pieces_.size(); ++i) {
        const auto& piece = pieces_[i];
        uint32_t len = static_cast<uint32_t>(piece.text.size());
        bin_file.write(reinterpret_cast<const char*>(&len), 4);
        bin_file.write(piece.text.data(), len);
        bin_file.write(reinterpret_cast<const char*>(&piece.score), 4);
        int tid = tokenIdForIndex(static_cast<int>(i));
        bin_file.write(reinterpret_cast<const char*>(&tid), 4);
    }
    
    bin_file.close();
    std::cout << "[UnigramLM] Saved binary vocab (" << total_vocab_size
              << " embedding vocab = " << NUM_SPECIAL_TOKENS << " special + "
              << BYTE_VOCAB_SIZE << " bytes + " << ATOM_VOCAB_SIZE
              << " atom type placeholders + " << pieces_.size()
              << " unigram pieces) to " << bin_path << std::endl;
    
    // Optional: Save text format (.txt) for human readability
    if (save_text_format) {
        std::string txt_path = bin_path.substr(0, bin_path.rfind('.')) + ".txt";
        std::ofstream txt_file(txt_path);
        if (txt_file.is_open()) {
            for (const auto& piece : pieces_) {
                txt_file << piece.text << "\t" << piece.score << "\n";
            }
            txt_file.close();
            std::cout << "[UnigramLM] Saved text vocab (human-readable) to " << txt_path << std::endl;
        } else {
            std::cerr << "[UnigramLM] Warning: Failed to create text vocab file: " << txt_path << std::endl;
        }
    }
    
    return true;
}

bool UnigramLM::trainFromCorpus(const std::vector<std::string>& texts,
                                 int target_vocab_size,
                                 float character_coverage,
                                 int min_subword_freq,
                                 bool prune_during_mining,
                                 bool enable_parallel_subword_mining,
                                 int subword_mining_workers,
                                 size_t subword_mining_max_bytes) {
    // Delegate to atom-aware overload with empty spans (no atom skipping).
    std::vector<std::vector<AtomSpan>> empty_spans(texts.size());
    return trainFromCorpus(texts, empty_spans, target_vocab_size,
                           character_coverage, min_subword_freq,
                           prune_during_mining, enable_parallel_subword_mining,
                           subword_mining_workers, subword_mining_max_bytes);
}

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
    
    // CRITICAL: Segment documents into sentences BEFORE subword mining.
    // This prevents learning garbage tokens that cross sentence boundaries
    // (e.g., ". The", "., B", "). This" which are meaningless).
    std::vector<std::string> sentences;
    std::vector<std::vector<AtomSpan>> sentence_atom_spans;  // parallel to sentences
    sentences.reserve(texts.size() * 5);  // Estimate ~5 sentences per document
    sentence_atom_spans.reserve(texts.size() * 5);
    
    // Helper: clip document-level atom spans to a sentence byte range and offset-adjust.
    // doc_spans must be sorted by start. sent_start/sent_end are byte offsets in the document.
    auto clipAtomSpans = [](const std::vector<AtomSpan>& doc_spans,
                            size_t sent_start, size_t sent_end) -> std::vector<AtomSpan> {
        std::vector<AtomSpan> result;
        for (const auto& span : doc_spans) {
            if (span.end <= sent_start) continue;   // Entirely before sentence
            if (span.start >= sent_end) break;       // Past sentence (sorted, done)
            // Clip to sentence range and offset-adjust
            size_t clipped_start = std::max(span.start, sent_start) - sent_start;
            size_t clipped_end = std::min(span.end, sent_end) - sent_start;
            if (clipped_end > clipped_start) {
                result.push_back({clipped_start, clipped_end});
            }
        }
        return result;
    };
    
    // Helper: add a sentence extracted from text[start..end) with whitespace trimming,
    // also producing the corresponding atom spans adjusted for trimming offset.
    auto addSentence = [&](const std::string& text, size_t raw_start, size_t raw_end,
                           const std::vector<AtomSpan>& doc_spans) {
        std::string sentence = text.substr(raw_start, raw_end - raw_start);
        size_t first = sentence.find_first_not_of(" \t\n\r");
        if (first != std::string::npos) {
            size_t last = sentence.find_last_not_of(" \t\n\r");
            size_t trimmed_len = last - first + 1;
            if (trimmed_len >= 3) {
                // The trimmed sentence maps to doc bytes [raw_start + first, raw_start + first + trimmed_len)
                size_t doc_sent_start = raw_start + first;
                size_t doc_sent_end = doc_sent_start + trimmed_len;
                auto spans = clipAtomSpans(doc_spans, doc_sent_start, doc_sent_end);
                sentences.push_back(sentence.substr(first, trimmed_len));
                sentence_atom_spans.push_back(std::move(spans));
            }
        }
    };
    
    // Simple sentence segmentation: split on [.!?] followed by whitespace and capital
    // Also split on newlines (paragraph boundaries)
    for (size_t text_idx = 0; text_idx < texts.size(); ++text_idx) {
        const auto& text = texts[text_idx];
        const auto& doc_spans = atom_spans[text_idx];
        if (text.empty()) continue;
        
        size_t start = 0;
        for (size_t i = 0; i < text.size(); ++i) {
            char c = text[i];
            bool is_sentence_end = false;
            
            // Check for sentence-ending punctuation followed by whitespace + capital
            if ((c == '.' || c == '!' || c == '?') && i + 2 < text.size()) {
                char next = text[i + 1];
                char after = text[i + 2];
                // Pattern: [.!?] + space + uppercase letter
                if ((next == ' ' || next == '\n' || next == '\t') && 
                    (after >= 'A' && after <= 'Z')) {
                    is_sentence_end = true;
                }
            }
            // Also split on newlines (paragraph boundaries)
            else if (c == '\n' && i > start) {
                is_sentence_end = true;
            }
            
            if (is_sentence_end) {
                // Include the punctuation in this sentence
                size_t end = (c == '\n') ? i : i + 1;
                addSentence(text, start, end, doc_spans);
                start = (c == '\n') ? i + 1 : i + 2;  // Skip past whitespace
            }
        }
        // Add remaining text as final sentence
        if (start < text.size()) {
            addSentence(text, start, text.size(), doc_spans);
        }
    }
    
    std::cout << "[UnigramLM] Segmented " << texts.size() << " documents into " 
              << sentences.size() << " sentences for subword mining" << std::endl;
    
    // SentencePiece-style whitespace normalization: replace spaces with ▁ (U+2581).
    // Applied to both full documents (for char counting + EM) and sentences (for mining).
    // Sentence segmentation ran on ORIGINAL text (to find ". T" patterns correctly),
    // now we normalize everything before any tokenizer operations.
    std::vector<std::string> norm_texts;
    std::vector<std::vector<AtomSpan>> norm_atom_spans;
    norm_texts.reserve(texts.size());
    norm_atom_spans.reserve(texts.size());
    for (size_t i = 0; i < texts.size(); ++i) {
        auto spans_copy = atom_spans[i];
        norm_texts.push_back(normalizeWithSpans(texts[i], spans_copy));
        norm_atom_spans.push_back(std::move(spans_copy));
    }
    for (size_t i = 0; i < sentences.size(); ++i) {
        sentences[i] = normalizeWithSpans(sentences[i], sentence_atom_spans[i]);
    }
    std::cout << "[UnigramLM] Applied SentencePiece whitespace normalization (space -> ▁)" << std::endl;

    // Use sentences instead of full documents for the rest of training
    const std::vector<std::string>& training_units = sentences;
    
    // Minimum frequency: subwords appearing fewer times than this are not included
    // This prevents noise from rare subwords while keeping everything that matters
    const int MIN_SUBWORD_FREQ = min_subword_freq;
    
    // Calculate total corpus size (for logging)
    size_t total_corpus_bytes = 0;
    for (const auto& text : texts) {
        total_corpus_bytes += text.size();
    }
    std::cout << "[UnigramLM] Total corpus size: " << (total_corpus_bytes / (1024*1024)) << " MB" << std::endl;
    
    // Calculate sentence corpus size for sampling
    size_t total_sentence_bytes = 0;
    for (const auto& sent : training_units) {
        total_sentence_bytes += sent.size();
    }
    
    // For large corpora, sample SENTENCES to avoid OOM during subword generation.
    // Allow runtime override from config: 0 means use compile-time default.
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
        std::cout << "[UnigramLM] Sampling " << sample_indices.size() << " sentences (" 
                  << (sampled_bytes / (1024*1024)) << " MB) for subword mining" << std::endl;
    }
    
    // Step 1: Count character frequencies (use ALL normalized texts)
    // Skip bytes inside atom spans — atom internals (://@.com digits etc.)
    // should NOT inflate character counts.
    std::unordered_map<std::string, int> char_counts;
    size_t total_chars = 0;
    size_t atom_chars_skipped = 0;
    
    for (size_t text_idx = 0; text_idx < norm_texts.size(); ++text_idx) {
        const auto& text = norm_texts[text_idx];
        const auto& spans = norm_atom_spans[text_idx];
        size_t span_i = 0;  // Current atom span index
        
        for (size_t i = 0; i < text.size(); ) {
            const size_t seq_len = utf8SequenceLength(static_cast<unsigned char>(text[i]));
            
            // Advance past atom spans that end before current position
            while (span_i < spans.size() && spans[span_i].end <= i) {
                ++span_i;
            }
            // Skip if current byte is inside an atom span
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
    
    // Special tokens are at absolute IDs 0-3, NOT stored in pieces_.
    // pieces_ contains only regular unigram vocabulary (multi-char subwords only).
    // Single chars are NOT added to pieces_ — they are covered by the byte token layer
    // [BYTE_TOKEN_OFFSET .. BYTE_TOKEN_OFFSET+255] at runtime. Keeping them out of
    // pieces_ avoids duplicate token IDs and wastes no vocab slots on chars that are
    // already handled.
    //
    // char_seeds is a temp filter set used in Step 4 to prevent single chars from
    // being re-added from subword mining candidates.
    
    size_t covered = 0;
    size_t coverage_target = static_cast<size_t>(total_chars * character_coverage);
    
    // Minimum frequency threshold: reject characters appearing less than this many times.
    const int MIN_CHAR_FREQUENCY = 10;
    
    // Temporary holder: valid single-char EM seeds — never committed to pieces_.
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
        
        // Record as byte-covered char — do NOT call addPiece().
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
    
    // Diagnostic: show byte-covered chars (they go through BYTE_TOKEN_OFFSET at runtime,
    // not through pieces_, so they never appear as unigram vocab entries).
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
        // Byte token ID = raw byte value + BYTE_TOKEN_OFFSET
        int byte_id = (int)(unsigned char)ch[0] + BYTE_TOKEN_OFFSET;
        std::cout << "  [byte:" << byte_id << "] \"" << display_text
                  << "\" (0x" << hex_bytes.str() << ") count=" << std::dec << count << std::endl;
    }
    
    // Step 3: Generate candidate subwords from SENTENCES (never cross sentence boundaries)
    std::unordered_map<std::string, int> subword_counts;
    subword_counts.reserve(1000000);  // Pre-allocate reasonable amount

    const size_t num_texts_to_process = use_sampling ? sample_indices.size() : training_units.size();
    const size_t progress_interval = std::max<size_t>(1, num_texts_to_process / 20);
    const size_t max_len = use_sampling
        ? static_cast<size_t>(MAX_PIECE_LENGTH)
        : std::min(static_cast<size_t>(MAX_PIECE_LENGTH), size_t(16));

    auto sentenceForIndex = [&](size_t idx) -> const std::string& {
        return use_sampling ? training_units[sample_indices[idx]] : training_units[idx];
    };

    auto sentenceAtomsForIndex = [&](size_t idx) -> const std::vector<AtomSpan>& {
        return use_sampling ? sentence_atom_spans[sample_indices[idx]] : sentence_atom_spans[idx];
    };

    unsigned int mining_workers = resolveSubwordMiningWorkerCount(
        enable_parallel_subword_mining, subword_mining_workers, num_texts_to_process);
    if (prune_during_mining && mining_workers > 1) {
        // Pruning semantics depend on seeing global counts during mining.
        // Keep legacy behavior when prune_during_mining is enabled.
        std::cout << "[UnigramLM] prune_during_mining=true; using single-thread mining to preserve pruning semantics"
                  << std::endl;
        mining_workers = 1;
    }

    std::cout << "[UnigramLM] Mining subwords from " << num_texts_to_process
              << " sentences (workers=" << mining_workers
              << ", max_len=" << max_len << ")..." << std::endl;
    const auto mining_start = std::chrono::steady_clock::now();

    if (mining_workers <= 1) {
        for (size_t ti = 0; ti < num_texts_to_process; ++ti) {
            const std::string& text = sentenceForIndex(ti);

            // Progress reporting
            if (ti % progress_interval == 0) {
                std::cout << "[UnigramLM] Subword mining: " << ti << "/" << num_texts_to_process
                          << " (" << (100 * ti / std::max<size_t>(1, num_texts_to_process)) << "%), "
                          << subword_counts.size() << " unique subwords" << std::endl;
            }

            mineSubwordsFromSentence(text, max_len, sentenceAtomsForIndex(ti), subword_counts);

            // Safety valve: if enabled and we've accumulated too many unique subwords, prune low-frequency ones
            if (prune_during_mining && subword_counts.size() > 50000000) {  // 50M entries ~= 2GB memory
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
        // 6M slots per worker: ~600MB per map at ~100-byte amortised entry cost.
        // With 128GB RAM and 18 workers that is ~10.8GB total — well within budget.
        // Pre-reserving prevents the dozens of rehash events that occur when growing
        // from the old 333K default to the actual 5-11M entries per worker.
        constexpr size_t kReservePerWorker = 6000000;
        for (auto& map : local_counts) {
            map.reserve(kReservePerWorker);
        }
        // Local high-water mark: when a worker's map exceeds this size we prune
        // entries with count == 1 (true singletons in this worker's slice are almost
        // certainly global singletons — keeping them wastes memory and dominates
        // merge cost). 5M chosen so pruning fires before first rehash above reserve.
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
                    // Local pruning: drop singleton entries to keep map small.
                    // Fires at most a handful of times per worker lifetime.
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

        // Parallel tree-reduction merge.
        // Each round halves the number of maps by merging adjacent pairs in parallel.
        // This keeps individual working sets small and saturates all cores during merge.
        const auto merge_start = std::chrono::steady_clock::now();
        std::cout << "[UnigramLM] Merging " << local_counts.size()
                  << " maps via parallel tree-reduction..." << std::endl;

        // Inline helper: merge src into dst, then release src memory.
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
                // Merge map[pi*2+1] into map[pi*2] in parallel.
                merge_threads.emplace_back([&, pi]() {
                    mergeInto(local_counts[pi * 2], local_counts[pi * 2 + 1]);
                });
            }
            for (auto& t : merge_threads) t.join();

            // Compact: keep only the even-indexed (merged) maps.
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

        // local_counts[0] is now the fully merged global map.
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
    // Sort by frequency descending, add until we hit target OR run out of candidates meeting min_freq
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
    const int max_to_add = target_vocab_size > 0 ? target_vocab_size : std::numeric_limits<int>::max();

    // Edge-trim canonical form: one trie entry per trimmed identity; first ranked
    // variant supplies the mining count; stored piece text is always canonical.
    std::unordered_set<std::string> dedup_keys_seen;
    for (const auto& ch : char_seeds) {
        std::string key = structuralDedupKeyForCandidate(ch);
        if (!key.empty()) dedup_keys_seen.insert(key);
    }
    std::cout << "[UnigramLM] Pre-seeded " << dedup_keys_seen.size()
              << " structural dedup keys from byte-layer chars" << std::endl;

    // Pass A: encoding-safety, repetition, then structural edge-trim dedup.
    // Initial scores are log(count / sum_accepted_counts).
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
        dedup_keys_seen.insert(dedup_key);

        // Store canonical trimmed bytes, not the raw mined surface (avoids " this"
        // and "this " as separate vocab entries).
        accepted.push_back({dedup_key, count});
    }

    // Sum of all accepted mining counts — correct denominator for initial log-probs.
    double total_accepted_count = 0.0;
    for (const auto& ap : accepted) total_accepted_count += ap.count;
    if (total_accepted_count < 1.0) total_accepted_count = 1.0;  // safety

    // Pass B: commit to pieces_ with correct initial log-probability scores.
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
    std::cout << "[UnigramLM] Added " << added << " subwords (min_freq=" << MIN_SUBWORD_FREQ 
              << ", target=" << target_vocab_size << "), total vocab: " << pieces_.size() << std::endl;
    
    // Step 5: EM with convergence detection + dead token pruning + backfill.
    //
    // Production EM loop:
    //   1. Iterate until log-likelihood converges (relative change < threshold).
    //   2. After convergence, prune tokens that Viterbi never selected (dead weight).
    //   3. Backfill pruned slots with next-best candidates from ranked_subwords.
    //   4. Re-run EM to convergence so replacements get proper scores.
    //
    // This replaces the old fixed-5-iteration loop that stopped mid-optimization
    // and kept 1300+ dead tokens wasting vocab slots.

    constexpr int    EM_MAX_ITERATIONS     = 50;      // Safety cap — never spin forever
    constexpr double EM_CONVERGENCE_THRESH = 0.0001; // 0.01% relative change in log-likelihood
    constexpr double SMOOTHING             = 0.1;    // Add-k smoothing for M-step

    // Lambda: run one E-step (Viterbi over normalized corpus), return {token_counts, total_tokens, log_likelihood}.
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
                // Log-likelihood = score of best Viterbi path for this segment.
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

    // Lambda: run one M-step (re-estimate scores from counts), return # unused tokens.
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

    // Lambda: run EM to convergence, return final {token_counts, iterations_run}.
    auto runEMToConvergence = [&](const char* phase_label) -> std::pair<std::unordered_map<int, double>, int> {
        double prev_ll = -1e30;
        std::unordered_map<int, double> last_counts;
        int iter = 0;
        for (; iter < EM_MAX_ITERATIONS; ++iter) {
            buildTrie();
            auto [token_counts, total_tokens, log_likelihood] = runEStep();
            int unused = runMStep(token_counts, total_tokens);

            double relative_change = (prev_ll < -1e20)
                ? 1.0  // First iteration — no baseline yet
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

    // ---- Phase A: initial EM to convergence ----
    auto [phase_a_counts, phase_a_iters] = runEMToConvergence("Phase-A");

    // ---- Phase B: prune dead tokens, backfill from ranked_subwords ----
    int pruned = 0;
    {
        // Identify dead tokens (zero Viterbi usage after full convergence).
        // token_counts from Phase A are keyed by token_id = UNIGRAM_VOCAB_OFFSET + index.
        // After pruning, indices change, so we identify by INDEX first, then compact.
        std::unordered_set<int> dead_indices;  // indices into pieces_
        for (size_t i = 0; i < pieces_.size(); ++i) {
            if (pieces_[i].is_user_defined) continue;
            int tid = tokenIdForIndex(static_cast<int>(i));
            if (!phase_a_counts.count(tid) || phase_a_counts.at(tid) == 0.0) {
                dead_indices.insert(static_cast<int>(i));
            }
        }

        if (!dead_indices.empty()) {
            std::cout << "[UnigramLM] Pruning " << dead_indices.size()
                      << " dead tokens (zero Viterbi usage after convergence)" << std::endl;

            // Compact pieces_, removing dead entries.
            std::vector<UnigramPiece> surviving;
            surviving.reserve(pieces_.size() - dead_indices.size());
            for (size_t i = 0; i < pieces_.size(); ++i) {
                if (!dead_indices.count(static_cast<int>(i))) {
                    surviving.push_back(std::move(pieces_[i]));
                }
            }
            pieces_ = std::move(surviving);
            pruned = static_cast<int>(dead_indices.size());

            // CRITICAL: Rebuild piece_to_id_ after compacting pieces_.
            // Old indices are stale — new indices are 0..pieces_.size()-1.
            piece_to_id_.clear();
            for (size_t i = 0; i < pieces_.size(); ++i) {
                piece_to_id_[pieces_[i].text] = static_cast<int>(i);
            }

            // Backfill: scan ranked_subwords for next-best candidates not already in vocab.
            int backfilled = 0;
            const int slots = pruned;
            // Recompute total accepted count for scoring backfills.
            double backfill_total = 0.0;
            for (const auto& p : pieces_) backfill_total += std::exp(static_cast<double>(p.score));
            if (backfill_total < 1e-30) backfill_total = 1.0;

            for (const SubwordEntry* entry : ranked_subwords) {
                if (backfilled >= slots) break;
                const std::string& subword = entry->first;
                const int count = entry->second;
                if (hasPiece(subword)) continue;
                if (char_seeds.count(subword)) continue;
                if (isRepetitionNoise(subword)) continue;
                if (!isValidSubword(subword)) continue;
                std::string dedup_key = structuralDedupKeyForCandidate(subword);
                if (dedup_key.empty()) continue;
                if (dedup_keys_seen.count(dedup_key)) continue;
                dedup_keys_seen.insert(dedup_key);

                // Seed with mining log-prob (will be refined by Phase C EM).
                float score = static_cast<float>(std::log(static_cast<double>(count) / total_accepted_count));
                addPiece(dedup_key, score, false);
                backfilled++;
            }
            std::cout << "[UnigramLM] Backfilled " << backfilled << " replacement tokens" << std::endl;
        }
    }

    // ---- Phase C: re-converge if we changed the vocab ----
    if (pruned > 0) {
        auto [phase_c_counts, phase_c_iters] = runEMToConvergence("Phase-C");
        (void)phase_c_counts;
        (void)phase_c_iters;
    }

    // Final trie build with converged scores
    buildTrie();
    
    std::cout << "[UnigramLM] Training complete. Final vocab size: " << pieces_.size() << std::endl;
    return true;
}

//--------------------------------------------------//
// Trie Building
//--------------------------------------------------//

void UnigramLM::buildTrie() {
    trie_.clear();
    trie_.push_back(TrieNode());  // Root node
    
    for (size_t i = 0; i < pieces_.size(); ++i) {
        const auto& piece = pieces_[i];
        int node = 0;
        
        for (unsigned char c : piece.text) {
            if (trie_[node].children[c] < 0) {
                trie_[node].children[c] = static_cast<int>(trie_.size());
                trie_.push_back(TrieNode());
            }
            node = trie_[node].children[c];
        }
        
        // Token ID is ALWAYS position-derived. No stored field.
        trie_[node].token_id = tokenIdForIndex(static_cast<int>(i));
        trie_[node].score = piece.score;
    }
}

//--------------------------------------------------//
// CPU Viterbi Encoding
//--------------------------------------------------//

std::vector<ViterbiNode> UnigramLM::viterbi(const std::string& text) const {
    size_t n = text.size();
    std::vector<ViterbiNode> nodes(n + 1);
    
    // Initialize
    nodes[0].score = 0.0f;
    nodes[0].prev_pos = -1;
    nodes[0].token_id = -1;
    nodes[0].piece_length = 0;
    
    for (size_t i = 1; i <= n; ++i) {
        nodes[i].score = -1e30f;
        nodes[i].prev_pos = -1;
        nodes[i].token_id = unk_id_;  // Absolute UNK_TOKEN_ID = 0
        nodes[i].piece_length = 1;
    }
    
    // Forward pass
    for (size_t pos = 0; pos < n; ++pos) {
        if (nodes[pos].score < -1e20f) continue;  // Unreachable
        
        unsigned char cur_byte = static_cast<unsigned char>(text[pos]);
        bool cur_is_punct = isPunctBoundary(cur_byte);
        
        // PUNCTUATION ISOLATION GUARD:
        // If current position is a punctuation character, force it to be emitted
        // as a single byte token and skip trie matching entirely.
        // This prevents tokens like "however," or "al." from ever being selected.
        if (cur_is_punct) {
            float byte_score = nodes[pos].score + UNKNOWN_SCORE;
            if (byte_score > nodes[pos + 1].score) {
                nodes[pos + 1].score = byte_score;
                nodes[pos + 1].prev_pos = static_cast<int>(pos);
                nodes[pos + 1].token_id = static_cast<int>(cur_byte) + BYTE_TOKEN_OFFSET;
                nodes[pos + 1].piece_length = 1;
            }
            continue;  // Skip trie search — punctuation is always isolated
        }
        
        // Try all pieces starting at pos
        {
            if (trie_.empty()) {
                throw std::runtime_error("viterbi(): trie_ is empty — buildTrie() was never called. "
                                         "Caller MUST build trie before encoding at " + 
                                         std::string(__FILE__) + ":" + std::to_string(__LINE__));
            }
            int node = 0;
            for (size_t len = 1; len <= MAX_PIECE_LENGTH && pos + len <= n; ++len) {
                unsigned char c = static_cast<unsigned char>(text[pos + len - 1]);
                
                // PUNCTUATION ISOLATION GUARD:
                // Stop extending the piece if we hit a punctuation character.
                // This prevents the trie from matching tokens that contain
                // punctuation mixed with letters (e.g., "al.", "et al.,").
                if (isPunctBoundary(c)) break;
                
                if (trie_[node].children[c] < 0) break;
                node = trie_[node].children[c];
                
                if (trie_[node].token_id >= 0) {
                    float score = nodes[pos].score + trie_[node].score;
                    
                    if (score > nodes[pos + len].score) {
                        nodes[pos + len].score = score;
                        nodes[pos + len].prev_pos = static_cast<int>(pos);
                        nodes[pos + len].token_id = trie_[node].token_id;
                        nodes[pos + len].piece_length = static_cast<int>(len);
                    }
                }
            }
        }
        
        if (enable_byte_fallback_) {
            // Byte fallback: allow single byte advance
            float byte_score = nodes[pos].score + UNKNOWN_SCORE;
            if (byte_score > nodes[pos + 1].score) {
                unsigned char byte_val = static_cast<unsigned char>(text[pos]);
                nodes[pos + 1].score = byte_score;
                nodes[pos + 1].prev_pos = static_cast<int>(pos);
                nodes[pos + 1].token_id = static_cast<int>(byte_val) + BYTE_TOKEN_OFFSET;  // Byte token ID (offset by specials)
                nodes[pos + 1].piece_length = 1;
            }
        } else {
            // Byte fallback disabled: advance with <unk> instead of byte tokens.
            std::cout << "[UnigramLM] Warning: byte fallback disabled, using <unk> token for unknown bytes" << std::endl;
            float unk_score = nodes[pos].score + UNKNOWN_SCORE;
            if (unk_score > nodes[pos + 1].score) {
                nodes[pos + 1].score = unk_score;
                nodes[pos + 1].prev_pos = static_cast<int>(pos);
                nodes[pos + 1].token_id = unk_id_;  // Absolute UNK_TOKEN_ID = 0
                nodes[pos + 1].piece_length = 1;
            }
        }
    }
    
    return nodes;
}

std::vector<int> UnigramLM::backtrack(const std::vector<ViterbiNode>& nodes, int end_pos) const {
    std::vector<int> tokens;
    int pos = end_pos;
    
    while (pos > 0) {
        tokens.push_back(nodes[pos].token_id);
        pos = nodes[pos].prev_pos;
    }
    
    std::reverse(tokens.begin(), tokens.end());
    return tokens;
}

std::vector<int> UnigramLM::encode(const std::string& text) const {
    if (text.empty()) return {};

    // SentencePiece-style normalization: spaces → ▁, prepend ▁
    std::string normalized = normalizeSpaces(text);
    auto nodes = viterbi(normalized);
    return backtrack(nodes, static_cast<int>(normalized.size()));
}

std::vector<UnigramPiece> UnigramLM::encodeWithPieces(const std::string& text) const {
    auto token_ids = encode(text);
    std::vector<UnigramPiece> result;
    result.reserve(token_ids.size());
    
    for (int tid : token_ids) {
        if (tid >= SPECIAL_TOKEN_OFFSET && tid < NUM_SPECIAL_TOKENS) {
            // Special token — token_id is NOT stored on UnigramPiece (it's position-derived).
            // Caller should use the token_ids from encode(), not from pieces.
            UnigramPiece piece;
            if (tid == UNK_TOKEN_ID) piece.text = "<unk>";
            else if (tid == PAD_TOKEN_ID) piece.text = "<pad>";
            else if (tid == BOS_TOKEN_ID) piece.text = "<s>";
            else if (tid == EOS_TOKEN_ID) piece.text = "</s>";
            piece.score = -10.0f;
            piece.is_special = true;
            piece.is_user_defined = true;
            result.push_back(piece);
        } else if (tid >= BYTE_TOKEN_OFFSET && tid < ATOM_TOKEN_OFFSET) {
            // Byte token
            UnigramPiece piece;
            piece.text = std::string(1, static_cast<char>(tid - BYTE_TOKEN_OFFSET));
            piece.score = UNKNOWN_SCORE;
            piece.is_special = false;
            piece.is_user_defined = false;
            result.push_back(piece);
        } else if (tid >= UNIGRAM_VOCAB_OFFSET) {
            const UnigramPiece* p = getPiece(tid);
            if (p) {
                result.push_back(*p);
            }
        }
    }
    
    return result;
}

std::string UnigramLM::decode(const std::vector<int>& token_ids) const {
    return decode(token_ids.data(), token_ids.size());
}

std::string UnigramLM::decode(const int* token_ids, size_t count) const {
    std::string result;
    
    for (size_t i = 0; i < count; ++i) {
        int tid = token_ids[i];
        
        if (tid >= SPECIAL_TOKEN_OFFSET && tid < NUM_SPECIAL_TOKENS) {
            // Special token — decode as their text repr
            if (tid == UNK_TOKEN_ID) result += "<unk>";
            else if (tid == PAD_TOKEN_ID) { /* skip padding */ }
            else if (tid == BOS_TOKEN_ID) result += "<s>";
            else if (tid == EOS_TOKEN_ID) result += "</s>";
        } else if (tid >= BYTE_TOKEN_OFFSET && tid < BYTE_TOKEN_OFFSET + BYTE_VOCAB_SIZE) {
            // Byte token (subtract BYTE_TOKEN_OFFSET to get raw byte)
            result.push_back(static_cast<char>(tid - BYTE_TOKEN_OFFSET));
        } else if (tid >= UNIGRAM_VOCAB_OFFSET) {
            // Unigram token
            const UnigramPiece* p = getPiece(tid);
            if (p) {
                result += p->text;
            }
        }
        // Atom tokens are handled by ScratchBlock
    }
    
    return denormalizeSpaces(result);
}

void UnigramLM::capVocabSize(int max_size) {
    if (max_size >= static_cast<int>(pieces_.size())) {
        return;  // Already smaller than cap
    }
    
    if (max_size < 4) {
        throw std::runtime_error("capVocabSize: max_size must be >= 4 to include minimum vocabulary");
    }
    
    // Sort pieces by score (descending) to keep most frequent
    // But always keep user-defined tokens (special tokens) regardless of score
    std::vector<size_t> indices(pieces_.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    std::stable_sort(indices.begin(), indices.end(), [this](size_t a, size_t b) {
        // User-defined tokens always come first
        if (pieces_[a].is_user_defined != pieces_[b].is_user_defined) {
            return pieces_[a].is_user_defined;  // user-defined = true sorts before false
        }
        return pieces_[a].score > pieces_[b].score;  // Higher score = more frequent
    });
    
    // Keep top max_size pieces
    std::vector<UnigramPiece> new_pieces;
    new_pieces.reserve(max_size);
    
    std::unordered_map<std::string, int> new_piece_to_id;
    
    for (int i = 0; i < max_size && i < static_cast<int>(indices.size()); ++i) {
        UnigramPiece piece = pieces_[indices[i]];
        // Token ID is always UNIGRAM_VOCAB_OFFSET + index — no field to reassign.
        new_piece_to_id[piece.text] = static_cast<int>(new_pieces.size());
        new_pieces.push_back(piece);
    }
    
    pieces_ = std::move(new_pieces);
    piece_to_id_ = std::move(new_piece_to_id);
    
    // Rebuild trie for fast encoding (uses new token_ids)
    buildTrie();
    
    std::cout << "[UnigramLM] Capped vocab to " << pieces_.size() << " pieces" << std::endl;
}


//--------------------------------------------------//
// GPU Implementation
//--------------------------------------------------//

bool UnigramLM::initGPU() {
    if (gpu_->initialized) return true;
    
    if (trie_.empty()) {
        buildTrie();
    }
    
    return uploadTrieToGPU();
}

bool UnigramLM::uploadTrieToGPU() {
    cudaError_t err;
    size_t num_nodes = trie_.size();
    
    // Helper lambda for cleanup on failure
    auto cleanup = [this]() {
        if (gpu_->d_trie_children) { cudaFree(gpu_->d_trie_children); gpu_->d_trie_children = nullptr; }
        if (gpu_->d_trie_token_ids) { cudaFree(gpu_->d_trie_token_ids); gpu_->d_trie_token_ids = nullptr; }
        if (gpu_->d_trie_scores) { cudaFree(gpu_->d_trie_scores); gpu_->d_trie_scores = nullptr; }
        if (gpu_->d_piece_data) { cudaFree(gpu_->d_piece_data); gpu_->d_piece_data = nullptr; }
        if (gpu_->d_piece_offsets) { cudaFree(gpu_->d_piece_offsets); gpu_->d_piece_offsets = nullptr; }
        if (gpu_->d_piece_lengths) { cudaFree(gpu_->d_piece_lengths); gpu_->d_piece_lengths = nullptr; }
        if (gpu_->d_viterbi_scores) { cudaFree(gpu_->d_viterbi_scores); gpu_->d_viterbi_scores = nullptr; }
        if (gpu_->d_viterbi_prev) { cudaFree(gpu_->d_viterbi_prev); gpu_->d_viterbi_prev = nullptr; }
        if (gpu_->d_viterbi_tokens) { cudaFree(gpu_->d_viterbi_tokens); gpu_->d_viterbi_tokens = nullptr; }
    };
    
    // Allocate trie arrays
    err = cudaMalloc(&gpu_->d_trie_children, num_nodes * 256 * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "[UnigramLM] Failed to allocate trie_children" << std::endl;
        return false;
    }
    
    err = cudaMalloc(&gpu_->d_trie_token_ids, num_nodes * sizeof(int));
    if (err != cudaSuccess) {
        cleanup();
        return false;
    }
    
    err = cudaMalloc(&gpu_->d_trie_scores, num_nodes * sizeof(float));
    if (err != cudaSuccess) {
        cleanup();
        return false;
    }
    
    // Flatten and upload trie data
    std::vector<int> children_flat(num_nodes * 256);
    std::vector<int> token_ids_flat(num_nodes);
    std::vector<float> scores_flat(num_nodes);
    
    for (size_t i = 0; i < num_nodes; ++i) {
        for (int c = 0; c < 256; ++c) {
            children_flat[i * 256 + c] = trie_[i].children[c];
        }
        token_ids_flat[i] = trie_[i].token_id;
        scores_flat[i] = trie_[i].score;
    }
    
    cudaMemcpy(gpu_->d_trie_children, children_flat.data(), 
               num_nodes * 256 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_->d_trie_token_ids, token_ids_flat.data(),
               num_nodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_->d_trie_scores, scores_flat.data(),
               num_nodes * sizeof(float), cudaMemcpyHostToDevice);
    
    gpu_->num_nodes = static_cast<int>(num_nodes);
    
    // Upload piece data for decoding
    size_t total_piece_length = 0;
    for (const auto& p : pieces_) {
        total_piece_length += p.text.size();
    }
    
    std::vector<char> piece_data(total_piece_length);
    std::vector<int> piece_offsets(pieces_.size());
    std::vector<int> piece_lengths(pieces_.size());
    
    size_t offset = 0;
    for (size_t i = 0; i < pieces_.size(); ++i) {
        piece_offsets[i] = static_cast<int>(offset);
        piece_lengths[i] = static_cast<int>(pieces_[i].text.size());
        std::copy(pieces_[i].text.begin(), pieces_[i].text.end(), 
                  piece_data.begin() + offset);
        offset += pieces_[i].text.size();
    }
    
    err = cudaMalloc(&gpu_->d_piece_data, total_piece_length > 0 ? total_piece_length : 1);
    if (err != cudaSuccess) { cleanup(); return false; }
    
    err = cudaMalloc(&gpu_->d_piece_offsets, pieces_.size() * sizeof(int));
    if (err != cudaSuccess) { cleanup(); return false; }
    
    err = cudaMalloc(&gpu_->d_piece_lengths, pieces_.size() * sizeof(int));
    if (err != cudaSuccess) { cleanup(); return false; }
    
    // Pre-allocate Viterbi workspace with fixed capacity
    constexpr size_t MAX_SEQUENCE_LENGTH = HyperParameters::UNIGRAM_MAX_SEQUENCE_LENGTH;
    gpu_->workspace_max_length = MAX_SEQUENCE_LENGTH;
    
    err = cudaMalloc(&gpu_->d_viterbi_scores, (MAX_SEQUENCE_LENGTH + 1) * sizeof(float));
    if (err != cudaSuccess) { cleanup(); return false; }
    
    err = cudaMalloc(&gpu_->d_viterbi_prev, (MAX_SEQUENCE_LENGTH + 1) * sizeof(int));
    if (err != cudaSuccess) { cleanup(); return false; }
    
    err = cudaMalloc(&gpu_->d_viterbi_tokens, (MAX_SEQUENCE_LENGTH + 1) * sizeof(int));
    if (err != cudaSuccess) { cleanup(); return false; }
    
    cudaMemcpy(gpu_->d_piece_data, piece_data.data(), 
               total_piece_length, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_->d_piece_offsets, piece_offsets.data(),
               pieces_.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_->d_piece_lengths, piece_lengths.data(),
               pieces_.size() * sizeof(int), cudaMemcpyHostToDevice);
    
    gpu_->initialized = true;
    std::cout << "[UnigramLM] GPU initialized with " << num_nodes << " trie nodes" << std::endl;
    return true;
}

bool UnigramLM::encodeGPU(const char* d_text,
                          size_t length,
                          int* d_token_ids,
                          int* d_token_count,
                          int max_tokens,
                          bool* d_needs_byte_fallback) {
    if (!gpu_->initialized) {
        if (!initGPU()) return false;
    }
    
    // Validate length against pre-allocated workspace capacity
    if (length > gpu_->workspace_max_length) {
        std::cerr << "[UnigramLM] Input length " << length 
                  << " exceeds workspace capacity " << gpu_->workspace_max_length << std::endl;
        return false;
    }
    
    const bool enable_fallback = enable_byte_fallback_ && d_needs_byte_fallback;
    if (enable_fallback) {
        // Initialize fallback flags to false
        // NOTE: Using default stream (nullptr) is intentional here - Unigram operates
        // in the data loading path which may run on separate thread from training
        cudaMemsetAsync(d_needs_byte_fallback, 0, length * sizeof(bool), nullptr);
    }
    
    // Forward pass (single-threaded kernel due to Viterbi sequential dependency)
    bool* fallback_ptr = enable_fallback ? d_needs_byte_fallback : nullptr;
    // INTENTIONAL: Stream 0 for synchronous Viterbi forward pass (cudaMemcpy follows)
    // NOTE: Kernel runs single-threaded because Viterbi has O(n) sequential dependency
    kernelViterbiForward<<<1, 1, 0, 0>>>(
        d_text, length,
        gpu_->d_trie_children, gpu_->d_trie_token_ids, gpu_->d_trie_scores,
        gpu_->num_nodes,
        gpu_->d_viterbi_scores, gpu_->d_viterbi_prev, gpu_->d_viterbi_tokens,
        fallback_ptr,
        unk_id_,  // Absolute UNK_TOKEN_ID = 0
        UNKNOWN_SCORE,
        enable_fallback
    );
    
    // INTENTIONAL: Stream 0 for synchronous Viterbi backtrack (result copied back immediately)
    // Backtrack with max_tokens to prevent buffer overflow
    kernelViterbiBacktrack<<<1, 1, 0, 0>>>(
        length,
        gpu_->d_viterbi_prev, gpu_->d_viterbi_tokens,
        d_token_ids, d_token_count,
        max_tokens
    );
    
    return cudaGetLastError() == cudaSuccess;
}

bool UnigramLM::encodeBatchGPU(const char* const* d_texts,
                                const size_t* lengths,
                                int** d_token_ids,
                                int* d_token_counts,
                                int max_tokens_per_seq,
                                size_t batch_size) {
    if (batch_size == 0) return true;
    
    // NOTE: Sequences are processed sequentially because:
    // 1. Viterbi algorithm has inherent sequential dependency (each position depends on all previous)
    // 2. The pre-allocated workspace (d_viterbi_scores/prev/tokens) is shared across all sequences
    // True batch parallelization would require per-sequence workspace allocation, which trades
    // memory for parallelism. For typical batch sizes (8-32), sequential processing is adequate
    // since the bottleneck is usually elsewhere (embedding lookup, attention).
    
    // Find max length to allocate fallback buffer once (avoid per-iteration malloc!)
    size_t max_len = 0;
    for (size_t i = 0; i < batch_size; ++i) {
        max_len = std::max(max_len, lengths[i]);
    }
    
    // Single allocation for entire batch
    bool* d_fallback = nullptr;
    cudaError_t err = cudaMalloc(&d_fallback, max_len > 0 ? max_len * sizeof(bool) : sizeof(bool));
    if (err != cudaSuccess) {
        std::cerr << "[UnigramLM] Failed to allocate batch fallback buffer" << std::endl;
        return false;
    }
    
    // Process each sequence reusing the same buffer
    bool success = true;
    for (size_t i = 0; i < batch_size && success; ++i) {
        success = encodeGPU(d_texts[i], lengths[i], d_token_ids[i], 
                            &d_token_counts[i], max_tokens_per_seq, d_fallback);
    }
    
    cudaFree(d_fallback);
    return success;
}

bool UnigramLM::decodeGPU(const int* d_token_ids,
                          size_t count,
                          char* d_output,
                          size_t* d_output_length,
                          size_t max_output_length) {
    if (!gpu_->initialized) {
        if (!initGPU()) return false;
    }
    
    // INTENTIONAL: Stream 0 for synchronous decode (result copied back immediately)
    kernelUnigramDecode<<<1, 1, 0, 0>>>(
        d_token_ids, count,
        gpu_->d_piece_data, gpu_->d_piece_offsets, gpu_->d_piece_lengths,
        UNIGRAM_VOCAB_OFFSET, static_cast<int>(pieces_.size()),
        d_output, d_output_length, max_output_length
    );
    
    return cudaGetLastError() == cudaSuccess;
}

} // namespace Tokenizer
} // namespace GRIM
