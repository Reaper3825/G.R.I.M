// Character coverage telemetry and initial vocabulary reset.

#include "UnigramCharacterDiagnostics.hpp"
#include "SubwordMining.hpp"
#include "../TextUtils.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unordered_set>

namespace GRIM {
namespace Tokenizer {

void runUnigramCharacterDiagnostics(
    const std::vector<std::string>& norm_texts,
    const std::vector<std::vector<AtomSpan>>& norm_training_exclusion_spans,
    float character_coverage, UnigramVocabWriteTarget target) {
    auto& pieces_ = target.pieces;
    auto& piece_to_id_ = target.piece_to_id;
    // Step 1: Count character frequencies (use ALL normalized texts)
    std::unordered_map<std::string, int> char_counts;
    size_t total_chars = 0;
    size_t excluded_chars_skipped = 0;

    for (size_t text_idx = 0; text_idx < norm_texts.size(); ++text_idx) {
        const auto& text = norm_texts[text_idx];
        const auto& spans = norm_training_exclusion_spans[text_idx];
        size_t span_i = 0;

        for (size_t i = 0; i < text.size(); ) {
            const size_t seq_len = utf8SequenceLength(static_cast<unsigned char>(text[i]));

            while (span_i < spans.size() && spans[span_i].end <= i) {
                ++span_i;
            }
            if (span_i < spans.size() && i >= spans[span_i].start && i < spans[span_i].end) {
                excluded_chars_skipped++;
                i += seq_len;
                continue;
            }

            if (i + seq_len <= text.size()) {
                std::string ch = text.substr(i, seq_len);
                if (ch != "\n") {
                    char_counts[ch]++;
                    total_chars++;
                }
            }
            i += seq_len;
        }
    }

    if (excluded_chars_skipped > 0) {
        std::cout << "[UnigramLM] Char counting: skipped " << excluded_chars_skipped
                  << " characters inside atom, numeric, and exact-manual spans" << std::endl;
    }
    if (total_chars == 0) {
        throw std::runtime_error("UnigramLM::trainFromCorpus: corpus has zero trainable characters outside atom, numeric, and exact-manual spans after normalization");
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

        if (!isValidUnigramVocabCharacter(ch)) {
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

}

} // namespace Tokenizer
} // namespace GRIM
