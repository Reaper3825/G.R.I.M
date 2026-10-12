// Mined candidate filtering, structural dedup, and initial scoring.

#include "UnigramCandidateAdmission.hpp"
#include "HyperParameters/HyperparameterGroupings.hpp"
#include "UnigramTrainingValidation.hpp"
#include "../ManualVocabSource.hpp"
#include "../TextUtils.hpp"
#include "../VocabWriteOp.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace GRIM {
namespace Tokenizer {

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

void admitUnigramVocabularyCandidates(
    UnigramLM& unigram, const UnigramSubwordMiningResult& mining_result, const ::GRIM::HyperParameters::TokenizerHP& hp,
    const std::vector<UnigramPiece>& manual_pieces) {
    const int min_subword_freq = hp.min_subword_freq;
    const int target_vocab_size = hp.target_vocab_size;
    auto& pieces_ = unigram.pieces_;
    auto& piece_to_id_ = unigram.piece_to_id_;
    const UnigramSubwordCount MIN_SUBWORD_FREQ = static_cast<UnigramSubwordCount>(min_subword_freq);
    // Step 3: Generate candidate subwords from SENTENCES
    const UnigramSubwordCountMap& subword_counts = mining_result.subword_counts;

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

    appendManualVocabPieces(unigram, manual_pieces, target_vocab_size);

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
        if (unigram.hasPiece(subword)) continue;

        if (isUnigramRepetitionNoise(subword, hp)) {
            repetition_filtered++;
            continue;
        }
        if (!isValidUnigramSubword(subword, hp)) {
            filtered++;
            continue;
        }

        std::string dedup_key = hp.structural_dedup_trim_edges ? structuralDedupKeyForCandidate(subword) : subword;
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
    if (!accepted.empty() || manual_pieces.empty()) {
        requireUnigramAcceptedCandidateSetIsScorable(
            accepted.size(),
            total_accepted_count,
            "UnigramLM::trainFromCorpus initial candidate score normalization");
    }

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
              << "; data-driven vocab, no target cap), total candidate vocab: " << pieces_.size() << std::endl;
    if (pieces_.empty()) {
        throw std::runtime_error("UnigramLM::trainFromCorpus: no learned subword candidates survived frequency/validity filters; lower min_subword_freq or provide more corpus text");
    }

}

} // namespace Tokenizer
} // namespace GRIM
