// Likelihood-loss scoring and survivor selection for Unigram vocabulary pruning.

#include "UnigramLikelihoodPruning.hpp"
#include "../TextUtils.hpp"
#include "HyperParameters/HyperparameterGroupings.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace GRIM {
namespace Tokenizer {

//======================================================//
//  Likelihood-loss pruning (SentencePiece-style)
//
//  Gold-standard Unigram-LM vocabulary reduction. After EM has fit the
//  piece probabilities, each learned piece is scored by its contribution
//  to corpus likelihood: how much log-probability is lost per use if the
//  piece is removed and its surface is re-encoded by the next-best
//  combination of OTHER learned pieces. The lowest-contribution pieces
//  (dead prefix fragments like "traditio"/"recogn" sit at ~0) are dropped
//  toward target_vocab_size.
//
//  Coverage is always preserved. A piece is protected from pruning when:
//    - it is user-defined / special, OR
//    - it is a single Unicode codepoint (single-char coverage comes from
//      single-char UNIGRAM pieces, never from byte fallback), OR
//    - its surface has no alternative learned-piece segmentation at all
//      (removing it would force byte fallback).
//  Byte fallback (the fixed UNKNOWN_SCORE path) is never a protected vocab
//  entry; it stays the unnormalized last-resort coverage path only.
//======================================================//

static bool isSingleCodepointPiece(const std::string& text) {
    if (text.empty()) return false;
    uint32_t cp = 0;
    size_t len = 0;
    if (!utf8DecodeAt(text, 0, &cp, &len)) return false;
    return len == text.size();
}

namespace {
struct LikelihoodLossTrieNode {
    std::array<int, 256> children;
    int piece_index = -1;
    double score = 0.0;
    LikelihoodLossTrieNode() { children.fill(-1); }
};
}  // namespace

// Best total log-probability of segmenting `surface` using OTHER learned
// pieces only — i.e., excluding the single whole-surface piece. Returns
// -inf when no alternative learned-piece segmentation exists; such pieces
// are irreplaceable coverage and must never be pruned.
static double bestAlternativeSegmentationScore(
    const std::vector<LikelihoodLossTrieNode>& trie,
    const std::string& surface) {
    const size_t n = surface.size();
    const double neg_inf = -std::numeric_limits<double>::infinity();
    std::vector<double> best(n + 1, neg_inf);
    best[0] = 0.0;
    for (size_t start = 0; start < n; ++start) {
        if (best[start] == neg_inf) continue;
        int node = 0;
        const size_t max_end = std::min(n, start + static_cast<size_t>(MAX_PIECE_LENGTH));
        for (size_t end = start + 1; end <= max_end; ++end) {
            const unsigned char byte = static_cast<unsigned char>(surface[end - 1]);
            const int child = trie[static_cast<size_t>(node)].children[byte];
            if (child < 0) break;
            node = child;
            const LikelihoodLossTrieNode& cur = trie[static_cast<size_t>(node)];
            if (cur.piece_index < 0) continue;
            if (start == 0 && end == n) continue;  // exclude the whole-surface piece itself
            const double candidate = best[start] + cur.score;
            if (candidate > best[end]) best[end] = candidate;
        }
    }
    return best[n];
}

// Select the highest-likelihood-contribution pieces toward a target size,
// always retaining protected (single-codepoint / user-defined / no-alt)
// pieces. Survivors are moved out of `pieces` in original order so token
// IDs stay position-derived after rewriteUnigramVocab().
UnigramLikelihoodPruningResult runUnigramLikelihoodPruning(
    UnigramTrainingSession& session, const char* caller) {
    requireUnigramTrainingSessionReady(session, caller);
    if (session.expectation_generation != session.vocabulary_generation) {
        throw std::runtime_error("Unigram pruning: expected counts are stale");
    }
    auto& pieces = *session.pieces;
    const auto& expected_counts_by_piece_index = session.storage->expected_counts;
    const int target_vocab_size = session.total_piece_target;
    const double shrink_factor = session.hp->prune_shrink_factor;
    const int worker_count = session.hp->subword_mining_workers;
    if (caller == nullptr || caller[0] == '\0') {
        throw std::runtime_error("selectUnigramSurvivorsByLikelihoodLoss: caller label is empty");
    }
    const size_t n = pieces.size();
    if (expected_counts_by_piece_index.size() != n) {
        throw std::runtime_error(std::string(caller) +
                                 ": freq_by_index size=" + std::to_string(expected_counts_by_piece_index.size()) +
                                 " != pieces size=" + std::to_string(n));
    }
    if (!(shrink_factor > 0.0 && shrink_factor < 1.0)) {
        throw std::runtime_error(std::string(caller) +
                                 ": shrink_factor must be in (0,1), got " + std::to_string(shrink_factor));
    }

    std::vector<LikelihoodLossTrieNode> trie;
    trie.reserve(n + 1);
    trie.emplace_back();
    for (size_t i = 0; i < n; ++i) {
        if (pieces[i].is_user_defined) continue;
        int node = 0;
        for (unsigned char byte : pieces[i].text) {
            int child = trie[static_cast<size_t>(node)].children[byte];
            if (child < 0) {
                child = static_cast<int>(trie.size());
                trie[static_cast<size_t>(node)].children[byte] = child;
                trie.emplace_back();
            }
            node = child;
        }
        trie[static_cast<size_t>(node)].piece_index = static_cast<int>(i);
        trie[static_cast<size_t>(node)].score = static_cast<double>(pieces[i].score);
    }

    std::vector<char> is_protected(n, 0);
    std::vector<double> loss(n, 0.0);
    std::atomic<int> irreplaceable{0};

    auto compute_range = [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            if (pieces[i].is_user_defined || isSingleCodepointPiece(pieces[i].text)) {
                is_protected[i] = 1;
                continue;
            }
            const double alt = bestAlternativeSegmentationScore(trie, pieces[i].text);
            if (!std::isfinite(alt)) {
                is_protected[i] = 1;
                irreplaceable.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            const double keep_logprob = static_cast<double>(pieces[i].score);
            loss[i] = expected_counts_by_piece_index[i] * (keep_logprob - alt);
        }
    };

    int workers = worker_count > 0 ? worker_count : static_cast<int>(std::thread::hardware_concurrency());
    if (workers < 1) workers = 1;
    if (static_cast<size_t>(workers) > n) workers = static_cast<int>(std::max<size_t>(1, n));
    if (workers == 1) {
        compute_range(0, n);
    } else {
        std::vector<std::thread> pool;
        const size_t chunk = (n + static_cast<size_t>(workers) - 1) / static_cast<size_t>(workers);
        for (int w = 0; w < workers; ++w) {
            const size_t begin = static_cast<size_t>(w) * chunk;
            if (begin >= n) break;
            const size_t end = std::min(n, begin + chunk);
            pool.emplace_back(compute_range, begin, end);
        }
        for (auto& t : pool) t.join();
    }

    int protected_count = 0;
    std::vector<int> prunable;
    prunable.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (is_protected[i]) {
            ++protected_count;
        } else {
            prunable.push_back(static_cast<int>(i));
        }
    }

    const int desired = std::max(target_vocab_size,
                                 static_cast<int>(std::floor(shrink_factor * static_cast<double>(n))));
    int keep_prunable = desired - protected_count;
    if (keep_prunable < 0) keep_prunable = 0;
    if (keep_prunable > static_cast<int>(prunable.size())) {
        keep_prunable = static_cast<int>(prunable.size());
    }

    // Highest likelihood contribution = most valuable = kept first.
    std::sort(prunable.begin(), prunable.end(), [&](int a, int b) {
        if (loss[static_cast<size_t>(a)] != loss[static_cast<size_t>(b)]) {
            return loss[static_cast<size_t>(a)] > loss[static_cast<size_t>(b)];
        }
        return a < b;
    });

    std::vector<char> keep(n, 0);
    for (size_t i = 0; i < n; ++i) {
        if (is_protected[i]) keep[i] = 1;
    }
    for (int k = 0; k < keep_prunable; ++k) {
        keep[static_cast<size_t>(prunable[static_cast<size_t>(k)])] = 1;
    }

    UnigramLikelihoodPruningResult result;
    result.protected_count = protected_count;
    result.irreplaceable_no_alt = irreplaceable.load();
    // A no-progress round must leave the live vocabulary intact.
    if (protected_count + keep_prunable == static_cast<int>(n)) return result;
    // Moving survivors invalidates model contents before the coordinator rewrite.
    session.matches_generation = 0;
    session.expectation_generation = 0;
    result.survivors.reserve(static_cast<size_t>(protected_count) + static_cast<size_t>(keep_prunable));
    int kept = 0;
    for (size_t i = 0; i < n; ++i) {
        if (keep[i]) {
            result.survivors.push_back(std::move(pieces[i]));
            ++kept;
        }
    }
    result.removed = static_cast<int>(n) - kept;
    return result;
}

} // namespace Tokenizer
} // namespace GRIM
