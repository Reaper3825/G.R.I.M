//======================================================//
//  UnigramForwardBackward.cu
//  True Unigram LM forward-backward estimator for training
//======================================================//

#include "UnigramForwardBackward.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace GRIM {
namespace Tokenizer {

namespace {

constexpr double kLogZero = -std::numeric_limits<double>::infinity();
constexpr double kForwardBackwardConsistencyTolerance = 1.0e-5;
constexpr size_t kByteFallbackSpanBytes = 1;

static void requireCallerLabel(const char* caller) {
    if (caller == nullptr) {
        throw std::runtime_error("UnigramForwardBackward requires a non-null caller label at " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    if (caller[0] == '\0') {
        throw std::runtime_error("UnigramForwardBackward requires a non-empty caller label at " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
}

static bool isLogReachable(double value) {
    return value != kLogZero;
}

static double logAddExp(double lhs, double rhs) {
    if (!std::isfinite(lhs)) {
        if (lhs == kLogZero) return rhs;
        throw std::runtime_error("logAddExp received non-finite lhs that is not log-zero at " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    if (!std::isfinite(rhs)) {
        if (rhs == kLogZero) return lhs;
        throw std::runtime_error("logAddExp received non-finite rhs that is not log-zero at " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    if (rhs > lhs) {
        std::swap(lhs, rhs);
    }
    return lhs + std::log1p(std::exp(rhs - lhs));
}

static void requireStatsShape(const UnigramForwardBackwardStats& stats,
                              size_t piece_count,
                              const char* caller) {
    if (stats.piece_expected_counts.size() != piece_count) {
        throw std::runtime_error(std::string(caller) +
                                 ": UnigramForwardBackwardStats piece count mismatch: stats=" +
                                 std::to_string(stats.piece_expected_counts.size()) +
                                 ", lattice=" + std::to_string(piece_count) +
                                 " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
}

} // namespace

UnigramForwardBackwardStats::UnigramForwardBackwardStats(size_t piece_count)
    : piece_expected_counts(piece_count, 0.0) {}

UnigramForwardBackwardLattice::TrieNode::TrieNode() {
    children.fill(-1);
}

UnigramForwardBackwardLattice::UnigramForwardBackwardLattice(
    const std::vector<UnigramPiece>& pieces,
    bool enable_byte_fallback,
    const char* caller)
    : piece_count_(pieces.size()), enable_byte_fallback_(enable_byte_fallback) {
    requireCallerLabel(caller);
    if (pieces.empty()) {
        throw std::runtime_error(std::string(caller) +
                                 ": cannot build forward-backward lattice from an empty learned vocabulary at " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    trie_.reserve(pieces.size() + 1);
    trie_.emplace_back();
    for (size_t i = 0; i < pieces.size(); ++i) {
        if (i > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error(std::string(caller) +
                                     ": learned piece index exceeds int range during forward-backward lattice build: index=" +
                                     std::to_string(i) + " at " + std::string(__FILE__) + ":" +
                                     std::to_string(__LINE__));
        }
        insertPiece(pieces[i], static_cast<int>(i), caller);
    }
}

void UnigramForwardBackwardLattice::insertPiece(const UnigramPiece& piece,
                                                int piece_index,
                                                const char* caller) {
    requireCallerLabel(caller);
    if (piece.text.empty()) {
        throw std::runtime_error(std::string(caller) +
                                 ": learned piece text is empty during forward-backward lattice build at " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    if (piece.text.size() > static_cast<size_t>(MAX_PIECE_LENGTH)) {
        throw std::runtime_error(std::string(caller) +
                                 ": learned piece exceeds MAX_PIECE_LENGTH during forward-backward lattice build: text='" +
                                 piece.text + "', bytes=" + std::to_string(piece.text.size()) +
                                 ", max=" + std::to_string(MAX_PIECE_LENGTH) + " at " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    if (!std::isfinite(static_cast<double>(piece.score))) {
        throw std::runtime_error(std::string(caller) +
                                 ": learned piece score is not finite during forward-backward lattice build: text='" +
                                 piece.text + "' at " + std::string(__FILE__) + ":" +
                                 std::to_string(__LINE__));
    }

    int node = 0;
    for (unsigned char byte : piece.text) {
        int child = trie_[node].children[byte];
        if (child < 0) {
            if (trie_.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error(std::string(caller) +
                                         ": forward-backward trie node count exceeds int range at " +
                                         std::string(__FILE__) + ":" + std::to_string(__LINE__));
            }
            child = static_cast<int>(trie_.size());
            trie_[node].children[byte] = child;
            trie_.emplace_back();
        }
        node = child;
    }

    if (trie_[node].piece_index >= 0) {
        throw std::runtime_error(std::string(caller) +
                                 ": duplicate learned piece text in forward-backward lattice: text='" +
                                 piece.text + "', existing_index=" + std::to_string(trie_[node].piece_index) +
                                 ", duplicate_index=" + std::to_string(piece_index) + " at " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    trie_[node].piece_index = piece_index;
    trie_[node].score = static_cast<double>(piece.score);
}

void UnigramForwardBackwardLattice::accumulateSegment(
    const std::string& normalized_segment,
    UnigramForwardBackwardStats& stats,
    const char* caller) const {
    requireCallerLabel(caller);
    requireStatsShape(stats, piece_count_, caller);

    const size_t n = normalized_segment.size();
    if (n == 0) {
        return;
    }

    std::vector<double> alpha(n + 1, kLogZero);
    std::vector<double> beta(n + 1, kLogZero);
    alpha[0] = 0.0;

    for (size_t start = 0; start < n; ++start) {
        if (!isLogReachable(alpha[start])) {
            continue;
        }

        int node = 0;
        const size_t max_end = std::min(n, start + static_cast<size_t>(MAX_PIECE_LENGTH));
        for (size_t end = start + 1; end <= max_end; ++end) {
            const unsigned char byte = static_cast<unsigned char>(normalized_segment[end - 1]);
            const int child = trie_[node].children[byte];
            if (child < 0) {
                break;
            }
            node = child;

            const int piece_index = trie_[node].piece_index;
            if (piece_index >= 0) {
                const double candidate = alpha[start] + trie_[node].score;
                alpha[end] = logAddExp(alpha[end], candidate);
            }
        }

        if (enable_byte_fallback_) {
            // UNKNOWN_SCORE is an unnormalized fixed per-byte coverage penalty, not a
            // trainable probability mass competing in the M-step normalizer.
            // Fallback granularity is deliberately one raw byte, not one UTF-8 codepoint.
            const double fallback_candidate = alpha[start] + static_cast<double>(UNKNOWN_SCORE);
            alpha[start + kByteFallbackSpanBytes] = logAddExp(alpha[start + kByteFallbackSpanBytes], fallback_candidate);
        }
    }

    const double log_z = alpha[n];
    if (!isLogReachable(log_z)) {
        throw std::runtime_error(std::string(caller) +
                                 ": forward-backward E-step cannot segment normalized segment and byte fallback is disabled; bytes=" +
                                 std::to_string(n) + " at " + std::string(__FILE__) + ":" +
                                 std::to_string(__LINE__));
    }
    if (!std::isfinite(log_z)) {
        throw std::runtime_error(std::string(caller) +
                                 ": forward-backward partition value is not finite for segment bytes=" +
                                 std::to_string(n) + " at " + std::string(__FILE__) + ":" +
                                 std::to_string(__LINE__));
    }

    beta[n] = 0.0;
    for (size_t reverse_start = n; reverse_start > 0; --reverse_start) {
        const size_t start = reverse_start - 1;

        int node = 0;
        const size_t max_end = std::min(n, start + static_cast<size_t>(MAX_PIECE_LENGTH));
        for (size_t end = start + 1; end <= max_end; ++end) {
            const unsigned char byte = static_cast<unsigned char>(normalized_segment[end - 1]);
            const int child = trie_[node].children[byte];
            if (child < 0) {
                break;
            }
            node = child;

            const int piece_index = trie_[node].piece_index;
            if (piece_index >= 0 && isLogReachable(beta[end])) {
                const double candidate = trie_[node].score + beta[end];
                beta[start] = logAddExp(beta[start], candidate);
            }
        }

        if (enable_byte_fallback_ && isLogReachable(beta[start + kByteFallbackSpanBytes])) {
            // Keep the backward lattice consistent with the same fixed penalty
            // used by production Viterbi; do not infer a normalized fallback mass.
            // Fallback granularity is deliberately one raw byte, not one UTF-8 codepoint.
            const double fallback_candidate = static_cast<double>(UNKNOWN_SCORE) + beta[start + kByteFallbackSpanBytes];
            beta[start] = logAddExp(beta[start], fallback_candidate);
        }
    }

    if (!isLogReachable(beta[0])) {
        throw std::runtime_error(std::string(caller) +
                                 ": backward lattice cannot reach the segment start for bytes=" +
                                 std::to_string(n) + " at " + std::string(__FILE__) + ":" +
                                 std::to_string(__LINE__));
    }
    const double consistency_error = std::abs(beta[0] - log_z);
    const double scaled_tolerance = kForwardBackwardConsistencyTolerance *
                                    std::max(1.0, std::abs(log_z));
    if (consistency_error > scaled_tolerance) {
        throw std::runtime_error(std::string(caller) +
                                 ": forward/backward partition mismatch: forward=" +
                                 std::to_string(log_z) + ", backward=" + std::to_string(beta[0]) +
                                 ", abs_error=" + std::to_string(consistency_error) +
                                 ", tolerance=" + std::to_string(scaled_tolerance) + " at " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    for (size_t start = 0; start < n; ++start) {
        if (!isLogReachable(alpha[start])) {
            continue;
        }

        int node = 0;
        const size_t max_end = std::min(n, start + static_cast<size_t>(MAX_PIECE_LENGTH));
        for (size_t end = start + 1; end <= max_end; ++end) {
            const unsigned char byte = static_cast<unsigned char>(normalized_segment[end - 1]);
            const int child = trie_[node].children[byte];
            if (child < 0) {
                break;
            }
            node = child;

            const int piece_index = trie_[node].piece_index;
            if (piece_index >= 0 && isLogReachable(beta[end])) {
                const double log_posterior = alpha[start] + trie_[node].score + beta[end] - log_z;
                const double posterior = std::exp(log_posterior);
                if (!std::isfinite(posterior)) {
                    throw std::runtime_error(std::string(caller) +
                                             ": non-finite posterior in forward-backward E-step: start=" +
                                             std::to_string(start) + ", end=" + std::to_string(end) +
                                             ", piece_index=" + std::to_string(piece_index) + " at " +
                                             std::string(__FILE__) + ":" + std::to_string(__LINE__));
                }
                stats.piece_expected_counts[static_cast<size_t>(piece_index)] += posterior;
                stats.expected_learned_piece_tokens += posterior;
            }
        }

        if (enable_byte_fallback_ && isLogReachable(beta[start + kByteFallbackSpanBytes])) {
            // Telemetry only: this posterior measures how often the fixed
            // penalty path is used under the current lattice. The tokenizer
            // M-step deliberately excludes it from learned-piece normalization.
            // A multibyte UTF-8 codepoint therefore contributes one fallback transition per byte.
            const double log_posterior = alpha[start] + static_cast<double>(UNKNOWN_SCORE) + beta[start + kByteFallbackSpanBytes] - log_z;
            const double posterior = std::exp(log_posterior);
            if (!std::isfinite(posterior)) {
                throw std::runtime_error(std::string(caller) +
                                         ": non-finite byte-fallback posterior in forward-backward E-step: start=" +
                                         std::to_string(start) + " at " + std::string(__FILE__) + ":" +
                                         std::to_string(__LINE__));
            }
            stats.expected_fixed_penalty_byte_fallback_tokens += posterior;
        }
    }

    stats.log_likelihood += log_z;
}

} // namespace Tokenizer
} // namespace GRIM
