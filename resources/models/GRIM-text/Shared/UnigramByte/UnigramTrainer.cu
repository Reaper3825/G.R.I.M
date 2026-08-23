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
#include "NumericTokens.hpp"
#include "TextUtils.hpp"
#include "Training/SubwordMining.hpp"
#include "Training/UnigramForwardBackward.hpp"
#include "VocabWriteOp.hpp"
#include "HyperParameters/HyperparameterGroupings.hpp"

#include <algorithm>
#include <array>
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

struct UnigramLikelihoodLossPruneResult {
    std::vector<UnigramPiece> survivors;
    int removed = 0;
    int protected_count = 0;
    int irreplaceable_no_alt = 0;
};

// Select the highest-likelihood-contribution pieces toward a target size,
// always retaining protected (single-codepoint / user-defined / no-alt)
// pieces. Survivors are moved out of `pieces` in original order so token
// IDs stay position-derived after rewriteUnigramVocab().
static UnigramLikelihoodLossPruneResult selectUnigramSurvivorsByLikelihoodLoss(
    std::vector<UnigramPiece>& pieces,
    const std::vector<double>& freq_by_index,
    int target_vocab_size,
    double shrink_factor,
    int worker_count,
    const char* caller) {
    if (caller == nullptr || caller[0] == '\0') {
        throw std::runtime_error("selectUnigramSurvivorsByLikelihoodLoss: caller label is empty");
    }
    const size_t n = pieces.size();
    if (freq_by_index.size() != n) {
        throw std::runtime_error(std::string(caller) +
                                 ": freq_by_index size=" + std::to_string(freq_by_index.size()) +
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
            loss[i] = freq_by_index[i] * (keep_logprob - alt);
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

    UnigramLikelihoodLossPruneResult result;
    result.protected_count = protected_count;
    result.irreplaceable_no_alt = irreplaceable.load();
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
    size_t total_numeric_spans = 0;
    const bool detect = tokenizer_hp.enable_atom_reasoning;
    for (const auto& text : texts) {
        std::vector<AtomSpan> spans;
        if (detect) {
            const Detector::RawTextDetectorOptions detector_options(
                true,
                true);
            const auto detections = detector_registry.scan(text, detector_options);
            const AtomTableFromDetectionsResult atom_table_build = createAtomTableFromRawTextDetections(
                std::string_view(text.data(), text.size()),
                detections,
                "UnigramLM::trainFromCorpus");
            spans.reserve(atom_table_build.atom_tokens.size());
            for (const AtomTokenizationPayload& atom_payload : atom_table_build.atom_tokens) {
                spans.push_back({atom_payload.span.start, atom_payload.span.end});
            }
            total_atoms += spans.size();
        }

        std::vector<NumericTokenSpan> atom_exclusions;
        atom_exclusions.reserve(spans.size());
        for (const AtomSpan& span : spans) {
            atom_exclusions.push_back(NumericTokenSpan{span.start, span.end});
        }
        const std::vector<NumericTokenSpan> numeric_spans = findNumericTokenSpans(
            std::string_view(text.data(), text.size()), atom_exclusions);
        spans.reserve(spans.size() + numeric_spans.size());
        for (const NumericTokenSpan& span : numeric_spans) {
            spans.push_back(AtomSpan{span.start, span.end});
        }
        std::sort(spans.begin(), spans.end(), [](const AtomSpan& lhs, const AtomSpan& rhs) {
            return lhs.start < rhs.start;
        });
        total_numeric_spans += numeric_spans.size();
        all_atom_spans.push_back(std::move(spans));
    }

    std::cout << "[UnigramLM] Detected " << total_atoms << " atoms across "
              << texts.size() << " texts (will skip during vocab training); "
              << "atom_reasoning=" << (detect ? "on" : "off") << std::endl;
    std::cout << "[UnigramLM] Identified " << total_numeric_spans
              << " non-atom numeric spans (fixed numeric vocabulary; will skip during learned-vocab training)"
              << std::endl;

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
    
    const UnigramSubwordMiningResult mining_result = mineUnigramSubwordsFromTrainingUnits(
        UnigramSubwordMiningRequest{
            training_units,
            norm_atom_spans,
            enable_parallel_subword_mining,
            subword_mining_workers,
            subword_mining_max_bytes,
            "[UnigramLM]"});
    
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

        if (isUnigramRepetitionNoise(subword)) {
            repetition_filtered++;
            continue;
        }
        if (!isValidUnigramSubword(subword)) {
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
              << "; data-driven vocab, no target cap), total candidate vocab: " << pieces_.size() << std::endl;
    if (pieces_.empty()) {
        throw std::runtime_error("UnigramLM::trainFromCorpus: no learned subword candidates survived frequency/validity filters; lower min_subword_freq or provide more corpus text");
    }
    
    // Step 5: Soft EM + likelihood-loss prune-to-target (SentencePiece-style)
    //
    // Starting from the full data-selected candidate vocab, we:
    //   1. Run true Unigram forward-backward EM to convergence (Phase-A).
    //   2. Iteratively prune the lowest-likelihood-contribution pieces toward
    //      target_vocab_size, re-fitting with a few EM sub-iterations each
    //      round, always protecting single-codepoint / user-defined / no-alt
    //      coverage pieces (Prune rounds).
    //   3. Re-converge EM on the final vocab (Final).
    //
    // Raw frequency cannot separate a real word from its dead prefixes
    // (they share the same substring count); only the EM likelihood signal
    // can, which is why pruning is posterior/likelihood-driven, not a
    // frequency cap. Single-char coverage is provided by single-char unigram
    // pieces; byte fallback is never a protected vocab entry.
    constexpr int    EM_MAX_ITERATIONS     = 50;
    constexpr double EM_CONVERGENCE_THRESH = 0.0001;
    constexpr double SMOOTHING             = 0.1;
    // SentencePiece-style EM + likelihood-loss prune-to-target controls.
    constexpr double PRUNE_SHRINK_FACTOR   = 0.75;  // max fraction kept per prune round
    constexpr int    PRUNE_SUB_ITERATIONS  = 2;     // EM re-fit iterations after each prune round

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

    // ---- Phase A: EM to convergence on the full data-selected candidate vocab ----
    std::cout << "[UnigramLM] Phase-A: forward-backward EM on data-selected candidate vocab ("
              << pieces_.size() << " pieces; pruning toward target_vocab_size=" << target_vocab_size
              << ")" << std::endl;
    EMConvergenceResult em_result = runEMToConvergence("Phase-A");
    requireConvergedPhaseNotFallbackDominated(em_result, "Phase-A");

    // ---- Prune rounds: drop lowest-likelihood-contribution pieces toward target ----
    // Each round computes per-piece likelihood loss from the current EM
    // posterior, prunes toward max(target, shrink*current) while protecting
    // single-codepoint / user-defined / no-alternative coverage pieces, then
    // re-fits scores with a few EM sub-iterations.
    int prune_round = 0;
    while (static_cast<int>(pieces_.size()) > target_vocab_size) {
        std::vector<double> freq_by_index(pieces_.size(), 0.0);
        for (const auto& token_count : em_result.learned_token_counts) {
            const int idx = indexForTokenId(token_count.first);
            if (idx >= 0 && idx < static_cast<int>(freq_by_index.size())) {
                freq_by_index[static_cast<size_t>(idx)] = token_count.second;
            }
        }

        const int before = static_cast<int>(pieces_.size());
        UnigramLikelihoodLossPruneResult prune = selectUnigramSurvivorsByLikelihoodLoss(
            pieces_,
            freq_by_index,
            target_vocab_size,
            PRUNE_SHRINK_FACTOR,
            subword_mining_workers,
            "UnigramLM::trainFromCorpus likelihood-loss prune");

        if (prune.removed <= 0) {
            std::cout << "[UnigramLM] Likelihood-loss prune: no further prunable pieces (protected="
                      << prune.protected_count << ", irreplaceable_no_alt=" << prune.irreplaceable_no_alt
                      << "); stopping at " << before << " pieces, above target_vocab_size="
                      << target_vocab_size << " (coverage pieces cannot be pruned)" << std::endl;
            break;
        }

        rewriteUnigramVocab(
            UnigramVocabWriteTarget{pieces_, piece_to_id_},
            std::move(prune.survivors),
            "UnigramLM::trainFromCorpus likelihood-loss prune compaction");
        ++prune_round;
        std::cout << "[UnigramLM] Likelihood-loss prune round " << prune_round << ": "
                  << before << " -> " << pieces_.size() << " pieces (removed " << prune.removed
                  << ", protected=" << prune.protected_count
                  << ", irreplaceable_no_alt=" << prune.irreplaceable_no_alt << ")" << std::endl;

        // Re-fit scores on the smaller vocab before the next loss estimate.
        EStepResult sub_e_step;
        for (int sub_iter = 0; sub_iter < PRUNE_SUB_ITERATIONS; ++sub_iter) {
            sub_e_step = runEStep();
            runMStep(sub_e_step.learned_token_counts, sub_e_step.expected_learned_piece_tokens);
        }
        em_result.learned_token_counts = std::move(sub_e_step.learned_token_counts);
        em_result.expected_learned_piece_tokens = sub_e_step.expected_learned_piece_tokens;
        em_result.expected_fixed_penalty_byte_fallback_tokens =
            sub_e_step.expected_fixed_penalty_byte_fallback_tokens;
    }

    // ---- Final: re-converge EM on the pruned vocab ----
    std::cout << "[UnigramLM] Final: re-converging EM after " << prune_round
              << " prune round(s) (vocab now " << pieces_.size() << " pieces)" << std::endl;
    EMConvergenceResult final_result = runEMToConvergence("Final");
    requireConvergedPhaseNotFallbackDominated(final_result, "Final");

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
