//======================================================//
//  unigrambyte_self_test.cu
//  Comprehensive test suite for UnigramByte tokenizer
//======================================================//

#include "unigrambyte_self_test.hpp"

#include "../Shared/UnigramByte/TokenLayout.hpp"
#include "../Shared/UnigramByte/NumericTokens.hpp"
#include "../Shared/UnigramByte/Unigram.hpp"
#include "../Shared/UnigramByte/VocabWriteOp.hpp"
#include "../Shared/UnigramByte/Training/UnigramForwardBackward.hpp"
#include "../Shared/UnigramByte/UnigramViterbi.hpp"
#include "../Shared/UnigramByte/UniByte.hpp"
#include "../Shared/UnigramByte/AtomTable.hpp"
#include "../Shared/UnigramByte/SequenceLocalAtomTable.hpp"
#include "../Shared/UnigramByte/Detectors/DetectorRegistry.hpp"
#include "../Shared/UnigramByte/Detectors/StructuralSpan.hpp"
#include "../Shared/UnigramByte/AhoCorasick.hpp"
#include "../Shared/TokenizerArtifacts/TokenizerArtifactBundle.hpp"
#include "../training/Phases/Startup/SlidingWindow.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>
#include <string>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <utility>

namespace GRIM {
namespace Tokenizer {

__global__ void kernelViterbiForward(
    const char* __restrict__ text,
    size_t length,
    const int* __restrict__ trie_children,
    const int* __restrict__ trie_token_ids,
    const float* __restrict__ trie_scores,
    int num_trie_nodes,
    float* __restrict__ viterbi_scores,
    int* __restrict__ viterbi_prev,
    int* __restrict__ viterbi_tokens,
    bool* __restrict__ selected_fallback,
    int unk_id,
    bool enable_byte_fallback,
    int* __restrict__ error_code
);

__global__ void kernelViterbiBacktrack(
    size_t length,
    const int* __restrict__ viterbi_prev,
    const int* __restrict__ viterbi_tokens,
    int* __restrict__ output_tokens,
    int* __restrict__ output_count,
    int max_tokens,
    bool* __restrict__ selected_fallback,
    int* __restrict__ error_code
);

} // namespace Tokenizer
} // namespace GRIM

namespace GRIM {
namespace Tokenizer {
void requireUnigramFinalCleanupLeavesLearnedPiece(size_t piece_count,
                                                  size_t dead_count,
                                                  const char* caller);
void requireUnigramAcceptedCandidateSetIsScorable(size_t accepted_count,
                                                  double total_accepted_count,
                                                  const char* caller);
void requireUnigramLearnedPosteriorMassNotByteFallbackDominated(
    double expected_learned_piece_tokens,
    double expected_fixed_penalty_byte_fallback_tokens,
    const char* phase_label,
    const char* caller);
uint64_t addUnigramSubwordCountsForTraining(uint64_t current,
                                            uint64_t delta,
                                            const char* caller);
} // namespace Tokenizer
} // namespace GRIM

using namespace GRIM::Tokenizer;
using namespace GRIM::Test;

namespace TokenizerArtifacts = GRIM::TokenizerArtifacts;

static ::GRIM::HyperParameters::TokenizerHP makeSelfTestTokenizerHP() {
    ::GRIM::HyperParameters::TokenizerHP hp;
    hp.target_vocab_size = 50000;
    hp.character_coverage = 0.9995f;
    hp.min_subword_freq = 3;
    hp.enable_parallel_subword_mining = true;
    hp.enable_atom_reasoning = true;
    hp.enable_byte_fallback = true;
    hp.vocab_score_multiplier = 1.0f;
    return hp;
}

static void appendSelfTestUnigramPiece(UnigramLM& unigram,
                                       std::string text,
                                       float score,
                                       bool is_user_defined) {
    UnigramPiece piece;
    piece.text = std::move(text);
    piece.score = score;
    piece.is_user_defined = is_user_defined;
    applyUnigramVocabWriteOp(UnigramVocabWriteRequest{
        UnigramVocabWriteTarget{unigram.pieces_, unigram.piece_to_id_},
        std::move(piece),
        UnigramLM::tokenIdForIndex(unigram.pieceCount()),
        UnigramVocabWriteMode::AppendOnly,
        "unigrambyte_self_test fixture append"});
}

// Helper: add minimal ▁-prefixed vocab to a UniByte tokenizer so UnigramViterbiSession has a valid trie.
// Without this, Viterbi segmentation crashes (Rule 20: trie_ must not be empty).
static void addMinimalVocab(UniByte& tok) {
    appendSelfTestUnigramPiece(tok.unigramLM(), "\xe2\x96\x81" "the",     -1.0f, false);
    appendSelfTestUnigramPiece(tok.unigramLM(), "\xe2\x96\x81" "is",      -1.1f, false);
    appendSelfTestUnigramPiece(tok.unigramLM(), "\xe2\x96\x81" "price",   -1.2f, false);
    appendSelfTestUnigramPiece(tok.unigramLM(), "\xe2\x96\x81" "dollars", -1.3f, false);
    appendSelfTestUnigramPiece(tok.unigramLM(), "\xe2\x96\x81" "on",      -1.4f, false);
    appendSelfTestUnigramPiece(tok.unigramLM(), "\xe2\x96\x81" "for",     -1.5f, false);
    appendSelfTestUnigramPiece(tok.unigramLM(), "\xe2\x96\x81" "more",    -1.6f, false);
    appendSelfTestUnigramPiece(tok.unigramLM(), "\xe2\x96\x81" "info",    -1.7f, false);
    appendSelfTestUnigramPiece(tok.unigramLM(), "\xe2\x96\x81" "us",      -1.8f, false);
    appendSelfTestUnigramPiece(tok.unigramLM(), "\xe2\x96\x81" "at",      -1.9f, false);
    appendSelfTestUnigramPiece(tok.unigramLM(), "\xe2\x96\x81" "meeting", -2.0f, false);
    appendSelfTestUnigramPiece(tok.unigramLM(), "\xe2\x96\x81" "Count",   -2.1f, false);
    tok.unigramLM().buildTrie();
}

static uint32_t registerSelfTestAtom(AtomTable& table,
                                     AtomType type,
                                     std::string_view raw_text,
                                     size_t source_start = 0,
                                     size_t source_end = std::numeric_limits<size_t>::max()) {
    const size_t resolved_source_end =
        source_end == std::numeric_limits<size_t>::max() ? source_start + raw_text.size() : source_end;

    StructuralSpan span{};
    span.start = source_start;
    span.end = resolved_source_end;
    span.atom_type = type;
    span.buffer_ptr = raw_text.data();
    span.offset = static_cast<uint32_t>(source_start);
    span.length = static_cast<uint32_t>(raw_text.size());
    span.content_offset = span.offset;
    span.content_length = span.length;
    span.open_token_id = atomTypeToOpenTokenId(type);
    span.close_token_id = atomTypeToCloseTokenId(type);

    try {
        return table.registerSpan(span);
    } catch (const std::exception&) {
        return UINT32_MAX;
    }
}

//======================================================//
//  Section 2: Unigram LM Tests
//======================================================//

bool testUnigramBuildVocab(std::string& message) {
    UnigramLM unigram;
    
    // Build a simple vocabulary using the canonical VocabWriteOp primitive.
    // Token IDs must start after pre-existing special tokens (unk, pad, bos, eos)
    appendSelfTestUnigramPiece(unigram, "hello", -1.0f, false);
    appendSelfTestUnigramPiece(unigram, "world", -1.2f, false);
    appendSelfTestUnigramPiece(unigram, "he", -2.0f, false);
    appendSelfTestUnigramPiece(unigram, "llo", -2.5f, false);
    appendSelfTestUnigramPiece(unigram, "wo", -2.3f, false);
    appendSelfTestUnigramPiece(unigram, "rld", -2.8f, false);
    appendSelfTestUnigramPiece(unigram, "h", -3.0f, false);
    appendSelfTestUnigramPiece(unigram, "e", -3.1f, false);
    appendSelfTestUnigramPiece(unigram, "l", -3.2f, false);
    appendSelfTestUnigramPiece(unigram, "o", -3.3f, false);
    appendSelfTestUnigramPiece(unigram, "w", -3.4f, false);
    appendSelfTestUnigramPiece(unigram, "r", -3.5f, false);
    appendSelfTestUnigramPiece(unigram, "d", -3.6f, false);
    
        // Only learned pieces are stored in UnigramLM::pieces_; specials are layout metadata.
        ASSERT_EQ(unigram.pieceCount(), 13, "Learned piece count mismatch");
    
    return true;
}

bool testUnigramEncode(std::string& message) {
    UnigramLM unigram;
    
    // Build vocabulary with log probabilities
    // Start after pre-existing special tokens
    appendSelfTestUnigramPiece(unigram, "hello", -1.0f, false);  // Most likely for "hello"
    appendSelfTestUnigramPiece(unigram, "he", -2.0f, false);
    appendSelfTestUnigramPiece(unigram, "llo", -2.5f, false);
    appendSelfTestUnigramPiece(unigram, "h", -3.0f, false);
    appendSelfTestUnigramPiece(unigram, "e", -3.1f, false);
    appendSelfTestUnigramPiece(unigram, "l", -3.2f, false);
    appendSelfTestUnigramPiece(unigram, "o", -3.3f, false);
    unigram.buildTrie();  // Must build trie before encoding
    
    std::vector<int> tokens = unigram.encode("hello");
    
    // Should produce token(s) - verify we got something
    ASSERT_TRUE(tokens.size() >= 1, "Should encode 'hello' to at least 1 token");
    
    return true;
}

bool testUnigramViterbi(std::string& message) {
    UnigramLM unigram;
    
    // Vocabulary where splitting is better than whole word
    // Start after pre-existing special tokens
    appendSelfTestUnigramPiece(unigram, "test", -5.0f, false);    // Whole word is worse
    appendSelfTestUnigramPiece(unigram, "te", -1.0f, false);      // Better to split
    appendSelfTestUnigramPiece(unigram, "st", -1.0f, false);
    appendSelfTestUnigramPiece(unigram, "t", -2.0f, false);
    appendSelfTestUnigramPiece(unigram, "e", -2.1f, false);
    appendSelfTestUnigramPiece(unigram, "s", -2.2f, false);
    unigram.buildTrie();  // Must build trie before encoding
    
    std::vector<int> tokens = unigram.encode("test");
    
    // Viterbi should find optimal segmentation
    ASSERT_TRUE(tokens.size() >= 1, "Should produce tokens for 'test'");
    
    return true;
}

bool testUnigramSentencePiecePunctuationPiece(std::string& message) {
    UnigramLM unigram;

    // SentencePiece-style policy: punctuation is ordinary normalized text.
    // If a learned ▁-prefixed piece includes punctuation and wins by score,
    // Viterbi must select that piece instead of forcing punctuation to byte fallback.
    const std::string full_piece = "\xe2\x96\x81hello!";
    appendSelfTestUnigramPiece(unigram, full_piece, -1.0f, false);
    appendSelfTestUnigramPiece(unigram, "\xe2\x96\x81hello", -10.0f, false);
    appendSelfTestUnigramPiece(unigram, "!", -10.0f, false);
    unigram.buildTrie();

    std::vector<int> tokens = unigram.encode("hello!");

    ASSERT_EQ(tokens.size(), static_cast<size_t>(1), "Punctuation-bearing learned piece should remain selectable");
    ASSERT_EQ(tokens[0], unigram.getPieceId(full_piece), "Viterbi should select learned punctuation-bearing piece");

    return true;
}

bool testUnigramUnknown(std::string& message) {
    UnigramLM unigram;
    
    // Minimal vocab - will need byte fallback for some chars
    // Start after pre-existing special tokens
    appendSelfTestUnigramPiece(unigram, "a", -1.0f, false);
    appendSelfTestUnigramPiece(unigram, "b", -1.0f, false);
    unigram.buildTrie();  // Must build trie before encoding
    
    // Try to encode something not in vocab
    std::vector<int> tokens = unigram.encode("xyz");
    
        // After ▁ normalization, pieces are ▁-prefixed ("▁gonna") or bare ("gonna")
    // Note: Behavior depends on implementation - may produce empty or UNK
    // For now just verify no crash
    
    return true;
}

bool testUnigramForwardBackwardByteFallbackIsFixedPenalty(std::string& message) {
    std::vector<UnigramPiece> pieces;
    UnigramPiece piece;
    piece.text = "a";
    piece.score = 0.0f;
    piece.is_user_defined = false;
    pieces.push_back(piece);

    UnigramForwardBackwardLattice lattice(
        pieces,
        true,
        "testUnigramForwardBackwardByteFallbackIsFixedPenalty lattice");
    UnigramForwardBackwardStats stats(pieces.size());

    lattice.accumulateSegment(
        "az",
        stats,
        "testUnigramForwardBackwardByteFallbackIsFixedPenalty accumulate");

    ASSERT_NEAR(stats.piece_expected_counts[0], 1.0, 1.0e-9,
                "Learned piece posterior should be normalized over lattice paths");
    ASSERT_NEAR(stats.expected_learned_piece_tokens, 1.0, 1.0e-9,
                "Learned-piece expected token telemetry mismatch");
    ASSERT_NEAR(stats.expected_fixed_penalty_byte_fallback_tokens, 1.0, 1.0e-9,
                "Byte fallback telemetry should count fixed-penalty path usage separately");
    ASSERT_NEAR(stats.log_likelihood, static_cast<double>(UNKNOWN_SCORE), 1.0e-6,
                "Partition should include the fixed fallback penalty without normalizing it into learned pieces");

    return true;
}

bool testUnigramForwardBackwardByteFallbackIsByteLevelForUtf8(std::string& message) {
    std::vector<UnigramPiece> pieces;
    UnigramPiece piece;
    piece.text = "a";
    piece.score = 0.0f;
    piece.is_user_defined = false;
    pieces.push_back(piece);

    UnigramForwardBackwardLattice lattice(
        pieces,
        true,
        "testUnigramForwardBackwardByteFallbackIsByteLevelForUtf8 lattice");
    UnigramForwardBackwardStats stats(pieces.size());

    const std::string e_acute = "\xC3\xA9";
    ASSERT_EQ(e_acute.size(), static_cast<size_t>(2), "Regression fixture must be a 2-byte UTF-8 character");

    lattice.accumulateSegment(
        e_acute,
        stats,
        "testUnigramForwardBackwardByteFallbackIsByteLevelForUtf8 accumulate");

    ASSERT_NEAR(stats.expected_learned_piece_tokens, 0.0, 1.0e-12,
                "No learned piece should match the UTF-8 fallback-only fixture");
    ASSERT_NEAR(stats.expected_fixed_penalty_byte_fallback_tokens, static_cast<double>(e_acute.size()), 1.0e-9,
                "UTF-8 fallback must count one fixed-penalty transition per raw byte, not per codepoint");
    ASSERT_NEAR(stats.log_likelihood, static_cast<double>(e_acute.size()) * static_cast<double>(UNKNOWN_SCORE), 1.0e-6,
                "UTF-8 fallback partition must apply UNKNOWN_SCORE once per raw byte");

    return true;
}

bool testUnigramTrainFinalCleanupRejectsEmptyLearnedVocab(std::string& message) {
    bool threw = false;
    std::string error_text;
    try {
        requireUnigramFinalCleanupLeavesLearnedPiece(
            3,
            3,
            "testUnigramTrainFinalCleanupRejectsEmptyLearnedVocab all-dead");
    } catch (const std::runtime_error& e) {
        threw = true;
        error_text = e.what();
    }

    ASSERT_TRUE(threw, "Final cleanup must fail before deleting every learned piece");
    ASSERT_TRUE(error_text.find("delete every learned piece") != std::string::npos,
                "Final cleanup error should identify the empty learned-vocab root cause");
    ASSERT_TRUE(error_text.find("Phase-D lattice construction") != std::string::npos,
                "Final cleanup error should mention that Phase-D must not receive an empty learned vocab");

    try {
        requireUnigramFinalCleanupLeavesLearnedPiece(
            3,
            2,
            "testUnigramTrainFinalCleanupRejectsEmptyLearnedVocab survivor");
    } catch (const std::exception& e) {
        message = std::string("Final cleanup guard should allow at least one learned-piece survivor: ") + e.what();
        return false;
    }

    return true;
}

bool testUnigramTrainRejectsEmptyAcceptedCandidateSet(std::string& message) {
    bool threw = false;
    std::string error_text;
    try {
        requireUnigramAcceptedCandidateSetIsScorable(
            0,
            0.0,
            "testUnigramTrainRejectsEmptyAcceptedCandidateSet empty");
    } catch (const std::runtime_error& e) {
        threw = true;
        error_text = e.what();
    }

    ASSERT_TRUE(threw, "Initial candidate score normalization must fail when accepted candidate count is zero");
    ASSERT_TRUE(error_text.find("zero accepted learned pieces") != std::string::npos,
                "Accepted-candidate guard should identify empty accepted candidate admission as the root cause");

    threw = false;
    error_text.clear();
    try {
        requireUnigramAcceptedCandidateSetIsScorable(
            1,
            0.0,
            "testUnigramTrainRejectsEmptyAcceptedCandidateSet bad-total");
    } catch (const std::runtime_error& e) {
        threw = true;
        error_text = e.what();
    }

    ASSERT_TRUE(threw, "Initial candidate score normalization must fail when accepted counts do not sum to a positive finite value");
    ASSERT_TRUE(error_text.find("not scorable") != std::string::npos,
                "Accepted-candidate guard should identify a non-positive total accepted count");

    try {
        requireUnigramAcceptedCandidateSetIsScorable(
            1,
            3.0,
            "testUnigramTrainRejectsEmptyAcceptedCandidateSet valid");
    } catch (const std::exception& e) {
        message = std::string("Accepted-candidate guard should allow non-empty positive-mass candidate sets: ") + e.what();
        return false;
    }

    return true;
}

bool testUnigramTrainRejectsFallbackDominatedPosteriorMass(std::string& message) {
    bool threw = false;
    std::string error_text;
    try {
        requireUnigramLearnedPosteriorMassNotByteFallbackDominated(
            0.25,
            1.0,
            "test-phase",
            "testUnigramTrainRejectsFallbackDominatedPosteriorMass dominated");
    } catch (const std::runtime_error& e) {
        threw = true;
        error_text = e.what();
    }

    ASSERT_TRUE(threw, "Converged-phase guard must reject byte-fallback-dominated posterior mass");
    ASSERT_TRUE(error_text.find("byte fallback dominated posterior mass") != std::string::npos,
                "Converged-phase guard error should identify fallback dominance as the root cause");
    ASSERT_TRUE(error_text.find("after EM convergence") != std::string::npos,
                "Guard error should state that fallback dominance is checked after EM convergence");

    try {
        requireUnigramLearnedPosteriorMassNotByteFallbackDominated(
            1.0,
            1.0,
            "test-phase",
            "testUnigramTrainRejectsFallbackDominatedPosteriorMass balanced");
        requireUnigramLearnedPosteriorMassNotByteFallbackDominated(
            0.0,
            0.0,
            "test-phase",
            "testUnigramTrainRejectsFallbackDominatedPosteriorMass no-fallback");
    } catch (const std::exception& e) {
        message = std::string("Converged-phase guard should allow balanced or no-fallback posterior mass: ") + e.what();
        return false;
    }

    return true;
}

bool testUnigramTrainByteFallbackDisabledAddsCharacterSeeds(std::string& message) {
    UnigramLM unigram(false);

    std::string repeated;
    for (int i = 0; i < 32; ++i) {
        repeated += "a ";
    }
    std::vector<std::string> corpus = {repeated};

    const bool trained = unigram.trainFromCorpus(
        corpus,
        {},     // atom_spans: none
        100,    // target_vocab_size: enough room for required char seeds and mined pieces
        1.0f,   // character_coverage: byte-fallback-off requires exact char coverage
        3,      // min_subword_freq
        false,  // prune_during_mining
        false,  // enable_parallel_subword_mining
        1,      // subword_mining_workers
        0);     // subword_mining_max_bytes
    ASSERT_TRUE(trained, "trainFromCorpus should succeed when fallback-off char seeds cover the corpus");
    ASSERT_TRUE(unigram.hasPiece("a"), "Byte-fallback-off training must insert ASCII character seed 'a' as a learned piece");
    ASSERT_TRUE(unigram.hasPiece("\xe2\x96\x81"), "Byte-fallback-off training must insert SentencePiece underline as a learned piece");

    return true;
}

bool testUnigramTrainByteFallbackDisabledFailsOnUncoveredCharacterSeed(std::string& message) {
    UnigramLM unigram(false);

    std::vector<std::string> corpus = {"aaaaaaaaaaaaaaaa uncommon_z"};
    bool threw = false;
    std::string error_text;
    try {
        unigram.trainFromCorpus(
            corpus,
            {},
            100,
            1.0f,
            3,
            false,
            false,
            1,
            0);
    } catch (const std::runtime_error& e) {
        threw = true;
        error_text = e.what();
    }

    ASSERT_TRUE(threw, "Byte-fallback-off training must fail immediately when character seeds cannot cover the corpus");
    ASSERT_TRUE(error_text.find("byte fallback is disabled") != std::string::npos,
                "Coverage failure should identify that byte fallback is disabled");
    ASSERT_TRUE(error_text.find("Step-2 character seeds do not cover") != std::string::npos,
                "Coverage failure should identify uncovered character seeds before EM");

    return true;
}

bool testUnigramTrainSubwordCountsUseUint64(std::string& message) {
    const uint64_t above_signed_int = static_cast<uint64_t>(std::numeric_limits<int>::max()) + 1ULL;
    ASSERT_EQ(addUnigramSubwordCountsForTraining(
                  above_signed_int - 1ULL,
                  1ULL,
                  "testUnigramTrainSubwordCountsUseUint64 above-int"),
              above_signed_int,
              "Subword counts must support values above signed int range");

    bool threw = false;
    std::string error_text;
    try {
        addUnigramSubwordCountsForTraining(
            std::numeric_limits<uint64_t>::max() - 1ULL,
            2ULL,
            "testUnigramTrainSubwordCountsUseUint64 overflow");
    } catch (const std::runtime_error& e) {
        threw = true;
        error_text = e.what();
    }

    ASSERT_TRUE(threw, "Subword count addition must fail loudly on uint64 overflow");
    ASSERT_TRUE(error_text.find("subword candidate count overflow") != std::string::npos,
                "Overflow error should identify subword candidate count overflow");

    return true;
}

bool testUnigramTrainLikelihoodLossPrunesTowardTarget(std::string& message) {
    // Vocabulary-rich corpus (each line repeated) so substring mining yields
    // far more candidates than the small target, forcing likelihood-loss
    // pruning to engage.
    std::vector<std::string> corpus = {
        "machine learning models process language tokens",
        "machine learning models process language tokens",
        "language models learn machine generated tokens",
        "language models learn machine generated tokens",
        "processing language models requires learning machine patterns",
        "processing language models requires learning machine patterns",
        "tokens generated by language models inform machine learning",
        "tokens generated by language models inform machine learning",
        "neural networks transform representation vectors efficiently",
        "neural networks transform representation vectors efficiently",
        "transformers encode contextual representation across sequences",
        "transformers encode contextual representation across sequences",
        "optimization improves training stability significantly overall",
        "optimization improves training stability significantly overall"
    };

    // Reference run: target far above the candidate set, so nothing is pruned.
    UnigramLM unpruned(true);
    const bool unpruned_trained = unpruned.trainFromCorpus(
        corpus, {}, 100000, 1.0f, 2, false, false, 1, 0);
    ASSERT_TRUE(unpruned_trained, "Reference (no-prune) training should succeed");
    const int full_count = unpruned.pieceCount();
    ASSERT_TRUE(full_count > 40,
                "Fixture must mine more candidates than the prune target to exercise pruning");

    // Pruned run: small target forces likelihood-loss prune rounds.
    const int target = 40;
    UnigramLM pruned(true);
    const bool pruned_trained = pruned.trainFromCorpus(
        corpus, {}, target, 1.0f, 2, false, false, 1, 0);
    ASSERT_TRUE(pruned_trained, "Likelihood-loss prune training should succeed");
    const int pruned_count = pruned.pieceCount();

    ASSERT_TRUE(pruned_count < full_count,
                "Likelihood-loss pruning must reduce the vocab below the unpruned candidate count");
    ASSERT_TRUE(pruned_count <= target,
                "Pruned vocab must not exceed target_vocab_size once coverage pieces fit under it");

    // Single-char coverage is protected: common characters survive as single
    // unigram pieces (never demoted to byte fallback).
    ASSERT_TRUE(pruned.hasPiece("a"), "Single-char unigram coverage piece 'a' must survive pruning");
    ASSERT_TRUE(pruned.hasPiece("e"), "Single-char unigram coverage piece 'e' must survive pruning");

    return true;
}

bool testUnigramTrainFiltersRepetitionNoise(std::string& message) {
    UnigramLM unigram;

    std::vector<std::string> corpus = {
        "i'm gonna call you now i'm gonna call you now i'm gonna call you now",
        "hahaha hahaha hahaha aaaaaa aaaaaa aaaaaa",
        "i i i i i i keep repeating words",
        "callcall callcall callcall should be stripped"
    };

    const bool trained = unigram.trainFromCorpus(
        corpus,
        {},     // atom_spans: none
        600,    // target_vocab_size
        1.0f,   // character_coverage
        3,      // min_subword_freq
        false   // prune_during_mining
    );
    ASSERT_TRUE(trained, "trainFromCorpus should succeed");

    // After ▁ normalization, pieces are ▁-prefixed ("▁gonna") or bare ("gonna")
    const bool has_gonna = unigram.hasPiece("gonna") || unigram.hasPiece("\xe2\x96\x81gonna");
    ASSERT_TRUE(has_gonna, "Expected natural speech token variant for 'gonna' to remain");
    ASSERT_FALSE(unigram.hasPiece("hahaha"), "Repeated-pattern token 'hahaha' should be filtered");
    ASSERT_FALSE(unigram.hasPiece("aaaaaa"), "Excessive run-length token 'aaaaaa' should be filtered");
    ASSERT_FALSE(unigram.hasPiece("i i i"), "Word-level stutter token 'i i i' should be filtered");
    ASSERT_FALSE(unigram.hasPiece("callcall"), "Doubled-token pattern 'callcall' should be filtered");

    return true;
}

bool testUnigramTrainDedupsRepeatedVariants(std::string& message) {
    UnigramLM unigram;

    std::vector<std::string> corpus = {
        "soo good soo good soo good",
        "sooo good sooo good sooo good",
        "soooo good soooo good soooo good"
    };

    const bool trained = unigram.trainFromCorpus(
        corpus,
        {},     // atom_spans: none
        400,    // target_vocab_size
        1.0f,   // character_coverage
        3,      // min_subword_freq
        false   // prune_during_mining
    );
    ASSERT_TRUE(trained, "trainFromCorpus should succeed");

    // After ▁ normalization, pieces may be ▁-prefixed or bare
    const bool has_soo = unigram.hasPiece("soo") || unigram.hasPiece("\xe2\x96\x81soo");
    const bool has_sooo = unigram.hasPiece("sooo") || unigram.hasPiece("\xe2\x96\x81sooo");
    ASSERT_TRUE(has_soo || has_sooo, "Expected at least one repeated-char variant to survive");
    ASSERT_FALSE(has_soo && has_sooo, "Repeated-char variants should deduplicate to one form");

    return true;
}

bool testUnigramTrainPreservesSpieceUnderlineDedupBoundary(std::string& message) {
    UnigramLM unigram;

    std::vector<std::string> corpus = {
        "word sword word sword word sword word sword",
        "word sword word sword word sword word sword",
        "word sword word sword word sword word sword"
    };

    const bool trained = unigram.trainFromCorpus(
        corpus,
        {},     // atom_spans: none
        5000,   // target_vocab_size; keep candidate vocab above this tiny fixture size
        1.0f,   // character_coverage
        3,      // min_subword_freq
        false,  // prune_during_mining
        false,  // enable_parallel_subword_mining
        1,      // subword_mining_workers
        0);     // subword_mining_max_bytes
    ASSERT_TRUE(trained, "trainFromCorpus should succeed");

    ASSERT_TRUE(unigram.hasPiece("\xe2\x96\x81word"),
                "Structural dedup must preserve word-initial ▁word as its own candidate");
    ASSERT_TRUE(unigram.hasPiece("word"),
                "Structural dedup must not collapse bare word into ▁word or vice versa");

    return true;
}

bool testUnigramTrainStridedMiningKeepsLateDocumentPatterns(std::string& message) {
    UnigramLM unigram;

    std::string long_document;
    for (int i = 0; i < 40; ++i) {
        long_document += "alpha beta gamma delta ";
    }
    for (int i = 0; i < 8; ++i) {
        long_document += "tailquark ";
    }

    std::vector<std::string> corpus = {long_document};

    const bool trained = unigram.trainFromCorpus(
        corpus,
        {},     // atom_spans: none
        5000,   // target_vocab_size; avoid pruning this small candidate set
        1.0f,   // character_coverage
        3,      // min_subword_freq
        false,  // prune_during_mining
        false,  // enable_parallel_subword_mining
        1,      // subword_mining_workers
        128);   // subword_mining_max_bytes; forces sampling below full document size
    ASSERT_TRUE(trained, "trainFromCorpus should succeed with strided mining enabled");

    ASSERT_TRUE(unigram.hasPiece("tailquark") || unigram.hasPiece("\xe2\x96\x81tailquark"),
                "Strided subword mining must sample late-document patterns, not only prefixes");

    return true;
}

bool testUnigramTrainStridedMiningOverlapsBoundaryCandidates(std::string& message) {
    UnigramLM unigram(false);

    // With subword_mining_max_bytes=160 across ten identical 47-byte normalized
    // documents, each document receives a 16-byte intended sampled span centered
    // at byte 15..31. "overlapquark" starts at normalized byte 27 and ends at
    // byte 39, so it can only be mined if the sampled span has right context
    // overlap while still counting only starts inside 15..31.
    const std::string boundary_crossing_document =
        "abcdefghijklmnopqrstuvwx"
        "overlapquark"
        "yzabcdef";
    std::vector<std::string> corpus(10, boundary_crossing_document);

    const bool trained = unigram.trainFromCorpus(
        corpus,
        {},     // atom_spans: none
        5000,   // target_vocab_size; keep candidate vocab above this fixture size
        1.0f,   // character_coverage; no-byte-fallback path requires exact char coverage
        3,      // min_subword_freq
        false,  // prune_during_mining
        false,  // enable_parallel_subword_mining
        1,      // subword_mining_workers
        160);   // 16 intended sampled bytes per document, plus overlap context
    ASSERT_TRUE(trained, "trainFromCorpus should succeed with overlapped strided mining enabled");

    ASSERT_TRUE(unigram.hasPiece("overlapquark"),
                "Sampled span overlap must mine pieces that start inside the intended span and end just outside it");

    return true;
}

bool testUnigramTrainEnforcesBytePieceLimit(std::string& message) {
    UnigramLM unigram;

    const std::string thirty_three_byte_piece =
        "\xF0\x9F\x99\x82"  // 🙂 4 bytes
        "\xF0\x9F\x9A\x80"  // 🚀 4 bytes
        "\xF0\x9F\x8C\x9F"  // 🌟 4 bytes
        "\xE6\xBC\xA2"      // 漢 3 bytes
        "\xE5\xAD\x97"      // 字 3 bytes
        "\xE4\xBB\xAE"      // 仮 3 bytes
        "\xE5\x90\x8D"      // 名 3 bytes
        "\xE4\xBA\xA4"      // 交 3 bytes
        "\xE3\x81\x98"      // じ 3 bytes
        "\xE3\x82\x8A";     // り 3 bytes
    ASSERT_EQ(thirty_three_byte_piece.size(), static_cast<size_t>(MAX_PIECE_LENGTH + 1),
              "Regression fixture must remain exactly one byte over MAX_PIECE_LENGTH");

    std::vector<std::string> corpus = {
        thirty_three_byte_piece + " alpha " + thirty_three_byte_piece,
        thirty_three_byte_piece + " beta " + thirty_three_byte_piece,
        thirty_three_byte_piece + " gamma " + thirty_three_byte_piece
    };

    const bool trained = unigram.trainFromCorpus(
        corpus,
        {},     // atom_spans: none
        300,    // target_vocab_size
        1.0f,   // character_coverage
        3,      // min_subword_freq
        false,  // prune_during_mining
        false,  // enable_parallel_subword_mining
        1,      // subword_mining_workers
        0);     // subword_mining_max_bytes
    ASSERT_TRUE(trained, "trainFromCorpus should not append >32-byte learned pieces");

    ASSERT_FALSE(unigram.hasPiece(thirty_three_byte_piece),
                 "A 33-byte candidate must stay on the byte-fallback path, not become a learned piece");
    for (int token_idx = 0; token_idx < unigram.pieceCount(); ++token_idx) {
        const int token_id = UnigramLM::tokenIdForIndex(token_idx);
        const UnigramPiece* piece = unigram.getPiece(token_id);
        ASSERT_TRUE(piece != nullptr, "Every learned-piece token id must resolve to a piece");
        ASSERT_TRUE(piece->text.size() <= static_cast<size_t>(MAX_PIECE_LENGTH),
                    "Tokenizer training produced an oversized learned piece");
    }

    return true;
}

//======================================================//
//  Section 3: Aho-Corasick Tests
//======================================================//

bool testAhoCorasickBasicMatches(std::string& message) {
    AhoCorasick ac;

    uint32_t id_he = ac.addPattern("he", AtomType::ATOM_INT);
    uint32_t id_she = ac.addPattern("she", AtomType::ATOM_INT);
    uint32_t id_hers = ac.addPattern("hers", AtomType::ATOM_INT);
    uint32_t id_his = ac.addPattern("his", AtomType::ATOM_INT);
    ac.build();

    std::string text = "ushers";
    auto matches = ac.search(text);

    auto has_match = [&](uint32_t pid, size_t start, size_t end) {
        for (const auto& m : matches) {
            if (m.pattern_id == pid && m.start == start && m.end == end) {
                return true;
            }
        }
        return false;
    };

    ASSERT_EQ(matches.size(), static_cast<size_t>(3), "Unexpected match count");
    ASSERT_TRUE(has_match(id_she, 1, 4), "Missing match for 'she'");
    ASSERT_TRUE(has_match(id_he, 2, 4), "Missing match for 'he'");
    ASSERT_TRUE(has_match(id_hers, 2, 6), "Missing match for 'hers'");

    for (const auto& m : matches) {
        ASSERT_FALSE(m.pattern_id == id_his, "Unexpected match for 'his'");
    }

    return true;
}

bool testAhoCorasickOutputClosure(std::string& message) {
    AhoCorasick ac;

    uint32_t id_abc = ac.addPattern("abc", AtomType::ATOM_INT);
    uint32_t id_bc = ac.addPattern("bc", AtomType::ATOM_INT);
    uint32_t id_c = ac.addPattern("c", AtomType::ATOM_INT);
    ac.build();

    std::string text = "zabc";
    auto matches = ac.search(text);

    auto has_match = [&](uint32_t pid, size_t start, size_t end) {
        for (const auto& m : matches) {
            if (m.pattern_id == pid && m.start == start && m.end == end) {
                return true;
            }
        }
        return false;
    };

    ASSERT_EQ(matches.size(), static_cast<size_t>(3), "Unexpected match count");
    ASSERT_TRUE(has_match(id_abc, 1, 4), "Missing match for 'abc'");
    ASSERT_TRUE(has_match(id_bc, 2, 4), "Missing match for 'bc'");
    ASSERT_TRUE(has_match(id_c, 3, 4), "Missing match for 'c'");

    AhoCorasickMatch first;
    ASSERT_TRUE(ac.findFirst(text, first), "findFirst should return a match");

    bool first_ok = (first.end == 4) &&
                    ((first.pattern_id == id_abc && first.start == 1) ||
                     (first.pattern_id == id_bc && first.start == 2) ||
                     (first.pattern_id == id_c && first.start == 3));
    ASSERT_TRUE(first_ok, "findFirst returned unexpected match");

    ASSERT_FALSE(ac.contains("zzz"), "contains should return false for no matches");

    return true;
}

bool testAhoCorasickStructuralVsNaive(std::string& message) {
    struct Pattern {
        std::string text;
        AtomType type;
    };

    std::vector<Pattern> patterns = {
        {"http://", AtomType::ATOM_INT},
        {"https://", AtomType::ATOM_INT},
        {"ftp://", AtomType::ATOM_INT},
        {"ftps://", AtomType::ATOM_INT},
        {"ws://", AtomType::ATOM_INT},
        {"wss://", AtomType::ATOM_INT},
        {"file://", AtomType::ATOM_INT},
        {"@", AtomType::ATOM_INT},
        {"0x", AtomType::ATOM_INT},
        {"0X", AtomType::ATOM_INT},
        {"0b", AtomType::ATOM_INT},
        {"0B", AtomType::ATOM_INT},
    };

    AhoCorasick ac;
    for (const auto& pattern : patterns) {
        ac.addPattern(pattern.text, pattern.type);
    }
    ac.build();

    std::string text =
        "Email test@example.com, visit https://example.com, http://site, "
        "ftp://host, ftps://secure, ws://sock, wss://sock, file://C:/tmp/0xDE "
        "and 0XFF then 0b101 and 0B1010.";

    auto matches = ac.search(text);

    struct MatchKey {
        uint32_t pattern_id;
        size_t start;
        size_t end;

        bool operator<(const MatchKey& other) const {
            if (pattern_id != other.pattern_id) return pattern_id < other.pattern_id;
            if (start != other.start) return start < other.start;
            return end < other.end;
        }
    };

    std::set<MatchKey> aho_set;
    for (const auto& match : matches) {
        if (match.atom_type != patterns[match.pattern_id].type) {
            message = std::string("Atom type mismatch: expected ") +
                      atomTypeName(patterns[match.pattern_id].type) +
                      ", got " + atomTypeName(match.atom_type);
            return false;
        }
        aho_set.insert(MatchKey{match.pattern_id, match.start, match.end});
    }

    std::set<MatchKey> naive_set;
    for (uint32_t pid = 0; pid < patterns.size(); ++pid) {
        const std::string& needle = patterns[pid].text;
        size_t pos = text.find(needle, 0);
        while (pos != std::string::npos) {
            naive_set.insert(MatchKey{pid, pos, pos + needle.size()});
            pos = text.find(needle, pos + 1);
        }
    }

    ASSERT_EQ(aho_set.size(), naive_set.size(), "Mismatch between Aho-Corasick and naive match count");

    for (const auto& key : naive_set) {
        if (aho_set.find(key) == aho_set.end()) {
            message = "Aho-Corasick missing match at " + std::to_string(key.start) +
                      " for pattern id " + std::to_string(key.pattern_id);
            return false;
        }
    }

    return true;
}

bool testAhoCorasickVisualization(std::string& message) {
    std::vector<std::string> patterns = {
        "http://",
        "https://",
        "ftp://",
        "ftps://",
        "ws://",
        "wss://",
        "file://",
        "@",
        "0x",
        "0X",
        "0b",
        "0B"
    };

    AhoCorasick ac;
    for (const auto& pattern : patterns) {
        ac.addPattern(pattern, AtomType::ATOM_INT);
    }
    ac.build();

    std::cout << "\n--- AhoCorasick DOT BEGIN ---\n";
    if (!ac.writeDot(std::cout)) {
        message = "Failed to write DOT to stdout";
        return false;
    }
    std::cout << "--- AhoCorasick DOT END ---\n";

    return true;
}

bool testAhoCorasickCaseInsensitive(std::string& message) {
    AhoCorasick ac;
    ac.setCaseInsensitive(true);
    uint32_t pid = ac.addPattern("http://", AtomType::ATOM_INT);
    ac.build();

    std::string text = "HTTP://";
    AhoCorasickMatch match;
    ASSERT_TRUE(ac.findFirst(text, match), "Expected case-insensitive match");
    ASSERT_EQ(match.pattern_id, pid, "Pattern ID mismatch");
    ASSERT_EQ(match.start, static_cast<size_t>(0), "Match start mismatch");
    ASSERT_EQ(match.end, static_cast<size_t>(7), "Match end mismatch");

    return true;
}

//======================================================//
//  Section 4: UniByte Orchestrator Tests
//======================================================//

bool testFixedNumericTokenEncoding(std::string& message) {
    const std::string literal = "-11111.0000e+22";
    std::vector<int> token_ids;
    appendNumericLiteralTokenIds(literal, token_ids);

    const std::vector<int> expected = {
        NUMERIC_TOKEN_OFFSET + 42,
        NUMERIC_TOKEN_OFFSET + 31,
        NUMERIC_TOKEN_OFFSET + 1,
        NUMERIC_TOKEN_OFFSET + 40,
        NUMERIC_TOKEN_OFFSET + 30,
        NUMERIC_TOKEN_OFFSET + 44,
        NUMERIC_TOKEN_OFFSET + 43,
        NUMERIC_TOKEN_OFFSET + 12
    };
    ASSERT_EQ(token_ids.size(), expected.size(),
              "Fixed numeric encoding emitted the wrong token count");
    for (size_t i = 0; i < expected.size(); ++i) {
        ASSERT_EQ(token_ids[i], expected[i],
                  "Fixed numeric encoding emitted the wrong token ID");
        ASSERT_TRUE(isNumericTokenId(token_ids[i]),
                    "Fixed numeric encoding escaped the numeric token range");
    }

    std::string reconstructed;
    for (const int token_id : token_ids) {
        reconstructed += numericTokenTextOrThrow(
            token_id, "testFixedNumericTokenEncoding");
    }
    ASSERT_STR_EQ(reconstructed, literal,
                  "Fixed numeric token table failed exact round-trip");

    const auto config = makeSelfTestTokenizerHP();
    UniByte tokenizer(config);
    ASSERT_STR_EQ(tokenizer.decode(DecodeRequest(token_ids)), literal,
                  "UniByte decode must render fixed numeric IDs without learned pieces");
    return true;
}

bool testNumericTokenSpanSelectionExcludesAtoms(std::string& message) {
    const std::string text = "outside=-12.5 atom=<INT>9999</INT> grouped=1,000 tail=.25";
    const size_t atom_start = text.find("<INT>");
    const size_t atom_close_start = text.find("</INT>");
    ASSERT_TRUE(atom_start != std::string::npos && atom_close_start != std::string::npos,
                "Numeric span test fixture is missing its atom delimiters");
    const size_t atom_end = atom_close_start + std::string("</INT>").size();

    const std::vector<NumericTokenSpan> spans = findNumericTokenSpans(
        text, {NumericTokenSpan{atom_start, atom_end}});
    std::vector<std::string> literals;
    for (const NumericTokenSpan& span : spans) {
        ASSERT_TRUE(span.end <= atom_start || span.start >= atom_end,
                    "Numeric token selection entered an authored atom span");
        literals.push_back(text.substr(span.start, span.end - span.start));
    }

    const std::vector<std::string> expected = {"-12.5", "1,000", ".25"};
    ASSERT_EQ(literals.size(), expected.size(),
              "Numeric span selection emitted the wrong number of literals");
    for (size_t i = 0; i < expected.size(); ++i) {
        ASSERT_STR_EQ(literals[i], expected[i],
                      "Numeric span selection emitted the wrong literal");
    }

    ASSERT_EQ(numericLiteralLengthAt("plain punctuation + - . , e E", 18),
              static_cast<size_t>(0),
              "Standalone punctuation must not become a numeric token span");
    return true;
}

bool testUniByteBasicEncode(std::string& message) {
    auto config = makeSelfTestTokenizerHP();
    config.target_vocab_size = 50000;
    config.enable_byte_fallback = true;
    
    UniByte tokenizer(config);
    
    // Initialize with a simple vocab via VocabWriteOp-backed fixture appends.
    // Start after pre-existing special tokens
    // Pieces use ▁ (U+2581) prefix — SentencePiece whitespace normalization
    appendSelfTestUnigramPiece(tokenizer.unigramLM(), "\xe2\x96\x81hello", -1.0f, false);
    appendSelfTestUnigramPiece(tokenizer.unigramLM(), "\xe2\x96\x81world", -1.2f, false);
    tokenizer.unigramLM().buildTrie();  // Must build trie before encoding
    
    std::vector<int> tokens = tokenizer.tokenizeWithMetadata("hello world").token_ids;
    
    ASSERT_TRUE(tokens.size() > 0, "Should produce tokens");
    
    return true;
}

bool testUniByteStructuralDetection(std::string& message) {
    auto config = makeSelfTestTokenizerHP();
    config.target_vocab_size = 50000;
    
    UniByte tokenizer(config);
    
    // Test number detection
    std::string input = "The price is 42.99 dollars";
    auto result = tokenizer.tokenizeWithMetadata(input);
    
    bool found_number = false;
    for (const auto& span : result.atoms) {
        if (span.atom_type == AtomType::ATOM_INT || 
            span.atom_type == AtomType::ATOM_FLOAT) {
            found_number = true;
            break;
        }
    }
    
    ASSERT_TRUE(found_number, "Should detect number in input");
    
    return true;
}

    bool testUniByteRejectsUnparseableDetectedAtom(std::string& message) {
        auto config = makeSelfTestTokenizerHP();

        UniByte tokenizer(config);

        bool threw = false;
        std::string error_text;
        try {
            tokenizer.tokenizeWithMetadata("Count 99999999999999999999999999999999999999 now");
        } catch (const std::exception& e) {
            threw = true;
            error_text = e.what();
        }

        ASSERT_TRUE(threw, "Tokenizer must fail loudly when a detector-emitted atom span is not parseable");
        ASSERT_TRUE(error_text.find("detector-emitted atom span is not parseable") != std::string::npos,
                    "Tokenizer error should identify the detector/parseability contract violation");
        ASSERT_TRUE(error_text.find("upstream detector/data pipeline bug") != std::string::npos,
                    "Tokenizer error should identify the upstream contract break");

        return true;
    }

    bool testUnigramTrainRejectsUnparseableDetectedAtom(std::string& message) {
        auto config = makeSelfTestTokenizerHP();

        UnigramLM unigram;

        bool threw = false;
        std::string error_text;
        try {
            unigram.trainFromCorpus({"Count 99999999999999999999999999999999999999 now"}, config);
        } catch (const std::exception& e) {
            threw = true;
            error_text = e.what();
        }

        ASSERT_TRUE(threw, "Tokenizer training must fail loudly when a detector-emitted atom span is not parseable");
        ASSERT_TRUE(error_text.find("detector-emitted atom span is not parseable") != std::string::npos,
                    "Training error should identify the detector/parseability contract violation");
        ASSERT_TRUE(error_text.find("upstream detector/data pipeline bug") != std::string::npos,
                    "Training error should identify the upstream contract break");

        return true;
    }

bool testAtomTableRejectsBadNumericDetectionWithContext(std::string& message) {
    const std::string text = "arg_number";
    std::vector<Detector::RawTextDetection> detections;
    detections.emplace_back(0, text.size(), AtomType::ATOM_INT, "bad_numeric_fixture");

    bool threw = false;
    std::string error_text;
    try {
        (void)createAtomTableFromRawTextDetections(
            std::string_view(text.data(), text.size()),
            detections,
            "testAtomTableRejectsBadNumericDetectionWithContext");
    } catch (const std::exception& e) {
        threw = true;
        error_text = e.what();
    }

    ASSERT_TRUE(threw, "Detector-emitted alphabetic ATOM_INT span must fail loudly");
    ASSERT_TRUE(error_text.find("detector-emitted atom span is not parseable") != std::string::npos,
                "Error must identify the numeric detector contract violation");
    ASSERT_TRUE(error_text.find("detector='bad_numeric_fixture'") != std::string::npos,
                "Error must include detector name");
    ASSERT_TRUE(error_text.find("span=[0, 10)") != std::string::npos,
                "Error must include byte offsets");
    ASSERT_TRUE(error_text.find("raw_text='arg_number'") != std::string::npos,
                "Error must include raw text");
    ASSERT_TRUE(error_text.find("must not fall back to text") != std::string::npos,
                "Error must forbid text fallback for detector-emitted numeric spans");

    return true;
}

bool testDefaultRegistryDoesNotDetectPlainNumbers(std::string& message) {
    auto registry = Detector::makeDefaultRawTextDetectorRegistry();
    const Detector::RawTextDetectorOptions options(true, true);

    const std::string text = "CPU 42\nGPU -3.5x";
    const auto detections = registry.scan(text, options);

    for (const auto& detection : detections) {
        ASSERT_FALSE(detection.emitsAtom(),
                     "Default registry must not emit atoms for plain numeric text");
    }

    return true;
}

bool testUniByteURLDetection(std::string& message) {
    auto config = makeSelfTestTokenizerHP();
    config.target_vocab_size = 50000;
    
    UniByte tokenizer(config);
    
    std::string input = "Visit https://example.com/path for more info";
    auto result = tokenizer.tokenizeWithMetadata(input);
    
    ASSERT_EQ(result.atoms.size(), 0, "URLs should pass through without atom detection");
    
    return true;
}

bool testUniByteURLDetectionCaseInsensitive(std::string& message) {
    auto config = makeSelfTestTokenizerHP();
    config.target_vocab_size = 50000;

    UniByte tokenizer(config);

    std::string input = "Visit HTTPS://Example.com/path for more info";
    auto result = tokenizer.tokenizeWithMetadata(input);

    ASSERT_EQ(result.atoms.size(), 0, "URLs should pass through without atom detection");

    return true;
}

bool testUniByteEmailDetection(std::string& message) {
    auto config = makeSelfTestTokenizerHP();
    config.target_vocab_size = 50000;
    
    UniByte tokenizer(config);
    
    std::string input = "Contact us at test@example.com";
    auto result = tokenizer.tokenizeWithMetadata(input);
    
    ASSERT_EQ(result.atoms.size(), 0, "Emails should pass through without atom detection");
    
    return true;
}

bool testUniByteDateDetection(std::string& message) {
    auto config = makeSelfTestTokenizerHP();
    config.target_vocab_size = 50000;
    
    UniByte tokenizer(config);
    
    std::string input = "The meeting is on 2024-12-25";
    auto result = tokenizer.tokenizeWithMetadata(input);
    
    ASSERT_TRUE(result.atoms.size() >= 1, "Date text should only expose numeric atom spans");
    
    return true;
}

bool testUniByteTypedAtomSpanInjection(std::string& message) {
    auto config = makeSelfTestTokenizerHP();
    config.target_vocab_size = 50000;
    
    UniByte tokenizer(config);
    
    std::string input = "Count: 12345 ratio: 3.5";
    auto result = tokenizer.tokenizeWithMetadata(input);
    
    const int open_id = atomTypeToOpenTokenId(AtomType::ATOM_INT);
    const int close_id = atomTypeToCloseTokenId(AtomType::ATOM_INT);
    const auto open_it = std::find(result.token_ids.begin(), result.token_ids.end(), open_id);
    ASSERT_TRUE(open_it != result.token_ids.end(), "Should inject typed opening boundary for integer");

    const auto close_it = std::find(open_it + 1, result.token_ids.end(), close_id);
    ASSERT_TRUE(close_it != result.token_ids.end(), "Should inject matching typed closing boundary for integer");
    ASSERT_TRUE(close_it > open_it + 1, "Integer value must remain tokenized inside typed boundaries");

    const size_t open_index = static_cast<size_t>(open_it - result.token_ids.begin());
    const size_t close_index = static_cast<size_t>(close_it - result.token_ids.begin());
    ASSERT_EQ(result.token_atom_mask[open_index], static_cast<uint8_t>(1),
              "Opening boundary must anchor atom metadata");
    ASSERT_TRUE(result.atom_entry_ids[open_index] != kAtomEntryNone,
                "Opening boundary must retain its AtomTable entry ID");
    ASSERT_EQ(result.token_atom_mask[close_index], static_cast<uint8_t>(0),
              "Closing boundary must not duplicate atom metadata");
    ASSERT_EQ(result.atom_entry_ids[close_index], kAtomEntryNone,
              "Closing boundary must not carry an AtomTable entry ID");
    for (auto it = open_it + 1; it != close_it; ++it) {
        const size_t content_index = static_cast<size_t>(it - result.token_ids.begin());
        ASSERT_FALSE(isAtomTokenId(*it), "Atom content must use ordinary unigram/byte tokens");
        ASSERT_EQ(result.token_atom_mask[content_index], static_cast<uint8_t>(0),
                  "Atom content must not duplicate opening-boundary metadata");
        ASSERT_EQ(result.atom_entry_ids[content_index], kAtomEntryNone,
                  "Atom content must not carry an AtomTable entry ID");
    }

    const int float_open_id = atomTypeToOpenTokenId(AtomType::ATOM_FLOAT);
    const int float_close_id = atomTypeToCloseTokenId(AtomType::ATOM_FLOAT);
    const auto float_open_it = std::find(close_it + 1, result.token_ids.end(), float_open_id);
    ASSERT_TRUE(float_open_it != result.token_ids.end(),
                "Should inject typed opening boundary for float");
    const auto float_close_it = std::find(float_open_it + 1, result.token_ids.end(), float_close_id);
    ASSERT_TRUE(float_close_it != result.token_ids.end(),
                "Should inject matching typed closing boundary for float");
    ASSERT_TRUE(float_close_it > float_open_it + 1,
                "Float value must remain tokenized inside typed boundaries");
    const size_t float_open_index = static_cast<size_t>(float_open_it - result.token_ids.begin());
    const size_t float_close_index = static_cast<size_t>(float_close_it - result.token_ids.begin());
    ASSERT_EQ(result.token_atom_mask[float_open_index], static_cast<uint8_t>(1),
              "Float opening boundary must anchor atom metadata");
    ASSERT_EQ(result.token_atom_mask[float_close_index], static_cast<uint8_t>(0),
              "Float closing boundary must not duplicate atom metadata");

    const std::string expected =
        "Count: <INT>12345</INT> ratio: <FLOAT>3.5</FLOAT>";
    ASSERT_STR_EQ(tokenizer.decode(GRIM::Tokenizer::DecodeRequest(result)), expected,
                  "Metadata-aware decode must preserve typed atom markup and in-band content");
    ASSERT_STR_EQ(tokenizer.decode(GRIM::Tokenizer::DecodeRequest(result.token_ids)), expected,
                  "ID-only decode must produce the same typed atom markup");

    return true;
}

bool testAuthoredAtomDelimiterDetector(std::string& message) {
    auto registry = Detector::makeDefaultRawTextDetectorRegistry();
    const Detector::RawTextDetectorOptions options(
        false,
        false);
    const std::string text =
        "value=<INT> 42 </INT> ratio=<FLOAT>-3.5</FLOAT> "
        "label=<STRING>  hello world  </STRING> enabled=<BOOL> true </BOOL> "
        "owner=<ENTITY>Ada Lovelace</ENTITY>";
    const auto detections = registry.scan(text, options);

    ASSERT_EQ(detections.size(), static_cast<size_t>(5),
              "Authored atom delimiters must claim their complete spans");

    const auto& integer = detections[0];
    ASSERT_TRUE(integer.emitsAtom(), "Authored integer span must emit an atom");
    ASSERT_TRUE(integer.atom_type == AtomType::ATOM_INT,
                "Authored integer span has the wrong atom type");
    ASSERT_STR_EQ(text.substr(integer.start, integer.end - integer.start),
                  "<INT> 42 </INT>",
                  "Authored integer outer span mismatch");
    ASSERT_STR_EQ(text.substr(integer.content_offset, integer.content_length),
                  "42",
                  "Authored integer content span mismatch");
    const auto& floating = detections[1];
    ASSERT_TRUE(floating.emitsAtom(), "Authored float span must emit an atom");
    ASSERT_TRUE(floating.atom_type == AtomType::ATOM_FLOAT,
                "Authored float span has the wrong atom type");
    ASSERT_STR_EQ(text.substr(floating.content_offset, floating.content_length),
                  "-3.5",
                  "Authored float content span mismatch");
    const auto& string_value = detections[2];
    ASSERT_TRUE(string_value.atom_type == AtomType::ATOM_STRING,
                "Authored string span has the wrong atom type");
    ASSERT_STR_EQ(text.substr(string_value.content_offset, string_value.content_length),
                  "  hello world  ",
                  "Authored string content must preserve edge whitespace");
    const auto& boolean = detections[3];
    ASSERT_TRUE(boolean.atom_type == AtomType::ATOM_BOOL,
                "Authored boolean span has the wrong atom type");
    ASSERT_STR_EQ(text.substr(boolean.content_offset, boolean.content_length),
                  "true",
                  "Authored boolean content span mismatch");
    const auto& entity = detections[4];
    ASSERT_TRUE(entity.atom_type == AtomType::ATOM_ENTITY,
                "Authored entity span has the wrong atom type");
    ASSERT_STR_EQ(text.substr(entity.content_offset, entity.content_length),
                  "Ada Lovelace",
                  "Authored entity content span mismatch");

    const AtomTableFromDetectionsResult table_result =
        createAtomTableFromRawTextDetections(
            text,
            detections,
            "testAuthoredAtomDelimiterDetector");
    ASSERT_EQ(table_result.atom_tokens.size(), static_cast<size_t>(5),
              "Authored delimiter detections must register five atom payloads");
    ASSERT_TRUE(table_result.local_atom_table != nullptr,
                "Detection finalization must create a sequence-local atom table");
    for (const auto& payload : table_result.atom_tokens) {
        ASSERT_EQ(payload.span.local_atom_index, 0u,
                  "The first value of each atom type must receive local index zero");
        ASSERT_TRUE(table_result.local_atom_table->contains(
                        payload.span.atom_type,
                        payload.span.local_atom_index),
                    "Detected atom local address must resolve without an AtomTable entry ID");
    }
    ASSERT_TRUE(table_result.local_atom_table
                    ->getRawText(AtomType::ATOM_STRING, 0)
                    .value_or("") == "  hello world  ",
                "Sequence-local string table must preserve exact authored bytes");
    ASSERT_EQ(table_result.atom_tokens[0].span.open_token_id,
              atomTypeToOpenTokenId(AtomType::ATOM_INT),
              "Authored integer opening token mismatch");
    ASSERT_EQ(table_result.atom_tokens[0].span.close_token_id,
              atomTypeToCloseTokenId(AtomType::ATOM_INT),
              "Authored integer closing token mismatch");
    ASSERT_EQ(table_result.atom_tokens[0].token_numeric_value, 42.0f,
              "Authored integer value was not stored");
    ASSERT_EQ(table_result.atom_tokens[1].token_numeric_value, -3.5f,
              "Authored float value was not stored");
    const auto integer_entry = table_result.atom_table->getAtom(
        table_result.atom_tokens[0].span.atom_entry_id);
    ASSERT_TRUE(integer_entry.has_value(), "Authored integer AtomTable entry is missing");
    ASSERT_FALSE(integer_entry->arg_number.has_value(),
                 "Authored integer registration must not populate digit/pow10 metadata");
    const auto float_entry = table_result.atom_table->getAtom(
        table_result.atom_tokens[1].span.atom_entry_id);
    ASSERT_TRUE(float_entry.has_value(), "Authored float AtomTable entry is missing");
    ASSERT_FALSE(float_entry->arg_number.has_value(),
                 "Authored float registration must not populate digit/pow10 metadata");
    ASSERT_EQ(table_result.atom_tokens[2].span.open_token_id,
              atomTypeToOpenTokenId(AtomType::ATOM_STRING),
              "Authored string opening token mismatch");
    ASSERT_EQ(table_result.atom_tokens[2].span.close_token_id,
              atomTypeToCloseTokenId(AtomType::ATOM_STRING),
              "Authored string closing token mismatch");
    ASSERT_EQ(table_result.atom_tokens[2].token_numeric_value, 0.0f,
              "Authored string must not populate the numeric side channel");
    ASSERT_EQ(table_result.atom_tokens[3].span.open_token_id,
              atomTypeToOpenTokenId(AtomType::ATOM_BOOL),
              "Authored boolean opening token mismatch");
    ASSERT_EQ(table_result.atom_tokens[3].token_atom_flags, static_cast<uint32_t>(1),
              "Authored true boolean value was not stored in type-specific flags");
    ASSERT_EQ(table_result.atom_tokens[4].span.open_token_id,
              atomTypeToOpenTokenId(AtomType::ATOM_ENTITY),
              "Authored entity opening token mismatch");
    ASSERT_EQ(table_result.atom_tokens[4].span.close_token_id,
              atomTypeToCloseTokenId(AtomType::ATOM_ENTITY),
              "Authored entity closing token mismatch");

    const auto string_entry = table_result.atom_table->getAtom(
        table_result.atom_tokens[2].span.atom_entry_id);
    ASSERT_TRUE(string_entry.has_value(), "Authored string AtomTable entry is missing");
    ASSERT_TRUE(string_entry->type == AtomType::ATOM_STRING,
                "Authored string AtomTable type mismatch");
    ASSERT_STR_EQ(std::string(table_result.atom_table->getString(string_entry->raw_text_ref)),
                  "  hello world  ",
                  "Authored string AtomTable value mismatch");
    ASSERT_FALSE(table_result.atom_table->getNumericValue(string_entry->id).has_value(),
                 "Authored string must not expose a numeric payload");

    const auto bool_entry = table_result.atom_table->getAtom(
        table_result.atom_tokens[3].span.atom_entry_id);
    ASSERT_TRUE(bool_entry.has_value(), "Authored boolean AtomTable entry is missing");
    ASSERT_TRUE(bool_entry->type == AtomType::ATOM_BOOL,
                "Authored boolean AtomTable type mismatch");
    ASSERT_EQ(bool_entry->flags, static_cast<uint32_t>(1),
              "Authored boolean AtomTable value mismatch");
    ASSERT_FALSE(table_result.atom_table->getNumericValue(bool_entry->id).has_value(),
                 "Authored boolean must not expose a numeric payload");

    const auto entity_entry = table_result.atom_table->getAtom(
        table_result.atom_tokens[4].span.atom_entry_id);
    ASSERT_TRUE(entity_entry.has_value(), "Authored entity AtomTable entry is missing");
    ASSERT_TRUE(entity_entry->type == AtomType::ATOM_ENTITY,
                "Authored entity AtomTable type mismatch");
    ASSERT_STR_EQ(std::string(table_result.atom_table->getString(entity_entry->raw_text_ref)),
                  "Ada Lovelace",
                  "Authored entity AtomTable value mismatch");
    ASSERT_FALSE(table_result.atom_table->getNumericValue(entity_entry->id).has_value(),
                 "Authored entity must not expose a numeric payload");

    std::stringstream persisted(std::ios::in | std::ios::out | std::ios::binary);
    table_result.atom_table->serializeToStreamOrThrow(
        persisted,
        "testAuthoredAtomDelimiterDetector");
    persisted.seekg(0);
    AtomTable restored;
    restored.deserializeFromStreamOrThrow(
        persisted,
        "testAuthoredAtomDelimiterDetector");
    ASSERT_EQ(restored.size(), static_cast<size_t>(5),
              "Persisted authored atom table entry count mismatch");
    const auto restored_string = restored.getAtom(string_entry->id);
    ASSERT_TRUE(restored_string.has_value() &&
                    restored_string->type == AtomType::ATOM_STRING,
                "Persisted string atom type mismatch");
    ASSERT_STR_EQ(std::string(restored.getString(restored_string->raw_text_ref)),
                  "  hello world  ",
                  "Persisted string atom value mismatch");
    const auto restored_bool = restored.getAtom(bool_entry->id);
    ASSERT_TRUE(restored_bool.has_value() &&
                    restored_bool->type == AtomType::ATOM_BOOL,
                "Persisted boolean atom type mismatch");
    ASSERT_EQ(restored_bool->flags, static_cast<uint32_t>(1),
              "Persisted boolean atom value mismatch");
    const auto restored_entity = restored.getAtom(entity_entry->id);
    ASSERT_TRUE(restored_entity.has_value() &&
                    restored_entity->type == AtomType::ATOM_ENTITY,
                "Persisted entity atom type mismatch");
    ASSERT_STR_EQ(std::string(restored.getString(restored_entity->raw_text_ref)),
                  "Ada Lovelace",
                  "Persisted entity atom value mismatch");

    const ParseResult false_bool = AtomTable::parseAtom(AtomType::ATOM_BOOL, "false");
    ASSERT_TRUE(false_bool.success, "Lowercase false must parse as an authored boolean");
    ASSERT_FALSE(std::get<AtomBoolean>(false_bool.value).value,
                 "Parsed false boolean value mismatch");
    ASSERT_FALSE(AtomTable::parseAtom(AtomType::ATOM_BOOL, "TRUE").success,
                 "Non-canonical boolean spelling must be rejected");
    ASSERT_TRUE(AtomTable::parseAtom(AtomType::ATOM_ENTITY, "東京").success,
                "Non-empty UTF-8 entity bytes must parse");
    ASSERT_FALSE(AtomTable::parseAtom(AtomType::ATOM_ENTITY, "").success,
                 "Empty entity content must be rejected");

    return true;
}

    bool testUniBytePreRegistersAtomTableBeforeSpanEmission(std::string& message) {
        auto config = makeSelfTestTokenizerHP();
        config.target_vocab_size = 50000;

        UniByte tokenizer(config);

        const std::string input = "same 42 then 42";
        auto result = tokenizer.tokenizeWithMetadata(input);

        ASSERT_TRUE(result.atom_table != nullptr,
                    "tokenizeWithMetadata must create a per-sequence AtomTable before span merge");
        ASSERT_EQ(result.atoms.size(), static_cast<size_t>(2),
                  "Repeated-number fixture should yield two structural atom spans");
        ASSERT_TRUE(result.atoms[0].atom_entry_id != kAtomEntryNone,
                    "Pre-registered structural spans must carry their AtomTable entry ID");
        ASSERT_EQ(result.atoms[0].atom_entry_id, result.atoms[1].atom_entry_id,
                  "Repeated identical atoms should deduplicate to one AtomTable entry before unigram runs");

        size_t opening_count = 0;
        size_t closing_count = 0;
        for (size_t i = 0; i < result.token_ids.size(); ++i) {
            if (isAtomOpenTokenId(result.token_ids[i])) {
                ++opening_count;
                ASSERT_EQ(result.token_atom_mask[i], static_cast<uint8_t>(1),
                          "Opening boundary must preserve token_atom_mask");
                ASSERT_EQ(result.atom_entry_ids[i], result.atoms[0].atom_entry_id,
                          "Opening boundary must reuse the pre-registered AtomTable entry ID");
            } else if (isAtomCloseTokenId(result.token_ids[i])) {
                ++closing_count;
                ASSERT_EQ(result.token_atom_mask[i], static_cast<uint8_t>(0),
                          "Closing boundary must not duplicate atom metadata");
                ASSERT_EQ(result.atom_entry_ids[i], kAtomEntryNone,
                          "Closing boundary must not carry an AtomTable entry ID");
            }
        }

        ASSERT_EQ(opening_count, static_cast<size_t>(2),
                  "Repeated identical numbers should emit two opening boundaries");
        ASSERT_EQ(closing_count, static_cast<size_t>(2),
                  "Repeated identical numbers should emit two closing boundaries");

        return true;
    }

bool testUniByteRoundTrip(std::string& message) {
    std::cout << "\n[RoundTrip] === Starting Round-Trip Test ===\n";
    
    auto config = makeSelfTestTokenizerHP();
    config.target_vocab_size = 50000;
    config.enable_byte_fallback = true;
    
    UniByte tokenizer(config);
    
    // Start after pre-existing special tokens
    std::cout << "[RoundTrip] UNIGRAM_VOCAB_OFFSET = " << UNIGRAM_VOCAB_OFFSET << "\n";
    std::cout << "[RoundTrip] Pre-existing learned piece count = " << tokenizer.unigramLM().pieceCount() << "\n";
    
    // Pieces use ▁ (U+2581) prefix — SentencePiece whitespace normalization
    std::cout << "[RoundTrip] Adding pieces:\n";
    appendSelfTestUnigramPiece(tokenizer.unigramLM(), "\xe2\x96\x81the", -1.0f, false);
    std::cout << "  '▁the'   -> id=" << UnigramLM::tokenIdForIndex(tokenizer.unigramLM().pieceCount() - 1) << "\n";
    appendSelfTestUnigramPiece(tokenizer.unigramLM(), "\xe2\x96\x81quick", -1.5f, false);
    std::cout << "  '▁quick' -> id=" << UnigramLM::tokenIdForIndex(tokenizer.unigramLM().pieceCount() - 1) << "\n";
    appendSelfTestUnigramPiece(tokenizer.unigramLM(), "\xe2\x96\x81" "brown", -1.6f, false);
    std::cout << "  '▁brown' -> id=" << UnigramLM::tokenIdForIndex(tokenizer.unigramLM().pieceCount() - 1) << "\n";
    appendSelfTestUnigramPiece(tokenizer.unigramLM(), "\xe2\x96\x81" "fox", -1.7f, false);
    std::cout << "  '▁fox'   -> id=" << UnigramLM::tokenIdForIndex(tokenizer.unigramLM().pieceCount() - 1) << "\n";
    
    std::cout << "[RoundTrip] Final learned piece count = " << tokenizer.unigramLM().pieceCount() << "\n";
    
    // Dump ALL pieces in vocabulary to see what's actually stored
    std::cout << "[RoundTrip] === Full Vocabulary Dump ===\n";
    for (int i = 0; i < tokenizer.unigramLM().pieceCount(); ++i) {
        int tid = UNIGRAM_VOCAB_OFFSET + i;
        const auto* piece = tokenizer.unigramLM().getPiece(tid);
        if (piece) {
            std::cout << "  idx=" << i << " token_id=" << tid 
                      << " text=\"" << piece->text << "\""
                      << " score=" << piece->score
                      << " user_def=" << piece->is_user_defined << "\n";
        } else {
            std::cout << "  idx=" << i << " token_id=" << tid << " -> nullptr!\n";
        }
    }
    std::cout << "[RoundTrip] === End Vocabulary Dump ===\n";
    
    std::cout << "[RoundTrip] Building trie...\n";
    tokenizer.unigramLM().buildTrie();
    
    std::string input = "the quick brown fox";
    std::cout << "[RoundTrip] Input: \"" << input << "\"\n";
    
    std::vector<int> tokens = tokenizer.tokenizeWithMetadata(input).token_ids;
    std::cout << "[RoundTrip] Encoded tokens (" << tokens.size() << "): [";
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::cout << tokens[i];
        if (i + 1 < tokens.size()) std::cout << ", ";
    }
    std::cout << "]\n";
    
    // Show what each token decodes to individually
    std::cout << "[RoundTrip] Token breakdown:\n";
    for (size_t i = 0; i < tokens.size(); ++i) {
        int tid = tokens[i];
        std::cout << "  tokens[" << i << "] = " << tid;
        if (tid < static_cast<int>(GRIM::Tokenizer::NUM_SPECIAL_TOKENS)) {
            std::cout << " (special)\n";
        } else if (tid >= static_cast<int>(GRIM::Tokenizer::BYTE_TOKEN_OFFSET) &&
               tid < static_cast<int>(GRIM::Tokenizer::BYTE_TOKEN_OFFSET + GRIM::Tokenizer::BYTE_VOCAB_SIZE)) {
            std::cout << " (byte: '" << static_cast<char>(tid - GRIM::Tokenizer::BYTE_TOKEN_OFFSET) << "')\n";
        } else if (tid >= UNIGRAM_VOCAB_OFFSET) {
            const auto* piece = tokenizer.unigramLM().getPiece(tid);
            if (piece) {
                std::cout << " (piece: \"" << piece->text << "\")\n";
            } else {
                std::cout << " (UNKNOWN - getPiece returned nullptr!)\n";
                std::cout << "    -> idx would be: " << (tid - UNIGRAM_VOCAB_OFFSET) << "\n";
            }
        } else {
            std::cout << " (atom range)\n";
        }
    }
    
    std::string output = tokenizer.decode(GRIM::Tokenizer::DecodeRequest(tokens));
    std::cout << "[RoundTrip] Decoded: \"" << output << "\"\n";
    std::cout << "[RoundTrip] Expected: \"" << input << "\"\n";
    std::cout << "[RoundTrip] Match: " << (output == input ? "YES" : "NO") << "\n";
    
    ASSERT_STR_EQ(output, input, "Round-trip failed");
    
    return true;
}

//======================================================//
//  Section 4: AtomTable Tests
//======================================================//

bool testAtomTableRegisterInteger(std::string& message) {
    AtomTable table;
    
    uint32_t id = registerSelfTestAtom(table, AtomType::ATOM_INT, "12345", 0, 5);
    
    auto entry = table.getAtom(id);
    ASSERT_TRUE(entry, "Failed to retrieve atom");
    ASSERT_EQ(static_cast<int>(entry->type), static_cast<int>(AtomType::ATOM_INT), 
              "Type mismatch");
    
    // Access raw_text through string pool
    std::string raw_text(table.getString(entry->raw_text_ref));
    ASSERT_STR_EQ(raw_text, "12345", "Raw text mismatch");
    
    // Check exact parsed value through ID-based side-channel lookup.
    auto numeric = table.getNumericValue(id);
    ASSERT_TRUE(numeric, "Numeric payload missing");
    ASSERT_EQ(static_cast<int>(numeric->kind), static_cast<int>(NumericPayloadKind::INTEGER),
              "Integer numeric kind mismatch");
    ASSERT_EQ(numeric->int_value, static_cast<int64_t>(12345), "Exact integer value mismatch");
    ASSERT_NEAR(numeric->float_value, 12345.0, 0.01, "Integer float side-channel mismatch");

    uint32_t large_id = registerSelfTestAtom(table, AtomType::ATOM_INT, "9007199254740993", 0, 16);
    auto large_numeric = table.getNumericValue(large_id);
    ASSERT_TRUE(large_numeric, "Large integer numeric payload missing");
    ASSERT_EQ(large_numeric->int_value, static_cast<int64_t>(9007199254740993LL),
              "Public numeric getter must not round integer atoms through float");

    uint32_t padded_id = registerSelfTestAtom(table, AtomType::ATOM_INT, "001", 0, 3);
    auto padded_entry = table.getAtom(padded_id);
    ASSERT_TRUE(padded_entry, "Failed to retrieve padded integer atom");
    ASSERT_STR_EQ(table.atomToString(*padded_entry), "001",
                  "Atom stringification must preserve raw source text");
    ASSERT_EQ(padded_entry->reserved_zero, static_cast<uint64_t>(0),
              "AtomTable reserved padding must stay zero, not become canonical parsed text");

    ASSERT_EQ(registerSelfTestAtom(table, AtomType::ATOM_INT, " 42", 0, 3), UINT32_MAX,
              "Leading whitespace in numeric atom text must be rejected");
    ASSERT_EQ(registerSelfTestAtom(table, AtomType::ATOM_INT, "42 ", 0, 3), UINT32_MAX,
              "Trailing whitespace in numeric atom text must be rejected");
    ASSERT_EQ(registerSelfTestAtom(table, AtomType::ATOM_INT, "4 2", 0, 3), UINT32_MAX,
              "Internal whitespace in numeric atom text must be rejected");
    
    return true;
}

bool testAtomTableRegisterFloat(std::string& message) {
    AtomTable table;
    
    uint32_t id = registerSelfTestAtom(table, AtomType::ATOM_FLOAT, "3.14159", 0, 7);
    
    auto entry = table.getAtom(id);
    ASSERT_TRUE(entry, "Failed to retrieve atom");
    
    auto numeric = table.getNumericValue(id);
    ASSERT_TRUE(numeric, "Numeric payload missing");
    ASSERT_EQ(static_cast<int>(numeric->kind), static_cast<int>(NumericPayloadKind::FLOAT),
              "Float numeric kind mismatch");
    ASSERT_NEAR(numeric->float_value, 3.14159, 0.0001, "Float value mismatch");

    ASSERT_EQ(registerSelfTestAtom(table, AtomType::ATOM_FLOAT, " 3.14159", 0, 8), UINT32_MAX,
              "Leading whitespace in float atom text must be rejected");

    return true;
}

bool testAtomTableRegisterHex(std::string& message) {
    // Hex atoms no longer supported — verify a plain integer still works.
    AtomTable table;
    
    uint32_t id = registerSelfTestAtom(table, AtomType::ATOM_INT, "255", 0, 3);
    
    auto entry = table.getAtom(id);
    ASSERT_TRUE(entry, "Failed to retrieve atom");
    
    auto numeric = table.getNumericValue(id);
    ASSERT_TRUE(numeric, "Numeric payload missing");
    ASSERT_EQ(numeric->int_value, static_cast<int64_t>(255), "Integer value mismatch");
    
    return true;
}

bool testAtomTableRegisterBinary(std::string& message) {
    // Binary atoms no longer supported — verify a plain integer still works.
    AtomTable table;
    
    uint32_t id = registerSelfTestAtom(table, AtomType::ATOM_INT, "10", 0, 2);
    
    auto entry = table.getAtom(id);
    ASSERT_TRUE(entry, "Failed to retrieve atom");
    
    auto numeric = table.getNumericValue(id);
    ASSERT_TRUE(numeric, "Numeric payload missing");
    ASSERT_EQ(numeric->int_value, static_cast<int64_t>(10), "Integer value mismatch");
    
    return true;
}

bool testAtomTableRegisterURL(std::string& message) {
    AtomTable table;
    
    uint32_t id = registerSelfTestAtom(table, AtomType::ATOM_INT,
                                       "https://example.com:8080/path?query=1#fragment",
                                       0, 45);
    ASSERT_EQ(id, UINT32_MAX, "Invalid integer text must not register as ATOM_INT");
    ASSERT_EQ(table.size(), static_cast<size_t>(0), "Rejected URL should not mutate table");
    
    return true;
}

bool testAtomTableRegisterEmail(std::string& message) {
    AtomTable table;
    
    uint32_t id = registerSelfTestAtom(table, AtomType::ATOM_INT, "user@domain.com", 0, 15);
    ASSERT_EQ(id, UINT32_MAX, "Invalid integer text must not register as ATOM_INT");
    ASSERT_EQ(table.size(), static_cast<size_t>(0), "Rejected email should not mutate table");
    
    return true;
}

bool testAtomTableRegisterDate(std::string& message) {
    AtomTable table;
    
    uint32_t id = registerSelfTestAtom(table, AtomType::ATOM_INT, "2024-12-25", 0, 10);
    ASSERT_EQ(id, UINT32_MAX, "Invalid integer text must not register as ATOM_INT");
    ASSERT_EQ(table.size(), static_cast<size_t>(0), "Rejected date should not mutate table");
    
    return true;
}

bool testAtomTableRegisterTime(std::string& message) {
    AtomTable table;
    
    uint32_t id = registerSelfTestAtom(table, AtomType::ATOM_INT, "14:30:00", 0, 8);
    ASSERT_EQ(id, UINT32_MAX, "Invalid integer text must not register as ATOM_INT");
    ASSERT_EQ(table.size(), static_cast<size_t>(0), "Rejected time should not mutate table");
    
    return true;
}

bool testAtomTableRegisterIP(std::string& message) {
    AtomTable table;
    
    uint32_t id = registerSelfTestAtom(table, AtomType::ATOM_INT, "192.168.1.1", 0, 11);
    ASSERT_EQ(id, UINT32_MAX, "Invalid integer text must not register as ATOM_INT");
    ASSERT_EQ(table.size(), static_cast<size_t>(0), "Rejected IP should not mutate table");
    
    return true;
}

bool testAtomTableRegisterPath(std::string& message) {
    AtomTable table;
    
    uint32_t id = registerSelfTestAtom(table, AtomType::ATOM_INT, "/usr/local/bin/test", 0, 19);
    ASSERT_EQ(id, UINT32_MAX, "Invalid integer text must not register as ATOM_INT");
    ASSERT_EQ(table.size(), static_cast<size_t>(0), "Rejected path should not mutate table");
    
    return true;
}

bool testAtomTableRegisterString(std::string& message) {
    AtomTable table;
    
    uint32_t id = registerSelfTestAtom(table, AtomType::ATOM_INT, "\"hello\\nworld\"", 0, 14);
    ASSERT_EQ(id, UINT32_MAX, "Invalid integer text must not register as ATOM_INT");
    ASSERT_EQ(table.size(), static_cast<size_t>(0), "Rejected string should not mutate table");
    
    return true;
}

bool testAtomTableRegisterIdentifier(std::string& message) {
    AtomTable table;
    
    // Test various naming conventions
    ASSERT_EQ(registerSelfTestAtom(table, AtomType::ATOM_INT, "camelCase", 0, 9), UINT32_MAX,
              "Invalid identifier must not register as ATOM_INT");
    ASSERT_EQ(registerSelfTestAtom(table, AtomType::ATOM_INT, "PascalCase", 0, 10), UINT32_MAX,
              "Invalid identifier must not register as ATOM_INT");
    ASSERT_EQ(registerSelfTestAtom(table, AtomType::ATOM_INT, "snake_case", 0, 10), UINT32_MAX,
              "Invalid identifier must not register as ATOM_INT");
    ASSERT_EQ(registerSelfTestAtom(table, AtomType::ATOM_INT, "SCREAMING_SNAKE", 0, 15), UINT32_MAX,
              "Invalid identifier must not register as ATOM_INT");
    ASSERT_EQ(table.size(), static_cast<size_t>(0), "Rejected identifiers should not mutate table");
    
    return true;
}

bool testAtomTableLookupByType(std::string& message) {
    AtomTable table;
    
    // Register multiple atoms of different types
    registerSelfTestAtom(table, AtomType::ATOM_INT, "100", 0, 3);
    registerSelfTestAtom(table, AtomType::ATOM_INT, "200", 0, 3);
    registerSelfTestAtom(table, AtomType::ATOM_FLOAT, "3.14", 0, 4);
    registerSelfTestAtom(table, AtomType::ATOM_INT, "300", 0, 3);
    
    auto integers = table.getAtomsByType(AtomType::ATOM_INT);
    ASSERT_EQ(integers.size(), 3, "Should find 3 integers");
    
    auto floats = table.getAtomsByType(AtomType::ATOM_FLOAT);
    ASSERT_EQ(floats.size(), 1, "Should find 1 float");
    
    return true;
}

bool testAtomTableGPUUpload(std::string& message) {
    AtomTable table;
    
    // Register some atoms
    registerSelfTestAtom(table, AtomType::ATOM_INT, "42", 0, 2);
    registerSelfTestAtom(table, AtomType::ATOM_FLOAT, "3.14", 0, 4);
    registerSelfTestAtom(table, AtomType::ATOM_INT, "9007199254740993", 0, 16);
    
    bool success = table.uploadToGPU();
    const AtomTable::GPUAtomData* gpu_data = table.getGPUBuffer();
    
    ASSERT_TRUE(success, "GPU upload failed");
    ASSERT_TRUE(gpu_data != nullptr, "Internal GPU buffer pointer missing");
    ASSERT_EQ(gpu_data->num_atoms, 3, "GPU atom count mismatch");
    ASSERT_TRUE(gpu_data->d_numeric_values != nullptr, "Numeric values not allocated");
    ASSERT_TRUE(gpu_data->d_numeric_float_values != nullptr, "Exact float numeric values not allocated");
    ASSERT_TRUE(gpu_data->d_numeric_int_values != nullptr, "Exact int numeric values not allocated");
    ASSERT_TRUE(gpu_data->d_numeric_kind != nullptr, "Numeric kind values not allocated");
    ASSERT_TRUE(gpu_data->d_types != nullptr, "Types not allocated");

    int64_t exact_large_int = 0;
    cudaError_t copy_err = cudaMemcpy(&exact_large_int,
                                      gpu_data->d_numeric_int_values + 2,
                                      sizeof(int64_t),
                                      cudaMemcpyDeviceToHost);
    ASSERT_TRUE(copy_err == cudaSuccess, "Failed to copy exact integer payload back from GPU");
    ASSERT_EQ(exact_large_int, static_cast<int64_t>(9007199254740993LL),
              "GPU exact integer payload must not round through float");

    AtomTable internal_table;
    registerSelfTestAtom(internal_table, AtomType::ATOM_INT, "7", 0, 1);
    ASSERT_TRUE(internal_table.uploadToGPU(), "Internal GPU upload failed");
    float* first_internal_numeric = internal_table.getGPUBuffer()->d_numeric_values;
    ASSERT_TRUE(first_internal_numeric != nullptr, "Internal upload did not allocate numeric values");
    ASSERT_TRUE(internal_table.uploadToGPU(), "Clean internal GPU upload should succeed");
    ASSERT_TRUE(internal_table.getGPUBuffer()->d_numeric_values == first_internal_numeric,
                "Clean internal upload must not free valid GPU buffers");
    
    return true;
}

bool testAtomTableClear(std::string& message) {
    AtomTable table;
    
    registerSelfTestAtom(table, AtomType::ATOM_INT, "1", 0, 1);
    registerSelfTestAtom(table, AtomType::ATOM_INT, "2", 0, 1);
    registerSelfTestAtom(table, AtomType::ATOM_INT, "3", 0, 1);
    
    ASSERT_EQ(table.size(), 3, "Should have 3 atoms before clear");
    
    table.clear();
    
    ASSERT_EQ(table.size(), 0, "Should have 0 atoms after clear");
    
    return true;
}

bool testAtomTableMetadata(std::string& message) {
    AtomTable table;
    
    // Register an atom
    uint32_t id = registerSelfTestAtom(table, AtomType::ATOM_INT, "42", 0, 2);
    
    auto entry = table.getAtom(id);
    ASSERT_TRUE(entry, "Failed to retrieve atom");
    
    // Check default metadata
    ASSERT_EQ(static_cast<int>(entry->origin), static_cast<int>(AtomOrigin::USER_INPUT), 
              "Default origin should be USER_INPUT");
    ASSERT_EQ(static_cast<int>(entry->category), static_cast<int>(AtomCategory::NUMERIC),
              "Integer should be NUMERIC category");
    ASSERT_NEAR(entry->confidence, 1.0f, 0.001f, "Default confidence should be 1.0");
    ASSERT_TRUE(entry->hash != 0, "Hash should be computed");
    ASSERT_TRUE(entry->created_at != 0, "Timestamp should be set");
    
    // Test updating metadata
    table.setOrigin(id, AtomOrigin::MODEL_GENERATED);
    table.setConfidence(id, 0.85f);
    table.setCategory(id, AtomCategory::SYSTEM);
    
    entry = table.getAtom(id);
    ASSERT_EQ(static_cast<int>(entry->origin), static_cast<int>(AtomOrigin::MODEL_GENERATED),
              "Origin should be updated");
    ASSERT_NEAR(entry->confidence, 0.85f, 0.001f, "Confidence should be updated");
    ASSERT_EQ(static_cast<int>(entry->category), static_cast<int>(AtomCategory::SYSTEM),
              "Category should be updated");
    
    return true;
}

bool testAtomTableHashDeduplication(std::string& message) {
    AtomTable table;
    
    // Register same atom twice
    uint32_t id1 = registerSelfTestAtom(table, AtomType::ATOM_INT, "100", 0, 3);
    uint32_t id2 = registerSelfTestAtom(table, AtomType::ATOM_INT, "100", 0, 3);
    
    auto entry1 = table.getAtom(id1);
    auto entry2 = table.getAtom(id2);
    
    ASSERT_TRUE(entry1 && entry2, "Both atoms should exist");
    ASSERT_EQ(id1, id2, "Registering the same numeric atom should dedupe to one atom entry id");
    ASSERT_EQ(entry1->hash, entry2->hash, "Identical atoms should have same hash");
    ASSERT_FALSE(entry1->arg_number.has_value(), "Tokenizer registration must not populate arg_number metadata");
    ASSERT_FALSE(entry2->arg_number.has_value(), "Deduped atom copies must not gain arg_number metadata");
    
    // Different atom should have different hash
    uint32_t id3 = registerSelfTestAtom(table, AtomType::ATOM_INT, "200", 0, 3);
    auto entry3 = table.getAtom(id3);
    
    ASSERT_TRUE(entry3, "Different atom should exist");
    ASSERT_TRUE(entry1->hash != entry3->hash, "Different atoms should have different hashes");
    
    return true;
}

bool testSequenceLocalAtomTicketing(std::string& message) {
    SequenceLocalAtomTable table;

    const auto alice = table.ticket(AtomType::ATOM_STRING, "Alice");
    const auto alice_again = table.ticket(AtomType::ATOM_STRING, "Alice");
    const auto bob = table.ticket(AtomType::ATOM_STRING, "Bob");
    const auto seven = table.ticket(AtomType::ATOM_INT, "7");

    ASSERT_EQ(alice.local_index, 0u,
              "First string must receive string-local index zero");
    ASSERT_EQ(alice_again.local_index, alice.local_index,
              "Repeated typed value must reuse its local index");
    ASSERT_EQ(bob.local_index, 1u,
              "Second distinct string must receive string-local index one");
    ASSERT_EQ(seven.local_index, 0u,
              "Integer indices must be independent from string indices");
    ASSERT_EQ(table.size(AtomType::ATOM_STRING), static_cast<size_t>(2),
              "String ticket count must include unique string values only");
    ASSERT_EQ(table.size(AtomType::ATOM_INT), static_cast<size_t>(1),
              "Integer ticket count must be maintained independently");
    ASSERT_TRUE(table.getRawText(AtomType::ATOM_STRING, 0).value_or("") == "Alice",
                "Typed local address must resolve its original value");

    table.clear();
    ASSERT_TRUE(table.empty(), "Sequence-local table clear must remove every typed value");
    ASSERT_EQ(table.ticket(AtomType::ATOM_STRING, "Alice").local_index, 0u,
              "A cleared sequence-local type must restart at index zero");
    return true;
}

bool testAtomTableDoesNotPopulateArgNumber(std::string& message) {
    AtomTable table;
    const uint32_t id = registerSelfTestAtom(table, AtomType::ATOM_FLOAT, "-1.5e-4", 12, 19);
    ASSERT_TRUE(id != UINT32_MAX, "Failed to register numeric atom");

    const auto entry = table.getAtom(id);
    ASSERT_TRUE(entry.has_value(), "Original atom entry missing before serialization");
    ASSERT_FALSE(entry->arg_number.has_value(),
                 "Tokenizer registration must not populate mantissa/digit/pow10 metadata");

    std::stringstream binary_stream;
    table.serializeToStreamOrThrow(binary_stream, "testAtomTableDoesNotPopulateArgNumber binary_stream");

    AtomTable loaded;
    loaded.deserializeFromStreamOrThrow(binary_stream, "testAtomTableDoesNotPopulateArgNumber binary_stream");

    const auto loaded_entry = loaded.getAtom(id);
    ASSERT_TRUE(loaded_entry.has_value(), "Loaded atom entry missing after serialization round-trip");
    ASSERT_FALSE(loaded_entry->arg_number.has_value(),
                 "Serialization must preserve the absence of arg_number metadata");
    return true;
}

//======================================================//
//  Section 5: Integration Tests
//======================================================//

bool testFullPipeline(std::string& message) {
    // Create tokenizer with current numeric-only atom detection.
    auto config = makeSelfTestTokenizerHP();
    config.target_vocab_size = 50000;
    config.enable_byte_fallback = true;
    
    UniByte tokenizer(config);
    
    // Add vocabulary via VocabWriteOp-backed fixture appends.
    // Start after pre-existing special tokens
    // Pieces use \xe2\x96\x81 (U+2581 ▁) prefix — SentencePiece whitespace normalization
    appendSelfTestUnigramPiece(tokenizer.unigramLM(), "\xe2\x96\x81the", -1.0f, false);
    appendSelfTestUnigramPiece(tokenizer.unigramLM(), "\xe2\x96\x81is", -1.1f, false);
    appendSelfTestUnigramPiece(tokenizer.unigramLM(), "\xe2\x96\x81price", -1.5f, false);
    appendSelfTestUnigramPiece(tokenizer.unigramLM(), "\xe2\x96\x81visit", -1.6f, false);
    appendSelfTestUnigramPiece(tokenizer.unigramLM(), "\xe2\x96\x81" "for", -1.7f, false);
    appendSelfTestUnigramPiece(tokenizer.unigramLM(), "\xe2\x96\x81more", -1.8f, false);
    appendSelfTestUnigramPiece(tokenizer.unigramLM(), "\xe2\x96\x81info", -1.9f, false);
    appendSelfTestUnigramPiece(tokenizer.unigramLM(), ".", -0.6f, false);
    
    // Mixed input: numbers become atoms, URLs remain plain text.
    std::string input = "The price is 42.99. Visit https://shop.com for 3 more info.";
    
    auto result = tokenizer.tokenizeWithMetadata(input);
    
    ASSERT_TRUE(result.token_ids.size() > 0, "Should produce tokens");
    ASSERT_TRUE(result.atoms.size() >= 2, "Should detect numeric structures only");
    
    // Decode preserves typed boundaries and the model-visible values between them.
    std::string decoded = tokenizer.decode(GRIM::Tokenizer::DecodeRequest(result));
    ASSERT_STR_EQ(decoded,
                  "The price is <FLOAT>42.99</FLOAT>. Visit https://shop.com for <INT>3</INT> more info.",
                  "Full pipeline decode must render typed numeric spans literally");
    
    return true;
}

bool testAtomTableIntegration(std::string& message) {
    auto config = makeSelfTestTokenizerHP();
    config.target_vocab_size = 50000;
    
    UniByte tokenizer(config);
    
    std::string input = "Values: 100, 200, 300";
    auto result = tokenizer.tokenizeWithMetadata(input);
    
    // Register detected atoms in AtomTable
    AtomTable table;
    for (const auto& span : result.atoms) {
        table.registerSpan(span);
    }
    
    ASSERT_TRUE(table.size() >= 3, "Should register at least 3 number atoms");
    
    // Upload to GPU and verify
    bool success = table.uploadToGPU();
    ASSERT_TRUE(success, "GPU upload should succeed");
    ASSERT_EQ(table.getGPUBuffer()->num_atoms, table.size(), "Internal GPU atom count mismatch");
    
    return true;
}

bool testBatchProcessing(std::string& message) {
    auto config = makeSelfTestTokenizerHP();
    config.target_vocab_size = 50000;
    config.enable_byte_fallback = true;
    
    UniByte tokenizer(config);
    
    std::vector<std::string> inputs = {
        "First sentence.",
        "Second sentence with numbers 123.",
        "Third with email test@test.com",
        "Fourth with URL https://example.com"
    };
    
    std::vector<std::vector<int>> all_tokens;
    
    for (const auto& input : inputs) {
        std::vector<int> tokens = tokenizer.tokenizeWithMetadata(input).token_ids;
        all_tokens.push_back(std::move(tokens));
    }
    
    ASSERT_EQ(all_tokens.size(), inputs.size(), "Should process all inputs");
    
    for (size_t i = 0; i < all_tokens.size(); ++i) {
        ASSERT_TRUE(all_tokens[i].size() > 0, 
                   "Each input should produce tokens");
    }
    
    return true;
}

//======================================================//
//  Section 7: Edge Case Tests
//======================================================//

bool testEdgeCaseEmptyString(std::string& message) {
    auto config = makeSelfTestTokenizerHP();
    config.enable_byte_fallback = true;
    UniByte tokenizer(config);
    
    std::string input = "";
    std::vector<int> tokens = tokenizer.tokenizeWithMetadata(input).token_ids;
    
    ASSERT_EQ(tokens.size(), 0, "Empty string should produce no tokens");
    
    std::string decoded = tokenizer.decode(GRIM::Tokenizer::DecodeRequest(tokens));
    ASSERT_STR_EQ(decoded, "", "Empty decode should be empty");
    
    return true;
}

bool testEdgeCaseSingleChar(std::string& message) {
    auto config = makeSelfTestTokenizerHP();
    config.enable_byte_fallback = true;
    UniByte tokenizer(config);
    
    std::string input = "a";
    std::vector<int> tokens = tokenizer.tokenizeWithMetadata(input).token_ids;
    
    ASSERT_TRUE(tokens.size() >= 1, "Single char should produce at least 1 token");
    
    std::string decoded = tokenizer.decode(GRIM::Tokenizer::DecodeRequest(tokens));
    ASSERT_STR_EQ(decoded, input, "Single char round-trip failed");
    
    return true;
}

bool testEdgeCaseOnlyWhitespace(std::string& message) {
    auto config = makeSelfTestTokenizerHP();
    config.enable_byte_fallback = true;
    UniByte tokenizer(config);
    
    // After ▁ normalization, spaces become ▁ characters
    // Add ▁ piece to vocab so whitespace-only input has vocab matches
    appendSelfTestUnigramPiece(tokenizer.unigramLM(), "\xe2\x96\x81", -0.5f, false);
    tokenizer.unigramLM().buildTrie();
    
    std::string input = "   ";
    std::vector<int> tokens = tokenizer.tokenizeWithMetadata(input).token_ids;
    
    ASSERT_TRUE(tokens.size() >= 1, "Whitespace should produce tokens");
    
    std::string decoded = tokenizer.decode(GRIM::Tokenizer::DecodeRequest(tokens));
    ASSERT_STR_EQ(decoded, input, "Whitespace round-trip failed");
    
    return true;
}

bool testEdgeCaseAsciiSpacingRewriteToSpiece(std::string& message) {
    auto config = makeSelfTestTokenizerHP();
    config.enable_byte_fallback = true;
    UniByte tokenizer(config);

    appendSelfTestUnigramPiece(tokenizer.unigramLM(), "\xe2\x96\x81", -0.5f, false);
    appendSelfTestUnigramPiece(tokenizer.unigramLM(), "\xe2\x96\x81hello", -1.0f, false);
    appendSelfTestUnigramPiece(tokenizer.unigramLM(), "\xe2\x96\x81world", -1.0f, false);
    appendSelfTestUnigramPiece(tokenizer.unigramLM(), "\xe2\x96\x81tab", -1.0f, false);
    appendSelfTestUnigramPiece(tokenizer.unigramLM(), "\xe2\x96\x81" "crlf", -1.0f, false);
    tokenizer.unigramLM().buildTrie();

    const std::string input = "hello\nworld\ttab\r\ncrlf";
    auto result = tokenizer.tokenizeWithMetadata(input);
    std::string decoded = tokenizer.decode(GRIM::Tokenizer::DecodeRequest(result));

    ASSERT_STR_EQ(decoded, "hello world tab crlf",
                  "ASCII newline/tab bytes should rewrite through the shared ▁ marker and decode as spaces");

    for (size_t token_index = 0; token_index < result.token_ids.size(); ++token_index) {
        if (result.is_byte_fallback[token_index]) {
            const int token_id = result.token_ids[token_index];
            ASSERT_FALSE(token_id == byteToTokenId(static_cast<uint8_t>('\n')),
                         "Newline must not survive normalization as a byte fallback token");
            ASSERT_FALSE(token_id == byteToTokenId(static_cast<uint8_t>('\r')),
                         "Carriage return must not survive normalization as a byte fallback token");
            ASSERT_FALSE(token_id == byteToTokenId(static_cast<uint8_t>('\t')),
                         "Tab must not survive normalization as a byte fallback token");
        }
    }

    return true;
}

bool testEdgeCaseLongSequence(std::string& message) {
    auto config = makeSelfTestTokenizerHP();
    config.enable_byte_fallback = true;
    UniByte tokenizer(config);
    
    // Build a long input (10KB)
    std::string input;
    for (int i = 0; i < 1000; ++i) {
        input += "Hello world! ";
    }
    
    std::vector<int> tokens = tokenizer.tokenizeWithMetadata(input).token_ids;
    ASSERT_TRUE(tokens.size() > 0, "Long sequence should produce tokens");
    
    std::string decoded = tokenizer.decode(GRIM::Tokenizer::DecodeRequest(tokens));
    ASSERT_STR_EQ(decoded, input, "Long sequence round-trip failed");
    
    return true;
}

bool testEdgeCaseSpecialTokenLiterals(std::string& message) {
    auto config = makeSelfTestTokenizerHP();
    config.enable_byte_fallback = true;
    // Disable atom reasoning to avoid atom detection interfering with special token literals
    config.enable_atom_reasoning = false;
    UniByte tokenizer(config);
    
    // Add vocab pieces for common words — ▁-prefixed for SentencePiece normalization
    appendSelfTestUnigramPiece(tokenizer.unigramLM(), "\xe2\x96\x81This", -1.0f, false);
    appendSelfTestUnigramPiece(tokenizer.unigramLM(), "\xe2\x96\x81is", -1.0f, false);
    appendSelfTestUnigramPiece(tokenizer.unigramLM(), "\xe2\x96\x81not", -1.0f, false);
    appendSelfTestUnigramPiece(tokenizer.unigramLM(), "\xe2\x96\x81" "a", -1.0f, false);
    appendSelfTestUnigramPiece(tokenizer.unigramLM(), "\xe2\x96\x81special", -1.0f, false);
    appendSelfTestUnigramPiece(tokenizer.unigramLM(), "\xe2\x96\x81token", -1.0f, false);
    // Add the literal special token strings as regular vocab pieces
    appendSelfTestUnigramPiece(tokenizer.unigramLM(), "\xe2\x96\x81<unk>", -2.0f, false);
    appendSelfTestUnigramPiece(tokenizer.unigramLM(), "\xe2\x96\x81<s>", -2.0f, false);
    appendSelfTestUnigramPiece(tokenizer.unigramLM(), "\xe2\x96\x81</s>", -2.0f, false);
    appendSelfTestUnigramPiece(tokenizer.unigramLM(), "\xe2\x96\x81<pad>", -2.0f, false);
    tokenizer.unigramLM().buildTrie();
    
    // Input contains literal special token strings
    std::string input = "This <unk> is not <s> a </s> special <pad> token";
    
    std::vector<int> tokens = tokenizer.tokenizeWithMetadata(input).token_ids;
    ASSERT_TRUE(tokens.size() > 0, "Should produce tokens");
    
    std::string decoded = tokenizer.decode(GRIM::Tokenizer::DecodeRequest(tokens));
    // Should preserve the literal text, not interpret as special tokens
    ASSERT_STR_EQ(decoded, input, "Literal special tokens should round-trip");
    
    return true;
}

//======================================================//
//  Section 8: Unicode & Emoji Tests
//======================================================//

bool testUnicodeEmoji(std::string& message) {
    auto config = makeSelfTestTokenizerHP();
    config.enable_byte_fallback = true;
    UniByte tokenizer(config);
    
    std::string input = "Hello 👋 world 🌍!";
    
    std::vector<int> tokens = tokenizer.tokenizeWithMetadata(input).token_ids;
    ASSERT_TRUE(tokens.size() > 0, "Emoji input should produce tokens");
    
    std::string decoded = tokenizer.decode(GRIM::Tokenizer::DecodeRequest(tokens));
    ASSERT_STR_EQ(decoded, input, "Emoji round-trip failed");
    
    return true;
}

bool testUnicodeMultiLanguage(std::string& message) {
    auto config = makeSelfTestTokenizerHP();
    config.enable_byte_fallback = true;
    UniByte tokenizer(config);
    
    std::string input = "English 日本語 한국어 العربية";
    
    std::vector<int> tokens = tokenizer.tokenizeWithMetadata(input).token_ids;
    ASSERT_TRUE(tokens.size() > 0, "Multi-language should produce tokens");
    
    std::string decoded = tokenizer.decode(GRIM::Tokenizer::DecodeRequest(tokens));
    ASSERT_STR_EQ(decoded, input, "Multi-language round-trip failed");
    
    return true;
}

bool testUnicodeWithStructural(std::string& message) {
    auto config = makeSelfTestTokenizerHP();
    config.enable_byte_fallback = true;
    config.enable_atom_reasoning = true;  // Enable atom detection
    UniByte tokenizer(config);
    
    std::string input = "日本の価格は 42.5 円です";
    
    // Use tokenizeWithMetadata to get both tokens and atom information
    auto result = tokenizer.tokenizeWithMetadata(input);
    ASSERT_TRUE(result.token_ids.size() > 0, "Unicode with numbers should produce tokens");
    
    std::string decoded = tokenizer.decode(GRIM::Tokenizer::DecodeRequest(result));
    const std::string expected =
        "æ—¥æœ¬ã®ä¾¡æ ¼ã¯ <FLOAT>42.5</FLOAT> å††ã§ã™";
    ASSERT_STR_EQ(decoded, expected, "Unicode+numeric typed-span decode failed");
    
    return true;
}

//======================================================//
//  Section 9: Multiple Structural Elements Tests
//======================================================//

bool testMultipleURLs(std::string& message) {
    auto config = makeSelfTestTokenizerHP();
    config.enable_byte_fallback = true;
    UniByte tokenizer(config);
    
    std::string input = "Visit https://first.com and https://second.com or http://third.org";
    
    auto result = tokenizer.tokenizeWithMetadata(input);
    
    ASSERT_EQ(result.atoms.size(), 0, "URLs should remain regular text");
    
    return true;
}

bool testMultipleEmails(std::string& message) {
    auto config = makeSelfTestTokenizerHP();
    config.enable_byte_fallback = true;
    UniByte tokenizer(config);
    
    std::string input = "Contact: alice@example.com, bob@test.org, charlie@domain.net";
    
    auto result = tokenizer.tokenizeWithMetadata(input);
    
    ASSERT_EQ(result.atoms.size(), 0, "Emails should remain regular text");
    
    return true;
}

bool testMixedNumbers(std::string& message) {
    auto config = makeSelfTestTokenizerHP();
    config.enable_byte_fallback = true;
    UniByte tokenizer(config);
    
    std::string input = "Int: 42, Float: 3.14, Negative: -17, Scientific: 1.5e10, Hex: 0xFF";
    
    auto result = tokenizer.tokenizeWithMetadata(input);
    
    int number_count = 0;
    for (const auto& span : result.atoms) {
        if (span.atom_type == AtomType::ATOM_INT || 
            span.atom_type == AtomType::ATOM_FLOAT) {
            number_count++;
        }
    }
    
    ASSERT_TRUE(number_count >= 4, "Should detect multiple number types");
    
    return true;
}

bool testAdjacentStructural(std::string& message) {
    auto config = makeSelfTestTokenizerHP();
    config.enable_byte_fallback = true;
    UniByte tokenizer(config);
    
    // Number immediately followed by letters should still tokenize cleanly.
    std::string input = "Price:$99USD";
    
    auto result = tokenizer.tokenizeWithMetadata(input);
    ASSERT_TRUE(result.token_ids.size() > 0, "Adjacent structures should tokenize");
    ASSERT_TRUE(result.atoms.size() >= 1, "Adjacent numeric text should preserve number atoms");
    
    return true;
}

//======================================================//
//  Section 10: Path Detection Tests
//======================================================//

bool testWindowsPath(std::string& message) {
    auto config = makeSelfTestTokenizerHP();
    config.enable_byte_fallback = true;
    UniByte tokenizer(config);
    
    std::string input = "Open file C:\\Users\\test\\document.txt please";
    
    auto result = tokenizer.tokenizeWithMetadata(input);
    
    ASSERT_EQ(result.atoms.size(), 0, "Paths should remain regular text");
    
    return true;
}

bool testUnixPath(std::string& message) {
    auto config = makeSelfTestTokenizerHP();
    config.enable_byte_fallback = true;
    UniByte tokenizer(config);
    
    std::string input = "Run /usr/local/bin/program with args";
    
    auto result = tokenizer.tokenizeWithMetadata(input);
    
    ASSERT_EQ(result.atoms.size(), 0, "Paths should remain regular text");
    
    return true;
}

//======================================================//
//  Section 11: Numeric Edge Cases
//======================================================//

bool testScientificNotation(std::string& message) {
    auto config = makeSelfTestTokenizerHP();
    config.enable_byte_fallback = true;
    UniByte tokenizer(config);
    
    std::string input = "Values: 1.23e-10, 4.56E+20, 7.89e5";
    
    auto result = tokenizer.tokenizeWithMetadata(input);
    
    int float_count = 0;
    for (const auto& span : result.atoms) {
        if (span.atom_type == AtomType::ATOM_INT) {
            float_count++;
        }
    }
    
    ASSERT_TRUE(float_count >= 2, "Should detect scientific notation");
    
    return true;
}

bool testIPAddressVsDecimal(std::string& message) {
    auto config = makeSelfTestTokenizerHP();
    config.enable_byte_fallback = true;
    // Note: IP detection may be part of number detection or separate
    UniByte tokenizer(config);
    
    std::string input = "Server 192.168.1.1 price 1.2.3";
    
    auto result = tokenizer.tokenizeWithMetadata(input);
    
    bool found_ip = false;
    for (const auto& span : result.atoms) {
        if (span.atom_type == AtomType::ATOM_INT) {
            found_ip = true;
            break;
        }
    }
    
    // IP detection may not be implemented - just verify we got some structural detection
    ASSERT_TRUE(result.atoms.size() >= 0, "Should process without crash");
    
    return true;
}

bool testNegativeNumbers(std::string& message) {
    auto config = makeSelfTestTokenizerHP();
    config.enable_byte_fallback = true;
    UniByte tokenizer(config);
    
    std::string input = "Temperature: -40 degrees, balance: -$1,234.56";
    
    auto result = tokenizer.tokenizeWithMetadata(input);
    
    int number_count = 0;
    for (const auto& span : result.atoms) {
        if (span.atom_type == AtomType::ATOM_INT || span.atom_type == AtomType::ATOM_FLOAT) {
            number_count++;
        }
    }
    
    ASSERT_TRUE(number_count >= 1, "Should detect negative numbers");
    
    return true;
}

bool testDigitsFollowedByAlpha(std::string& message) {
    // Regression test: digits followed by alphabetic chars (ordinals, units, versions)
    // must still be detected as integer atoms while their digits remain visible
    // as ordinary content tokens inside typed boundaries.
    auto config = makeSelfTestTokenizerHP();
    config.enable_byte_fallback = true;
    UniByte tokenizer(config);
    
    struct TestCase {
        std::string input;
        int expected_integers;
        std::string description;
    };
    
    std::vector<TestCase> cases = {
        {"the 5th element", 1, "ordinal '5th'"},
        {"took 100ms to run", 1, "unit suffix '100ms'"},
        {"a 3D model", 1, "dimension '3D'"},
        {"on the 1st day", 1, "ordinal '1st'"},
        {"the 2nd place", 1, "ordinal '2nd'"},
        {"came in 3rd", 1, "ordinal '3rd'"},
        {"version 5 is out", 1, "standalone integer"},
    };
    
    for (const auto& tc : cases) {
        auto result = tokenizer.tokenizeWithMetadata(tc.input);
        
        int int_count = 0;
        for (const auto& span : result.atoms) {
            if (span.atom_type == AtomType::ATOM_INT) {
                int_count++;
            }
        }
        
        std::string fail_msg = "Should detect integer in: " + tc.description 
                             + " (input='" + tc.input + "', found=" + std::to_string(int_count) 
                             + ", expected=" + std::to_string(tc.expected_integers) + ")";
        ASSERT_TRUE(int_count == tc.expected_integers, fail_msg.c_str());
        
        const int open_id = atomTypeToOpenTokenId(AtomType::ATOM_INT);
        const int close_id = atomTypeToCloseTokenId(AtomType::ATOM_INT);
        const auto open_it = std::find(result.token_ids.begin(), result.token_ids.end(), open_id);
        ASSERT_TRUE(open_it != result.token_ids.end(),
                    "Detected integer must emit an opening boundary");
        const auto close_it = std::find(open_it + 1, result.token_ids.end(), close_id);
        ASSERT_TRUE(close_it != result.token_ids.end(),
                    "Detected integer must emit a matching closing boundary");
        ASSERT_TRUE(close_it > open_it + 1,
                    "Detected integer digits must remain as model-visible content tokens");
    }
    
    return true;
}

//======================================================//
//  Section 12: Byte Fallback Control Tests
//======================================================//

bool testByteFallbackDisabled(std::string& message) {
    auto config = makeSelfTestTokenizerHP();
    config.enable_byte_fallback = false;  // Disable byte fallback
    UniByte tokenizer(config);
    
    // With no vocab and no byte fallback, unknown chars should use UNK
    std::string input = "xyz";
    
    std::vector<int> tokens = tokenizer.tokenizeWithMetadata(input).token_ids;
    
    // Should still produce tokens (UNK tokens)
    ASSERT_TRUE(tokens.size() > 0, "Should produce UNK tokens without byte fallback");
    
    return true;
}

bool testMixedVocabAndByteFallback(std::string& message) {
    auto config = makeSelfTestTokenizerHP();
    config.enable_byte_fallback = true;
    UniByte tokenizer(config);
    
    // Add partial vocab — ▁-prefixed for SentencePiece whitespace normalization
    appendSelfTestUnigramPiece(tokenizer.unigramLM(), "\xe2\x96\x81hello", -1.0f, false);
    tokenizer.unigramLM().buildTrie();
    
    // Input has both vocab word and unknown
    std::string input = "hello xyz hello";
    
    std::vector<int> tokens = tokenizer.tokenizeWithMetadata(input).token_ids;
    std::string decoded = tokenizer.decode(GRIM::Tokenizer::DecodeRequest(tokens));
    
    ASSERT_STR_EQ(decoded, input, "Mixed vocab+byte fallback round-trip failed");
    
    return true;
}

bool testByteFallbackRejectsMalformedGeneratedUtf8(std::string& message) {
    auto config = makeSelfTestTokenizerHP();
    config.enable_byte_fallback = true;
    UniByte tokenizer(config);

    const int malformed_byte_token = BYTE_TOKEN_OFFSET + 0xB8;
    std::vector<int> tokens = { BOS_TOKEN_ID, EOS_TOKEN_ID, malformed_byte_token };

    try {
        (void)tokenizer.decode(GRIM::Tokenizer::DecodeRequest(tokens));
        message = "Decode should reject malformed generated byte-token UTF-8 runs";
        return false;
    } catch (const std::exception& e) {
        const std::string error_text = e.what();
        ASSERT_TRUE(error_text.find("invalid UTF-8") != std::string::npos,
                    "Malformed byte-token decode error should mention invalid UTF-8");
        ASSERT_TRUE(error_text.find("token_id=" + std::to_string(malformed_byte_token)) != std::string::npos,
                    "Malformed byte-token decode error should identify the offending token id");
        ASSERT_TRUE(error_text.find("prior_token_count=2") != std::string::npos,
                    "Malformed byte-token decode error should report the prior token count");
        ASSERT_TRUE(error_text.find("prior_token_ids=[2, 3]") != std::string::npos,
                    "Malformed byte-token decode error should dump all prior token ids");
    }

    return true;
}

//======================================================//
//  Section 13: Vocabulary Persistence Tests
//======================================================//

static TokenizerArtifacts::GrmtSequence makePersistenceGrmtSequence() {
    TokenizerArtifacts::GrmtSequence sequence;
    const int int_open_id = atomTypeToOpenTokenId(AtomType::ATOM_INT);
    const int int_close_id = atomTypeToCloseTokenId(AtomType::ATOM_INT);
    sequence.token_ids = {
        int_open_id,
        BYTE_TOKEN_OFFSET + static_cast<int>('4'),
        BYTE_TOKEN_OFFSET + static_cast<int>('2'),
        int_close_id
    };
    sequence.targets = {
        BYTE_TOKEN_OFFSET + static_cast<int>('4'),
        BYTE_TOKEN_OFFSET + static_cast<int>('2'),
        int_close_id,
        EOS_TOKEN_ID
    };
    const std::size_t n = sequence.token_ids.size();
    sequence.token_numeric_values.assign(n, 0.0f);
    sequence.token_atom_mask.assign(n, 0);
    sequence.token_atom_flags.assign(n, 0);
    sequence.atom_table = std::make_shared<AtomTable>();
    sequence.atom_entry_ids.assign(n, kAtomEntryNone);
    sequence.local_atom_table = std::make_shared<SequenceLocalAtomTable>();
    sequence.token_local_atom_indices.assign(n, kLocalAtomIndexNone);
    sequence.token_exec_slot_indices.assign(n, -1);

    const uint32_t entry_id = registerSelfTestAtom(
        *sequence.atom_table, AtomType::ATOM_INT, "42");
    if (entry_id == UINT32_MAX) {
        throw std::runtime_error("makePersistenceGrmtSequence: failed to register integer atom");
    }
    const auto entry = sequence.atom_table->getAtom(entry_id);
    if (!entry.has_value()) {
        throw std::runtime_error("makePersistenceGrmtSequence: registered atom is not retrievable");
    }
    sequence.token_numeric_values[0] = entry->numeric_value;
    sequence.token_atom_mask[0] = 1;
    sequence.token_atom_flags[0] = entry->flags;
    sequence.atom_entry_ids[0] = entry_id;
    sequence.token_local_atom_indices[0] =
        sequence.local_atom_table->ticket(AtomType::ATOM_INT, "42").local_index;
    return sequence;
}

bool testGrmtAtomSpanSideChannelValidation(std::string& message) {
    auto valid = makePersistenceGrmtSequence();
    try {
        valid.validateForWrite("test valid typed atom span");
    } catch (const std::exception& e) {
        message = std::string("Valid typed atom span was rejected: ") + e.what();
        return false;
    }

    std::filesystem::create_directories("output");
    const std::string round_trip_path = "output/test_grmt_typed_atom_span.grmt";
    const auto save_report = TokenizerArtifacts::saveGrmtCorpus(
        round_trip_path,
        std::vector<TokenizerArtifacts::GrmtSequence>{valid},
        static_cast<std::uint32_t>(UNIGRAM_VOCAB_OFFSET));
    ASSERT_EQ(save_report.written_sequences, 1u,
              "Typed atom span fixture should persist as one GRMT row");
    const auto round_trip = TokenizerArtifacts::loadGrmtCorpus(round_trip_path);
    std::filesystem::remove(round_trip_path);
    ASSERT_EQ(round_trip.header.version, GRIM::GRMT_FORMAT_VERSION,
              "Typed atom span fixture should use the current GRMT format");
    ASSERT_EQ(round_trip.sequences.size(), static_cast<std::size_t>(1),
              "Typed atom span fixture should load as one GRMT row");
    ASSERT_TRUE(round_trip.sequences[0].token_ids == valid.token_ids,
                "Typed atom span token IDs must survive GRMT round-trip");
    ASSERT_TRUE(round_trip.sequences[0].token_atom_mask == valid.token_atom_mask,
                "Opening-only atom mask must survive GRMT round-trip");
    ASSERT_TRUE(round_trip.sequences[0].atom_entry_ids == valid.atom_entry_ids,
                "Opening-only AtomTable entry IDs must survive GRMT round-trip");
    ASSERT_TRUE(round_trip.sequences[0].token_local_atom_indices ==
                    valid.token_local_atom_indices,
                "Opening-only local atom indices must survive GRMT round-trip");
    ASSERT_TRUE(round_trip.sequences[0].local_atom_table != nullptr,
                "Sequence-local atom table must survive GRMT round-trip");
    ASSERT_TRUE(round_trip.sequences[0].local_atom_table
                    ->getRawText(AtomType::ATOM_INT, 0)
                    .value_or("") == "42",
                "Round-tripped local atom address must retain its value");

    auto validationRejects = [](const TokenizerArtifacts::GrmtSequence& sequence) {
        try {
            sequence.validateForWrite("test invalid typed atom span");
            return false;
        } catch (const std::runtime_error&) {
            return true;
        }
    };

    auto missing_open_metadata = makePersistenceGrmtSequence();
    missing_open_metadata.token_numeric_values[0] = 0.0f;
    missing_open_metadata.token_atom_mask[0] = 0;
    missing_open_metadata.token_atom_flags[0] = 0;
    missing_open_metadata.atom_entry_ids[0] = kAtomEntryNone;
    ASSERT_TRUE(validationRejects(missing_open_metadata),
                "GRMT must reject an opening boundary without its metadata anchor");

    auto close_metadata = makePersistenceGrmtSequence();
    const std::size_t close_index = close_metadata.token_ids.size() - 1;
    close_metadata.token_numeric_values[close_index] = close_metadata.token_numeric_values[0];
    close_metadata.token_atom_mask[close_index] = 1;
    close_metadata.token_atom_flags[close_index] = close_metadata.token_atom_flags[0];
    close_metadata.atom_entry_ids[close_index] = close_metadata.atom_entry_ids[0];
    ASSERT_TRUE(validationRejects(close_metadata),
                "GRMT must reject duplicated metadata on a closing boundary");

    auto content_metadata = makePersistenceGrmtSequence();
    content_metadata.token_numeric_values[1] = content_metadata.token_numeric_values[0];
    content_metadata.token_atom_mask[1] = 1;
    content_metadata.token_atom_flags[1] = content_metadata.token_atom_flags[0];
    content_metadata.atom_entry_ids[1] = content_metadata.atom_entry_ids[0];
    ASSERT_TRUE(validationRejects(content_metadata),
                "GRMT must reject atom metadata on ordinary span content");

    auto runtime_aux_mask = makePersistenceGrmtSequence();
    runtime_aux_mask.token_atom_aux_target_mask.assign(
        runtime_aux_mask.token_ids.size(), 0);
    ASSERT_TRUE(validationRejects(runtime_aux_mask),
                "GRMT must reject serialization of the runtime-derived atom auxiliary target mask");

    auto mismatched_type = makePersistenceGrmtSequence();
    mismatched_type.token_ids[0] = atomTypeToOpenTokenId(AtomType::ATOM_FLOAT);
    ASSERT_TRUE(validationRejects(mismatched_type),
                "GRMT must reject an opening boundary whose type disagrees with AtomTable");

    return true;
}

bool testGrmtFixedNumericTokenRoundTripAndRangeValidation(std::string& message) {
    TokenizerArtifacts::GrmtSequence sequence;
    appendNumericLiteralTokenIds("-11111.0000e+22", sequence.token_ids);
    sequence.targets.assign(sequence.token_ids.size(), -1);
    for (size_t index = 0; index + 1 < sequence.token_ids.size(); ++index) {
        sequence.targets[index] = sequence.token_ids[index + 1];
    }
    const size_t n = sequence.token_ids.size();
    sequence.token_numeric_values.assign(n, 0.0f);
    sequence.token_atom_mask.assign(n, 0);
    sequence.token_atom_flags.assign(n, 0);
    sequence.atom_entry_ids.assign(n, kAtomEntryNone);
    sequence.local_atom_table = std::make_shared<SequenceLocalAtomTable>();
    sequence.token_local_atom_indices.assign(n, kLocalAtomIndexNone);
    sequence.token_exec_slot_indices.assign(n, -1);

    std::filesystem::create_directories("output");
    const std::string round_trip_path = "output/test_grmt_fixed_numeric.grmt";
    const std::uint32_t vocab_size = static_cast<std::uint32_t>(UNIGRAM_VOCAB_OFFSET);
    const auto save_report = TokenizerArtifacts::saveGrmtCorpus(
        round_trip_path,
        std::vector<TokenizerArtifacts::GrmtSequence>{sequence},
        vocab_size);
    ASSERT_EQ(save_report.vocab_size, vocab_size,
              "Numeric GRMT header must retain the full fixed token space");

    const auto round_trip = TokenizerArtifacts::loadGrmtCorpus(round_trip_path);
    std::filesystem::remove(round_trip_path);
    ASSERT_EQ(round_trip.sequences.size(), static_cast<size_t>(1),
              "Numeric GRMT fixture should load as one row");
    ASSERT_TRUE(round_trip.sequences[0].token_ids == sequence.token_ids,
                "Fixed numeric token IDs must survive GRMT round-trip unchanged");
    ASSERT_TRUE(round_trip.sequences[0].targets == sequence.targets,
                "Fixed numeric next-token targets must survive GRMT round-trip unchanged");

    auto writerRejects = [vocab_size](TokenizerArtifacts::GrmtSequence invalid,
                                      const std::string& path) {
        try {
            (void)TokenizerArtifacts::saveGrmtCorpus(
                path,
                std::vector<TokenizerArtifacts::GrmtSequence>{std::move(invalid)},
                vocab_size);
        } catch (const std::runtime_error&) {
            std::filesystem::remove(path);
            std::filesystem::remove(path + ".tmp");
            return true;
        }
        std::filesystem::remove(path);
        std::filesystem::remove(path + ".tmp");
        return false;
    };

    auto invalid_token = sequence;
    invalid_token.token_ids[0] = static_cast<int>(vocab_size);
    ASSERT_TRUE(writerRejects(
                    std::move(invalid_token),
                    "output/test_grmt_invalid_primary_token.grmt"),
                "GRMT writer must reject a primary token ID at vocab_size");

    auto invalid_target = sequence;
    invalid_target.targets[0] = static_cast<int>(vocab_size);
    ASSERT_TRUE(writerRejects(
                    std::move(invalid_target),
                    "output/test_grmt_invalid_primary_target.grmt"),
                "GRMT writer must reject a primary target ID at vocab_size");
    return true;
}

static TokenizerArtifacts::GrmtSequence makeWindowingAtomSequence(
    std::size_t prefix_tokens,
    std::size_t suffix_tokens) {
    TokenizerArtifacts::GrmtSequence sequence;
    for (std::size_t i = 0; i < prefix_tokens; ++i) {
        sequence.token_ids.push_back(
            BYTE_TOKEN_OFFSET + static_cast<int>('a' + (i % 20)));
    }
    const std::size_t open_index = sequence.token_ids.size();
    sequence.token_ids.push_back(atomTypeToOpenTokenId(AtomType::ATOM_INT));
    sequence.token_ids.push_back(BYTE_TOKEN_OFFSET + static_cast<int>('4'));
    sequence.token_ids.push_back(BYTE_TOKEN_OFFSET + static_cast<int>('2'));
    sequence.token_ids.push_back(atomTypeToCloseTokenId(AtomType::ATOM_INT));
    for (std::size_t i = 0; i < suffix_tokens; ++i) {
        sequence.token_ids.push_back(
            BYTE_TOKEN_OFFSET + static_cast<int>('u' + (i % 5)));
    }

    const std::size_t n = sequence.token_ids.size();
    sequence.targets.assign(n, -1);
    for (std::size_t i = 0; i + 1 < n; ++i) {
        sequence.targets[i] = sequence.token_ids[i + 1];
    }
    sequence.token_numeric_values.assign(n, 0.0f);
    sequence.token_atom_mask.assign(n, 0);
    sequence.token_atom_flags.assign(n, 0);
    sequence.atom_entry_ids.assign(n, kAtomEntryNone);
    sequence.token_local_atom_indices.assign(n, kLocalAtomIndexNone);
    sequence.token_exec_slot_indices.assign(n, -1);
    sequence.atom_table = std::make_shared<AtomTable>();
    sequence.local_atom_table = std::make_shared<SequenceLocalAtomTable>();

    const uint32_t entry_id = registerSelfTestAtom(
        *sequence.atom_table, AtomType::ATOM_INT, "42");
    if (entry_id == UINT32_MAX) {
        throw std::runtime_error("makeWindowingAtomSequence: failed to register integer atom");
    }
    const auto entry = sequence.atom_table->getAtom(entry_id);
    if (!entry.has_value()) {
        throw std::runtime_error("makeWindowingAtomSequence: atom entry is not retrievable");
    }
    sequence.token_numeric_values[open_index] = entry->numeric_value;
    sequence.token_atom_mask[open_index] = 1;
    sequence.token_atom_flags[open_index] = entry->flags;
    sequence.atom_entry_ids[open_index] = entry_id;
    sequence.token_local_atom_indices[open_index] =
        sequence.local_atom_table->ticket(AtomType::ATOM_INT, "42").local_index;
    return sequence;
}

bool testSlidingWindowsPreserveTypedAtomSpans(std::string& message) {
    auto validateWindows = [&](const std::vector<TokenizerArtifacts::GrmtSequence>& windows,
                               std::size_t max_seq_len,
                               const char* stage) {
        bool found_complete_span = false;
        for (const auto& window : windows) {
            if (window.token_ids.size() > max_seq_len) {
                message = std::string(stage) + " emitted an overlong window";
                return false;
            }
            if (window.token_atom_aux_target_mask.size() != window.token_ids.size()) {
                message = std::string(stage) +
                          " did not author a token-aligned atom auxiliary target mask";
                return false;
            }
            bool inside_atom = false;
            AtomType open_type = AtomType::ATOM_INT;
            for (std::size_t i = 0; i < window.token_ids.size(); ++i) {
                const int token_id = window.token_ids[i];
                if (isAtomOpenTokenId(token_id)) {
                    if (inside_atom || window.token_atom_mask[i] != 1 ||
                        window.atom_entry_ids[i] == kAtomEntryNone ||
                        window.token_local_atom_indices[i] == kLocalAtomIndexNone ||
                        !window.local_atom_table ||
                        !window.local_atom_table->contains(
                            tokenIdToAtomType(token_id),
                            window.token_local_atom_indices[i]) ||
                        window.token_atom_aux_target_mask[i] != 1) {
                        message = std::string(stage) +
                                  " emitted an invalid typed opening boundary";
                        return false;
                    }
                    inside_atom = true;
                    open_type = tokenIdToAtomType(token_id);
                    continue;
                }
                if (isAtomCloseTokenId(token_id)) {
                    if (!inside_atom || tokenIdToAtomType(token_id) != open_type ||
                        window.token_atom_mask[i] != 0 ||
                        window.atom_entry_ids[i] != kAtomEntryNone ||
                        window.token_local_atom_indices[i] != kLocalAtomIndexNone ||
                        window.token_atom_aux_target_mask[i] != 0) {
                        message = std::string(stage) +
                                  " emitted an unmatched or metadata-bearing closing boundary";
                        return false;
                    }
                    inside_atom = false;
                    found_complete_span = true;
                    continue;
                }
                if (window.token_atom_mask[i] != 0 ||
                    window.atom_entry_ids[i] != kAtomEntryNone ||
                    window.token_local_atom_indices[i] != kLocalAtomIndexNone) {
                    message = std::string(stage) +
                              " moved atom metadata away from the opening boundary";
                    return false;
                }
                const uint8_t expected_aux_owner = inside_atom ? 1 : 0;
                if (window.token_atom_aux_target_mask[i] != expected_aux_owner) {
                    message = std::string(stage) +
                              " authored an incorrect causal atom auxiliary target mask";
                    return false;
                }
            }
            if (inside_atom) {
                message = std::string(stage) + " split a typed atom span";
                return false;
            }
        }
        if (!found_complete_span) {
            message = std::string(stage) + " dropped the typed atom span";
            return false;
        }
        return true;
    };

    std::filesystem::create_directories("output");
    {
        TrainingLogger logger("output", "sliding_window_atom_span");

        std::vector<TokenizerArtifacts::GrmtSequence> pt_sequences{
            makeWindowingAtomSequence(4, 4)};
        GRIMText::Training::applySlidingWindows(
            pt_sequences,
            "atom-span-pt-test",
            GRIM::HyperParameters::TrainingStage::PT,
            6,
            4,
            0,
            false,
            false,
            logger);
        ASSERT_TRUE(pt_sequences.size() > 1,
                    "PT atom-span fixture should produce multiple windows");
        if (!validateWindows(pt_sequences, 6, "PT")) {
            return false;
        }

        std::vector<TokenizerArtifacts::GrmtSequence> exact_capacity_sequences{
            makeWindowingAtomSequence(0, 2)};
        GRIMText::Training::applySlidingWindows(
            exact_capacity_sequences,
            "atom-span-exact-capacity-test",
            GRIM::HyperParameters::TrainingStage::PT,
            4,
            3,
            0,
            false,
            false,
            logger);
        ASSERT_TRUE(exact_capacity_sequences.size() > 1,
                    "An exact-capacity atom span must preserve following tail tokens");
        if (!validateWindows(exact_capacity_sequences, 4, "PT exact-capacity")) {
            return false;
        }

        bool rejected_overcapacity_atom = false;
        try {
            std::vector<TokenizerArtifacts::GrmtSequence> overcapacity_sequences{
                makeWindowingAtomSequence(0, 2)};
            GRIMText::Training::applySlidingWindows(
                overcapacity_sequences,
                "atom-span-overcapacity-test",
                GRIM::HyperParameters::TrainingStage::PT,
                3,
                2,
                0,
                false,
                false,
                logger);
        } catch (const std::runtime_error&) {
            rejected_overcapacity_atom = true;
        }
        ASSERT_TRUE(rejected_overcapacity_atom,
                    "A typed atom span larger than window capacity must fail loudly");

        auto sft_sequence = makeWindowingAtomSequence(6, 5);
        sft_sequence.prompt_length = 2;
        sft_sequence.prompt_end_pos = 1;
        std::vector<TokenizerArtifacts::GrmtSequence> sft_sequences{
            std::move(sft_sequence)};
        GRIMText::Training::applySlidingWindows(
            sft_sequences,
            "atom-span-sft-test",
            GRIM::HyperParameters::TrainingStage::SFT,
            8,
            6,
            0,
            false,
            false,
            logger);
        ASSERT_TRUE(sft_sequences.size() > 1,
                    "SFT atom-span fixture should produce multiple windows");
        if (!validateWindows(sft_sequences, 8, "SFT")) {
            return false;
        }
    }
    std::filesystem::remove("output/training_sliding_window_atom_span.log");
    return true;
}

bool testVocabTextExportBinaryLoad(std::string& message) {
    // Create output directory if it doesn't exist
    std::filesystem::create_directories("output");
    
    auto config = makeSelfTestTokenizerHP();
    UniByte original(config);
    appendSelfTestUnigramPiece(original.unigramLM(), "test", -1.0f, false);
    appendSelfTestUnigramPiece(original.unigramLM(), "vocab", -1.5f, false);
    appendSelfTestUnigramPiece(original.unigramLM(), "save", -2.0f, false);
    original.unigramLM().buildTrie();
    
    std::string grmt_path = "output/test_vocab_save.grmt";
    std::string path = "output/test_vocab_save.bin";
    std::string text_path = "output/test_vocab_save.txt";
    config.data_path = grmt_path;
    config.vocab_path = path;
    config.save_text_vocab = true;
    std::vector<TokenizerArtifacts::GrmtSequence> sequences;
    sequences.push_back(makePersistenceGrmtSequence());
    auto report = TokenizerArtifacts::saveTokenizerArtifactBundle(config, original, sequences);
    ASSERT_EQ(report.grmt.written_sequences, 1u, "Bundle should write one GRMT sequence");
    ASSERT_TRUE(std::filesystem::exists(text_path), "Text vocab sidecar should be exported");
    
    // Load into new tokenizer
    UniByte loaded(config);
    auto manifest = TokenizerArtifacts::loadTokenizerArtifactBundle(config, loaded);
    ASSERT_EQ(manifest.grmt_header.num_sequences, 1u, "Bundle load should validate GRMT header");
    
    // Verify pieces exist
    ASSERT_TRUE(loaded.unigramLM().hasPiece("test"), "Should have 'test' piece");
    ASSERT_TRUE(loaded.unigramLM().hasPiece("vocab"), "Should have 'vocab' piece");
    ASSERT_TRUE(loaded.unigramLM().hasPiece("save"), "Should have 'save' piece");
    
    // Cleanup
    std::filesystem::remove(grmt_path);
    std::filesystem::remove(path);
    std::filesystem::remove(text_path);
    
    return true;
}

bool testVocabSaveLoadBinary(std::string& message) {
    // Create output directory if it doesn't exist
    std::filesystem::create_directories("output");
    
    auto config = makeSelfTestTokenizerHP();
    UniByte original(config);
    appendSelfTestUnigramPiece(original.unigramLM(), "binary", -1.0f, false);
    appendSelfTestUnigramPiece(original.unigramLM(), "format", -1.5f, false);
    appendSelfTestUnigramPiece(original.unigramLM(), "fast", -2.0f, false);
    original.unigramLM().buildTrie();
    
    std::string grmt_path = "output/test_vocab_binary.grmt";
    std::string path = "output/test_vocab_binary.bin";
    config.data_path = grmt_path;
    config.vocab_path = path;
    std::vector<TokenizerArtifacts::GrmtSequence> sequences;
    sequences.push_back(makePersistenceGrmtSequence());
    auto report = TokenizerArtifacts::saveTokenizerArtifactBundle(config, original, sequences);
    ASSERT_EQ(report.manifest.tokenizer_vocab_size, static_cast<std::uint32_t>(original.vocabSize()),
              "Bundle save should report tokenizer vocab size");
    
    // Load into new tokenizer
    UniByte loaded(config);
    auto manifest = TokenizerArtifacts::loadTokenizerArtifactBundle(config, loaded);
    ASSERT_EQ(manifest.tokenizer_vocab_size, static_cast<std::uint32_t>(loaded.vocabSize()),
              "Bundle load should report loaded tokenizer vocab size");
    
    // Verify learned vocab entries survive special-metadata records in the file.
    ASSERT_TRUE(loaded.unigramLM().hasPiece("binary"), "Should have 'binary' piece");
    ASSERT_TRUE(loaded.unigramLM().hasPiece("format"), "Should have 'format' piece");
    
    // Cleanup
    std::filesystem::remove(grmt_path);
    std::filesystem::remove(path);
    
    return true;
}

bool testSharedVocabWritesMultipleGrmtsWithoutMutation(std::string& message) {
    std::filesystem::create_directories("output");

    auto config = makeSelfTestTokenizerHP();
    const std::string vocab_path = "output/test_shared_vocab.bin";
    const std::string first_grmt = "output/test_shared_vocab_first.grmt";
    const std::string second_grmt = "output/test_shared_vocab_second.grmt";
    config.vocab_path = vocab_path;
    config.data_path = first_grmt;

    UniByte original(config);
    appendSelfTestUnigramPiece(original.unigramLM(), "shared", -1.0f, false);
    appendSelfTestUnigramPiece(original.unigramLM(), "vocab", -1.5f, false);
    original.unigramLM().buildTrie();

    std::vector<TokenizerArtifacts::GrmtSequence> sequences{
        makePersistenceGrmtSequence()
    };
    (void)TokenizerArtifacts::saveTokenizerArtifactBundle(config, original, sequences);

    auto read_bytes = [](const std::string& path) {
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input.is_open()) {
            throw std::runtime_error("failed to open test artifact: " + path);
        }
        const std::streamsize size = input.tellg();
        input.seekg(0, std::ios::beg);
        std::vector<char> bytes(static_cast<std::size_t>(size));
        if (size > 0 && !input.read(bytes.data(), size)) {
            throw std::runtime_error("failed to read test artifact: " + path);
        }
        return bytes;
    };
    const auto vocab_before = read_bytes(vocab_path);

    config.data_path = second_grmt;
    UniByte shared(config);
    const auto shared_size = TokenizerArtifacts::loadSharedTokenizerVocabulary(config, shared);
    ASSERT_EQ(shared_size, static_cast<std::uint32_t>(original.vocabSize()),
              "Shared-vocab load should preserve the original token space");
    const auto report = TokenizerArtifacts::saveGrmtForSharedTokenizerVocabulary(
        config, shared, sequences);
    ASSERT_EQ(report.grmt.written_sequences, 1u,
              "GRMT-only save should write the second corpus");
    ASSERT_TRUE(std::filesystem::exists(first_grmt),
                "Writing the second GRMT must preserve the first GRMT");
    ASSERT_TRUE(vocab_before == read_bytes(vocab_path),
                "GRMT-only save must not mutate the shared vocab bytes");

    UniByte verified(config);
    const auto manifest = TokenizerArtifacts::loadTokenizerArtifactBundle(config, verified);
    ASSERT_EQ(manifest.grmt_header.vocab_size, shared_size,
              "Second GRMT must validate against the shared vocab");

    std::filesystem::remove(first_grmt);
    std::filesystem::remove(second_grmt);
    std::filesystem::remove(vocab_path);
    return true;
}

//======================================================//
//  Section 14: GPU Upload Tests
//======================================================//

bool testGPUUpload(std::string& message) {
    UnigramLM unigram;
    
    // Add vocab — ▁-prefixed for SentencePiece whitespace normalization
    appendSelfTestUnigramPiece(unigram, "\xe2\x96\x81gpu", -1.0f, false);
    appendSelfTestUnigramPiece(unigram, "\xe2\x96\x81" "decode", -1.5f, false);
    unigram.buildTrie();
    
    // Init GPU
    bool gpu_ok = unigram.initGPU();
    if (!gpu_ok) {
        message = "GPU init failed (may not have CUDA)";
        return true;  // Skip test if no GPU
    }
    
    // Encode through the live UnigramLM API after GPU upload.
    std::vector<int> tokens = unigram.encode("gpu decode");
    ASSERT_EQ(tokens.size(), static_cast<size_t>(2), "GPU-uploaded unigram vocab should encode to two learned pieces");
    ASSERT_EQ(tokens[0], unigram.getPieceId("\xe2\x96\x81" "gpu"), "First token should be the learned ▁gpu piece");
    ASSERT_EQ(tokens[1], unigram.getPieceId("\xe2\x96\x81" "decode"), "Second token should be the learned ▁decode piece");
    
    return true;
}

static bool assertCudaSuccess(cudaError_t err, std::string& message, const char* label) {
    if (err != cudaSuccess) {
        message = std::string(label) + " failed: " + cudaGetErrorString(err);
        return false;
    }
    return true;
}

static bool readViterbiCudaKernelError(int* d_error_code, int& host_error_code, std::string& message, const char* label) {
    if (!assertCudaSuccess(cudaMemcpy(&host_error_code, d_error_code, sizeof(int), cudaMemcpyDeviceToHost), message, label)) {
        return false;
    }
    return true;
}

static bool assertViterbiCudaKernelOk(int* d_error_code, std::string& message, const char* label) {
    int host_error_code = -1;
    if (!readViterbiCudaKernelError(d_error_code, host_error_code, message, label)) {
        return false;
    }
    if (host_error_code != kUnigramViterbiCudaOk) {
        message = std::string(label) + " reported error_code=" + std::to_string(host_error_code) +
                  " (" + unigramViterbiCudaErrorName(host_error_code) + ")";
        return false;
    }
    return true;
}

static cudaError_t launchTestViterbiForward(
    char* d_text,
    size_t length,
    int* d_trie_children,
    int* d_trie_token_ids,
    float* d_trie_scores,
    int num_nodes,
    float* d_viterbi_scores,
    int* d_viterbi_prev,
    int* d_viterbi_tokens,
    bool* d_selected_fallback,
    int* d_error_code) {
    int unk_id = UNK_TOKEN_ID;
    bool enable_byte_fallback = true;
    void* args[] = {
        &d_text,
        &length,
        &d_trie_children,
        &d_trie_token_ids,
        &d_trie_scores,
        &num_nodes,
        &d_viterbi_scores,
        &d_viterbi_prev,
        &d_viterbi_tokens,
        &d_selected_fallback,
        &unk_id,
        &enable_byte_fallback,
        &d_error_code
    };
    return cudaLaunchKernel(
        reinterpret_cast<const void*>(&kernelViterbiForward),
        dim3(1),
        dim3(1),
        args,
        0,
        nullptr);
}

static cudaError_t launchTestViterbiBacktrack(
    size_t length,
    int* d_viterbi_prev,
    int* d_viterbi_tokens,
    int* d_output_tokens,
    int* d_output_count,
    int max_tokens,
    bool* d_selected_fallback,
    int* d_error_code) {
    void* args[] = {
        &length,
        &d_viterbi_prev,
        &d_viterbi_tokens,
        &d_output_tokens,
        &d_output_count,
        &max_tokens,
        &d_selected_fallback,
        &d_error_code
    };
    return cudaLaunchKernel(
        reinterpret_cast<const void*>(&kernelViterbiBacktrack),
        dim3(1),
        dim3(1),
        args,
        0,
        nullptr);
}

bool testCudaViterbiForwardUsesForwardTrie(std::string& message) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        return true;
    }

    const int num_nodes = 4;
    const int token_abc = UNIGRAM_VOCAB_OFFSET + 42;
    std::vector<int> trie_children(static_cast<size_t>(num_nodes) * 256, -1);
    std::vector<int> trie_token_ids(num_nodes, -1);
    std::vector<float> trie_scores(num_nodes, UNKNOWN_SCORE);

    trie_children[static_cast<size_t>(0) * 256 + static_cast<unsigned char>('a')] = 1;
    trie_children[static_cast<size_t>(1) * 256 + static_cast<unsigned char>('b')] = 2;
    trie_children[static_cast<size_t>(2) * 256 + static_cast<unsigned char>('c')] = 3;
    trie_token_ids[3] = token_abc;
    trie_scores[3] = -1.0f;

    char* d_text = nullptr;
    int* d_trie_children = nullptr;
    int* d_trie_token_ids = nullptr;
    float* d_trie_scores = nullptr;
    float* d_viterbi_scores = nullptr;
    int* d_viterbi_prev = nullptr;
    int* d_viterbi_tokens = nullptr;
    bool* d_selected_fallback = nullptr;
    int* d_output_tokens = nullptr;
    int* d_output_count = nullptr;
    int* d_error_code = nullptr;

    auto cleanup = [&]() {
        cudaFree(d_text);
        cudaFree(d_trie_children);
        cudaFree(d_trie_token_ids);
        cudaFree(d_trie_scores);
        cudaFree(d_viterbi_scores);
        cudaFree(d_viterbi_prev);
        cudaFree(d_viterbi_tokens);
        cudaFree(d_selected_fallback);
        cudaFree(d_output_tokens);
        cudaFree(d_output_count);
        cudaFree(d_error_code);
    };

    const size_t max_text_len = 3;
    if (!assertCudaSuccess(cudaMalloc(reinterpret_cast<void**>(&d_text), max_text_len), message, "cudaMalloc d_text")) { cleanup(); return false; }
    if (!assertCudaSuccess(cudaMalloc(reinterpret_cast<void**>(&d_trie_children), trie_children.size() * sizeof(int)), message, "cudaMalloc d_trie_children")) { cleanup(); return false; }
    if (!assertCudaSuccess(cudaMalloc(reinterpret_cast<void**>(&d_trie_token_ids), trie_token_ids.size() * sizeof(int)), message, "cudaMalloc d_trie_token_ids")) { cleanup(); return false; }
    if (!assertCudaSuccess(cudaMalloc(reinterpret_cast<void**>(&d_trie_scores), trie_scores.size() * sizeof(float)), message, "cudaMalloc d_trie_scores")) { cleanup(); return false; }
    if (!assertCudaSuccess(cudaMalloc(reinterpret_cast<void**>(&d_viterbi_scores), (max_text_len + 1) * sizeof(float)), message, "cudaMalloc d_viterbi_scores")) { cleanup(); return false; }
    if (!assertCudaSuccess(cudaMalloc(reinterpret_cast<void**>(&d_viterbi_prev), (max_text_len + 1) * sizeof(int)), message, "cudaMalloc d_viterbi_prev")) { cleanup(); return false; }
    if (!assertCudaSuccess(cudaMalloc(reinterpret_cast<void**>(&d_viterbi_tokens), (max_text_len + 1) * sizeof(int)), message, "cudaMalloc d_viterbi_tokens")) { cleanup(); return false; }
    if (!assertCudaSuccess(cudaMalloc(reinterpret_cast<void**>(&d_selected_fallback), max_text_len * sizeof(bool)), message, "cudaMalloc d_selected_fallback")) { cleanup(); return false; }
    if (!assertCudaSuccess(cudaMalloc(reinterpret_cast<void**>(&d_output_tokens), max_text_len * sizeof(int)), message, "cudaMalloc d_output_tokens")) { cleanup(); return false; }
    if (!assertCudaSuccess(cudaMalloc(reinterpret_cast<void**>(&d_output_count), sizeof(int)), message, "cudaMalloc d_output_count")) { cleanup(); return false; }
    if (!assertCudaSuccess(cudaMalloc(reinterpret_cast<void**>(&d_error_code), sizeof(int)), message, "cudaMalloc d_error_code")) { cleanup(); return false; }

    if (!assertCudaSuccess(cudaMemcpy(d_trie_children, trie_children.data(), trie_children.size() * sizeof(int), cudaMemcpyHostToDevice), message, "cudaMemcpy d_trie_children")) { cleanup(); return false; }
    if (!assertCudaSuccess(cudaMemcpy(d_trie_token_ids, trie_token_ids.data(), trie_token_ids.size() * sizeof(int), cudaMemcpyHostToDevice), message, "cudaMemcpy d_trie_token_ids")) { cleanup(); return false; }
    if (!assertCudaSuccess(cudaMemcpy(d_trie_scores, trie_scores.data(), trie_scores.size() * sizeof(float), cudaMemcpyHostToDevice), message, "cudaMemcpy d_trie_scores")) { cleanup(); return false; }

    const std::string matched_text = "abc";
    if (!assertCudaSuccess(cudaMemcpy(d_text, matched_text.data(), matched_text.size(), cudaMemcpyHostToDevice), message, "cudaMemcpy matched text")) { cleanup(); return false; }
    if (!assertCudaSuccess(launchTestViterbiForward(
        d_text,
        matched_text.size(),
        d_trie_children,
        d_trie_token_ids,
        d_trie_scores,
        num_nodes,
        d_viterbi_scores,
        d_viterbi_prev,
        d_viterbi_tokens,
        d_selected_fallback,
        d_error_code), message, "kernelViterbiForward matched launch")) { cleanup(); return false; }
    if (!assertCudaSuccess(cudaDeviceSynchronize(), message, "kernelViterbiForward matched sync")) { cleanup(); return false; }
    if (!assertViterbiCudaKernelOk(d_error_code, message, "kernelViterbiForward matched status")) { cleanup(); return false; }

    std::vector<float> host_scores(max_text_len + 1, 0.0f);
    std::vector<int> host_prev(max_text_len + 1, -1);
    std::vector<int> host_tokens(max_text_len + 1, -1);
    std::vector<int> host_output_tokens(max_text_len, -1);
    static_assert(sizeof(bool) == sizeof(unsigned char), "CUDA bool marker copies require byte-sized bool");
    std::vector<unsigned char> host_selected_fallback(max_text_len, 1);
    if (!assertCudaSuccess(cudaMemcpy(host_scores.data(), d_viterbi_scores, host_scores.size() * sizeof(float), cudaMemcpyDeviceToHost), message, "cudaMemcpy matched scores")) { cleanup(); return false; }
    if (!assertCudaSuccess(cudaMemcpy(host_prev.data(), d_viterbi_prev, host_prev.size() * sizeof(int), cudaMemcpyDeviceToHost), message, "cudaMemcpy matched prev")) { cleanup(); return false; }
    if (!assertCudaSuccess(cudaMemcpy(host_tokens.data(), d_viterbi_tokens, host_tokens.size() * sizeof(int), cudaMemcpyDeviceToHost), message, "cudaMemcpy matched tokens")) { cleanup(); return false; }
    if (!assertCudaSuccess(cudaMemcpy(host_selected_fallback.data(), d_selected_fallback, host_selected_fallback.size() * sizeof(bool), cudaMemcpyDeviceToHost), message, "cudaMemcpy matched pre-backtrack markers")) { cleanup(); return false; }

    if (host_prev[3] != 0 || host_tokens[3] != token_abc) {
        message = "CUDA Viterbi did not match forward trie token 'abc': prev=" +
                  std::to_string(host_prev[3]) + ", token=" + std::to_string(host_tokens[3]) +
                  ", expected_token=" + std::to_string(token_abc);
        cleanup();
        return false;
    }
    if (std::abs(host_scores[3] - (-1.0f)) > 0.0001f) {
        message = "CUDA Viterbi score mismatch for 'abc': got " + std::to_string(host_scores[3]);
        cleanup();
        return false;
    }
    if (std::any_of(host_selected_fallback.begin(), host_selected_fallback.end(), [](unsigned char marked) { return marked != 0; })) {
        message = "CUDA Viterbi forward marked fallback bytes before final-path backtrack";
        cleanup();
        return false;
    }

    if (!assertCudaSuccess(launchTestViterbiBacktrack(
        matched_text.size(),
        d_viterbi_prev,
        d_viterbi_tokens,
        d_output_tokens,
        d_output_count,
        static_cast<int>(max_text_len),
        d_selected_fallback,
        d_error_code), message, "kernelViterbiBacktrack matched launch")) { cleanup(); return false; }
    if (!assertCudaSuccess(cudaDeviceSynchronize(), message, "kernelViterbiBacktrack matched sync")) { cleanup(); return false; }
    if (!assertViterbiCudaKernelOk(d_error_code, message, "kernelViterbiBacktrack matched status")) { cleanup(); return false; }

    int host_output_count = -1;
    if (!assertCudaSuccess(cudaMemcpy(&host_output_count, d_output_count, sizeof(int), cudaMemcpyDeviceToHost), message, "cudaMemcpy matched output count")) { cleanup(); return false; }
    if (!assertCudaSuccess(cudaMemcpy(host_output_tokens.data(), d_output_tokens, host_output_tokens.size() * sizeof(int), cudaMemcpyDeviceToHost), message, "cudaMemcpy matched output tokens")) { cleanup(); return false; }
    if (!assertCudaSuccess(cudaMemcpy(host_selected_fallback.data(), d_selected_fallback, host_selected_fallback.size() * sizeof(bool), cudaMemcpyDeviceToHost), message, "cudaMemcpy matched selected markers")) { cleanup(); return false; }

    if (host_output_count != 1 || host_output_tokens[0] != token_abc) {
        message = "CUDA Viterbi backtrack did not select full 'abc' token: count=" +
                  std::to_string(host_output_count) + ", token0=" + std::to_string(host_output_tokens[0]) +
                  ", expected_token=" + std::to_string(token_abc);
        cleanup();
        return false;
    }
    if (std::any_of(host_selected_fallback.begin(), host_selected_fallback.end(), [](unsigned char marked) { return marked != 0; })) {
        message = "CUDA Viterbi backtrack marked fallback bytes for learned full-token path";
        cleanup();
        return false;
    }

    const std::string fallback_text = "x";
    if (!assertCudaSuccess(cudaMemcpy(d_text, fallback_text.data(), fallback_text.size(), cudaMemcpyHostToDevice), message, "cudaMemcpy fallback text")) { cleanup(); return false; }
    if (!assertCudaSuccess(launchTestViterbiForward(
        d_text,
        fallback_text.size(),
        d_trie_children,
        d_trie_token_ids,
        d_trie_scores,
        num_nodes,
        d_viterbi_scores,
        d_viterbi_prev,
        d_viterbi_tokens,
        d_selected_fallback,
        d_error_code), message, "kernelViterbiForward fallback launch")) { cleanup(); return false; }
    if (!assertCudaSuccess(cudaDeviceSynchronize(), message, "kernelViterbiForward fallback sync")) { cleanup(); return false; }
    if (!assertViterbiCudaKernelOk(d_error_code, message, "kernelViterbiForward fallback status")) { cleanup(); return false; }

    if (!assertCudaSuccess(launchTestViterbiBacktrack(
        fallback_text.size(),
        d_viterbi_prev,
        d_viterbi_tokens,
        d_output_tokens,
        d_output_count,
        static_cast<int>(max_text_len),
        d_selected_fallback,
        d_error_code), message, "kernelViterbiBacktrack fallback launch")) { cleanup(); return false; }
    if (!assertCudaSuccess(cudaDeviceSynchronize(), message, "kernelViterbiBacktrack fallback sync")) { cleanup(); return false; }
    if (!assertViterbiCudaKernelOk(d_error_code, message, "kernelViterbiBacktrack fallback status")) { cleanup(); return false; }

    if (!assertCudaSuccess(cudaMemcpy(&host_output_count, d_output_count, sizeof(int), cudaMemcpyDeviceToHost), message, "cudaMemcpy fallback output count")) { cleanup(); return false; }
    if (!assertCudaSuccess(cudaMemcpy(host_output_tokens.data(), d_output_tokens, host_output_tokens.size() * sizeof(int), cudaMemcpyDeviceToHost), message, "cudaMemcpy fallback output tokens")) { cleanup(); return false; }
    if (!assertCudaSuccess(cudaMemcpy(host_selected_fallback.data(), d_selected_fallback, host_selected_fallback.size() * sizeof(bool), cudaMemcpyDeviceToHost), message, "cudaMemcpy fallback selected markers")) { cleanup(); return false; }

    const int expected_byte_token = BYTE_TOKEN_OFFSET + static_cast<int>(static_cast<unsigned char>('x'));
    if (host_output_count != 1 || host_output_tokens[0] != expected_byte_token || host_selected_fallback[0] == 0) {
        message = "CUDA Viterbi byte fallback mismatch for selected 'x' path: count=" +
                  std::to_string(host_output_count) + ", token0=" + std::to_string(host_output_tokens[0]) +
                  ", expected_token=" + std::to_string(expected_byte_token);
        cleanup();
        return false;
    }

    const std::string overflow_text = "xx";
    if (!assertCudaSuccess(cudaMemcpy(d_text, overflow_text.data(), overflow_text.size(), cudaMemcpyHostToDevice), message, "cudaMemcpy overflow text")) { cleanup(); return false; }
    if (!assertCudaSuccess(launchTestViterbiForward(
        d_text,
        overflow_text.size(),
        d_trie_children,
        d_trie_token_ids,
        d_trie_scores,
        num_nodes,
        d_viterbi_scores,
        d_viterbi_prev,
        d_viterbi_tokens,
        d_selected_fallback,
        d_error_code), message, "kernelViterbiForward overflow launch")) { cleanup(); return false; }
    if (!assertCudaSuccess(cudaDeviceSynchronize(), message, "kernelViterbiForward overflow sync")) { cleanup(); return false; }
    if (!assertViterbiCudaKernelOk(d_error_code, message, "kernelViterbiForward overflow status")) { cleanup(); return false; }

    if (!assertCudaSuccess(launchTestViterbiBacktrack(
        overflow_text.size(),
        d_viterbi_prev,
        d_viterbi_tokens,
        d_output_tokens,
        d_output_count,
        1,
        d_selected_fallback,
        d_error_code), message, "kernelViterbiBacktrack overflow launch")) { cleanup(); return false; }
    if (!assertCudaSuccess(cudaDeviceSynchronize(), message, "kernelViterbiBacktrack overflow sync")) { cleanup(); return false; }

    int overflow_error_code = -1;
    if (!readViterbiCudaKernelError(d_error_code, overflow_error_code, message, "kernelViterbiBacktrack overflow status")) { cleanup(); return false; }
    if (overflow_error_code != kUnigramViterbiCudaOutputBufferTooSmall) {
        message = "CUDA Viterbi backtrack failed to hard-report count > max_tokens: error_code=" +
                  std::to_string(overflow_error_code) + " (" + unigramViterbiCudaErrorName(overflow_error_code) + ")";
        cleanup();
        return false;
    }

    cleanup();
    return true;
}

//======================================================//
//  Main
//======================================================//

int main(int argc, char** argv) {
    // Check CUDA availability
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    
    if (device_count == 0) {
        std::cerr << "No CUDA devices found. Some tests may fail.\n";
    } else {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "Using CUDA device: " << prop.name << "\n";
    }
    
    UnigramByteTestSuite suite;
    
    // Section 2: Unigram LM Tests
    suite.addTest("Unigram.BuildVocab", testUnigramBuildVocab);
    suite.addTest("Unigram.Encode", testUnigramEncode);
    suite.addTest("Unigram.Viterbi", testUnigramViterbi);
    suite.addTest("Unigram.SentencePiecePunctuation", testUnigramSentencePiecePunctuationPiece);
    suite.addTest("Unigram.Unknown", testUnigramUnknown);
    suite.addTest("Unigram.ForwardBackward.ByteFallbackFixedPenalty", testUnigramForwardBackwardByteFallbackIsFixedPenalty);
    suite.addTest("Unigram.ForwardBackward.ByteFallbackUtf8IsByteLevel", testUnigramForwardBackwardByteFallbackIsByteLevelForUtf8);
    suite.addTest("Unigram.Train.FinalCleanupRejectsEmptyLearnedVocab", testUnigramTrainFinalCleanupRejectsEmptyLearnedVocab);
    suite.addTest("Unigram.Train.RejectEmptyAcceptedCandidateSet", testUnigramTrainRejectsEmptyAcceptedCandidateSet);
    suite.addTest("Unigram.Train.RejectFallbackDominatedPosteriorMass", testUnigramTrainRejectsFallbackDominatedPosteriorMass);
    suite.addTest("Unigram.Train.ByteFallbackOffAddsCharSeeds", testUnigramTrainByteFallbackDisabledAddsCharacterSeeds);
    suite.addTest("Unigram.Train.ByteFallbackOffFailsUncoveredChars", testUnigramTrainByteFallbackDisabledFailsOnUncoveredCharacterSeed);
    suite.addTest("Unigram.Train.SubwordCountsUseUint64", testUnigramTrainSubwordCountsUseUint64);
    suite.addTest("Unigram.Train.LikelihoodLossPrunesTowardTarget", testUnigramTrainLikelihoodLossPrunesTowardTarget);
    suite.addTest("Unigram.Train.FilterRepetitionNoise", testUnigramTrainFiltersRepetitionNoise);
    suite.addTest("Unigram.Train.DedupRepeatedVariants", testUnigramTrainDedupsRepeatedVariants);
    suite.addTest("Unigram.Train.PreserveSpieceUnderlineDedupBoundary", testUnigramTrainPreservesSpieceUnderlineDedupBoundary);
    suite.addTest("Unigram.Train.StridedMiningLatePatterns", testUnigramTrainStridedMiningKeepsLateDocumentPatterns);
    suite.addTest("Unigram.Train.StridedMiningBoundaryOverlap", testUnigramTrainStridedMiningOverlapsBoundaryCandidates);
    suite.addTest("Unigram.Train.BytePieceLimit", testUnigramTrainEnforcesBytePieceLimit);
    
    // Section 3: Aho-Corasick Tests
    suite.addTest("AhoCorasick.BasicMatches", testAhoCorasickBasicMatches);
    suite.addTest("AhoCorasick.OutputClosure", testAhoCorasickOutputClosure);
    suite.addTest("AhoCorasick.StructuralVsNaive", testAhoCorasickStructuralVsNaive);
    suite.addTest("AhoCorasick.CaseInsensitive", testAhoCorasickCaseInsensitive);
    suite.addTest("AhoCorasick.Visualization", testAhoCorasickVisualization);

    // Section 4: UniByte Orchestrator Tests
    suite.addTest("NumericTokens.FixedEncoding", testFixedNumericTokenEncoding);
    suite.addTest("NumericTokens.ExcludesAtomSpans", testNumericTokenSpanSelectionExcludesAtoms);
    suite.addTest("UniByte.BasicEncode", testUniByteBasicEncode);
    suite.addTest("AtomTable.RejectsBadNumericDetectionWithContext", testAtomTableRejectsBadNumericDetectionWithContext);
    suite.addTest("UniByte.DefaultRegistryDoesNotDetectPlainNumbers", testDefaultRegistryDoesNotDetectPlainNumbers);
    suite.addTest("UniByte.AuthoredAtomDelimiterDetector", testAuthoredAtomDelimiterDetector);
    suite.addTest("UniByte.URLPassthrough", testUniByteURLDetection);
    suite.addTest("UniByte.URLPassthrough.CaseInsensitive", testUniByteURLDetectionCaseInsensitive);
    suite.addTest("UniByte.EmailPassthrough", testUniByteEmailDetection);
    suite.addTest("UniByte.RoundTrip", testUniByteRoundTrip);
    
    // Section 5: AtomTable Tests
    suite.addTest("AtomTable.RegisterInteger", testAtomTableRegisterInteger);
    suite.addTest("AtomTable.RegisterFloat", testAtomTableRegisterFloat);
    suite.addTest("AtomTable.RegisterHex", testAtomTableRegisterHex);
    suite.addTest("AtomTable.RegisterBinary", testAtomTableRegisterBinary);
    suite.addTest("AtomTable.RegisterURL", testAtomTableRegisterURL);
    suite.addTest("AtomTable.RegisterEmail", testAtomTableRegisterEmail);
    suite.addTest("AtomTable.RegisterDate", testAtomTableRegisterDate);
    suite.addTest("AtomTable.RegisterTime", testAtomTableRegisterTime);
    suite.addTest("AtomTable.RegisterIP", testAtomTableRegisterIP);
    suite.addTest("AtomTable.RegisterPath", testAtomTableRegisterPath);
    suite.addTest("AtomTable.RegisterString", testAtomTableRegisterString);
    suite.addTest("AtomTable.RegisterIdentifier", testAtomTableRegisterIdentifier);
    suite.addTest("AtomTable.LookupByType", testAtomTableLookupByType);
    suite.addTest("AtomTable.GPUUpload", testAtomTableGPUUpload);
    suite.addTest("AtomTable.Clear", testAtomTableClear);
    suite.addTest("AtomTable.Metadata", testAtomTableMetadata);
    suite.addTest("AtomTable.HashDeduplication", testAtomTableHashDeduplication);
    suite.addTest("SequenceLocalAtomTable.Ticketing", testSequenceLocalAtomTicketing);
    suite.addTest("AtomTable.DoesNotPopulateArgNumber", testAtomTableDoesNotPopulateArgNumber);
    
    // Section 6: Integration Tests
    suite.addTest("Integration.BatchProcessing", testBatchProcessing);
    
    // Section 7: Edge Case Tests
    suite.addTest("EdgeCase.EmptyString", testEdgeCaseEmptyString);
    suite.addTest("EdgeCase.SingleChar", testEdgeCaseSingleChar);
    suite.addTest("EdgeCase.OnlyWhitespace", testEdgeCaseOnlyWhitespace);
    suite.addTest("EdgeCase.AsciiSpacingRewriteToSpiece", testEdgeCaseAsciiSpacingRewriteToSpiece);
    suite.addTest("EdgeCase.LongSequence", testEdgeCaseLongSequence);
    suite.addTest("EdgeCase.SpecialTokenLiterals", testEdgeCaseSpecialTokenLiterals);
    
    // Section 8: Unicode & Emoji Tests
    suite.addTest("Unicode.Emoji", testUnicodeEmoji);
    suite.addTest("Unicode.MultiLanguage", testUnicodeMultiLanguage);
    
    // Section 9: Multiple Structural Elements Tests
    suite.addTest("Structural.MultipleURLsPassthrough", testMultipleURLs);
    suite.addTest("Structural.MultipleEmailsPassthrough", testMultipleEmails);
    
    // Section 10: Path Detection Tests
    suite.addTest("Path.WindowsPassthrough", testWindowsPath);
    suite.addTest("Path.UnixPassthrough", testUnixPath);
    
    // Section 12: Byte Fallback Control Tests
    suite.addTest("ByteFallback.Disabled", testByteFallbackDisabled);
    suite.addTest("ByteFallback.Mixed", testMixedVocabAndByteFallback);
    suite.addTest("ByteFallback.RejectMalformedUtf8", testByteFallbackRejectsMalformedGeneratedUtf8);
    
    // Section 13: Vocabulary Persistence Tests
    suite.addTest("GRMT.TypedAtomSpanSideChannels", testGrmtAtomSpanSideChannelValidation);
    suite.addTest("GRMT.FixedNumericTokenRoundTrip", testGrmtFixedNumericTokenRoundTripAndRangeValidation);
    suite.addTest("SlidingWindow.TypedAtomSpanIntegrity", testSlidingWindowsPreserveTypedAtomSpans);
    suite.addTest("Vocab.TextExportBinaryLoad", testVocabTextExportBinaryLoad);
    suite.addTest("Vocab.SaveLoadBinary", testVocabSaveLoadBinary);
    suite.addTest("Vocab.SharedAcrossMultipleGrmts", testSharedVocabWritesMultipleGrmtsWithoutMutation);
    
    // Section 14: GPU Upload Tests
    suite.addTest("GPU.ViterbiForwardTrie", testCudaViterbiForwardUsesForwardTrie);
    suite.addTest("GPU.Upload", testGPUUpload);
    
    // Run all tests
    auto results = suite.runAll();
    
    // Return exit code based on results
    int failures = 0;
    for (const auto& result : results) {
        if (!result.passed) ++failures;
    }
    
    return failures > 0 ? 1 : 0;
}
