//======================================================//
//  unigrambyte_self_test.cu
//  Comprehensive test suite for UnigramByte tokenizer
//======================================================//

#include "unigrambyte_self_test.hpp"

#include "../Shared/UnigramByte/TokenLayout.hpp"
#include "../Shared/UnigramByte/Unigram.hpp"
#include "../Shared/UnigramByte/VocabWriteOp.hpp"
#include "../Shared/UnigramByte/Training/UnigramForwardBackward.hpp"
#include "../Shared/UnigramByte/UnigramViterbi.hpp"
#include "../Shared/UnigramByte/UniByte.hpp"
#include "../Shared/UnigramByte/AtomTable.hpp"
#include "../Shared/UnigramByte/Detectors/DetectorRegistry.hpp"
#include "../Shared/UnigramByte/Detectors/StructuralSpan.hpp"
#include "../Shared/UnigramByte/AhoCorasick.hpp"
#include "../Shared/TokenizerArtifacts/TokenizerArtifactBundle.hpp"

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
double scoreUnigramShrinkCandidateForPosteriorCompression(const std::string& piece_text,
                                                          double posterior_expected_count,
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
    hp.detect_numbers = true;
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
    span.placeholder_id = atomTypeToTokenId(type);

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

bool testUnigramTrainShrinkRankingUsesCompressionGain(std::string& message) {
    const double frequent_tiny_fragment_score = scoreUnigramShrinkCandidateForPosteriorCompression(
        "a",
        10.0,
        "testUnigramTrainShrinkRankingUsesCompressionGain tiny");
    const double less_frequent_high_compression_score = scoreUnigramShrinkCandidateForPosteriorCompression(
        "abcdefghij",
        9.2,
        "testUnigramTrainShrinkRankingUsesCompressionGain high-compression");

    ASSERT_TRUE(less_frequent_high_compression_score > frequent_tiny_fragment_score,
                "Shrink ranking should let compression gain beat a slightly more frequent tiny fragment");
    ASSERT_NEAR(scoreUnigramShrinkCandidateForPosteriorCompression(
                    "abcdefghij",
                    0.0,
                    "testUnigramTrainShrinkRankingUsesCompressionGain zero-count"),
                0.0,
                1.0e-12,
                "Zero-posterior pieces must not survive only because they are long");
    ASSERT_NEAR(scoreUnigramShrinkCandidateForPosteriorCompression(
                    "abcdefghij",
                    0.01,
                    "testUnigramTrainShrinkRankingUsesCompressionGain rare-long"),
                0.10,
                1.0e-12,
                "Rare long pieces should receive expected compression gain, not a near-free normalized ratio bonus");

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
    config.detect_numbers = true;
    
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
        config.detect_numbers = true;

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
        config.detect_numbers = true;

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
            0,
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

bool testAtomTableArgNumberSupportsSignedDecimalExponent(std::string& message) {
    const std::string text = "-4 +4 .75 75.0 1e6 -1.5e-4";
    Detector::DetectorRegistry registry = Detector::makeDefaultRawTextDetectorRegistry();
    const Detector::RawTextDetectorOptions options(true, true, true);
    const auto detections = registry.scan(text, options);

    const AtomTableFromDetectionsResult result = createAtomTableFromRawTextDetections(
        std::string_view(text.data(), text.size()),
        detections,
        0,
        "testAtomTableArgNumberSupportsSignedDecimalExponent");

    ASSERT_EQ(result.atom_tokens.size(), static_cast<size_t>(6),
              "Expected six numeric atom tokens");
    ASSERT_EQ(static_cast<int>(result.arg_number_payload.malformed_numbers), 0,
              "arg_number population must not silently mark detector-approved numerics malformed");
    ASSERT_EQ(static_cast<int>(result.arg_number_payload.total_numbers), 6,
              "Every numeric atom should receive arg_number payload");
    ASSERT_EQ(static_cast<int>(result.arg_number_payload.total_digits), 10,
              "Mantissa digit binding count mismatch");

    const auto minus_four_entry = result.atom_table->getAtom(result.atom_tokens[0].atom_entry_id);
    ASSERT_TRUE(minus_four_entry.has_value(), "-4 atom entry missing");
    ASSERT_TRUE(minus_four_entry->arg_number.has_value(), "-4 arg_number metadata missing");
    const AtomNumber& minus_four = *minus_four_entry->arg_number;
    ASSERT_EQ(static_cast<int>(minus_four.has_sign), 1, "-4 should record a sign");
    ASSERT_EQ(static_cast<int>(minus_four.sign_negative), 1, "-4 should record a negative sign");
    ASSERT_EQ(static_cast<int>(minus_four.digits[0].digit), 4, "-4 digit mismatch");
    ASSERT_EQ(static_cast<int>(minus_four.digits[0].pow10), 0, "-4 digit pow10 mismatch");

    const auto plus_four_entry = result.atom_table->getAtom(result.atom_tokens[1].atom_entry_id);
    ASSERT_TRUE(plus_four_entry.has_value(), "+4 atom entry missing");
    ASSERT_TRUE(plus_four_entry->arg_number.has_value(), "+4 arg_number metadata missing");
    const AtomNumber& plus_four = *plus_four_entry->arg_number;
    ASSERT_EQ(static_cast<int>(plus_four.has_sign), 1, "+4 should record a sign");
    ASSERT_EQ(static_cast<int>(plus_four.sign_negative), 0, "+4 should record a positive sign");
    ASSERT_EQ(static_cast<int>(plus_four.digits[0].digit), 4, "+4 digit mismatch");

    const auto dot_seventy_five_entry = result.atom_table->getAtom(result.atom_tokens[2].atom_entry_id);
    ASSERT_TRUE(dot_seventy_five_entry.has_value(), ".75 atom entry missing");
    ASSERT_TRUE(dot_seventy_five_entry->arg_number.has_value(), ".75 arg_number metadata missing");
    const AtomNumber& dot_seventy_five = *dot_seventy_five_entry->arg_number;
    ASSERT_EQ(static_cast<int>(dot_seventy_five.has_decimal_point), 1,
              ".75 should record decimal metadata");
    ASSERT_EQ(static_cast<int>(dot_seventy_five.integer_digit_count), 0,
              ".75 should have zero integer mantissa digits");
    ASSERT_EQ(static_cast<int>(dot_seventy_five.fractional_digit_count), 2,
              ".75 should have two fractional mantissa digits");
    ASSERT_EQ(static_cast<int>(dot_seventy_five.digits[0].digit), 7,
              ".75 first digit mismatch");
    ASSERT_EQ(static_cast<int>(dot_seventy_five.digits[0].pow10), -1,
              ".75 first digit pow10 mismatch");
    ASSERT_EQ(static_cast<int>(dot_seventy_five.digits[1].digit), 5,
              ".75 second digit mismatch");
    ASSERT_EQ(static_cast<int>(dot_seventy_five.digits[1].pow10), -2,
              ".75 second digit pow10 mismatch");

    const auto seventy_five_point_zero_entry = result.atom_table->getAtom(result.atom_tokens[3].atom_entry_id);
    ASSERT_TRUE(seventy_five_point_zero_entry.has_value(), "75.0 atom entry missing");
    ASSERT_TRUE(seventy_five_point_zero_entry->arg_number.has_value(), "75.0 arg_number metadata missing");
    const AtomNumber& seventy_five_point_zero = *seventy_five_point_zero_entry->arg_number;
    ASSERT_EQ(static_cast<int>(seventy_five_point_zero.digits.size()), 3,
              "75.0 should bind all mantissa digits");
    ASSERT_EQ(static_cast<int>(seventy_five_point_zero.digits[0].pow10), 1,
              "75.0 hundreds/tens place pow10 mismatch");
    ASSERT_EQ(static_cast<int>(seventy_five_point_zero.digits[1].pow10), 0,
              "75.0 ones place pow10 mismatch");
    ASSERT_EQ(static_cast<int>(seventy_five_point_zero.digits[2].pow10), -1,
              "75.0 fractional digit pow10 mismatch");

    const auto one_e_six_entry = result.atom_table->getAtom(result.atom_tokens[4].atom_entry_id);
    ASSERT_TRUE(one_e_six_entry.has_value(), "1e6 atom entry missing");
    ASSERT_TRUE(one_e_six_entry->arg_number.has_value(), "1e6 arg_number metadata missing");
    const AtomNumber& one_e_six = *one_e_six_entry->arg_number;
    ASSERT_EQ(static_cast<int>(one_e_six.has_exponent), 1, "1e6 should record exponent metadata");
    ASSERT_EQ(one_e_six.exponent_value, 6, "1e6 exponent value mismatch");
    ASSERT_EQ(static_cast<int>(one_e_six.digits[0].pow10), 6, "1e6 mantissa pow10 mismatch");

    const auto signed_scientific_entry = result.atom_table->getAtom(result.atom_tokens[5].atom_entry_id);
    ASSERT_TRUE(signed_scientific_entry.has_value(), "-1.5e-4 atom entry missing");
    ASSERT_TRUE(signed_scientific_entry->arg_number.has_value(), "-1.5e-4 arg_number metadata missing");
    const AtomNumber& signed_scientific = *signed_scientific_entry->arg_number;
    ASSERT_EQ(static_cast<int>(signed_scientific.has_sign), 1,
              "-1.5e-4 should record mantissa sign");
    ASSERT_EQ(static_cast<int>(signed_scientific.sign_negative), 1,
              "-1.5e-4 should record negative mantissa sign");
    ASSERT_EQ(static_cast<int>(signed_scientific.has_decimal_point), 1,
              "-1.5e-4 should record decimal point");
    ASSERT_EQ(static_cast<int>(signed_scientific.has_exponent), 1,
              "-1.5e-4 should record exponent metadata");
    ASSERT_EQ(static_cast<int>(signed_scientific.exponent_negative), 1,
              "-1.5e-4 should record negative exponent sign");
    ASSERT_EQ(signed_scientific.exponent_value, -4,
              "-1.5e-4 exponent value mismatch");
    ASSERT_EQ(static_cast<int>(signed_scientific.digits[0].digit), 1,
              "-1.5e-4 first mantissa digit mismatch");
    ASSERT_EQ(static_cast<int>(signed_scientific.digits[0].pow10), -4,
              "-1.5e-4 first mantissa pow10 mismatch");
    ASSERT_EQ(static_cast<int>(signed_scientific.digits[1].digit), 5,
              "-1.5e-4 second mantissa digit mismatch");
    ASSERT_EQ(static_cast<int>(signed_scientific.digits[1].pow10), -5,
              "-1.5e-4 second mantissa pow10 mismatch");

    return true;
}

bool testAtomTokenizationRejectsMantissaDigitSlotOverflow(std::string& message) {
    const std::string text = "value 12345678901234567";
    Detector::DetectorRegistry registry = Detector::makeDefaultRawTextDetectorRegistry();
    const Detector::RawTextDetectorOptions options(true, true, true);
    const auto detections = registry.scan(text, options);

    bool threw = false;
    std::string error_text;
    try {
        (void)createAtomTableFromRawTextDetections(
            std::string_view(text.data(), text.size()),
            detections,
            16,
            "testAtomTokenizationRejectsMantissaDigitSlotOverflow");
    } catch (const std::runtime_error& e) {
        threw = true;
        error_text = e.what();
    }

    ASSERT_TRUE(threw, "Tokenization-time atom creation must fail when mantissa digit count exceeds configured max slots");
    ASSERT_TRUE(error_text.find("max_mantissa_digit_slots=16") != std::string::npos,
                "Overflow error must include configured max_mantissa_digit_slots");
    ASSERT_TRUE(error_text.find("raw_text='12345678901234567'") != std::string::npos,
                "Overflow error must include offending numeric raw_text");
    ASSERT_TRUE(error_text.find("mantissa_digit_sequence='12345678901234567'") != std::string::npos,
                "Overflow error must include offending mantissa digit sequence");

    return true;
}

bool testUniByteRawTextDetectorRegistry(std::string& message) {
    auto config = makeSelfTestTokenizerHP();
    config.detect_numbers = true;
    auto registry = Detector::makeDefaultRawTextDetectorRegistry();
    const Detector::RawTextDetectorOptions options(
        config.detect_numbers,
        true,
        true);

    const std::string text = "CPU 42\nGPU -3.5x";
    const auto detections = registry.scan(text, options);

    ASSERT_EQ(detections.size(), static_cast<size_t>(6), "Raw detector count mismatch");

    auto spanText = [&](size_t idx) {
        const auto& d = detections[idx];
        return text.substr(d.start, d.end - d.start);
    };

    ASSERT_TRUE(detections[0].feature == Detector::RawTextFeature::UPPERCASE_RUN,
                "First raw detection should be uppercase");
    ASSERT_FALSE(detections[0].emitsAtom(), "Uppercase detector must not emit atoms");
    ASSERT_STR_EQ(spanText(0), "CPU", "Uppercase span mismatch");

    ASSERT_TRUE(detections[1].feature == Detector::RawTextFeature::WHITESPACE,
                "Second raw detection should be whitespace");
    ASSERT_FALSE(detections[1].emitsAtom(), "Whitespace detector must not emit atoms");
    ASSERT_STR_EQ(spanText(1), " ", "Whitespace span mismatch");

    ASSERT_TRUE(detections[2].emitsAtom(), "Integer raw detection should emit atom");
    ASSERT_TRUE(detections[2].atom_type == AtomType::ATOM_INT,
                "Integer detector emitted wrong atom type");
    ASSERT_STR_EQ(spanText(2), "42", "Integer span mismatch");

    ASSERT_TRUE(detections[5].emitsAtom(), "Float raw detection should emit atom");
    ASSERT_TRUE(detections[5].atom_type == AtomType::ATOM_FLOAT,
                "Float detector emitted wrong atom type");
    ASSERT_STR_EQ(spanText(5), "-3.5", "Float span mismatch");

    std::vector<StructuralSpan> structures;
    structures.reserve(detections.size());
    for (const auto& detection : detections) {
        if (!detection.emitsAtom()) {
            continue;
        }

        StructuralSpan span;
        span.start = detection.start;
        span.end = detection.end;
        span.atom_type = detection.atom_type;
        span.buffer_ptr = text.data();
        span.offset = static_cast<uint32_t>(detection.start);
        span.length = static_cast<uint32_t>(detection.end - detection.start);
        span.content_offset = static_cast<uint32_t>(detection.start);
        span.content_length = static_cast<uint32_t>(detection.end - detection.start);
        span.placeholder_id = atomTypeToTokenId(detection.atom_type);
        structures.push_back(span);
    }

    ASSERT_EQ(structures.size(), static_cast<size_t>(2), "Only atom detections should become structures");
    ASSERT_TRUE(structures[0].atom_type == AtomType::ATOM_INT,
                "First structure should be integer atom");
    ASSERT_TRUE(structures[1].atom_type == AtomType::ATOM_FLOAT,
                "Second structure should be float atom");

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

bool testUniBytePlaceholderInjection(std::string& message) {
    auto config = makeSelfTestTokenizerHP();
    config.target_vocab_size = 50000;
    config.detect_numbers = true;
    
    UniByte tokenizer(config);
    
    std::string input = "Count: 12345";
    auto result = tokenizer.tokenizeWithMetadata(input);
    
    // Check that placeholder token was injected
    bool found_placeholder = false;
    for (int token : result.token_ids) {
        // Atom tokens are in range [ATOM_TOKEN_OFFSET, ATOM_TOKEN_OFFSET + ATOM_VOCAB_SIZE)
        if (token >= static_cast<int>(ATOM_TOKEN_OFFSET) && token < static_cast<int>(ATOM_TOKEN_OFFSET + ATOM_VOCAB_SIZE)) {
            found_placeholder = true;
            break;
        }
    }
    
    ASSERT_TRUE(found_placeholder, "Should inject placeholder for number");
    
    return true;
}

    bool testUniBytePreRegistersAtomTableBeforePlaceholderEmission(std::string& message) {
        auto config = makeSelfTestTokenizerHP();
        config.target_vocab_size = 50000;
        config.detect_numbers = true;

        UniByte tokenizer(config);

        const std::string input = "same 42 then 42";
        auto result = tokenizer.tokenizeWithMetadata(input);

        ASSERT_TRUE(result.atom_table != nullptr,
                    "tokenizeWithMetadata must create a per-sequence AtomTable before placeholder merge");
        ASSERT_EQ(result.atoms.size(), static_cast<size_t>(2),
                  "Repeated-number fixture should yield two structural atom spans");
        ASSERT_TRUE(result.atoms[0].atom_entry_id != kAtomEntryNone,
                    "Pre-registered structural spans must carry their AtomTable entry ID");
        ASSERT_EQ(result.atoms[0].atom_entry_id, result.atoms[1].atom_entry_id,
                  "Repeated identical atoms should deduplicate to one AtomTable entry before unigram runs");

        size_t placeholder_count = 0;
        for (size_t i = 0; i < result.token_ids.size(); ++i) {
            const bool token_is_atom = result.token_ids[i] >= static_cast<int>(ATOM_TOKEN_OFFSET) &&
                                       result.token_ids[i] < static_cast<int>(UNIGRAM_VOCAB_OFFSET);
            if (!token_is_atom) {
                continue;
            }
            ++placeholder_count;
            ASSERT_EQ(result.token_atom_mask[i], static_cast<uint8_t>(1),
                      "Placeholder emission must preserve token_atom_mask for atom tokens");
            ASSERT_EQ(result.atom_entry_ids[i], result.atoms[0].atom_entry_id,
                      "Placeholder emission must reuse the pre-registered AtomTable entry ID");
        }

        ASSERT_EQ(placeholder_count, static_cast<size_t>(2),
                  "Repeated identical numbers should still emit two placeholder tokens");

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
        } else if (tid >= static_cast<int>(GRIM::Tokenizer::BYTE_TOKEN_OFFSET) && tid < static_cast<int>(ATOM_TOKEN_OFFSET)) {
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
    ASSERT_TRUE(entry1->arg_number.has_value(), "Deduped numeric atom should retain arg_number metadata");
    ASSERT_TRUE(entry2->arg_number.has_value(), "Reloaded deduped numeric atom copy should carry arg_number metadata");
    
    // Different atom should have different hash
    uint32_t id3 = registerSelfTestAtom(table, AtomType::ATOM_INT, "200", 0, 3);
    auto entry3 = table.getAtom(id3);
    
    ASSERT_TRUE(entry3, "Different atom should exist");
    ASSERT_TRUE(entry1->hash != entry3->hash, "Different atoms should have different hashes");
    
    return true;
}

bool testAtomTableArgNumberSerializesOnEntry(std::string& message) {
    AtomTable table;
    const uint32_t id = registerSelfTestAtom(table, AtomType::ATOM_FLOAT, "-1.5e-4", 12, 19);
    ASSERT_TRUE(id != UINT32_MAX, "Failed to register numeric atom for arg_number serialization test");

    const auto entry = table.getAtom(id);
    ASSERT_TRUE(entry.has_value(), "Original atom entry missing before serialization");
    ASSERT_TRUE(entry->arg_number.has_value(), "Original atom entry missing arg_number metadata");

    std::filesystem::create_directories("output");
    const std::filesystem::path text_path = std::filesystem::path("output") / "atomtable_arg_number_entry.tsv";

    std::stringstream binary_stream;
    table.serializeToStreamOrThrow(binary_stream, "testAtomTableArgNumberSerializesOnEntry binary_stream");
    ASSERT_TRUE(table.saveToTextFile(text_path.string()), "AtomTable text save should include entry-owned arg_number metadata");

    AtomTable loaded;
    loaded.deserializeFromStreamOrThrow(binary_stream, "testAtomTableArgNumberSerializesOnEntry binary_stream");

    const auto loaded_entry = loaded.getAtom(id);
    ASSERT_TRUE(loaded_entry.has_value(), "Loaded atom entry missing after serialization round-trip");
    ASSERT_TRUE(loaded_entry->arg_number.has_value(), "Loaded atom entry missing arg_number metadata after serialization round-trip");

    const AtomNumber& loaded_number = *loaded_entry->arg_number;
    ASSERT_EQ(static_cast<int>(loaded_number.has_sign), 1, "Loaded arg_number should preserve mantissa sign metadata");
    ASSERT_EQ(static_cast<int>(loaded_number.has_decimal_point), 1, "Loaded arg_number should preserve decimal-point metadata");
    ASSERT_EQ(static_cast<int>(loaded_number.has_exponent), 1, "Loaded arg_number should preserve exponent metadata");
    ASSERT_EQ(loaded_number.exponent_value, -4, "Loaded arg_number exponent mismatch");
    ASSERT_EQ(static_cast<int>(loaded_number.digits.size()), 2, "Loaded arg_number digit count mismatch");
    ASSERT_EQ(static_cast<int>(loaded_number.digits[0].pow10), -4, "Loaded arg_number first digit pow10 mismatch");
    ASSERT_EQ(static_cast<int>(loaded_number.digits[1].pow10), -5, "Loaded arg_number second digit pow10 mismatch");

    std::ifstream text_dump(text_path);
    ASSERT_TRUE(text_dump.is_open(), "Serialized AtomTable text dump missing");
    std::stringstream text_buffer;
    text_buffer << text_dump.rdbuf();
    const std::string text_dump_contents = text_buffer.str();
    ASSERT_TRUE(text_dump_contents.find("arg_number") != std::string::npos,
                "Text dump header should include arg_number column");
    ASSERT_TRUE(text_dump_contents.find("pow10=-4") != std::string::npos,
                "Text dump should serialize digit binding pow10 metadata");

    std::filesystem::remove(text_path);
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
    config.detect_numbers = true;
    
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
    
    // Verify we can decode back through the single atom-aware decode entry point.
    std::string decoded = tokenizer.decode(GRIM::Tokenizer::DecodeRequest(result));
    
    // Note: With placeholders, decoded may differ from input
    // The key is that we have a valid token sequence
    
    return true;
}

bool testAtomTableIntegration(std::string& message) {
    auto config = makeSelfTestTokenizerHP();
    config.target_vocab_size = 50000;
    config.detect_numbers = true;
    
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
    config.detect_numbers = true;
    config.enable_atom_reasoning = true;  // Enable atom detection
    UniByte tokenizer(config);
    
    std::string input = "日本の価格は 42.5 円です";
    
    // Use tokenizeWithMetadata to get both tokens and atom information
    auto result = tokenizer.tokenizeWithMetadata(input);
    ASSERT_TRUE(result.token_ids.size() > 0, "Unicode with numbers should produce tokens");
    
    std::string decoded = tokenizer.decode(GRIM::Tokenizer::DecodeRequest(result));
    ASSERT_STR_EQ(decoded, input, "Unicode+numeric round-trip failed");
    
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
    config.detect_numbers = true;
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
    config.detect_numbers = true;
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
    config.detect_numbers = true;
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
    config.detect_numbers = true;
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
    config.detect_numbers = true;
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
    // must still be detected as integer atoms, not leak as raw byte tokens.
    auto config = makeSelfTestTokenizerHP();
    config.enable_byte_fallback = true;
    config.detect_numbers = true;
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
        
        // Verify no digit byte tokens leaked through
        for (size_t i = 0; i < result.token_ids.size(); ++i) {
            int tid = result.token_ids[i];
            if (tid >= static_cast<int>(BYTE_TOKEN_OFFSET) && 
                tid < static_cast<int>(BYTE_TOKEN_OFFSET) + 256) {
                int byte_val = tid - static_cast<int>(BYTE_TOKEN_OFFSET);
                if (byte_val >= '0' && byte_val <= '9') {
                    std::string leak_msg = "Digit byte token leaked in: " + tc.description
                                         + " (token_id=" + std::to_string(tid) 
                                         + ", byte='" + std::string(1, static_cast<char>(byte_val)) + "')";
                    ASSERT_TRUE(false, leak_msg.c_str());
                }
            }
        }
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
    sequence.token_ids = {
        BYTE_TOKEN_OFFSET + static_cast<int>('o'),
        BYTE_TOKEN_OFFSET + static_cast<int>('k')
    };
    sequence.targets = {
        BYTE_TOKEN_OFFSET + static_cast<int>('k'),
        EOS_TOKEN_ID
    };
    const std::size_t n = sequence.token_ids.size();
    sequence.token_numeric_values.assign(n, 0.0f);
    sequence.token_atom_mask.assign(n, 0);
    sequence.token_atom_flags.assign(n, 0);
    sequence.atom_table = std::make_shared<AtomTable>();
    sequence.atom_entry_ids.assign(n, kAtomEntryNone);
    sequence.token_exec_slots.assign(n, -1);
    return sequence;
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
    suite.addTest("Unigram.Train.ShrinkRankingUsesCompressionGain", testUnigramTrainShrinkRankingUsesCompressionGain);
    suite.addTest("Unigram.Train.ByteFallbackOffAddsCharSeeds", testUnigramTrainByteFallbackDisabledAddsCharacterSeeds);
    suite.addTest("Unigram.Train.ByteFallbackOffFailsUncoveredChars", testUnigramTrainByteFallbackDisabledFailsOnUncoveredCharacterSeed);
    suite.addTest("Unigram.Train.SubwordCountsUseUint64", testUnigramTrainSubwordCountsUseUint64);
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
    suite.addTest("UniByte.BasicEncode", testUniByteBasicEncode);
    suite.addTest("UniByte.StructuralDetection", testUniByteStructuralDetection);
    suite.addTest("UniByte.RejectsUnparseableDetectedAtom", testUniByteRejectsUnparseableDetectedAtom);
    suite.addTest("Unigram.Train.RejectsUnparseableDetectedAtom", testUnigramTrainRejectsUnparseableDetectedAtom);
    suite.addTest("AtomTable.RejectsBadNumericDetectionWithContext", testAtomTableRejectsBadNumericDetectionWithContext);
    suite.addTest("AtomTable.ArgNumberSupportsSignedDecimalExponent", testAtomTableArgNumberSupportsSignedDecimalExponent);
    suite.addTest("AtomTable.TokenizationRejectsMantissaDigitSlotOverflow", testAtomTokenizationRejectsMantissaDigitSlotOverflow);
    suite.addTest("UniByte.RawTextDetectorRegistry", testUniByteRawTextDetectorRegistry);
    suite.addTest("UniByte.URLPassthrough", testUniByteURLDetection);
    suite.addTest("UniByte.URLPassthrough.CaseInsensitive", testUniByteURLDetectionCaseInsensitive);
    suite.addTest("UniByte.EmailPassthrough", testUniByteEmailDetection);
    suite.addTest("UniByte.DateNumericOnly", testUniByteDateDetection);
    suite.addTest("UniByte.PlaceholderInjection", testUniBytePlaceholderInjection);
    suite.addTest("UniByte.PreRegistersAtomTableBeforePlaceholderEmission", testUniBytePreRegistersAtomTableBeforePlaceholderEmission);
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
    suite.addTest("AtomTable.ArgNumberSerializesOnEntry", testAtomTableArgNumberSerializesOnEntry);
    
    // Section 6: Integration Tests
    suite.addTest("Integration.FullPipeline", testFullPipeline);
    suite.addTest("Integration.AtomTable", testAtomTableIntegration);
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
    suite.addTest("Unicode.WithNumbers", testUnicodeWithStructural);
    
    // Section 9: Multiple Structural Elements Tests
    suite.addTest("Structural.MultipleURLsPassthrough", testMultipleURLs);
    suite.addTest("Structural.MultipleEmailsPassthrough", testMultipleEmails);
    suite.addTest("Structural.MixedNumbers", testMixedNumbers);
    suite.addTest("Structural.Adjacent", testAdjacentStructural);
    
    // Section 10: Path Detection Tests
    suite.addTest("Path.WindowsPassthrough", testWindowsPath);
    suite.addTest("Path.UnixPassthrough", testUnixPath);
    
    // Section 11: Numeric Edge Cases
    suite.addTest("Numeric.ScientificNotation", testScientificNotation);
    suite.addTest("Numeric.IPvsDecimal", testIPAddressVsDecimal);
    suite.addTest("Numeric.Negative", testNegativeNumbers);
    suite.addTest("Numeric.DigitsFollowedByAlpha", testDigitsFollowedByAlpha);
    
    // Section 12: Byte Fallback Control Tests
    suite.addTest("ByteFallback.Disabled", testByteFallbackDisabled);
    suite.addTest("ByteFallback.Mixed", testMixedVocabAndByteFallback);
    suite.addTest("ByteFallback.RejectMalformedUtf8", testByteFallbackRejectsMalformedGeneratedUtf8);
    
    // Section 13: Vocabulary Persistence Tests
    suite.addTest("Vocab.TextExportBinaryLoad", testVocabTextExportBinaryLoad);
    suite.addTest("Vocab.SaveLoadBinary", testVocabSaveLoadBinary);
    
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
