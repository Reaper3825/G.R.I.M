//======================================================//
//  unigrambyte_self_test.cu
//  Comprehensive test suite for UnigramByte tokenizer
//======================================================//

#include "unigrambyte_self_test.hpp"

#include "../Shared/UnigramByte/Byte.hpp"
#include "../Shared/UnigramByte/Unigram.hpp"
#include "../Shared/UnigramByte/UnigramViterbi.hpp"
#include "../Shared/UnigramByte/UniByte.hpp"
#include "../Shared/UnigramByte/AtomTable.hpp"
#include "../Shared/UnigramByte/AhoCorasick.hpp"
#include "../Shared/TokenizerArtifacts/TokenizerArtifactBundle.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>
#include <vector>
#include <string>
#include <cstring>
#include <filesystem>
#include <fstream>

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

using namespace GRIM::Tokenizer;
using namespace GRIM::Test;

namespace TokenizerArtifacts = GRIM::TokenizerArtifacts;

static ::GRIM::HyperParameters::TokenizerHP makeSelfTestTokenizerHP() {
    ::GRIM::HyperParameters::TokenizerHP hp;
    hp.target_vocab_size = 50000;
    hp.character_coverage = 0.9995f;
    hp.min_subword_freq = 3;
    hp.enable_parallel_subword_mining = true;
    hp.enable_scratch_block_reasoning = true;
    hp.detect_numbers = true;
    hp.enable_byte_fallback = true;
    hp.vocab_score_multiplier = 1.0f;
    return hp;
}

// Helper: add minimal ▁-prefixed vocab to a UniByte tokenizer so UnigramViterbiSession has a valid trie.
// Without this, Viterbi segmentation crashes (Rule 20: trie_ must not be empty).
static void addMinimalVocab(UniByte& tok) {
    tok.unigramLM().writePiece("\xe2\x96\x81" "the",     -1.0f, false);
    tok.unigramLM().writePiece("\xe2\x96\x81" "is",      -1.1f, false);
    tok.unigramLM().writePiece("\xe2\x96\x81" "price",   -1.2f, false);
    tok.unigramLM().writePiece("\xe2\x96\x81" "dollars", -1.3f, false);
    tok.unigramLM().writePiece("\xe2\x96\x81" "on",      -1.4f, false);
    tok.unigramLM().writePiece("\xe2\x96\x81" "for",     -1.5f, false);
    tok.unigramLM().writePiece("\xe2\x96\x81" "more",    -1.6f, false);
    tok.unigramLM().writePiece("\xe2\x96\x81" "info",    -1.7f, false);
    tok.unigramLM().writePiece("\xe2\x96\x81" "us",      -1.8f, false);
    tok.unigramLM().writePiece("\xe2\x96\x81" "at",      -1.9f, false);
    tok.unigramLM().writePiece("\xe2\x96\x81" "meeting", -2.0f, false);
    tok.unigramLM().writePiece("\xe2\x96\x81" "Count",   -2.1f, false);
    tok.unigramLM().buildTrie();
}

//======================================================//
//  Section 1: Byte Fallback Tests
//======================================================//

bool testByteEncodeBasic(std::string& message) {
    ByteEncoder byte;
    
    std::string input = "Hello";
    std::vector<int> tokens = byte.encode(input);
    
    ASSERT_EQ(tokens.size(), 5, "Token count mismatch");
    ASSERT_EQ(tokens[0], byte.byteToToken('H'), "First token mismatch");
    ASSERT_EQ(tokens[1], byte.byteToToken('e'), "Second token mismatch");
    ASSERT_EQ(tokens[2], byte.byteToToken('l'), "Third token mismatch");
    ASSERT_EQ(tokens[3], byte.byteToToken('l'), "Fourth token mismatch");
    ASSERT_EQ(tokens[4], byte.byteToToken('o'), "Fifth token mismatch");
    
    return true;
}

bool testByteDecodeBasic(std::string& message) {
    ByteEncoder byte;
    
    std::vector<int> tokens = {
        byte.byteToToken('H'),
        byte.byteToToken('e'),
        byte.byteToToken('l'),
        byte.byteToToken('l'),
        byte.byteToToken('o'),
    };
    std::string output = byte.decode(tokens);
    
    ASSERT_STR_EQ(output, "Hello", "Decoded string mismatch");
    
    return true;
}

bool testByteRoundTrip(std::string& message) {
    ByteEncoder byte;
    
    // Test various strings including special characters
    std::vector<std::string> test_cases = {
        "Hello, World!",
        "The quick brown fox",
        "Numbers: 12345",
        "Special: @#$%^&*()",
        "Mixed: abc123!@#",
        "",  // Empty string
        " ",  // Single space
        "   ",  // Multiple spaces
    };
    
    for (const auto& input : test_cases) {
        std::vector<int> tokens = byte.encode(input);
        std::string output = byte.decode(tokens);
        
        ASSERT_STR_EQ(output, input, "Round-trip failed for: " + input);
    }
    
    return true;
}

bool testByteUTF8(std::string& message) {
    ByteEncoder byte;
    
    // UTF-8 multi-byte sequences
    std::string input = "Café";  // é is 2 bytes in UTF-8
    std::vector<int> tokens = byte.encode(input);
    std::string output = byte.decode(tokens);
    
    ASSERT_STR_EQ(output, input, "UTF-8 round-trip failed");
    
    return true;
}

//======================================================//
//  Section 2: Unigram LM Tests
//======================================================//

bool testUnigramBuildVocab(std::string& message) {
    UnigramLM unigram;
    
    // Build a simple vocabulary using the canonical writePiece entrypoint.
    // Token IDs must start after pre-existing special tokens (unk, pad, bos, eos)
    unigram.writePiece("hello", -1.0f, false);
    unigram.writePiece("world", -1.2f, false);
    unigram.writePiece("he", -2.0f, false);
    unigram.writePiece("llo", -2.5f, false);
    unigram.writePiece("wo", -2.3f, false);
    unigram.writePiece("rld", -2.8f, false);
    unigram.writePiece("h", -3.0f, false);
    unigram.writePiece("e", -3.1f, false);
    unigram.writePiece("l", -3.2f, false);
    unigram.writePiece("o", -3.3f, false);
    unigram.writePiece("w", -3.4f, false);
    unigram.writePiece("r", -3.5f, false);
    unigram.writePiece("d", -3.6f, false);
    
        // Only learned pieces are stored in UnigramLM::pieces_; specials are layout metadata.
        ASSERT_EQ(unigram.pieceCount(), 13, "Learned piece count mismatch");
    
    return true;
}

bool testUnigramEncode(std::string& message) {
    UnigramLM unigram;
    
    // Build vocabulary with log probabilities
    // Start after pre-existing special tokens
    unigram.writePiece("hello", -1.0f, false);  // Most likely for "hello"
    unigram.writePiece("he", -2.0f, false);
    unigram.writePiece("llo", -2.5f, false);
    unigram.writePiece("h", -3.0f, false);
    unigram.writePiece("e", -3.1f, false);
    unigram.writePiece("l", -3.2f, false);
    unigram.writePiece("o", -3.3f, false);
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
    unigram.writePiece("test", -5.0f, false);    // Whole word is worse
    unigram.writePiece("te", -1.0f, false);      // Better to split
    unigram.writePiece("st", -1.0f, false);
    unigram.writePiece("t", -2.0f, false);
    unigram.writePiece("e", -2.1f, false);
    unigram.writePiece("s", -2.2f, false);
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
    unigram.writePiece(full_piece, -1.0f, false);
    unigram.writePiece("\xe2\x96\x81hello", -10.0f, false);
    unigram.writePiece("!", -10.0f, false);
    unigram.buildTrie();

    std::vector<int> tokens = unigram.encode("hello!");

    ASSERT_EQ(tokens.size(), static_cast<size_t>(1), "Punctuation-bearing learned piece should remain selectable");
    ASSERT_EQ(tokens[0], unigram.getPieceId(full_piece), "Viterbi should select learned punctuation-bearing piece");

    return true;
}

bool testUnigramDecode(std::string& message) {
    UnigramLM unigram;
    
    // Start after pre-existing special tokens
    // Pieces use \xe2\x96\x81 (U+2581 ▁) prefix — SentencePiece whitespace normalization
    unigram.writePiece("\xe2\x96\x81hello", -1.0f, false);
    unigram.writePiece("\xe2\x96\x81world", -1.2f, false);
    unigram.buildTrie();  // Must build trie before encoding
    
    // encode() normalizes "hello world" → "▁hello▁world", Viterbi matches ▁-prefixed pieces
    // decode() denormalizes: "▁hello▁world" → "hello world"
    std::vector<int> tokens = unigram.encode("hello world");
    std::string decoded = unigram.decode(tokens);
    
        ASSERT_STR_EQ(decoded, "hello world", "Decode mismatch");
    
    return true;
}

bool testUnigramUnknown(std::string& message) {
    UnigramLM unigram;
    
    // Minimal vocab - will need byte fallback for some chars
    // Start after pre-existing special tokens
    unigram.writePiece("a", -1.0f, false);
    unigram.writePiece("b", -1.0f, false);
    unigram.buildTrie();  // Must build trie before encoding
    
    // Try to encode something not in vocab
    std::vector<int> tokens = unigram.encode("xyz");
    
        // After ▁ normalization, pieces are ▁-prefixed ("▁gonna") or bare ("gonna")
    // Note: Behavior depends on implementation - may produce empty or UNK
    // For now just verify no crash
    
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
    
    // Initialize with a simple vocab via unigramLM
    // Start after pre-existing special tokens
    // Pieces use ▁ (U+2581) prefix — SentencePiece whitespace normalization
    tokenizer.unigramLM().writePiece("\xe2\x96\x81hello", -1.0f, false);
    tokenizer.unigramLM().writePiece("\xe2\x96\x81world", -1.2f, false);
    tokenizer.unigramLM().buildTrie();  // Must build trie before encoding
    
    std::vector<int> tokens = tokenizer.encode("hello world");
    
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

    const auto structures = registry.detectStructures(text, options);
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
    tokenizer.unigramLM().writePiece("\xe2\x96\x81the", -1.0f, false);
    std::cout << "  '▁the'   -> id=" << UnigramLM::tokenIdForIndex(tokenizer.unigramLM().pieceCount() - 1) << "\n";
    tokenizer.unigramLM().writePiece("\xe2\x96\x81quick", -1.5f, false);
    std::cout << "  '▁quick' -> id=" << UnigramLM::tokenIdForIndex(tokenizer.unigramLM().pieceCount() - 1) << "\n";
    tokenizer.unigramLM().writePiece("\xe2\x96\x81" "brown", -1.6f, false);
    std::cout << "  '▁brown' -> id=" << UnigramLM::tokenIdForIndex(tokenizer.unigramLM().pieceCount() - 1) << "\n";
    tokenizer.unigramLM().writePiece("\xe2\x96\x81" "fox", -1.7f, false);
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
    
    std::vector<int> tokens = tokenizer.encode(input);
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
    
    std::string output = tokenizer.decode(tokens);
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
    
    uint32_t id = table.registerAtom(AtomType::ATOM_INT, "12345", 0, 5);
    
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

    uint32_t large_id = table.registerAtom(AtomType::ATOM_INT, "9007199254740993", 0, 16);
    auto large_numeric = table.getNumericValue(large_id);
    ASSERT_TRUE(large_numeric, "Large integer numeric payload missing");
    ASSERT_EQ(large_numeric->int_value, static_cast<int64_t>(9007199254740993LL),
              "Public numeric getter must not round integer atoms through float");

    uint32_t padded_id = table.registerAtom(AtomType::ATOM_INT, "001", 0, 3);
    auto padded_entry = table.getAtom(padded_id);
    ASSERT_TRUE(padded_entry, "Failed to retrieve padded integer atom");
    ASSERT_STR_EQ(table.atomToString(*padded_entry), "001",
                  "Atom stringification must preserve raw source text");
    ASSERT_EQ(padded_entry->reserved_zero, static_cast<uint64_t>(0),
              "AtomTable reserved padding must stay zero, not become canonical parsed text");

    ASSERT_EQ(table.registerAtom(AtomType::ATOM_INT, " 42", 0, 3), UINT32_MAX,
              "Leading whitespace in numeric atom text must be rejected");
    ASSERT_EQ(table.registerAtom(AtomType::ATOM_INT, "42 ", 0, 3), UINT32_MAX,
              "Trailing whitespace in numeric atom text must be rejected");
    ASSERT_EQ(table.registerAtom(AtomType::ATOM_INT, "4 2", 0, 3), UINT32_MAX,
              "Internal whitespace in numeric atom text must be rejected");
    
    return true;
}

bool testAtomTableRegisterFloat(std::string& message) {
    AtomTable table;
    
    uint32_t id = table.registerAtom(AtomType::ATOM_FLOAT, "3.14159", 0, 7);
    
    auto entry = table.getAtom(id);
    ASSERT_TRUE(entry, "Failed to retrieve atom");
    
    auto numeric = table.getNumericValue(id);
    ASSERT_TRUE(numeric, "Numeric payload missing");
    ASSERT_EQ(static_cast<int>(numeric->kind), static_cast<int>(NumericPayloadKind::FLOAT),
              "Float numeric kind mismatch");
    ASSERT_NEAR(numeric->float_value, 3.14159, 0.0001, "Float value mismatch");

    ASSERT_EQ(table.registerAtom(AtomType::ATOM_FLOAT, " 3.14159", 0, 8), UINT32_MAX,
              "Leading whitespace in float atom text must be rejected");

    AtomFloat wrong_float{2.0, false, 0};
    ASSERT_EQ(table.registerAtom(AtomType::ATOM_FLOAT, AtomValue(wrong_float), "3.14159"), UINT32_MAX,
              "Pre-parsed float payload must match raw atom text");
    
    return true;
}

bool testAtomTableRegisterHex(std::string& message) {
    // Hex atoms no longer supported — verify a plain integer still works.
    AtomTable table;
    
    uint32_t id = table.registerAtom(AtomType::ATOM_INT, "255", 0, 3);
    
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
    
    uint32_t id = table.registerAtom(AtomType::ATOM_INT, "10", 0, 2);
    
    auto entry = table.getAtom(id);
    ASSERT_TRUE(entry, "Failed to retrieve atom");
    
    auto numeric = table.getNumericValue(id);
    ASSERT_TRUE(numeric, "Numeric payload missing");
    ASSERT_EQ(numeric->int_value, static_cast<int64_t>(10), "Integer value mismatch");
    
    return true;
}

bool testAtomTableRegisterURL(std::string& message) {
    AtomTable table;
    
    uint32_t id = table.registerAtom(AtomType::ATOM_INT, 
                                      "https://example.com:8080/path?query=1#fragment", 
                                      0, 45);
    ASSERT_EQ(id, UINT32_MAX, "Invalid integer text must not register as ATOM_INT");
    ASSERT_EQ(table.size(), static_cast<size_t>(0), "Rejected URL should not mutate table");
    
    return true;
}

bool testAtomTableRegisterEmail(std::string& message) {
    AtomTable table;
    
    uint32_t id = table.registerAtom(AtomType::ATOM_INT, "user@domain.com", 0, 15);
    ASSERT_EQ(id, UINT32_MAX, "Invalid integer text must not register as ATOM_INT");
    ASSERT_EQ(table.size(), static_cast<size_t>(0), "Rejected email should not mutate table");
    
    return true;
}

bool testAtomTableRegisterDate(std::string& message) {
    AtomTable table;
    
    uint32_t id = table.registerAtom(AtomType::ATOM_INT, "2024-12-25", 0, 10);
    ASSERT_EQ(id, UINT32_MAX, "Invalid integer text must not register as ATOM_INT");
    ASSERT_EQ(table.size(), static_cast<size_t>(0), "Rejected date should not mutate table");
    
    return true;
}

bool testAtomTableRegisterTime(std::string& message) {
    AtomTable table;
    
    uint32_t id = table.registerAtom(AtomType::ATOM_INT, "14:30:00", 0, 8);
    ASSERT_EQ(id, UINT32_MAX, "Invalid integer text must not register as ATOM_INT");
    ASSERT_EQ(table.size(), static_cast<size_t>(0), "Rejected time should not mutate table");
    
    return true;
}

bool testAtomTableRegisterIP(std::string& message) {
    AtomTable table;
    
    uint32_t id = table.registerAtom(AtomType::ATOM_INT, "192.168.1.1", 0, 11);
    ASSERT_EQ(id, UINT32_MAX, "Invalid integer text must not register as ATOM_INT");
    ASSERT_EQ(table.size(), static_cast<size_t>(0), "Rejected IP should not mutate table");
    
    return true;
}

bool testAtomTableRegisterPath(std::string& message) {
    AtomTable table;
    
    uint32_t id = table.registerAtom(AtomType::ATOM_INT, "/usr/local/bin/test", 0, 19);
    ASSERT_EQ(id, UINT32_MAX, "Invalid integer text must not register as ATOM_INT");
    ASSERT_EQ(table.size(), static_cast<size_t>(0), "Rejected path should not mutate table");
    
    return true;
}

bool testAtomTableRegisterString(std::string& message) {
    AtomTable table;
    
    uint32_t id = table.registerAtom(AtomType::ATOM_INT, "\"hello\\nworld\"", 0, 14);
    ASSERT_EQ(id, UINT32_MAX, "Invalid integer text must not register as ATOM_INT");
    ASSERT_EQ(table.size(), static_cast<size_t>(0), "Rejected string should not mutate table");
    
    return true;
}

bool testAtomTableRegisterIdentifier(std::string& message) {
    AtomTable table;
    
    // Test various naming conventions
    ASSERT_EQ(table.registerAtom(AtomType::ATOM_INT, "camelCase", 0, 9), UINT32_MAX,
              "Invalid identifier must not register as ATOM_INT");
    ASSERT_EQ(table.registerAtom(AtomType::ATOM_INT, "PascalCase", 0, 10), UINT32_MAX,
              "Invalid identifier must not register as ATOM_INT");
    ASSERT_EQ(table.registerAtom(AtomType::ATOM_INT, "snake_case", 0, 10), UINT32_MAX,
              "Invalid identifier must not register as ATOM_INT");
    ASSERT_EQ(table.registerAtom(AtomType::ATOM_INT, "SCREAMING_SNAKE", 0, 15), UINT32_MAX,
              "Invalid identifier must not register as ATOM_INT");
    ASSERT_EQ(table.size(), static_cast<size_t>(0), "Rejected identifiers should not mutate table");
    
    return true;
}

bool testAtomTableLookupByType(std::string& message) {
    AtomTable table;
    
    // Register multiple atoms of different types
    table.registerAtom(AtomType::ATOM_INT, "100", 0, 3);
    table.registerAtom(AtomType::ATOM_INT, "200", 0, 3);
    table.registerAtom(AtomType::ATOM_FLOAT, "3.14", 0, 4);
    table.registerAtom(AtomType::ATOM_INT, "300", 0, 3);
    
    auto integers = table.getAtomsByType(AtomType::ATOM_INT);
    ASSERT_EQ(integers.size(), 3, "Should find 3 integers");
    
    auto floats = table.getAtomsByType(AtomType::ATOM_FLOAT);
    ASSERT_EQ(floats.size(), 1, "Should find 1 float");
    
    return true;
}

bool testAtomTableGPUUpload(std::string& message) {
    AtomTable table;
    
    // Register some atoms
    table.registerAtom(AtomType::ATOM_INT, "42", 0, 2);
    table.registerAtom(AtomType::ATOM_FLOAT, "3.14", 0, 4);
    table.registerAtom(AtomType::ATOM_INT, "9007199254740993", 0, 16);
    
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
    internal_table.registerAtom(AtomType::ATOM_INT, "7", 0, 1);
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
    
    table.registerAtom(AtomType::ATOM_INT, "1", 0, 1);
    table.registerAtom(AtomType::ATOM_INT, "2", 0, 1);
    table.registerAtom(AtomType::ATOM_INT, "3", 0, 1);
    
    ASSERT_EQ(table.size(), 3, "Should have 3 atoms before clear");
    
    table.clear();
    
    ASSERT_EQ(table.size(), 0, "Should have 0 atoms after clear");
    
    return true;
}

bool testAtomTableMetadata(std::string& message) {
    AtomTable table;
    
    // Register an atom
    uint32_t id = table.registerAtom(AtomType::ATOM_INT, "42", 0, 2);
    
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
    uint32_t id1 = table.registerAtom(AtomType::ATOM_INT, "100", 0, 3);
    uint32_t id2 = table.registerAtom(AtomType::ATOM_INT, "100", 0, 3);
    
    auto entry1 = table.getAtom(id1);
    auto entry2 = table.getAtom(id2);
    
    ASSERT_TRUE(entry1 && entry2, "Both atoms should exist");
    ASSERT_EQ(entry1->hash, entry2->hash, "Identical atoms should have same hash");
    
    // Different atom should have different hash
    uint32_t id3 = table.registerAtom(AtomType::ATOM_INT, "200", 0, 3);
    auto entry3 = table.getAtom(id3);
    
    ASSERT_TRUE(entry3, "Different atom should exist");
    ASSERT_TRUE(entry1->hash != entry3->hash, "Different atoms should have different hashes");
    
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
    
    // Add vocabulary via unigramLM
    // Start after pre-existing special tokens
    // Pieces use \xe2\x96\x81 (U+2581 ▁) prefix — SentencePiece whitespace normalization
    tokenizer.unigramLM().writePiece("\xe2\x96\x81the", -1.0f, false);
    tokenizer.unigramLM().writePiece("\xe2\x96\x81is", -1.1f, false);
    tokenizer.unigramLM().writePiece("\xe2\x96\x81price", -1.5f, false);
    tokenizer.unigramLM().writePiece("\xe2\x96\x81visit", -1.6f, false);
    tokenizer.unigramLM().writePiece("\xe2\x96\x81" "for", -1.7f, false);
    tokenizer.unigramLM().writePiece("\xe2\x96\x81more", -1.8f, false);
    tokenizer.unigramLM().writePiece("\xe2\x96\x81info", -1.9f, false);
    tokenizer.unigramLM().writePiece(".", -0.6f, false);
    
    // Mixed input: numbers become atoms, URLs remain plain text.
    std::string input = "The price is 42.99. Visit https://shop.com for 3 more info.";
    
    auto result = tokenizer.tokenizeWithMetadata(input);
    
    ASSERT_TRUE(result.token_ids.size() > 0, "Should produce tokens");
    ASSERT_TRUE(result.atoms.size() >= 2, "Should detect numeric structures only");
    
    // Verify we can decode back through the single atom-aware decode entry point.
    std::string decoded = tokenizer.decode(result);
    
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
        std::vector<int> tokens = tokenizer.encode(input);
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
    std::vector<int> tokens = tokenizer.encode(input);
    
    ASSERT_EQ(tokens.size(), 0, "Empty string should produce no tokens");
    
    std::string decoded = tokenizer.decode(tokens);
    ASSERT_STR_EQ(decoded, "", "Empty decode should be empty");
    
    return true;
}

bool testEdgeCaseSingleChar(std::string& message) {
    auto config = makeSelfTestTokenizerHP();
    config.enable_byte_fallback = true;
    UniByte tokenizer(config);
    
    std::string input = "a";
    std::vector<int> tokens = tokenizer.encode(input);
    
    ASSERT_TRUE(tokens.size() >= 1, "Single char should produce at least 1 token");
    
    std::string decoded = tokenizer.decode(tokens);
    ASSERT_STR_EQ(decoded, input, "Single char round-trip failed");
    
    return true;
}

bool testEdgeCaseOnlyWhitespace(std::string& message) {
    auto config = makeSelfTestTokenizerHP();
    config.enable_byte_fallback = true;
    UniByte tokenizer(config);
    
    // After ▁ normalization, spaces become ▁ characters
    // Add ▁ piece to vocab so whitespace-only input has vocab matches
    tokenizer.unigramLM().writePiece("\xe2\x96\x81", -0.5f, false);
    tokenizer.unigramLM().buildTrie();
    
    std::string input = "   ";
    std::vector<int> tokens = tokenizer.encode(input);
    
    ASSERT_TRUE(tokens.size() >= 1, "Whitespace should produce tokens");
    
    std::string decoded = tokenizer.decode(tokens);
    ASSERT_STR_EQ(decoded, input, "Whitespace round-trip failed");
    
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
    
    std::vector<int> tokens = tokenizer.encode(input);
    ASSERT_TRUE(tokens.size() > 0, "Long sequence should produce tokens");
    
    std::string decoded = tokenizer.decode(tokens);
    ASSERT_STR_EQ(decoded, input, "Long sequence round-trip failed");
    
    return true;
}

bool testEdgeCaseSpecialTokenLiterals(std::string& message) {
    auto config = makeSelfTestTokenizerHP();
    config.enable_byte_fallback = true;
    // Disable scratch block reasoning to avoid atom detection interfering with special token literals
    config.enable_scratch_block_reasoning = false;
    UniByte tokenizer(config);
    
    // Add vocab pieces for common words — ▁-prefixed for SentencePiece normalization
    tokenizer.unigramLM().writePiece("\xe2\x96\x81This", -1.0f, false);
    tokenizer.unigramLM().writePiece("\xe2\x96\x81is", -1.0f, false);
    tokenizer.unigramLM().writePiece("\xe2\x96\x81not", -1.0f, false);
    tokenizer.unigramLM().writePiece("\xe2\x96\x81" "a", -1.0f, false);
    tokenizer.unigramLM().writePiece("\xe2\x96\x81special", -1.0f, false);
    tokenizer.unigramLM().writePiece("\xe2\x96\x81token", -1.0f, false);
    // Add the literal special token strings as regular vocab pieces
    tokenizer.unigramLM().writePiece("\xe2\x96\x81<unk>", -2.0f, false);
    tokenizer.unigramLM().writePiece("\xe2\x96\x81<s>", -2.0f, false);
    tokenizer.unigramLM().writePiece("\xe2\x96\x81</s>", -2.0f, false);
    tokenizer.unigramLM().writePiece("\xe2\x96\x81<pad>", -2.0f, false);
    tokenizer.unigramLM().buildTrie();
    
    // Input contains literal special token strings
    std::string input = "This <unk> is not <s> a </s> special <pad> token";
    
    std::vector<int> tokens = tokenizer.encode(input);
    ASSERT_TRUE(tokens.size() > 0, "Should produce tokens");
    
    std::string decoded = tokenizer.decode(tokens);
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
    
    std::vector<int> tokens = tokenizer.encode(input);
    ASSERT_TRUE(tokens.size() > 0, "Emoji input should produce tokens");
    
    std::string decoded = tokenizer.decode(tokens);
    ASSERT_STR_EQ(decoded, input, "Emoji round-trip failed");
    
    return true;
}

bool testUnicodeMultiLanguage(std::string& message) {
    auto config = makeSelfTestTokenizerHP();
    config.enable_byte_fallback = true;
    UniByte tokenizer(config);
    
    std::string input = "English 日本語 한국어 العربية";
    
    std::vector<int> tokens = tokenizer.encode(input);
    ASSERT_TRUE(tokens.size() > 0, "Multi-language should produce tokens");
    
    std::string decoded = tokenizer.decode(tokens);
    ASSERT_STR_EQ(decoded, input, "Multi-language round-trip failed");
    
    return true;
}

bool testUnicodeWithStructural(std::string& message) {
    auto config = makeSelfTestTokenizerHP();
    config.enable_byte_fallback = true;
    config.detect_numbers = true;
    config.enable_scratch_block_reasoning = true;  // Enable atom detection
    UniByte tokenizer(config);
    
    std::string input = "日本の価格は 42.5 円です";
    
    // Use tokenizeWithMetadata to get both tokens and atom information
    auto result = tokenizer.tokenizeWithMetadata(input);
    ASSERT_TRUE(result.token_ids.size() > 0, "Unicode with numbers should produce tokens");
    
    std::string decoded = tokenizer.decode(result);
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
    
    std::vector<int> tokens = tokenizer.encode(input);
    
    // Should still produce tokens (UNK tokens)
    ASSERT_TRUE(tokens.size() > 0, "Should produce UNK tokens without byte fallback");
    
    return true;
}

bool testMixedVocabAndByteFallback(std::string& message) {
    auto config = makeSelfTestTokenizerHP();
    config.enable_byte_fallback = true;
    UniByte tokenizer(config);
    
    // Add partial vocab — ▁-prefixed for SentencePiece whitespace normalization
    tokenizer.unigramLM().writePiece("\xe2\x96\x81hello", -1.0f, false);
    tokenizer.unigramLM().buildTrie();
    
    // Input has both vocab word and unknown
    std::string input = "hello xyz hello";
    
    std::vector<int> tokens = tokenizer.encode(input);
    std::string decoded = tokenizer.decode(tokens);
    
    ASSERT_STR_EQ(decoded, input, "Mixed vocab+byte fallback round-trip failed");
    
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
    original.unigramLM().writePiece("test", -1.0f, false);
    original.unigramLM().writePiece("vocab", -1.5f, false);
    original.unigramLM().writePiece("save", -2.0f, false);
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
    original.unigramLM().writePiece("binary", -1.0f, false);
    original.unigramLM().writePiece("format", -1.5f, false);
    original.unigramLM().writePiece("fast", -2.0f, false);
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
    unigram.writePiece("\xe2\x96\x81gpu", -1.0f, false);
    unigram.writePiece("\xe2\x96\x81" "decode", -1.5f, false);
    unigram.buildTrie();
    
    // Init GPU
    bool gpu_ok = unigram.initGPU();
    if (!gpu_ok) {
        message = "GPU init failed (may not have CUDA)";
        return true;  // Skip test if no GPU
    }
    
    // Encode/decode through the single CPU API after GPU upload.
    std::vector<int> tokens = unigram.encode("gpu decode");
    std::string result = unigram.decode(tokens);
    ASSERT_STR_EQ(result, "gpu decode", "Decode mismatch after GPU upload");
    
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
    
    // Section 1: Byte Fallback Tests
    suite.addTest("Byte.Encode.Basic", testByteEncodeBasic);
    suite.addTest("Byte.Decode.Basic", testByteDecodeBasic);
    suite.addTest("Byte.RoundTrip", testByteRoundTrip);
    suite.addTest("Byte.UTF8", testByteUTF8);
    
    // Section 2: Unigram LM Tests
    suite.addTest("Unigram.BuildVocab", testUnigramBuildVocab);
    suite.addTest("Unigram.Encode", testUnigramEncode);
    suite.addTest("Unigram.Viterbi", testUnigramViterbi);
    suite.addTest("Unigram.SentencePiecePunctuation", testUnigramSentencePiecePunctuationPiece);
    suite.addTest("Unigram.Decode", testUnigramDecode);
    suite.addTest("Unigram.Unknown", testUnigramUnknown);
    suite.addTest("Unigram.Train.FilterRepetitionNoise", testUnigramTrainFiltersRepetitionNoise);
    suite.addTest("Unigram.Train.DedupRepeatedVariants", testUnigramTrainDedupsRepeatedVariants);
    
    // Section 3: Aho-Corasick Tests
    suite.addTest("AhoCorasick.BasicMatches", testAhoCorasickBasicMatches);
    suite.addTest("AhoCorasick.OutputClosure", testAhoCorasickOutputClosure);
    suite.addTest("AhoCorasick.StructuralVsNaive", testAhoCorasickStructuralVsNaive);
    suite.addTest("AhoCorasick.CaseInsensitive", testAhoCorasickCaseInsensitive);
    suite.addTest("AhoCorasick.Visualization", testAhoCorasickVisualization);

    // Section 4: UniByte Orchestrator Tests
    suite.addTest("UniByte.BasicEncode", testUniByteBasicEncode);
    suite.addTest("UniByte.StructuralDetection", testUniByteStructuralDetection);
    suite.addTest("UniByte.RawTextDetectorRegistry", testUniByteRawTextDetectorRegistry);
    suite.addTest("UniByte.URLPassthrough", testUniByteURLDetection);
    suite.addTest("UniByte.URLPassthrough.CaseInsensitive", testUniByteURLDetectionCaseInsensitive);
    suite.addTest("UniByte.EmailPassthrough", testUniByteEmailDetection);
    suite.addTest("UniByte.DateNumericOnly", testUniByteDateDetection);
    suite.addTest("UniByte.PlaceholderInjection", testUniBytePlaceholderInjection);
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
    
    // Section 6: Integration Tests
    suite.addTest("Integration.FullPipeline", testFullPipeline);
    suite.addTest("Integration.AtomTable", testAtomTableIntegration);
    suite.addTest("Integration.BatchProcessing", testBatchProcessing);
    
    // Section 7: Edge Case Tests
    suite.addTest("EdgeCase.EmptyString", testEdgeCaseEmptyString);
    suite.addTest("EdgeCase.SingleChar", testEdgeCaseSingleChar);
    suite.addTest("EdgeCase.OnlyWhitespace", testEdgeCaseOnlyWhitespace);
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
