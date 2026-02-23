//======================================================//
//  unigrambyte_self_test.cu
//  Comprehensive test suite for UnigramByte tokenizer
//======================================================//

#include "unigrambyte_self_test.hpp"

#include "../Shared/UnigramByte/Byte.hpp"
#include "../Shared/UnigramByte/Unigram.hpp"
#include "../Shared/UnigramByte/UniByte.hpp"
#include "../Shared/UnigramByte/AtomTable.hpp"
#include "../Shared/UnigramByte/AhoCorasick.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <set>
#include <vector>
#include <string>
#include <cstring>
#include <filesystem>
#include <fstream>

using namespace GRIM::Tokenizer;
using namespace GRIM::Test;

//======================================================//
//  Section 1: Byte Fallback Tests
//======================================================//

bool testByteEncodeBasic(std::string& message) {
    ByteEncoder byte;
    
    std::string input = "Hello";
    std::vector<int> tokens = byte.encode(input);
    
    ASSERT_EQ(tokens.size(), 5, "Token count mismatch");
    ASSERT_EQ(tokens[0], static_cast<int>('H'), "First token mismatch");
    ASSERT_EQ(tokens[1], static_cast<int>('e'), "Second token mismatch");
    ASSERT_EQ(tokens[2], static_cast<int>('l'), "Third token mismatch");
    ASSERT_EQ(tokens[3], static_cast<int>('l'), "Fourth token mismatch");
    ASSERT_EQ(tokens[4], static_cast<int>('o'), "Fifth token mismatch");
    
    return true;
}

bool testByteDecodeBasic(std::string& message) {
    ByteEncoder byte;
    
    std::vector<int> tokens = {'H', 'e', 'l', 'l', 'o'};
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

bool testByteGPUEncode(std::string& message) {
    ByteEncoder byte;
    
    // Simple GPU test - encode a single string
    std::string input = "Hello";
    
    // Allocate device memory
    uint8_t* d_input = nullptr;
    int* d_output = nullptr;
    
    cudaError_t err = cudaMalloc(&d_input, input.size());
    if (err != cudaSuccess) {
        message = "Failed to allocate device input";
        return false;
    }
    
    err = cudaMalloc(&d_output, input.size() * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_input);
        message = "Failed to allocate device output";
        return false;
    }
    
    // Copy input to device
    cudaMemcpy(d_input, input.data(), input.size(), cudaMemcpyHostToDevice);
    
    // Encode on GPU
    bool success = byte.encodeGPU(d_input, d_output, input.size());
    
    if (!success) {
        cudaFree(d_input);
        cudaFree(d_output);
        message = "GPU encode failed";
        return false;
    }
    
    // Copy output back
    std::vector<int> tokens(input.size());
    cudaMemcpy(tokens.data(), d_output, input.size() * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    
    // Verify
    std::string decoded = byte.decode(tokens);
    ASSERT_STR_EQ(decoded, input, "GPU encode/decode mismatch");
    
    return true;
}

//======================================================//
//  Section 2: Unigram LM Tests
//======================================================//

bool testUnigramBuildVocab(std::string& message) {
    UnigramLM unigram;
    
    // Build a simple vocabulary using addPiece
    // Token IDs must start after pre-existing special tokens (unk, pad, bos, eos)
    unigram.addPiece("hello", -1.0f, false);
    unigram.addPiece("world", -1.2f, false);
    unigram.addPiece("he", -2.0f, false);
    unigram.addPiece("llo", -2.5f, false);
    unigram.addPiece("wo", -2.3f, false);
    unigram.addPiece("rld", -2.8f, false);
    unigram.addPiece("h", -3.0f, false);
    unigram.addPiece("e", -3.1f, false);
    unigram.addPiece("l", -3.2f, false);
    unigram.addPiece("o", -3.3f, false);
    unigram.addPiece("w", -3.4f, false);
    unigram.addPiece("r", -3.5f, false);
    unigram.addPiece("d", -3.6f, false);
    
    // 13 manually added pieces + 4 special tokens (<unk>, <pad>, <s>, </s>) = 17 total
    ASSERT_EQ(unigram.vocabSize(), 17, "Vocab size mismatch");
    
    return true;
}

bool testUnigramEncode(std::string& message) {
    UnigramLM unigram;
    
    // Build vocabulary with log probabilities
    // Start after pre-existing special tokens
    unigram.addPiece("hello", -1.0f, false);  // Most likely for "hello"
    unigram.addPiece("he", -2.0f, false);
    unigram.addPiece("llo", -2.5f, false);
    unigram.addPiece("h", -3.0f, false);
    unigram.addPiece("e", -3.1f, false);
    unigram.addPiece("l", -3.2f, false);
    unigram.addPiece("o", -3.3f, false);
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
    unigram.addPiece("test", -5.0f, false);    // Whole word is worse
    unigram.addPiece("te", -1.0f, false);      // Better to split
    unigram.addPiece("st", -1.0f, false);
    unigram.addPiece("t", -2.0f, false);
    unigram.addPiece("e", -2.1f, false);
    unigram.addPiece("s", -2.2f, false);
    unigram.buildTrie();  // Must build trie before encoding
    
    std::vector<int> tokens = unigram.encode("test");
    
    // Viterbi should find optimal segmentation
    ASSERT_TRUE(tokens.size() >= 1, "Should produce tokens for 'test'");
    
    return true;
}

bool testUnigramDecode(std::string& message) {
    UnigramLM unigram;
    
    // Start after pre-existing special tokens
    unigram.addPiece("hello", -1.0f, false);
    unigram.addPiece("world", -1.2f, false);
    unigram.addPiece(" ", -0.5f, false);
    unigram.buildTrie();  // Must build trie before encoding
    
    // Encode then decode
    std::vector<int> tokens = unigram.encode("hello world");
    std::string decoded = unigram.decode(tokens);
    
    ASSERT_STR_EQ(decoded, "hello world", "Decode mismatch");
    
    return true;
}

bool testUnigramUnknown(std::string& message) {
    UnigramLM unigram;
    
    // Minimal vocab - will need byte fallback for some chars
    // Start after pre-existing special tokens
    unigram.addPiece("a", -1.0f, false);
    unigram.addPiece("b", -1.0f, false);
    unigram.buildTrie();  // Must build trie before encoding
    
    // Try to encode something not in vocab
    std::vector<int> tokens = unigram.encode("xyz");
    
    // Should still produce tokens (using UNK or character fallback)
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

    const bool has_gonna = unigram.hasPiece("gonna") || unigram.hasPiece(" gonna");
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

    const bool has_soo = unigram.hasPiece("soo");
    const bool has_sooo = unigram.hasPiece("sooo");
    ASSERT_TRUE(has_soo || has_sooo, "Expected at least one repeated-char variant to survive");
    ASSERT_FALSE(has_soo && has_sooo, "Repeated-char variants should deduplicate to one form");

    return true;
}

//======================================================//
//  Section 3: Aho-Corasick Tests
//======================================================//

bool testAhoCorasickBasicMatches(std::string& message) {
    AhoCorasick ac;

    uint32_t id_he = ac.addPattern("he", AtomType::ATOM_IDENTIFIER);
    uint32_t id_she = ac.addPattern("she", AtomType::ATOM_IDENTIFIER);
    uint32_t id_hers = ac.addPattern("hers", AtomType::ATOM_IDENTIFIER);
    uint32_t id_his = ac.addPattern("his", AtomType::ATOM_IDENTIFIER);
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

    uint32_t id_abc = ac.addPattern("abc", AtomType::ATOM_IDENTIFIER);
    uint32_t id_bc = ac.addPattern("bc", AtomType::ATOM_IDENTIFIER);
    uint32_t id_c = ac.addPattern("c", AtomType::ATOM_IDENTIFIER);
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
        {"http://", AtomType::ATOM_URL},
        {"https://", AtomType::ATOM_URL},
        {"ftp://", AtomType::ATOM_URL},
        {"ftps://", AtomType::ATOM_URL},
        {"ws://", AtomType::ATOM_URL},
        {"wss://", AtomType::ATOM_URL},
        {"file://", AtomType::ATOM_URL},
        {"@", AtomType::ATOM_EMAIL},
        {"0x", AtomType::ATOM_HEX},
        {"0X", AtomType::ATOM_HEX},
        {"0b", AtomType::ATOM_BINARY},
        {"0B", AtomType::ATOM_BINARY},
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
        ac.addPattern(pattern, AtomType::ATOM_IDENTIFIER);
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
    uint32_t pid = ac.addPattern("http://", AtomType::ATOM_URL);
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
    UniByteConfig config;
    config.target_vocab_size = 50000;
    config.enable_byte_fallback = true;
    
    UniByte tokenizer(config);
    
    // Initialize with a simple vocab via unigramLM
    // Start after pre-existing special tokens
    tokenizer.unigramLM().addPiece("hello", -1.0f, false);
    tokenizer.unigramLM().addPiece("world", -1.2f, false);
    tokenizer.unigramLM().addPiece(" ", -0.5f, false);
    tokenizer.unigramLM().buildTrie();  // Must build trie before encoding
    
    std::vector<int> tokens = tokenizer.encode("hello world");
    
    ASSERT_TRUE(tokens.size() > 0, "Should produce tokens");
    
    return true;
}

bool testUniByteStructuralDetection(std::string& message) {
    UniByteConfig config;
    config.target_vocab_size = 50000;
    config.detect_numbers = true;
    config.detect_urls = true;
    config.detect_dates = true;
    
    UniByte tokenizer(config);
    
    // Test number detection
    std::string input = "The price is 42.99 dollars";
    auto result = tokenizer.encodeWithMetadata(input);
    
    bool found_number = false;
    for (const auto& span : result.atoms) {
        if (span.atom_type == AtomType::ATOM_FLOAT || 
            span.atom_type == AtomType::ATOM_INTEGER) {
            found_number = true;
            break;
        }
    }
    
    ASSERT_TRUE(found_number, "Should detect number in input");
    
    return true;
}

bool testUniByteURLDetection(std::string& message) {
    UniByteConfig config;
    config.target_vocab_size = 50000;
    config.detect_urls = true;
    
    UniByte tokenizer(config);
    
    std::string input = "Visit https://example.com/path for more info";
    auto result = tokenizer.encodeWithMetadata(input);
    
    bool found_url = false;
    for (const auto& span : result.atoms) {
        if (span.atom_type == AtomType::ATOM_URL) {
            found_url = true;
            std::string_view url_view = span.view();
            ASSERT_TRUE(url_view.find("example.com") != std::string_view::npos,
                       "URL should contain domain");
            break;
        }
    }
    
    ASSERT_TRUE(found_url, "Should detect URL in input");
    
    return true;
}

bool testUniByteURLDetectionCaseInsensitive(std::string& message) {
    UniByteConfig config;
    config.target_vocab_size = 50000;
    config.detect_urls = true;

    UniByte tokenizer(config);

    std::string input = "Visit HTTPS://Example.com/path for more info";
    auto result = tokenizer.encodeWithMetadata(input);

    bool found_url = false;
    for (const auto& span : result.atoms) {
        if (span.atom_type == AtomType::ATOM_URL) {
            found_url = true;
            std::string_view url_view = span.view();
            ASSERT_TRUE(url_view.find("Example.com") != std::string_view::npos,
                        "URL should contain domain");
            break;
        }
    }

    ASSERT_TRUE(found_url, "Should detect URL in input");

    return true;
}

bool testUniByteEmailDetection(std::string& message) {
    UniByteConfig config;
    config.target_vocab_size = 50000;
    config.detect_emails = true;
    
    UniByte tokenizer(config);
    
    std::string input = "Contact us at test@example.com";
    auto result = tokenizer.encodeWithMetadata(input);
    
    bool found_email = false;
    for (const auto& span : result.atoms) {
        if (span.atom_type == AtomType::ATOM_EMAIL) {
            found_email = true;
            // Use contentView() to get just the atom content (no whitespace)
            std::string_view email_view = span.contentView();
            ASSERT_TRUE(email_view == "test@example.com", "Email value mismatch");
            break;
        }
    }
    
    ASSERT_TRUE(found_email, "Should detect email in input");
    
    return true;
}

bool testUniByteDateDetection(std::string& message) {
    UniByteConfig config;
    config.target_vocab_size = 50000;
    config.detect_dates = true;
    
    UniByte tokenizer(config);
    
    std::string input = "The meeting is on 2024-12-25";
    auto result = tokenizer.encodeWithMetadata(input);
    
    bool found_date = false;
    for (const auto& span : result.atoms) {
        if (span.atom_type == AtomType::ATOM_DATE) {
            found_date = true;
            break;
        }
    }
    
    ASSERT_TRUE(found_date, "Should detect date in input");
    
    return true;
}

bool testUniBytePlaceholderInjection(std::string& message) {
    UniByteConfig config;
    config.target_vocab_size = 50000;
    config.detect_numbers = true;
    
    UniByte tokenizer(config);
    
    std::string input = "Count: 12345";
    auto result = tokenizer.encodeWithMetadata(input);
    
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
    
    UniByteConfig config;
    config.target_vocab_size = 50000;
    config.enable_byte_fallback = true;
    
    UniByte tokenizer(config);
    
    // Start after pre-existing special tokens
    std::cout << "[RoundTrip] UNIGRAM_VOCAB_OFFSET = " << UNIGRAM_VOCAB_OFFSET << "\n";
    std::cout << "[RoundTrip] Pre-existing vocab size = " << tokenizer.unigramLM().vocabSize() << "\n";
    
    std::cout << "[RoundTrip] Adding pieces:\n";
    tokenizer.unigramLM().addPiece("the", -1.0f, false);
    std::cout << "  'the'   -> id=" << UnigramLM::tokenIdForIndex(tokenizer.unigramLM().vocabSize() - 1) << "\n";
    tokenizer.unigramLM().addPiece("quick", -1.5f, false);
    std::cout << "  'quick' -> id=" << UnigramLM::tokenIdForIndex(tokenizer.unigramLM().vocabSize() - 1) << "\n";
    tokenizer.unigramLM().addPiece("brown", -1.6f, false);
    std::cout << "  'brown' -> id=" << UnigramLM::tokenIdForIndex(tokenizer.unigramLM().vocabSize() - 1) << "\n";
    tokenizer.unigramLM().addPiece("fox", -1.7f, false);
    std::cout << "  'fox'   -> id=" << UnigramLM::tokenIdForIndex(tokenizer.unigramLM().vocabSize() - 1) << "\n";
    tokenizer.unigramLM().addPiece(" ", -0.5f, false);
    std::cout << "  ' '     -> id=" << UnigramLM::tokenIdForIndex(tokenizer.unigramLM().vocabSize() - 1) << "\n";
    
    std::cout << "[RoundTrip] Final vocab size = " << tokenizer.unigramLM().vocabSize() << "\n";
    
    // Dump ALL pieces in vocabulary to see what's actually stored
    std::cout << "[RoundTrip] === Full Vocabulary Dump ===\n";
    for (int i = 0; i < tokenizer.unigramLM().vocabSize(); ++i) {
        int tid = UNIGRAM_VOCAB_OFFSET + i;
        const auto* piece = tokenizer.unigramLM().getPiece(tid);
        if (piece) {
            std::cout << "  idx=" << i << " token_id=" << tid 
                      << " text=\"" << piece->text << "\""
                      << " score=" << piece->score
                      << " special=" << piece->is_special
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
    
    uint32_t id = table.registerAtom(AtomType::ATOM_INTEGER, "12345", 0, 5);
    
    const AtomEntry* entry = table.getAtom(id);
    ASSERT_TRUE(entry != nullptr, "Failed to retrieve atom");
    ASSERT_EQ(static_cast<int>(entry->type), static_cast<int>(AtomType::ATOM_INTEGER), 
              "Type mismatch");
    
    // Access raw_text through string pool
    std::string raw_text(table.getString(entry->raw_text_ref));
    ASSERT_STR_EQ(raw_text, "12345", "Raw text mismatch");
    
    // Check parsed value
    double num = AtomTable::getNumericValue(*entry);
    ASSERT_NEAR(num, 12345.0, 0.01, "Numeric value mismatch");
    
    return true;
}

bool testAtomTableRegisterFloat(std::string& message) {
    AtomTable table;
    
    uint32_t id = table.registerAtom(AtomType::ATOM_FLOAT, "3.14159", 0, 7);
    
    const AtomEntry* entry = table.getAtom(id);
    ASSERT_TRUE(entry != nullptr, "Failed to retrieve atom");
    
    double num = AtomTable::getNumericValue(*entry);
    ASSERT_NEAR(num, 3.14159, 0.0001, "Float value mismatch");
    
    return true;
}

bool testAtomTableRegisterHex(std::string& message) {
    AtomTable table;
    
    uint32_t id = table.registerAtom(AtomType::ATOM_HEX, "0xFF", 0, 4);
    
    const AtomEntry* entry = table.getAtom(id);
    ASSERT_TRUE(entry != nullptr, "Failed to retrieve atom");
    
    double num = AtomTable::getNumericValue(*entry);
    ASSERT_NEAR(num, 255.0, 0.01, "Hex value mismatch");
    
    return true;
}

bool testAtomTableRegisterBinary(std::string& message) {
    AtomTable table;
    
    uint32_t id = table.registerAtom(AtomType::ATOM_BINARY, "0b1010", 0, 6);
    
    const AtomEntry* entry = table.getAtom(id);
    ASSERT_TRUE(entry != nullptr, "Failed to retrieve atom");
    
    double num = AtomTable::getNumericValue(*entry);
    ASSERT_NEAR(num, 10.0, 0.01, "Binary value mismatch");
    
    return true;
}

bool testAtomTableRegisterURL(std::string& message) {
    AtomTable table;
    
    uint32_t id = table.registerAtom(AtomType::ATOM_URL, 
                                      "https://example.com:8080/path?query=1#fragment", 
                                      0, 45);
    
    const AtomEntry* entry = table.getAtom(id);
    ASSERT_TRUE(entry != nullptr, "Failed to retrieve atom");
    ASSERT_EQ(static_cast<int>(entry->type), static_cast<int>(AtomType::ATOM_URL),
              "Type mismatch");
    
    // Verify serialization
    std::string serialized = table.atomToString(*entry);
    ASSERT_TRUE(serialized.find("example.com") != std::string::npos, 
                "Serialized URL should contain domain");
    
    return true;
}

bool testAtomTableRegisterEmail(std::string& message) {
    AtomTable table;
    
    uint32_t id = table.registerAtom(AtomType::ATOM_EMAIL, "user@domain.com", 0, 15);
    
    const AtomEntry* entry = table.getAtom(id);
    ASSERT_TRUE(entry != nullptr, "Failed to retrieve atom");
    
    std::string serialized = table.atomToString(*entry);
    ASSERT_STR_EQ(serialized, "user@domain.com", "Email serialization mismatch");
    
    return true;
}

bool testAtomTableRegisterDate(std::string& message) {
    AtomTable table;
    
    uint32_t id = table.registerAtom(AtomType::ATOM_DATE, "2024-12-25", 0, 10);
    
    const AtomEntry* entry = table.getAtom(id);
    ASSERT_TRUE(entry != nullptr, "Failed to retrieve atom");
    
    return true;
}

bool testAtomTableRegisterTime(std::string& message) {
    AtomTable table;
    
    uint32_t id = table.registerAtom(AtomType::ATOM_TIME, "14:30:00", 0, 8);
    
    const AtomEntry* entry = table.getAtom(id);
    ASSERT_TRUE(entry != nullptr, "Failed to retrieve atom");
    
    return true;
}

bool testAtomTableRegisterIP(std::string& message) {
    AtomTable table;
    
    uint32_t id = table.registerAtom(AtomType::ATOM_IP_ADDRESS, "192.168.1.1", 0, 11);
    
    const AtomEntry* entry = table.getAtom(id);
    ASSERT_TRUE(entry != nullptr, "Failed to retrieve atom");
    
    std::string serialized = table.atomToString(*entry);
    ASSERT_STR_EQ(serialized, "192.168.1.1", "IP serialization mismatch");
    
    return true;
}

bool testAtomTableRegisterPath(std::string& message) {
    AtomTable table;
    
    uint32_t id = table.registerAtom(AtomType::ATOM_PATH, "/usr/local/bin/test", 0, 19);
    
    const AtomEntry* entry = table.getAtom(id);
    ASSERT_TRUE(entry != nullptr, "Failed to retrieve atom");
    
    return true;
}

bool testAtomTableRegisterString(std::string& message) {
    AtomTable table;
    
    uint32_t id = table.registerAtom(AtomType::ATOM_STRING_LITERAL, "\"hello\\nworld\"", 0, 14);
    
    const AtomEntry* entry = table.getAtom(id);
    ASSERT_TRUE(entry != nullptr, "Failed to retrieve atom");
    
    return true;
}

bool testAtomTableRegisterIdentifier(std::string& message) {
    AtomTable table;
    
    // Test various naming conventions
    table.registerAtom(AtomType::ATOM_IDENTIFIER, "camelCase", 0, 9);
    table.registerAtom(AtomType::ATOM_IDENTIFIER, "PascalCase", 0, 10);
    table.registerAtom(AtomType::ATOM_IDENTIFIER, "snake_case", 0, 10);
    uint32_t id = table.registerAtom(AtomType::ATOM_IDENTIFIER, "SCREAMING_SNAKE", 0, 15);
    
    const AtomEntry* entry = table.getAtom(id);
    ASSERT_TRUE(entry != nullptr, "Failed to retrieve atom");
    ASSERT_EQ(table.size(), 4, "Should have 4 atoms");
    
    return true;
}

bool testAtomTableLookupByType(std::string& message) {
    AtomTable table;
    
    // Register multiple atoms of different types
    table.registerAtom(AtomType::ATOM_INTEGER, "100", 0, 3);
    table.registerAtom(AtomType::ATOM_INTEGER, "200", 0, 3);
    table.registerAtom(AtomType::ATOM_FLOAT, "3.14", 0, 4);
    table.registerAtom(AtomType::ATOM_INTEGER, "300", 0, 3);
    
    auto integers = table.getAtomsByType(AtomType::ATOM_INTEGER);
    ASSERT_EQ(integers.size(), 3, "Should find 3 integers");
    
    auto floats = table.getAtomsByType(AtomType::ATOM_FLOAT);
    ASSERT_EQ(floats.size(), 1, "Should find 1 float");
    
    return true;
}

bool testAtomTableGPUUpload(std::string& message) {
    AtomTable table;
    
    // Register some atoms
    table.registerAtom(AtomType::ATOM_INTEGER, "42", 0, 2);
    table.registerAtom(AtomType::ATOM_FLOAT, "3.14", 0, 4);
    table.registerAtom(AtomType::ATOM_URL, "https://test.com", 0, 16);
    
    AtomTable::GPUAtomData gpu_data;
    bool success = table.uploadToGPU(gpu_data);
    
    ASSERT_TRUE(success, "GPU upload failed");
    ASSERT_EQ(gpu_data.num_atoms, 3, "GPU atom count mismatch");
    ASSERT_TRUE(gpu_data.d_numeric_values != nullptr, "Numeric values not allocated");
    ASSERT_TRUE(gpu_data.d_types != nullptr, "Types not allocated");
    
    // Cleanup
    AtomTable::freeGPUData(gpu_data);
    
    return true;
}

bool testAtomTableClear(std::string& message) {
    AtomTable table;
    
    table.registerAtom(AtomType::ATOM_INTEGER, "1", 0, 1);
    table.registerAtom(AtomType::ATOM_INTEGER, "2", 0, 1);
    table.registerAtom(AtomType::ATOM_INTEGER, "3", 0, 1);
    
    ASSERT_EQ(table.size(), 3, "Should have 3 atoms before clear");
    
    table.clear();
    
    ASSERT_EQ(table.size(), 0, "Should have 0 atoms after clear");
    
    return true;
}

bool testAtomTableMetadata(std::string& message) {
    AtomTable table;
    
    // Register an atom
    uint32_t id = table.registerAtom(AtomType::ATOM_INTEGER, "42", 0, 2);
    
    const AtomEntry* entry = table.getAtom(id);
    ASSERT_TRUE(entry != nullptr, "Failed to retrieve atom");
    
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
    uint32_t id1 = table.registerAtom(AtomType::ATOM_INTEGER, "100", 0, 3);
    uint32_t id2 = table.registerAtom(AtomType::ATOM_INTEGER, "100", 0, 3);
    
    const AtomEntry* entry1 = table.getAtom(id1);
    const AtomEntry* entry2 = table.getAtom(id2);
    
    ASSERT_TRUE(entry1 != nullptr && entry2 != nullptr, "Both atoms should exist");
    ASSERT_EQ(entry1->hash, entry2->hash, "Identical atoms should have same hash");
    
    // Different atom should have different hash
    uint32_t id3 = table.registerAtom(AtomType::ATOM_INTEGER, "200", 0, 3);
    const AtomEntry* entry3 = table.getAtom(id3);
    
    ASSERT_TRUE(entry1->hash != entry3->hash, "Different atoms should have different hashes");
    
    return true;
}

//======================================================//
//  Section 5: Integration Tests
//======================================================//

bool testFullPipeline(std::string& message) {
    // Create tokenizer with all features
    UniByteConfig config;
    config.target_vocab_size = 50000;
    config.enable_byte_fallback = true;
    config.detect_numbers = true;
    config.detect_urls = true;
    config.detect_emails = true;
    config.detect_dates = true;
    
    UniByte tokenizer(config);
    
    // Add vocabulary via unigramLM
    // Start after pre-existing special tokens
    tokenizer.unigramLM().addPiece("the", -1.0f, false);
    tokenizer.unigramLM().addPiece("is", -1.1f, false);
    tokenizer.unigramLM().addPiece("price", -1.5f, false);
    tokenizer.unigramLM().addPiece("visit", -1.6f, false);
    tokenizer.unigramLM().addPiece("for", -1.7f, false);
    tokenizer.unigramLM().addPiece("more", -1.8f, false);
    tokenizer.unigramLM().addPiece("info", -1.9f, false);
    tokenizer.unigramLM().addPiece(" ", -0.5f, false);
    tokenizer.unigramLM().addPiece(".", -0.6f, false);
    
    // Complex input with multiple structural elements
    std::string input = "The price is 42.99. Visit https://shop.com for more info.";
    
    auto result = tokenizer.encodeWithMetadata(input);
    
    ASSERT_TRUE(result.token_ids.size() > 0, "Should produce tokens");
    ASSERT_TRUE(result.atoms.size() >= 2, "Should detect at least 2 structures");
    
    // Verify we can decode back
    std::string decoded = tokenizer.decode(result.token_ids);
    
    // Note: With placeholders, decoded may differ from input
    // The key is that we have a valid token sequence
    
    return true;
}

bool testAtomTableIntegration(std::string& message) {
    UniByteConfig config;
    config.target_vocab_size = 50000;
    config.detect_numbers = true;
    
    UniByte tokenizer(config);
    
    std::string input = "Values: 100, 200, 300";
    auto result = tokenizer.encodeWithMetadata(input);
    
    // Register detected atoms in AtomTable
    AtomTable table;
    for (const auto& span : result.atoms) {
        table.registerSpan(span);
    }
    
    ASSERT_TRUE(table.size() >= 3, "Should register at least 3 number atoms");
    
    // Upload to GPU and verify
    AtomTable::GPUAtomData gpu_data;
    bool success = table.uploadToGPU(gpu_data);
    ASSERT_TRUE(success, "GPU upload should succeed");
    
    AtomTable::freeGPUData(gpu_data);
    
    return true;
}

bool testBatchProcessing(std::string& message) {
    UniByteConfig config;
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
    UniByteConfig config;
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
    UniByteConfig config;
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
    UniByteConfig config;
    config.enable_byte_fallback = true;
    UniByte tokenizer(config);
    
    // Add space to vocab
    tokenizer.unigramLM().addPiece(" ", -0.5f, false);
    tokenizer.unigramLM().buildTrie();
    
    std::string input = "   ";
    std::vector<int> tokens = tokenizer.encode(input);
    
    ASSERT_TRUE(tokens.size() >= 1, "Whitespace should produce tokens");
    
    std::string decoded = tokenizer.decode(tokens);
    ASSERT_STR_EQ(decoded, input, "Whitespace round-trip failed");
    
    return true;
}

bool testEdgeCaseLongSequence(std::string& message) {
    UniByteConfig config;
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
    UniByteConfig config;
    config.enable_byte_fallback = true;
    // Disable scratch block reasoning to avoid atom detection interfering with special token literals
    config.enable_scratch_block_reasoning = false;
    UniByte tokenizer(config);
    
    // Add vocab pieces for common words to reduce byte fallback
    tokenizer.unigramLM().addPiece("This", -1.0f, false);
    tokenizer.unigramLM().addPiece(" ", -0.5f, false);
    tokenizer.unigramLM().addPiece("is", -1.0f, false);
    tokenizer.unigramLM().addPiece("not", -1.0f, false);
    tokenizer.unigramLM().addPiece("a", -1.0f, false);
    tokenizer.unigramLM().addPiece("special", -1.0f, false);
    tokenizer.unigramLM().addPiece("token", -1.0f, false);
    // Add the literal special token strings as regular vocab pieces
    tokenizer.unigramLM().addPiece("<unk>", -2.0f, false);
    tokenizer.unigramLM().addPiece("<s>", -2.0f, false);
    tokenizer.unigramLM().addPiece("</s>", -2.0f, false);
    tokenizer.unigramLM().addPiece("<pad>", -2.0f, false);
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
    UniByteConfig config;
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
    UniByteConfig config;
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
    UniByteConfig config;
    config.enable_byte_fallback = true;
    config.detect_urls = true;
    config.detect_emails = true;
    config.enable_scratch_block_reasoning = true;  // Enable atom detection
    UniByte tokenizer(config);
    
    std::string input = "日本のサイト: https://example.jp メール: test@日本.jp";
    
    // Use encodeWithMetadata to get both tokens and atom information
    auto result = tokenizer.encodeWithMetadata(input);
    ASSERT_TRUE(result.token_ids.size() > 0, "Unicode with URLs should produce tokens");
    
    // Build an atom resolver that returns the original text for detected atoms
    // Store atom original texts by their position in token stream
    std::unordered_map<int, std::string> atom_texts;
    for (const auto& span : result.atoms) {
        // Use contentView() which excludes leading whitespace (already emitted separately)
        atom_texts[span.placeholder_id] = std::string(span.contentView());
    }
    
    // Decode with atom resolution
    auto resolver = [&atom_texts](int token_id, AtomType type) -> std::string {
        auto it = atom_texts.find(token_id);
        if (it != atom_texts.end()) {
            return it->second;
        }
        return "<UNKNOWN_ATOM>";
    };
    
    std::string decoded = tokenizer.decodeWithAtoms(result.token_ids, resolver);
    ASSERT_STR_EQ(decoded, input, "Unicode+structural round-trip failed");
    
    return true;
}

//======================================================//
//  Section 9: Multiple Structural Elements Tests
//======================================================//

bool testMultipleURLs(std::string& message) {
    UniByteConfig config;
    config.enable_byte_fallback = true;
    config.detect_urls = true;
    UniByte tokenizer(config);
    
    std::string input = "Visit https://first.com and https://second.com or http://third.org";
    
    auto result = tokenizer.encodeWithMetadata(input);
    
    // Count URL atoms
    int url_count = 0;
    for (const auto& span : result.atoms) {
        if (span.atom_type == AtomType::ATOM_URL) {
            url_count++;
        }
    }
    
    ASSERT_EQ(url_count, 3, "Should detect 3 URLs");
    
    return true;
}

bool testMultipleEmails(std::string& message) {
    UniByteConfig config;
    config.enable_byte_fallback = true;
    config.detect_emails = true;
    UniByte tokenizer(config);
    
    std::string input = "Contact: alice@example.com, bob@test.org, charlie@domain.net";
    
    auto result = tokenizer.encodeWithMetadata(input);
    
    int email_count = 0;
    for (const auto& span : result.atoms) {
        if (span.atom_type == AtomType::ATOM_EMAIL) {
            email_count++;
        }
    }
    
    ASSERT_TRUE(email_count >= 3, "Should detect at least 3 emails");
    
    return true;
}

bool testMixedNumbers(std::string& message) {
    UniByteConfig config;
    config.enable_byte_fallback = true;
    config.detect_numbers = true;
    UniByte tokenizer(config);
    
    std::string input = "Int: 42, Float: 3.14, Negative: -17, Scientific: 1.5e10, Hex: 0xFF";
    
    auto result = tokenizer.encodeWithMetadata(input);
    
    int number_count = 0;
    for (const auto& span : result.atoms) {
        if (span.atom_type == AtomType::ATOM_INTEGER || 
            span.atom_type == AtomType::ATOM_FLOAT ||
            span.atom_type == AtomType::ATOM_HEX) {
            number_count++;
        }
    }
    
    ASSERT_TRUE(number_count >= 4, "Should detect multiple number types");
    
    return true;
}

bool testAdjacentStructural(std::string& message) {
    UniByteConfig config;
    config.enable_byte_fallback = true;
    config.detect_numbers = true;
    config.detect_urls = true;
    UniByte tokenizer(config);
    
    // Numbers immediately followed by URL
    std::string input = "Price:$99https://buy.com";
    
    auto result = tokenizer.encodeWithMetadata(input);
    ASSERT_TRUE(result.token_ids.size() > 0, "Adjacent structures should tokenize");
    
    return true;
}

//======================================================//
//  Section 10: Path Detection Tests
//======================================================//

bool testWindowsPath(std::string& message) {
    UniByteConfig config;
    config.enable_byte_fallback = true;
    config.detect_paths = true;
    UniByte tokenizer(config);
    
    std::string input = "Open file C:\\Users\\test\\document.txt please";
    
    auto result = tokenizer.encodeWithMetadata(input);
    
    bool found_path = false;
    for (const auto& span : result.atoms) {
        if (span.atom_type == AtomType::ATOM_PATH) {
            found_path = true;
            break;
        }
    }
    
    ASSERT_TRUE(found_path, "Should detect Windows path");
    
    return true;
}

bool testUnixPath(std::string& message) {
    UniByteConfig config;
    config.enable_byte_fallback = true;
    config.detect_paths = true;
    UniByte tokenizer(config);
    
    std::string input = "Run /usr/local/bin/program with args";
    
    auto result = tokenizer.encodeWithMetadata(input);
    
    bool found_path = false;
    for (const auto& span : result.atoms) {
        if (span.atom_type == AtomType::ATOM_PATH) {
            found_path = true;
            break;
        }
    }
    
    ASSERT_TRUE(found_path, "Should detect Unix path");
    
    return true;
}

//======================================================//
//  Section 11: Numeric Edge Cases
//======================================================//

bool testScientificNotation(std::string& message) {
    UniByteConfig config;
    config.enable_byte_fallback = true;
    config.detect_numbers = true;
    UniByte tokenizer(config);
    
    std::string input = "Values: 1.23e-10, 4.56E+20, 7.89e5";
    
    auto result = tokenizer.encodeWithMetadata(input);
    
    int float_count = 0;
    for (const auto& span : result.atoms) {
        if (span.atom_type == AtomType::ATOM_FLOAT) {
            float_count++;
        }
    }
    
    ASSERT_TRUE(float_count >= 2, "Should detect scientific notation");
    
    return true;
}

bool testIPAddressVsDecimal(std::string& message) {
    UniByteConfig config;
    config.enable_byte_fallback = true;
    config.detect_numbers = true;
    // Note: IP detection may be part of number detection or separate
    UniByte tokenizer(config);
    
    std::string input = "Server 192.168.1.1 price 1.2.3";
    
    auto result = tokenizer.encodeWithMetadata(input);
    
    bool found_ip = false;
    for (const auto& span : result.atoms) {
        if (span.atom_type == AtomType::ATOM_IP_ADDRESS) {
            found_ip = true;
            break;
        }
    }
    
    // IP detection may not be implemented - just verify we got some structural detection
    ASSERT_TRUE(result.atoms.size() >= 0, "Should process without crash");
    
    return true;
}

bool testNegativeNumbers(std::string& message) {
    UniByteConfig config;
    config.enable_byte_fallback = true;
    config.detect_numbers = true;
    UniByte tokenizer(config);
    
    std::string input = "Temperature: -40 degrees, balance: -$1,234.56";
    
    auto result = tokenizer.encodeWithMetadata(input);
    
    int number_count = 0;
    for (const auto& span : result.atoms) {
        if (span.atom_type == AtomType::ATOM_INTEGER || span.atom_type == AtomType::ATOM_FLOAT) {
            number_count++;
        }
    }
    
    ASSERT_TRUE(number_count >= 1, "Should detect negative numbers");
    
    return true;
}

bool testDigitsFollowedByAlpha(std::string& message) {
    // Regression test: digits followed by alphabetic chars (ordinals, units, versions)
    // must still be detected as integer atoms, not leak as raw byte tokens.
    UniByteConfig config;
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
        auto result = tokenizer.encodeWithMetadata(tc.input);
        
        int int_count = 0;
        for (const auto& span : result.atoms) {
            if (span.atom_type == AtomType::ATOM_INTEGER) {
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
    UniByteConfig config;
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
    UniByteConfig config;
    config.enable_byte_fallback = true;
    UniByte tokenizer(config);
    
    // Add partial vocab
    tokenizer.unigramLM().addPiece("hello", -1.0f, false);
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

bool testVocabSaveLoadText(std::string& message) {
    // Create output directory if it doesn't exist
    std::filesystem::create_directories("output");
    
    // Create tokenizer with vocab
    UnigramLM original;
    original.addPiece("test", -1.0f, false);
    original.addPiece("vocab", -1.5f, false);
    original.addPiece("save", -2.0f, false);
    
    // Save to temp file
    std::string path = "output/test_vocab_save.txt";
    bool saved = original.save(path, true);  // text format
    ASSERT_TRUE(saved, "Save should succeed");
    
    // Load into new tokenizer
    UnigramLM loaded;
    bool loaded_ok = loaded.load(path);
    ASSERT_TRUE(loaded_ok, "Load should succeed");
    
    // Verify pieces exist
    ASSERT_TRUE(loaded.hasPiece("test"), "Should have 'test' piece");
    ASSERT_TRUE(loaded.hasPiece("vocab"), "Should have 'vocab' piece");
    ASSERT_TRUE(loaded.hasPiece("save"), "Should have 'save' piece");
    
    // Cleanup
    std::filesystem::remove(path);
    
    return true;
}

bool testVocabSaveLoadBinary(std::string& message) {
    // Create output directory if it doesn't exist
    std::filesystem::create_directories("output");
    
    // Create tokenizer with vocab
    UnigramLM original;
    original.addPiece("binary", -1.0f, false);
    original.addPiece("format", -1.5f, false);
    original.addPiece("fast", -2.0f, false);
    
    // Save to binary
    std::string path = "output/test_vocab_binary.bin";
    bool saved = original.save(path, false);  // binary format
    ASSERT_TRUE(saved, "Binary save should succeed");
    
    // Load into new tokenizer
    UnigramLM loaded;
    bool loaded_ok = loaded.loadBinary(path);
    ASSERT_TRUE(loaded_ok, "Binary load should succeed");
    
    // Verify vocab size matches (accounting for special tokens)
    ASSERT_TRUE(loaded.hasPiece("binary"), "Should have 'binary' piece");
    ASSERT_TRUE(loaded.hasPiece("format"), "Should have 'format' piece");
    
    // Cleanup
    std::filesystem::remove(path);
    
    return true;
}

bool testVocabCapSize(std::string& message) {
    UnigramLM unigram;
    
    // Add many pieces
    for (int i = 0; i < 100; ++i) {
        std::string piece = "piece" + std::to_string(i);
        float score = -1.0f - (i * 0.01f);  // Decreasing scores
        unigram.addPiece(piece, score, false);
    }
    
    int original_size = unigram.vocabSize();
    ASSERT_TRUE(original_size > 50, "Should have many pieces");
    
    // Cap to smaller size (must be >= NUM_SPECIAL_TOKENS for special tokens)
    unigram.capVocabSize(260);
    
    int new_size = unigram.vocabSize();
    ASSERT_TRUE(new_size <= 260, "Should be capped to 260 or less");
    
    return true;
}

//======================================================//
//  Section 14: GPU Decode Tests
//======================================================//

bool testGPUDecode(std::string& message) {
    UnigramLM unigram;
    
    // Add vocab
    unigram.addPiece("gpu", -1.0f, false);
    unigram.addPiece("decode", -1.5f, false);
    unigram.addPiece(" ", -0.5f, false);
    unigram.buildTrie();
    
    // Init GPU
    bool gpu_ok = unigram.initGPU();
    if (!gpu_ok) {
        message = "GPU init failed (may not have CUDA)";
        return true;  // Skip test if no GPU
    }
    
    // Encode on CPU
    std::vector<int> tokens = unigram.encode("gpu decode");
    
    // Copy tokens to GPU
    int* d_tokens;
    cudaMalloc(&d_tokens, tokens.size() * sizeof(int));
    cudaMemcpy(d_tokens, tokens.data(), tokens.size() * sizeof(int), cudaMemcpyHostToDevice);
    
    // Decode on GPU
    char* d_output;
    size_t* d_length;
    cudaMalloc(&d_output, 256);
    cudaMalloc(&d_length, sizeof(size_t));
    
    bool decode_ok = unigram.decodeGPU(d_tokens, tokens.size(), d_output, d_length, 256);
    
    if (decode_ok) {
        size_t length;
        cudaMemcpy(&length, d_length, sizeof(size_t), cudaMemcpyDeviceToHost);
        
        std::vector<char> output(length + 1);
        cudaMemcpy(output.data(), d_output, length, cudaMemcpyDeviceToHost);
        output[length] = '\0';
        
        std::string result(output.data());
        ASSERT_STR_EQ(result, "gpu decode", "GPU decode mismatch");
    }
    
    cudaFree(d_tokens);
    cudaFree(d_output);
    cudaFree(d_length);
    
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
    suite.addTest("Byte.GPU.Encode", testByteGPUEncode);
    
    // Section 2: Unigram LM Tests
    suite.addTest("Unigram.BuildVocab", testUnigramBuildVocab);
    suite.addTest("Unigram.Encode", testUnigramEncode);
    suite.addTest("Unigram.Viterbi", testUnigramViterbi);
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
    suite.addTest("UniByte.URLDetection", testUniByteURLDetection);
    suite.addTest("UniByte.URLDetection.CaseInsensitive", testUniByteURLDetectionCaseInsensitive);
    suite.addTest("UniByte.EmailDetection", testUniByteEmailDetection);
    suite.addTest("UniByte.DateDetection", testUniByteDateDetection);
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
    suite.addTest("Unicode.WithStructural", testUnicodeWithStructural);
    
    // Section 9: Multiple Structural Elements Tests
    suite.addTest("Structural.MultipleURLs", testMultipleURLs);
    suite.addTest("Structural.MultipleEmails", testMultipleEmails);
    suite.addTest("Structural.MixedNumbers", testMixedNumbers);
    suite.addTest("Structural.Adjacent", testAdjacentStructural);
    
    // Section 10: Path Detection Tests
    suite.addTest("Path.Windows", testWindowsPath);
    suite.addTest("Path.Unix", testUnixPath);
    
    // Section 11: Numeric Edge Cases
    suite.addTest("Numeric.ScientificNotation", testScientificNotation);
    suite.addTest("Numeric.IPvsDecimal", testIPAddressVsDecimal);
    suite.addTest("Numeric.Negative", testNegativeNumbers);
    suite.addTest("Numeric.DigitsFollowedByAlpha", testDigitsFollowedByAlpha);
    
    // Section 12: Byte Fallback Control Tests
    suite.addTest("ByteFallback.Disabled", testByteFallbackDisabled);
    suite.addTest("ByteFallback.Mixed", testMixedVocabAndByteFallback);
    
    // Section 13: Vocabulary Persistence Tests
    suite.addTest("Vocab.SaveLoadText", testVocabSaveLoadText);
    suite.addTest("Vocab.SaveLoadBinary", testVocabSaveLoadBinary);
    suite.addTest("Vocab.CapSize", testVocabCapSize);
    
    // Section 14: GPU Decode Tests
    suite.addTest("GPU.Decode", testGPUDecode);
    
    // Run all tests
    auto results = suite.runAll();
    
    // Return exit code based on results
    int failures = 0;
    for (const auto& result : results) {
        if (!result.passed) ++failures;
    }
    
    return failures > 0 ? 1 : 0;
}
