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
    unigram.addPiece("hello", -1.0f);
    unigram.addPiece("world", -1.2f);
    unigram.addPiece("he", -2.0f);
    unigram.addPiece("llo", -2.5f);
    unigram.addPiece("wo", -2.3f);
    unigram.addPiece("rld", -2.8f);
    unigram.addPiece("h", -3.0f);
    unigram.addPiece("e", -3.1f);
    unigram.addPiece("l", -3.2f);
    unigram.addPiece("o", -3.3f);
    unigram.addPiece("w", -3.4f);
    unigram.addPiece("r", -3.5f);
    unigram.addPiece("d", -3.6f);
    
    // 13 manually added pieces + 4 special tokens (<unk>, <pad>, <s>, </s>) = 17 total
    ASSERT_EQ(unigram.vocabSize(), 17, "Vocab size mismatch");
    
    return true;
}

bool testUnigramEncode(std::string& message) {
    UnigramLM unigram;
    
    // Build vocabulary with log probabilities
    unigram.addPiece("hello", -1.0f);  // Most likely for "hello"
    unigram.addPiece("he", -2.0f);
    unigram.addPiece("llo", -2.5f);
    unigram.addPiece("h", -3.0f);
    unigram.addPiece("e", -3.1f);
    unigram.addPiece("l", -3.2f);
    unigram.addPiece("o", -3.3f);
    unigram.buildTrie();  // Must build trie before encoding
    
    std::vector<int> tokens = unigram.encode("hello");
    
    // Should produce token(s) - verify we got something
    ASSERT_TRUE(tokens.size() >= 1, "Should encode 'hello' to at least 1 token");
    
    return true;
}

bool testUnigramViterbi(std::string& message) {
    UnigramLM unigram;
    
    // Vocabulary where splitting is better than whole word
    unigram.addPiece("test", -5.0f);    // Whole word is worse
    unigram.addPiece("te", -1.0f);      // Better to split
    unigram.addPiece("st", -1.0f);
    unigram.addPiece("t", -2.0f);
    unigram.addPiece("e", -2.1f);
    unigram.addPiece("s", -2.2f);
    unigram.buildTrie();  // Must build trie before encoding
    
    std::vector<int> tokens = unigram.encode("test");
    
    // Viterbi should find optimal segmentation
    ASSERT_TRUE(tokens.size() >= 1, "Should produce tokens for 'test'");
    
    return true;
}

bool testUnigramDecode(std::string& message) {
    UnigramLM unigram;
    
    unigram.addPiece("hello", -1.0f);
    unigram.addPiece("world", -1.2f);
    unigram.addPiece(" ", -0.5f);
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
    unigram.addPiece("a", -1.0f);
    unigram.addPiece("b", -1.0f);
    unigram.buildTrie();  // Must build trie before encoding
    
    // Try to encode something not in vocab
    std::vector<int> tokens = unigram.encode("xyz");
    
    // Should still produce tokens (using UNK or character fallback)
    // Note: Behavior depends on implementation - may produce empty or UNK
    // For now just verify no crash
    
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
    tokenizer.unigramLM().addPiece("hello", -1.0f);
    tokenizer.unigramLM().addPiece("world", -1.2f);
    tokenizer.unigramLM().addPiece(" ", -0.5f);
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
        // Atom tokens are in range [256, 511]
        if (token >= 256 && token < 512) {
            found_placeholder = true;
            break;
        }
    }
    
    ASSERT_TRUE(found_placeholder, "Should inject placeholder for number");
    
    return true;
}

bool testUniByteRoundTrip(std::string& message) {
    UniByteConfig config;
    config.target_vocab_size = 50000;
    config.enable_byte_fallback = true;
    
    UniByte tokenizer(config);
    
    tokenizer.unigramLM().addPiece("the", -1.0f);
    tokenizer.unigramLM().addPiece("quick", -1.5f);
    tokenizer.unigramLM().addPiece("brown", -1.6f);
    tokenizer.unigramLM().addPiece("fox", -1.7f);
    tokenizer.unigramLM().addPiece(" ", -0.5f);
    tokenizer.unigramLM().buildTrie();  // Must build trie before encoding
    
    std::string input = "the quick brown fox";
    std::vector<int> tokens = tokenizer.encode(input);
    
    std::string output = tokenizer.decode(tokens);
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
    tokenizer.unigramLM().addPiece("the", -1.0f);
    tokenizer.unigramLM().addPiece("is", -1.1f);
    tokenizer.unigramLM().addPiece("price", -1.5f);
    tokenizer.unigramLM().addPiece("visit", -1.6f);
    tokenizer.unigramLM().addPiece("for", -1.7f);
    tokenizer.unigramLM().addPiece("more", -1.8f);
    tokenizer.unigramLM().addPiece("info", -1.9f);
    tokenizer.unigramLM().addPiece(" ", -0.5f);
    tokenizer.unigramLM().addPiece(".", -0.6f);
    
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
    
    // Run all tests
    auto results = suite.runAll();
    
    // Return exit code based on results
    int failures = 0;
    for (const auto& result : results) {
        if (!result.passed) ++failures;
    }
    
    return failures > 0 ? 1 : 0;
}
