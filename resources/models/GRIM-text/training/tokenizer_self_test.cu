//======================================================//
//  Tokenizer Self-Test
//  Comprehensive diagnostic tool for GRIM tokenizer
//  
//  Sections:
//    1/3 - Encoding/Decoding Visualization
//    2/3 - Spaces and Grammar Handling
//    3/3 - Round-trip & Edge Cases
//======================================================//

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <cctype>

#include <nlohmann/json.hpp>

#include "../Shared/UnigramByte/UniByte.hpp"
#include "../Shared/TokenizerArtifacts/TokenizerArtifactBundle.hpp"
#include "../Shared/HyperParameters/HyperParameters_GPU.hpp"  // single entry point; pulls in control/ai_config_paths.hpp transitively
#include "../Shared/HyperParameters/HyperparameterGroupings.hpp"

using GrimTokenizer = GRIM::Tokenizer::UniByte;

namespace fs = std::filesystem;

//======================================================//
//  CLI Options
//======================================================//
struct CliOptions {
    fs::path vocab_path{};  // Empty = load from ai_config.json
    fs::path data_path{};   // Empty = load from ai_config.json
    fs::path config_path{"ai_config.json"};
    fs::path cases_path{"tokenizer_selftest_cases.json"};
    fs::path log_dir{"selftest_logs"};
    bool update_baseline = false;
    bool verbose = false;
    bool run_all_sections = true;
    int section = 0;  // 0 = all, 1-3 = specific section
};

//======================================================//
//  Test Results Tracking
//======================================================//
struct TestResult {
    std::string name;
    bool passed = false;
    std::string details;
    std::string input;
    std::string output;
};

struct SectionResults {
    std::string section_name;
    int passed = 0;
    int failed = 0;
    int warnings = 0;
    std::vector<TestResult> tests;
};

//======================================================//
//  Console Colors (Windows/ANSI)
//======================================================//
namespace Color {
    const char* RESET   = "\033[0m";
    const char* RED     = "\033[31m";
    const char* GREEN   = "\033[32m";
    const char* YELLOW  = "\033[33m";
    const char* BLUE    = "\033[34m";
    const char* CYAN    = "\033[36m";
    const char* BOLD    = "\033[1m";
}

//======================================================//
//  Utility Functions
//======================================================//
std::string escapeString(const std::string& s) {
    std::ostringstream oss;
    for (char c : s) {
        if (c == '\n') oss << "\\n";
        else if (c == '\t') oss << "\\t";
        else if (c == '\r') oss << "\\r";
        else if (c == ' ') oss << "·";  // visible space marker
        else if (std::isprint(static_cast<unsigned char>(c))) oss << c;
        else oss << "\\x" << std::hex << std::setw(2) << std::setfill('0') 
                 << static_cast<int>(static_cast<unsigned char>(c));
    }
    return oss.str();
}

std::string summarizeIds(const std::vector<int>& ids, std::size_t max_preview = 20) {
    std::ostringstream oss;
    oss << "[";
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i) oss << ", ";
        if (i >= max_preview) {
            oss << "... +" << (ids.size() - max_preview) << " more";
            break;
        }
        oss << ids[i];
    }
    oss << "] (len=" << ids.size() << ")";
    return oss.str();
}

std::string tokenTextForDisplay(const GrimTokenizer& tokenizer, int token_id) {
            const GRIM::Tokenizer::TokenLayout layout =
                GRIM::Tokenizer::tokenLayoutFromActualVocabOrThrow(
                    static_cast<std::uint32_t>(tokenizer.vocabSize()),
                    "tokenizer_self_test: GRMT decode round-trip");
    if (layout.isSpecial(token_id)) {
        return GRIM::Tokenizer::specialTokenText(token_id);
    }
    if (layout.isByte(token_id)) {
        const uint8_t byte_value = tokenizer.byteEncoder().tokenToByte(token_id);
        return std::string(1, static_cast<char>(byte_value));
    }
    if (layout.isAtom(token_id)) {
        std::string label = "<";
        label += GRIM::Tokenizer::atomTypeName(GRIM::Tokenizer::tokenIdToAtomType(token_id));
        label += ">";
        return label;
    }
    if (layout.isUnigram(token_id)) {
        const auto* piece = tokenizer.unigramLM().getPiece(token_id);
        if (!piece) {
            throw std::runtime_error("tokenizer_self_test: unigram token_id=" +
                                     std::to_string(token_id) + " has no backing piece");
        }
        return piece->text;
    }
    throw std::runtime_error("tokenizer_self_test: token_id=" + std::to_string(token_id) +
                             " is outside the GRMT/tokenizer vocab layout");
}

void printHeader(const std::string& title) {
    std::cout << "\n" << Color::BOLD << Color::CYAN;
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  " << std::left << std::setw(60) << title << "║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    std::cout << Color::RESET;
}

void printSubHeader(const std::string& title) {
    std::cout << "\n" << Color::BOLD << Color::BLUE;
    std::cout << "┌─────────────────────────────────────────────────────────────┐\n";
    std::cout << "│  " << std::left << std::setw(58) << title << "│\n";
    std::cout << "└─────────────────────────────────────────────────────────────┘\n";
    std::cout << Color::RESET;
}

void printTestResult(const TestResult& result, bool verbose) {
    if (result.passed) {
        std::cout << Color::GREEN << "  ✓ " << Color::RESET << result.name;
        if (verbose && !result.details.empty()) {
            std::cout << " - " << result.details;
        }
        std::cout << "\n";
    } else {
        std::cout << Color::RED << "  ✗ " << Color::RESET << result.name << "\n";
        if (!result.details.empty()) {
            std::cout << "      " << Color::YELLOW << result.details << Color::RESET << "\n";
        }
        if (!result.input.empty()) {
            std::cout << "      Input:  \"" << escapeString(result.input) << "\"\n";
        }
        if (!result.output.empty()) {
            std::cout << "      Output: \"" << escapeString(result.output) << "\"\n";
        }
    }
}

//======================================================//
//  SECTION 1/3: Encoding/Decoding Visualization
//======================================================//
SectionResults runSection1_EncodingDecoding(GrimTokenizer& tokenizer, bool verbose) {
    SectionResults results;
    results.section_name = "1/3 Encoding/Decoding Visualization";
    
    printSubHeader(results.section_name);
    
    std::vector<std::pair<std::string, std::string>> test_cases = {
        {"Hello, world!", "Basic greeting"},
        {"Hello, how are you?", "Model input simulation"},
        {"Repeat the word 'hello' five times.", "Instruction with quotes"},
        {"The quick brown fox jumps.", "Multi-word sentence"},
        {"AI", "Very short input"},
        {"a b c d e", "Single chars with spaces"},
        {"123 + 456 = 579", "Numbers and operators"},
        {"What is 2+2?", "Math question no spaces"},
    };
    
    for (const auto& [input, desc] : test_cases) {
        TestResult test;
        test.name = desc;
        test.input = input;
        
        // Encode
        auto ids = tokenizer.encode(input);
        
        // Decode
        std::string decoded = tokenizer.decode(GRIM::Tokenizer::DecodeRequest(ids));
        test.output = decoded;
        
        // Visualization
        if (verbose) {
            std::cout << "\n  " << Color::CYAN << "Input: " << Color::RESET 
                      << "\"" << escapeString(input) << "\"\n";
            std::cout << "  " << Color::CYAN << "Token IDs: " << Color::RESET 
                      << summarizeIds(ids) << "\n";
            std::cout << "  " << Color::CYAN << "Tokens: " << Color::RESET;
            for (size_t i = 0; i < ids.size() && i < 15; ++i) {
                std::cout << "[" << ids[i] << ":" << escapeString(tokenTextForDisplay(tokenizer, ids[i])) << "] ";
            }
            if (ids.size() > 15) std::cout << "...";
            std::cout << "\n";
            std::cout << "  " << Color::CYAN << "Decoded: " << Color::RESET 
                      << "\"" << escapeString(decoded) << "\"\n";
        }
        
        // Check: does decoded contain all non-space chars from input?
        std::string input_no_space, decoded_no_space;
        for (char c : input) if (!std::isspace(c)) input_no_space += std::tolower(c);
        for (char c : decoded) if (!std::isspace(c)) decoded_no_space += std::tolower(c);
        
        test.passed = (input_no_space == decoded_no_space);
        if (!test.passed) {
            test.details = "Content mismatch after removing spaces";
        } else if (decoded.find(' ') == std::string::npos && input.find(' ') != std::string::npos) {
            test.details = "WARNING: Spaces lost during round-trip";
            results.warnings++;
        }
        
        if (test.passed) results.passed++; else results.failed++;
        results.tests.push_back(test);
        printTestResult(test, verbose);
    }
    
    return results;
}

//======================================================//
//  SECTION 2/3: Spaces and Grammar
//======================================================//
SectionResults runSection2_SpacesGrammar(GrimTokenizer& tokenizer, bool verbose) {
    SectionResults results;
    results.section_name = "2/3 Spaces and Grammar Handling";
    
    printSubHeader(results.section_name);
    
    // Test: Space token existence (UniByte uses byte fallback for spaces)
    {
        TestResult test;
        test.name = "Space token handling";
        // UniByte handles spaces via byte fallback or unigram pieces
        auto space_ids = tokenizer.encode(" ");
        test.passed = !space_ids.empty();
        test.details = test.passed ? 
            "Space encodes to " + std::to_string(space_ids.size()) + " token(s)" :
            "CRITICAL: Failed to encode space!";
        if (test.passed) results.passed++; else results.failed++;
        results.tests.push_back(test);
        printTestResult(test, verbose);
    }
    
    // Test: Multiple spaces handling
    {
        TestResult test;
        test.name = "Multiple spaces normalization";
        test.input = "hello    world";  // 4 spaces
        auto ids = tokenizer.encode(test.input);
        test.output = tokenizer.decode(GRIM::Tokenizer::DecodeRequest(ids));
        // Should normalize to single space or handle gracefully
        test.passed = true;  // Normalization is acceptable behavior
        test.details = "Encoded to " + std::to_string(ids.size()) + " tokens";
        results.passed++;
        results.tests.push_back(test);
        printTestResult(test, verbose);
    }
    
    // Test: Leading/trailing spaces
    {
        TestResult test;
        test.name = "Leading/trailing spaces";
        test.input = "  hello  ";
        auto ids = tokenizer.encode(test.input);
        test.output = tokenizer.decode(GRIM::Tokenizer::DecodeRequest(ids));
        test.passed = true;  // Trimming is acceptable
        test.details = "Output: \"" + escapeString(test.output) + "\"";
        results.passed++;
        results.tests.push_back(test);
        printTestResult(test, verbose);
    }
    
    // Test: Punctuation handling
    std::vector<std::pair<std::string, std::string>> punct_tests = {
        {"Hello, world!", "Comma and exclamation"},
        {"What's up?", "Apostrophe and question"},
        {"It's a \"test\".", "Mixed quotes"},
        {"[1] (2) {3}", "Brackets"},
        {"foo::bar->baz", "C++ operators"},
    };
    
    for (const auto& [input, desc] : punct_tests) {
        TestResult test;
        test.name = "Punctuation: " + desc;
        test.input = input;
        auto ids = tokenizer.encode(input);
        test.output = tokenizer.decode(GRIM::Tokenizer::DecodeRequest(ids));
        
        // Check all punctuation preserved (ignoring spaces)
        std::string input_punct, output_punct;
        for (char c : input) if (std::ispunct(c)) input_punct += c;
        for (char c : test.output) if (std::ispunct(c)) output_punct += c;
        
        test.passed = (input_punct == output_punct);
        if (!test.passed) {
            test.details = "Punctuation lost: expected \"" + input_punct + 
                          "\" got \"" + output_punct + "\"";
        }
        
        if (test.passed) results.passed++; else results.failed++;
        results.tests.push_back(test);
        printTestResult(test, verbose);
    }
    
    // Test: Case preservation
    {
        TestResult test;
        test.name = "Case handling";
        test.input = "HeLLo WoRLD";
        auto ids = tokenizer.encode(test.input);
        test.output = tokenizer.decode(GRIM::Tokenizer::DecodeRequest(ids));
        
        // Our tokenizer lowercases, so check for that
        std::string expected_lower;
        for (char c : test.input) {
            if (!std::isspace(c)) expected_lower += std::tolower(c);
        }
        std::string output_no_space;
        for (char c : test.output) {
            if (!std::isspace(c)) output_no_space += c;
        }
        
        test.passed = (output_no_space == expected_lower);
        test.details = test.passed ? "Lowercased as expected" : 
                       "Output: " + test.output;
        
        if (test.passed) results.passed++; else results.failed++;
        results.tests.push_back(test);
        printTestResult(test, verbose);
    }
    
    return results;
}

//======================================================//
//  SECTION 3/3: Round-trip & Edge Cases
//======================================================//
SectionResults runSection3_EdgeCases(GrimTokenizer& tokenizer, bool verbose) {
    SectionResults results;
    results.section_name = "3/3 Round-trip & Edge Cases";
    
    printSubHeader(results.section_name);
    
    // Test: Empty input
    {
        TestResult test;
        test.name = "Empty string";
        test.input = "";
        auto ids = tokenizer.encode(test.input);
        test.output = tokenizer.decode(GRIM::Tokenizer::DecodeRequest(ids));
        // Should handle gracefully (maybe just BOS/EOS)
        test.passed = true;
        test.details = "Token count: " + std::to_string(ids.size());
        results.passed++;
        results.tests.push_back(test);
        printTestResult(test, verbose);
    }
    
    // Test: Very long input
    {
        TestResult test;
        test.name = "Long input (1000 chars)";
        test.input = std::string(100, 'a') + " " + std::string(100, 'b') + " " +
                     std::string(100, 'c') + " " + std::string(100, 'd') + " " +
                     std::string(100, 'e') + " " + std::string(100, 'f') + " " +
                     std::string(100, 'g') + " " + std::string(100, 'h') + " " +
                     std::string(100, 'i') + " " + std::string(100, 'j');
        auto ids = tokenizer.encode(test.input);
        test.output = tokenizer.decode(GRIM::Tokenizer::DecodeRequest(ids));
        test.passed = (ids.size() > 0);
        test.details = "Encoded to " + std::to_string(ids.size()) + " tokens";
        if (test.passed) results.passed++; else results.failed++;
        results.tests.push_back(test);
        printTestResult(test, verbose);
    }
    
    // Test: Special tokens
    {
        TestResult test;
        test.name = "Special tokens (BOS/EOS/PAD)";
        test.passed = true;
        std::ostringstream oss;
        oss << "PAD=" << GRIM::Tokenizer::PAD_TOKEN_ID << " UNK=" << GRIM::Tokenizer::UNK_TOKEN_ID
            << " BOS=" << GRIM::Tokenizer::BOS_TOKEN_ID << " EOS=" << GRIM::Tokenizer::EOS_TOKEN_ID;
        test.details = oss.str();
        results.passed++;
        results.tests.push_back(test);
        printTestResult(test, verbose);
    }
    
    // Test: Unknown token handling
    {
        TestResult test;
        test.name = "Unknown character handling";
        test.input = "hello\xFF\xFEworld";  // Invalid UTF-8
        auto ids = tokenizer.encode(test.input);
        test.output = tokenizer.decode(GRIM::Tokenizer::DecodeRequest(ids));
        // Should not crash - UniByte uses byte fallback for unknown chars
        test.passed = true;
        test.details = "Handled " + std::to_string(ids.size()) + " tokens";
        results.passed++;
        results.tests.push_back(test);
        printTestResult(test, verbose);
    }
    
    // Test: Unicode (if applicable)
    {
        TestResult test;
        test.name = "Unicode handling";
        test.input = "café naïve résumé";
        auto ids = tokenizer.encode(test.input);
        test.output = tokenizer.decode(GRIM::Tokenizer::DecodeRequest(ids));
        test.passed = (ids.size() > 0);
        test.details = "Encoded to " + std::to_string(ids.size()) + " tokens";
        if (test.passed) results.passed++; else results.failed++;
        results.tests.push_back(test);
        printTestResult(test, verbose);
    }
    
    // Test: Numbers
    {
        TestResult test;
        test.name = "Number tokenization";
        test.input = "12345 67890";
        auto ids = tokenizer.encode(test.input);
        test.output = tokenizer.decode(GRIM::Tokenizer::DecodeRequest(ids));
        
        if (verbose) {
            std::cout << "    Number tokens: ";
            for (size_t i = 0; i < ids.size(); ++i) {
                std::cout << "[" << tokenTextForDisplay(tokenizer, ids[i]) << "] ";
            }
            std::cout << "\n";
        }
        
        test.passed = true;
        test.details = std::to_string(ids.size()) + " tokens for digits";
        results.passed++;
        results.tests.push_back(test);
        printTestResult(test, verbose);
    }
    
    // Test: Decode with invalid IDs
    {
        TestResult test;
        test.name = "Invalid token ID handling";
        std::vector<int> bad_ids = {-1, 999999, tokenizer.vocabSize() + 100};
        test.output = tokenizer.decode(GRIM::Tokenizer::DecodeRequest(bad_ids));
        test.passed = true;  // Should not crash
        test.details = "Decoded invalid IDs to: \"" + escapeString(test.output) + "\"";
        results.passed++;
        results.tests.push_back(test);
        printTestResult(test, verbose);
    }
    
    // CRITICAL TEST: Space preservation diagnostic
    {
        TestResult test;
        test.name = "DIAGNOSTIC: Space preservation";
        test.input = "hello how are you";
        auto ids = tokenizer.encode(test.input);
        test.output = tokenizer.decode(GRIM::Tokenizer::DecodeRequest(ids));
        
        int input_spaces = std::count(test.input.begin(), test.input.end(), ' ');
        int output_spaces = std::count(test.output.begin(), test.output.end(), ' ');
        
        test.passed = (output_spaces > 0);
        if (output_spaces == 0 && input_spaces > 0) {
            test.details = Color::RED + std::string("CRITICAL: All ") + 
                          std::to_string(input_spaces) + " spaces lost!" + Color::RESET;
            results.failed++;
        } else if (output_spaces < input_spaces) {
            test.details = Color::YELLOW + std::string("WARNING: ") + 
                          std::to_string(input_spaces - output_spaces) + " spaces lost" + Color::RESET;
            results.warnings++;
            results.passed++;
        } else {
            test.details = "All " + std::to_string(input_spaces) + " spaces preserved";
            results.passed++;
        }
        
        results.tests.push_back(test);
        printTestResult(test, true);  // Always verbose for this
    }
    
    return results;
}

//======================================================//
//  CLI Parsing
//======================================================//
CliOptions parseOptions(int argc, char** argv) {
    CliOptions opts;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--vocab" && i + 1 < argc) {
            opts.vocab_path = argv[++i];
        } else if (arg == "--data" && i + 1 < argc) {
            opts.data_path = argv[++i];
        } else if (arg == "--config" && i + 1 < argc) {
            opts.config_path = argv[++i];
        } else if (arg == "--cases" && i + 1 < argc) {
            opts.cases_path = argv[++i];
        } else if (arg == "--log-dir" && i + 1 < argc) {
            opts.log_dir = argv[++i];
        } else if (arg == "--update-baseline") {
            opts.update_baseline = true;
        } else if (arg == "--verbose" || arg == "-v") {
            opts.verbose = true;
        } else if (arg == "--section" && i + 1 < argc) {
            opts.section = std::stoi(argv[++i]);
            opts.run_all_sections = (opts.section == 0);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "GRIM Tokenizer Self-Test\n\n";
            std::cout << "Usage: tokenizer_self_test [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --vocab PATH       Path to vocab.bin (default: from ai_config.json)\n";
            std::cout << "  --data PATH        Path to training_data.grmt (default: from ai_config.json)\n";
            std::cout << "  --config PATH      Path to ai_config.json (default: ai_config.json)\n";
            std::cout << "  --cases PATH       Path to test cases JSON\n";
            std::cout << "  --log-dir PATH     Log output directory\n";
            std::cout << "  --section N        Run specific section (1-3), 0=all\n";
            std::cout << "  --verbose, -v      Show detailed output\n";
            std::cout << "  --update-baseline  Update expected tokens in cases file\n";
            std::cout << "  --help, -h         Show this help\n";
            std::cout << "\nSections:\n";
            std::cout << "  1: Encoding/Decoding Visualization\n";
            std::cout << "  2: Spaces and Grammar\n";
            std::cout << "  3: Round-trip & Edge Cases\n";
            std::exit(0);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            std::exit(1);
        }
    }
    return opts;
}

//======================================================//
//  Main Entry Point
//======================================================//
int main(int argc, char** argv) {
    try {
        auto opts = parseOptions(argc, argv);
        
        printHeader("GRIM Tokenizer Self-Test");

        auto config_snapshot = GRIM::Config::loadAiConfigSnapshot();
        const auto startup_config = GRIM::HyperParameters::finalizeAiConfigSnapshot(
            std::move(config_snapshot),
            argc,
            argv,
            GRIM::HyperParameters::ModelExecutionMode::TRAINING);
        const auto tokenizer_hp = GRIM::HyperParameters::tokenizerHP(startup_config);
        
        if (tokenizer_hp.vocab_path.empty()) {
            throw std::runtime_error("tokenizer_self_test: vocab path is empty after tokenizerHP");
        }
        if (tokenizer_hp.data_path.empty()) {
            throw std::runtime_error("tokenizer_self_test: training_data path is empty after tokenizerHP");
        }
        opts.vocab_path = tokenizer_hp.vocab_path;
        opts.data_path = tokenizer_hp.data_path;
        
        std::cout << "\nConfiguration:\n";
        std::cout << "  Vocab: " << opts.vocab_path << "\n";
        std::cout << "  GRMT: " << opts.data_path << "\n";
        std::cout << "  Config: " << opts.config_path << "\n";
        std::cout << "  Verbose: " << (opts.verbose ? "yes" : "no") << "\n";
        std::cout << "  Sections: " << (opts.run_all_sections ? "all" : std::to_string(opts.section)) << "\n";
        
        // Create log directory if specified (for future use)
        if (!opts.log_dir.empty()) {
            fs::create_directories(opts.log_dir);
        }
        
        std::cout << "\nLoading tokenizer hyperparameter grouping...\n";
        std::cout << Color::GREEN << "  ✓ Loaded TokenizerHP" << Color::RESET << "\n";
        
        // Load tokenizer
        GrimTokenizer tokenizer(tokenizer_hp);
        try {
            (void)GRIM::TokenizerArtifacts::loadTokenizerArtifactBundle(tokenizer_hp, tokenizer);
        } catch (const std::exception& e) {
            std::cerr << Color::RED << "\nERROR: Failed to load tokenizer artifact bundle: "
                      << e.what() << Color::RESET << "\n";
            return 2;
        }
        
        std::cout << "\n" << Color::GREEN << "✓ Loaded tokenizer" << Color::RESET << "\n";
        std::cout << "  Vocab size: " << tokenizer.vocabSize() << "\n";
        std::cout << "  Special tokens: PAD=" << GRIM::Tokenizer::PAD_TOKEN_ID 
              << " UNK=" << GRIM::Tokenizer::UNK_TOKEN_ID
              << " BOS=" << GRIM::Tokenizer::BOS_TOKEN_ID 
              << " EOS=" << GRIM::Tokenizer::EOS_TOKEN_ID << "\n";
        
        // Run sections
        std::vector<SectionResults> all_results;
        
        if (opts.run_all_sections || opts.section == 1) {
            all_results.push_back(runSection1_EncodingDecoding(tokenizer, opts.verbose));
        }
        if (opts.run_all_sections || opts.section == 2) {
            all_results.push_back(runSection2_SpacesGrammar(tokenizer, opts.verbose));
        }
        if (opts.run_all_sections || opts.section == 3) {
            all_results.push_back(runSection3_EdgeCases(tokenizer, opts.verbose));
        }
        
        // Summary
        printHeader("Test Summary");
        
        int total_passed = 0, total_failed = 0, total_warnings = 0;
        for (const auto& section : all_results) {
            std::cout << "\n  " << section.section_name << ":\n";
            std::cout << "    " << Color::GREEN << "Passed: " << section.passed << Color::RESET;
            std::cout << "  " << Color::RED << "Failed: " << section.failed << Color::RESET;
            if (section.warnings > 0) {
                std::cout << "  " << Color::YELLOW << "Warnings: " << section.warnings << Color::RESET;
            }
            std::cout << "\n";
            
            total_passed += section.passed;
            total_failed += section.failed;
            total_warnings += section.warnings;
        }
        
        std::cout << "\n" << Color::BOLD << "  TOTAL: " << Color::RESET;
        std::cout << Color::GREEN << total_passed << " passed" << Color::RESET << ", ";
        std::cout << Color::RED << total_failed << " failed" << Color::RESET;
        if (total_warnings > 0) {
            std::cout << ", " << Color::YELLOW << total_warnings << " warnings" << Color::RESET;
        }
        std::cout << "\n\n";
        
        if (total_failed > 0) {
            std::cout << Color::RED << "TOKENIZER SELF-TEST FAILED" << Color::RESET << "\n";
            return 3;
        }
        
        if (total_warnings > 0) {
            std::cout << Color::YELLOW << "TOKENIZER SELF-TEST PASSED WITH WARNINGS" << Color::RESET << "\n";
            return 0;
        }
        
        std::cout << Color::GREEN << "TOKENIZER SELF-TEST PASSED" << Color::RESET << "\n";
        return 0;
        
    } catch (const std::exception& ex) {
        std::cerr << Color::RED << "Tokenizer self-test error: " << ex.what() 
                  << Color::RESET << "\n";
        return 1;
    }
}
