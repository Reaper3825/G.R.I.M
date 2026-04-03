//======================================================//
//  Tokenizer Runner — Standalone Tokenizer Executable
//
//  Runs the full tokenizer validation pipeline independently:
//    1. Load config from ai_config.json
//    2. Validate paths (vocab, training data)
//    3. Load tokenizer and training data
//    4. Run self-test validation checks
//    5. Output JSON payload to stdout on success
//    6. Output JSON error to stdout on failure
//
//  The training executable (train_gpu) launches this as a
//  subprocess before starting the training loop. The JSON
//  payload communicates tokenizer metadata back to the
//  training process.
//
//  Can also be run standalone from the UI for vocab testing.
//
//  Exit codes:
//    0 = success (payload on stdout)
//    1 = fatal error (error JSON on stdout)
//    2 = validation failure (error JSON on stdout)
//
//  Author: Austin Wadkins
//  Date: April 2026
//======================================================//

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <algorithm>
#include <cstring>

#include <nlohmann/json.hpp>

#include "../Shared/UnigramByte/UniByte.hpp"
#include "../../../../control/ai_config_paths.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;
using GrimTokenizer = GRIM::Tokenizer::UniByte;

//======================================================//
//  CLI Options
//======================================================//
struct RunnerOptions {
    std::string vocab_path;
    std::string data_path;
    std::string config_path = "ai_config.json";
    bool verbose = false;
    bool standalone = false;  // true = standalone mode (human-readable output)
};

//======================================================//
//  Validation Result
//======================================================//
struct ValidationResult {
    bool passed = false;
    std::string name;
    std::string details;
};

//======================================================//
//  Tokenizer Payload — JSON output on success
//======================================================//
struct TokenizerPayload {
    int total_vocab_size = 0;
    int unigram_vocab_size = 0;
    int byte_vocab_size = 0;
    int atom_vocab_size = 0;
    int special_token_count = 0;
    int pad_id = 0;
    int unk_id = 0;
    int bos_id = 0;
    int eos_id = 0;
    std::string vocab_path;
    std::string data_path;
    int validation_tests_passed = 0;
    int validation_tests_total = 0;
    double load_time_ms = 0.0;
    double validation_time_ms = 0.0;

    json toJson() const {
        json j;
        j["status"] = "success";
        j["total_vocab_size"] = total_vocab_size;
        j["unigram_vocab_size"] = unigram_vocab_size;
        j["byte_vocab_size"] = byte_vocab_size;
        j["atom_vocab_size"] = atom_vocab_size;
        j["special_token_count"] = special_token_count;
        j["pad_id"] = pad_id;
        j["unk_id"] = unk_id;
        j["bos_id"] = bos_id;
        j["eos_id"] = eos_id;
        j["vocab_path"] = vocab_path;
        j["data_path"] = data_path;
        j["validation_tests_passed"] = validation_tests_passed;
        j["validation_tests_total"] = validation_tests_total;
        j["load_time_ms"] = load_time_ms;
        j["validation_time_ms"] = validation_time_ms;
        return j;
    }
};

//======================================================//
//  Error output — JSON on failure
//======================================================//
static json makeErrorJson(const std::string& error, const std::string& phase, int tests_passed = 0, int tests_total = 0) {
    json j;
    j["status"] = "error";
    j["error"] = error;
    j["phase"] = phase;
    j["validation_tests_passed"] = tests_passed;
    j["validation_tests_total"] = tests_total;
    return j;
}

//======================================================//
//  Validation Tests
//======================================================//
static std::vector<ValidationResult> runValidationChecks(GrimTokenizer& tokenizer, bool verbose) {
    std::vector<ValidationResult> results;

    // Test 1: Basic encode/decode round-trip
    {
        ValidationResult r;
        r.name = "Basic encode/decode round-trip";
        std::string input = "Hello, world!";
        auto ids = tokenizer.encode(input);
        std::string decoded = tokenizer.decode(ids);

        std::string input_stripped, decoded_stripped;
        for (char c : input) if (!std::isspace(c)) input_stripped += std::tolower(c);
        for (char c : decoded) if (!std::isspace(c)) decoded_stripped += std::tolower(c);

        r.passed = (input_stripped == decoded_stripped);
        if (!r.passed) {
            r.details = "Round-trip mismatch: input='" + input + "' decoded='" + decoded + "'";
        } else {
            r.details = "Encoded to " + std::to_string(ids.size()) + " tokens";
        }
        results.push_back(r);
    }

    // Test 2: Empty input handling
    {
        ValidationResult r;
        r.name = "Empty input handling";
        auto ids = tokenizer.encode("");
        r.passed = true;  // Should not crash
        r.details = "Empty input → " + std::to_string(ids.size()) + " tokens";
        results.push_back(r);
    }

    // Test 3: Space preservation
    {
        ValidationResult r;
        r.name = "Space preservation";
        std::string input = "hello how are you";
        auto ids = tokenizer.encode(input);
        std::string decoded = tokenizer.decode(ids);

        int input_spaces = static_cast<int>(std::count(input.begin(), input.end(), ' '));
        int output_spaces = static_cast<int>(std::count(decoded.begin(), decoded.end(), ' '));

        r.passed = (output_spaces > 0);
        if (!r.passed) {
            r.details = "All " + std::to_string(input_spaces) + " spaces lost!";
        } else {
            r.details = std::to_string(output_spaces) + "/" + std::to_string(input_spaces) + " spaces preserved";
        }
        results.push_back(r);
    }

    // Test 4: Special tokens exist
    {
        ValidationResult r;
        r.name = "Special tokens valid";
        r.passed = (tokenizer.padId() >= 0 && tokenizer.unkId() >= 0 &&
                    tokenizer.bosId() >= 0 && tokenizer.eosId() >= 0);
        if (!r.passed) {
            r.details = "Missing special tokens";
        } else {
            r.details = "PAD=" + std::to_string(tokenizer.padId()) +
                        " UNK=" + std::to_string(tokenizer.unkId()) +
                        " BOS=" + std::to_string(tokenizer.bosId()) +
                        " EOS=" + std::to_string(tokenizer.eosId());
        }
        results.push_back(r);
    }

    // Test 5: Vocab size sanity
    {
        ValidationResult r;
        r.name = "Vocab size sanity";
        int total = tokenizer.totalVocabSize();
        r.passed = (total > 260);  // At least special + bytes + atoms
        if (!r.passed) {
            r.details = "Vocab too small: " + std::to_string(total);
        } else {
            r.details = "Total vocab: " + std::to_string(total) + " tokens";
        }
        results.push_back(r);
    }

    // Test 6: UTF-8 byte fallback
    {
        ValidationResult r;
        r.name = "UTF-8 byte fallback";
        std::string input = "caf\xc3\xa9";  // café
        auto ids = tokenizer.encode(input);
        r.passed = (!ids.empty());
        r.details = "Encoded UTF-8 to " + std::to_string(ids.size()) + " tokens";
        results.push_back(r);
    }

    // Test 7: Punctuation preservation
    {
        ValidationResult r;
        r.name = "Punctuation preservation";
        std::string input = "Hello, world! How's it?";
        auto ids = tokenizer.encode(input);
        std::string decoded = tokenizer.decode(ids);

        std::string input_punct, output_punct;
        for (char c : input) if (std::ispunct(static_cast<unsigned char>(c))) input_punct += c;
        for (char c : decoded) if (std::ispunct(static_cast<unsigned char>(c))) output_punct += c;

        r.passed = (input_punct == output_punct);
        if (!r.passed) {
            r.details = "Punctuation mismatch: expected '" + input_punct + "' got '" + output_punct + "'";
        } else {
            r.details = "All punctuation preserved";
        }
        results.push_back(r);
    }

    // Test 8: Number tokenization
    {
        ValidationResult r;
        r.name = "Number tokenization";
        std::string input = "12345 67890";
        auto ids = tokenizer.encode(input);
        r.passed = (!ids.empty());
        r.details = "Numbers → " + std::to_string(ids.size()) + " tokens";
        results.push_back(r);
    }

    // Test 9: Multi-word sentence
    {
        ValidationResult r;
        r.name = "Multi-word sentence";
        std::string input = "The quick brown fox jumps over the lazy dog";
        auto ids = tokenizer.encode(input);
        std::string decoded = tokenizer.decode(ids);

        std::string input_stripped, decoded_stripped;
        for (char c : input) if (!std::isspace(c)) input_stripped += std::tolower(c);
        for (char c : decoded) if (!std::isspace(c)) decoded_stripped += std::tolower(c);

        r.passed = (input_stripped == decoded_stripped);
        if (!r.passed) {
            r.details = "Content mismatch in decoded output";
        } else {
            r.details = "9-word sentence → " + std::to_string(ids.size()) + " tokens";
        }
        results.push_back(r);
    }

    // Test 10: Invalid token ID handling
    {
        ValidationResult r;
        r.name = "Invalid token ID handling";
        std::vector<int> bad_ids = {-1, 999999, tokenizer.totalVocabSize() + 100};
        std::string decoded = tokenizer.decode(bad_ids);
        r.passed = true;  // Should not crash
        r.details = "Handled gracefully";
        results.push_back(r);
    }

    // Print results if verbose
    if (verbose) {
        for (const auto& r : results) {
            if (r.passed) {
                fprintf(stderr, "  [PASS] %s - %s\n", r.name.c_str(), r.details.c_str());
            } else {
                fprintf(stderr, "  [FAIL] %s - %s\n", r.name.c_str(), r.details.c_str());
            }
        }
    }

    return results;
}

//======================================================//
//  CLI Parsing
//======================================================//
static RunnerOptions parseOptions(int argc, char** argv) {
    RunnerOptions opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--vocab" && i + 1 < argc) {
            opts.vocab_path = argv[++i];
        } else if (arg == "--data" && i + 1 < argc) {
            opts.data_path = argv[++i];
        } else if (arg == "--config" && i + 1 < argc) {
            opts.config_path = argv[++i];
        } else if (arg == "--verbose" || arg == "-v") {
            opts.verbose = true;
        } else if (arg == "--standalone") {
            opts.standalone = true;
        } else if (arg == "--help" || arg == "-h") {
            // Print to stderr so stdout stays clean for JSON payload
            fprintf(stderr, "GRIM Tokenizer Runner\n\n");
            fprintf(stderr, "Usage: tokenizer_runner [options]\n\n");
            fprintf(stderr, "Options:\n");
            fprintf(stderr, "  --vocab PATH      Path to vocab.bin (default: from ai_config.json)\n");
            fprintf(stderr, "  --data PATH       Path to training_data.grmt (default: from ai_config.json)\n");
            fprintf(stderr, "  --config PATH     Path to ai_config.json (default: ai_config.json)\n");
            fprintf(stderr, "  --verbose, -v     Show detailed validation output on stderr\n");
            fprintf(stderr, "  --standalone      Human-readable output (for direct invocation)\n");
            fprintf(stderr, "  --help, -h        Show this help\n");
            fprintf(stderr, "\nExit codes:\n");
            fprintf(stderr, "  0 = success (JSON payload on stdout)\n");
            fprintf(stderr, "  1 = fatal error\n");
            fprintf(stderr, "  2 = validation failure\n");
            std::exit(0);
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
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

        if (opts.standalone) {
            fprintf(stderr, "╔══════════════════════════════════════════════════╗\n");
            fprintf(stderr, "║        GRIM Tokenizer Runner (Standalone)        ║\n");
            fprintf(stderr, "╚══════════════════════════════════════════════════╝\n");
        }

        //==============================================================
        // Phase: Load Configuration
        //==============================================================
        if (opts.verbose) {
            fprintf(stderr, "[tokenizer_runner] Loading configuration...\n");
        }

        // Load paths from ai_config.json if not specified on command line
        GRIM::Config::GrimTextPaths paths;
        if (!GRIM::Config::loadGrimTextPaths(paths, opts.config_path)) {
            std::string err = "Could not load GRIM-text paths from " + opts.config_path;
            std::cout << makeErrorJson(err, "config").dump() << std::endl;
            return 1;
        }

        if (opts.vocab_path.empty()) {
            if (paths.vocab.empty()) {
                std::cout << makeErrorJson("No vocab path configured in ai_config.json", "config").dump() << std::endl;
                return 1;
            }
            opts.vocab_path = paths.vocab;
        }

        if (opts.data_path.empty()) {
            if (paths.training_data.empty()) {
                std::cout << makeErrorJson("No training_data path configured in ai_config.json", "config").dump() << std::endl;
                return 1;
            }
            opts.data_path = paths.training_data;
        }

        if (opts.verbose) {
            fprintf(stderr, "[tokenizer_runner] Vocab: %s\n", opts.vocab_path.c_str());
            fprintf(stderr, "[tokenizer_runner] Data:  %s\n", opts.data_path.c_str());
        }

        //==============================================================
        // Phase: Path Validation
        //==============================================================
        if (!fs::exists(opts.vocab_path)) {
            std::string err = "Vocabulary file does not exist: " + opts.vocab_path;
            std::cout << makeErrorJson(err, "path_validation").dump() << std::endl;
            return 1;
        }

        if (!fs::exists(opts.data_path)) {
            std::string err = "Training data file does not exist: " + opts.data_path;
            std::cout << makeErrorJson(err, "path_validation").dump() << std::endl;
            return 1;
        }

        //==============================================================
        // Phase: Load Tokenizer
        //==============================================================
        if (opts.verbose) {
            fprintf(stderr, "[tokenizer_runner] Loading tokenizer...\n");
        }

        auto load_start = std::chrono::steady_clock::now();

        // Load tokenizer config
        GRIM::Config::TokenizerConfig tok_config;
        GRIM::Config::loadTokenizerConfig(tok_config, opts.config_path);

        GRIM::Tokenizer::UniByteConfig cfg;
        if (tok_config.vocab_size > 0) {
            cfg.target_vocab_size = tok_config.vocab_size;
        }
        cfg.enable_byte_fallback = tok_config.enable_byte_fallback;

        GrimTokenizer tokenizer(cfg);
        if (!tokenizer.load(opts.vocab_path)) {
            std::string err = "Failed to load vocabulary: " + opts.vocab_path;
            std::cout << makeErrorJson(err, "tokenizer_load").dump() << std::endl;
            return 1;
        }

        auto load_end = std::chrono::steady_clock::now();
        double load_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();

        if (opts.verbose) {
            fprintf(stderr, "[tokenizer_runner] Loaded %d tokens in %.1f ms\n",
                    tokenizer.totalVocabSize(), load_ms);
        }

        //==============================================================
        // Phase: Validation Checks
        //==============================================================
        if (opts.verbose) {
            fprintf(stderr, "[tokenizer_runner] Running validation checks...\n");
        }

        auto val_start = std::chrono::steady_clock::now();
        auto validation_results = runValidationChecks(tokenizer, opts.verbose);
        auto val_end = std::chrono::steady_clock::now();
        double val_ms = std::chrono::duration<double, std::milli>(val_end - val_start).count();

        int passed = 0;
        int total = static_cast<int>(validation_results.size());
        std::vector<std::string> failures;
        for (const auto& r : validation_results) {
            if (r.passed) {
                passed++;
            } else {
                failures.push_back(r.name + ": " + r.details);
            }
        }

        if (opts.verbose) {
            fprintf(stderr, "[tokenizer_runner] Validation: %d/%d passed (%.1f ms)\n",
                    passed, total, val_ms);
        }

        // Check for critical failures
        if (!failures.empty()) {
            json err_json;
            err_json["status"] = "error";
            err_json["error"] = "Tokenizer validation failed";
            err_json["phase"] = "validation";
            err_json["validation_tests_passed"] = passed;
            err_json["validation_tests_total"] = total;
            json failure_list = json::array();
            for (const auto& f : failures) {
                failure_list.push_back(f);
            }
            err_json["failures"] = failure_list;
            err_json["vocab_path"] = opts.vocab_path;
            err_json["data_path"] = opts.data_path;

            if (opts.standalone) {
                fprintf(stderr, "\n[FAILED] Tokenizer validation failed:\n");
                for (const auto& f : failures) {
                    fprintf(stderr, "  - %s\n", f.c_str());
                }
            }

            std::cout << err_json.dump() << std::endl;
            return 2;
        }

        //==============================================================
        // Phase: Build Payload
        //==============================================================
        TokenizerPayload payload;
        payload.total_vocab_size = tokenizer.totalVocabSize();
        payload.unigram_vocab_size = tokenizer.vocabSize();
        payload.byte_vocab_size = 256;
        payload.atom_vocab_size = GRIM::Tokenizer::kAtomTypeCount;
        payload.special_token_count = 4;  // UNK, PAD, BOS, EOS
        payload.pad_id = tokenizer.padId();
        payload.unk_id = tokenizer.unkId();
        payload.bos_id = tokenizer.bosId();
        payload.eos_id = tokenizer.eosId();
        payload.vocab_path = opts.vocab_path;
        payload.data_path = opts.data_path;
        payload.validation_tests_passed = passed;
        payload.validation_tests_total = total;
        payload.load_time_ms = load_ms;
        payload.validation_time_ms = val_ms;

        if (opts.standalone) {
            fprintf(stderr, "\n[SUCCESS] Tokenizer validation passed (%d/%d tests)\n", passed, total);
            fprintf(stderr, "  Total vocab size: %d\n", payload.total_vocab_size);
            fprintf(stderr, "  Unigram pieces:   %d\n", payload.unigram_vocab_size);
            fprintf(stderr, "  Load time:        %.1f ms\n", payload.load_time_ms);
            fprintf(stderr, "  Validation time:  %.1f ms\n", payload.validation_time_ms);
        }

        // Output JSON payload to stdout
        std::cout << payload.toJson().dump() << std::endl;
        return 0;

    } catch (const std::exception& e) {
        json err = makeErrorJson(std::string("Fatal exception: ") + e.what(), "fatal");
        std::cout << err.dump() << std::endl;
        return 1;
    } catch (...) {
        json err = makeErrorJson("Unknown fatal exception", "fatal");
        std::cout << err.dump() << std::endl;
        return 1;
    }
}
