// Standalone tokenizer training subprocess.
//
// Runs the full tokenizer pipeline on the entire corpus:
//   1. Load concept blocks from concept_blocks.jsonl
//   2. Train Unigram+ByteFallback tokenizer (with ▁ normalization)
//   3. Save vocab.bin + vocab.txt
//   4. Encode all sequences → training_data.grmt
//
// Invoked by GRIMText::Subprocess::run_tokenizer_subprocess.
//
// Usage:
//   train_tokenizer --status-file <path>
//
// Contract (Rule 20): the process MUST write a status JSON file at
// <status-file> before exiting, regardless of outcome:
//
//   { "outcome": "success", "vocab_size": <uint32> }
//
//   { "outcome": "error", "error_message": "<precise>" }
//
// vocab_path / training_data_path are NOT in the IPC payload — they are
// owned by TokenizerHP and the parent resolves them from the same
// hyperparameter grouping. Echoing them over IPC would
// create a second source of truth.
//
// The parent process refuses to proceed when the status file is missing or
// malformed, so all error paths funnel through writeStatusError().

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <nlohmann/json.hpp>

// HyperParameters_GPU.hpp is the single entry point for GRIM::Config structs;
// it transitively includes control/ai_config_paths.hpp in the correct order.
#include "../Shared/HyperParameters/HyperParameters_GPU.hpp"
#include "../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../Shared/GRMT/GrmtFormat.hpp"
#include "../Shared/DataLoader/DataLoader.hpp"
#include "Subprocess/subprocess_status_io.hpp" // Foundational status-file IPC. The envelope schema lives there; tokenizer-

namespace fs = std::filesystem;

namespace {

struct CliArgs {
    std::string status_file;
};

[[noreturn]] void usageError(const std::string& msg) {
    std::cerr << "train_tokenizer: " << msg << "\n"
              << "Usage: train_tokenizer --status-file <path>\n";
    std::exit(2);
}

CliArgs parseArgs(int argc, char** argv) {
    CliArgs out;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--status-file") {
            if (i + 1 >= argc) usageError("--status-file requires a path");
            out.status_file = argv[++i];
        } else {
            usageError("unknown argument: " + a);
        }
    }
    if (out.status_file.empty()) {
        usageError("--status-file is required (this is the parent IPC channel)");
    }
    return out;
}

// Atomic status-file writers and the schema itself live in
// Subprocess/subprocess_status_io.{hpp,cpp}. This file MUST NOT define them.

void writeStatusError(const std::string& path, const std::string& message) {
    (void)GRIMText::Subprocess::write_status_error(path, message);
}

void writeStatusSuccess(const std::string& path, std::uint32_t vocab_size) {
    // The IPC envelope is generic; tokenizer-specific fields go inside the
    // success payload. Only `vocab_size` crosses the wire — paths are owned
    // by TokenizerHP and the parent reads them from the same grouping.
    nlohmann::json payload;
    payload["vocab_size"] = vocab_size;
    (void)GRIMText::Subprocess::write_status_success(path, payload);
}

} // namespace

int main(int argc, char** argv) {
    CliArgs args;
    try {
        args = parseArgs(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "train_tokenizer: argument parsing failed: " << e.what() << "\n";
        return 2;
    }

    try {
        const auto startup_config = GRIM::HyperParameters::loadStartupConfig(argc, argv);
        const auto tokenizer_hp = GRIM::HyperParameters::tokenizerHP(startup_config);

        if (tokenizer_hp.data_path.empty() || tokenizer_hp.vocab_path.empty()) {
            throw std::runtime_error(
            "TokenizerHP missing required tokenizer artifact paths (vocab and/or training_data)");
        }

        std::cout << "=== GRIM Tokenizer Subprocess ===\n"
                  << "Status file: " << args.status_file << "\n"
                  << "Mode:        " << (tokenizer_hp.force_rebuild_vocab ? "FORCE REBUILD" : "build if missing/mismatched") << "\n"
                  << "Vocab path:  " << tokenizer_hp.vocab_path << "\n"
                  << "GRMT path:   " << tokenizer_hp.data_path << "\n" << std::endl;

        const bool ok = GRIM::PrepareTrainingDataFromCache(tokenizer_hp);

        if (!ok) {
            throw std::runtime_error(
                "PrepareTrainingDataFromCache returned false (vocab/training_data not produced)");
        }
        if (tokenizer_hp.vocab_path.empty() || !fs::exists(tokenizer_hp.vocab_path)) {
            throw std::runtime_error(
            "PrepareTrainingDataFromCache reported success but vocab file is missing: " + tokenizer_hp.vocab_path);
        }
        if (tokenizer_hp.data_path.empty() || !fs::exists(tokenizer_hp.data_path)) {
            throw std::runtime_error(
            "PrepareTrainingDataFromCache reported success but GRMT file is missing: " + tokenizer_hp.data_path);
        }

        const std::uint32_t vocab_size = GRIM::GRMT::readHeaderOrThrow(tokenizer_hp.data_path).vocab_size;

        std::cout << "\n=== Tokenizer Subprocess Complete ===\n"
              << "Vocab:      " << tokenizer_hp.vocab_path << "\n"
              << "GRMT:       " << tokenizer_hp.data_path << "\n"
                  << "Vocab size: " << vocab_size << std::endl;

        writeStatusSuccess(args.status_file, vocab_size);
        return 0;
    } catch (const std::exception& e) {
        const std::string msg =
            std::string("train_tokenizer subprocess failed: ") + e.what();
        std::cerr << msg << std::endl;
        writeStatusError(args.status_file, msg);
        return 1;
    } catch (...) {
        const std::string msg = "train_tokenizer subprocess failed: unknown C++ exception";
        std::cerr << msg << std::endl;
        writeStatusError(args.status_file, msg);
        return 1;
    }
}
