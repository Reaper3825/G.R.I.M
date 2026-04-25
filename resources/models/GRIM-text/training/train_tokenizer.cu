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
//   train_tokenizer --status-file <path> [--config <ai_config.json>] [--force]
//
// Contract (Rule 20): the process MUST write a status JSON file at
// <status-file> before exiting, regardless of outcome:
//
//   { "outcome": "success", "vocab_size": <uint32> }
//
//   { "outcome": "error", "error_message": "<precise>" }
//
// vocab_path / training_data_path are NOT in the IPC payload — they are
// owned by ai_config.json (GRIM::Config::GrimTextPaths) and the parent
// resolves them from the central config layer. Echoing them over IPC would
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
#include "../Shared/DataLoader/DataLoader.hpp"
#include "Subprocess/subprocess_status_io.hpp" // Foundational status-file IPC. The envelope schema lives there; tokenizer-

namespace fs = std::filesystem;

namespace {

struct CliArgs {
    std::string status_file;
    std::string config_file = "ai_config.json";
    bool force = false;
};

[[noreturn]] void usageError(const std::string& msg) {
    std::cerr << "train_tokenizer: " << msg << "\n"
              << "Usage: train_tokenizer --status-file <path> "
                 "[--config <ai_config.json>] [--force]\n";
    std::exit(2);
}

CliArgs parseArgs(int argc, char** argv) {
    CliArgs out;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--status-file") {
            if (i + 1 >= argc) usageError("--status-file requires a path");
            out.status_file = argv[++i];
        } else if (a == "--config") {
            if (i + 1 >= argc) usageError("--config requires a path");
            out.config_file = argv[++i];
        } else if (a == "--force") {
            out.force = true;
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
    // by ai_config.json and the parent reads them from GrimTextPaths.
    nlohmann::json payload;
    payload["vocab_size"] = vocab_size;
    (void)GRIMText::Subprocess::write_status_success(path, payload);
}

// Read vocab_size from the GRMT header. Throws if the file is missing or its
// header magic is wrong (Rule 20: no silent fallback to a guessed size).
std::uint32_t readGrmtVocabSize(const std::string& grmt_path) {
    std::ifstream f(grmt_path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("cannot open GRMT for header read: " + grmt_path);
    }
    std::uint32_t magic = 0, version = 0, num_sequences = 0, vocab_size = 0;
    f.read(reinterpret_cast<char*>(&magic), 4);
    f.read(reinterpret_cast<char*>(&version), 4);
    f.read(reinterpret_cast<char*>(&num_sequences), 4);
    f.read(reinterpret_cast<char*>(&vocab_size), 4);
    if (!f) {
        throw std::runtime_error("GRMT header truncated: " + grmt_path);
    }
    if (magic != 0x474D5254u) {
        throw std::runtime_error(
            "GRMT magic mismatch (expected 'GRMT' / 0x474D5254): " + grmt_path);
    }
    if (vocab_size == 0) {
        throw std::runtime_error("GRMT header reports vocab_size=0: " + grmt_path);
    }
    return vocab_size;
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
        if (!fs::exists(args.config_file)) {
            throw std::runtime_error(
                "ai_config.json not found at: " + args.config_file);
        }

        GRIM::Config::GrimTextPaths paths;
        if (!GRIM::Config::loadGrimTextPaths(paths, args.config_file)) {
            throw std::runtime_error(
                "could not load GRIM-text paths from: " + args.config_file);
        }
        if (!paths.isValid()) {
            throw std::runtime_error(
                "ai_config.json missing required paths (vocab and/or training_data)");
        }

        paths.printPaths();

        std::string out_training_data = paths.training_data;
        std::string out_vocab = paths.vocab;

        std::cout << "=== GRIM Tokenizer Subprocess ===\n"
                  << "Config:      " << args.config_file << "\n"
                  << "Status file: " << args.status_file << "\n"
                  << "Mode:        " << (args.force ? "FORCE REBUILD" : "build if missing/mismatched") << "\n"
                  << "Vocab path:  " << out_vocab << "\n"
                  << "GRMT path:   " << out_training_data << "\n" << std::endl;

        const bool ok = GRIM::PrepareTrainingDataFromCache(
            paths,
            out_training_data,
            out_vocab,
            args.force);

        if (!ok) {
            throw std::runtime_error(
                "PrepareTrainingDataFromCache returned false (vocab/training_data not produced)");
        }
        if (out_vocab.empty() || !fs::exists(out_vocab)) {
            throw std::runtime_error(
                "PrepareTrainingDataFromCache reported success but vocab file is missing: " + out_vocab);
        }
        if (out_training_data.empty() || !fs::exists(out_training_data)) {
            throw std::runtime_error(
                "PrepareTrainingDataFromCache reported success but GRMT file is missing: " + out_training_data);
        }

        const std::uint32_t vocab_size = readGrmtVocabSize(out_training_data);

        std::cout << "\n=== Tokenizer Subprocess Complete ===\n"
                  << "Vocab:      " << out_vocab << "\n"
                  << "GRMT:       " << out_training_data << "\n"
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
