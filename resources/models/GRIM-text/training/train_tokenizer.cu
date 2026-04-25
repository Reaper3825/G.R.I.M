// Standalone tokenizer training executable.
// Runs the full tokenizer pipeline on the entire corpus:
//   1. Load concept blocks from concept_blocks.jsonl
//   2. Train Unigram+ByteFallback tokenizer (with ▁ normalization)
//   3. Save vocab.bin + vocab.txt
//   4. Encode all sequences → training_data.grmt
//
// Usage:  train_tokenizer [--force]
//   --force   Rebuild even if vocab.bin and .grmt already exist
//
// Reads all paths and config from ai_config.json (same as train_gpu).

#include <iostream>
#include <string>
#include <stdexcept>

// HyperParameters_GPU.hpp is the single entry point for GRIM::Config structs;
// it transitively includes control/ai_config_paths.hpp in the correct order.
#include "../Shared/HyperParameters/HyperParameters_GPU.hpp"
#include "../Shared/DataLoader/DataLoader.hpp"

int main(int argc, char* argv[]) {
    bool force_rebuild = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--force") {
            force_rebuild = true;
        } else {
            std::cerr << "Usage: train_tokenizer [--force]\n";
            return 1;
        }
    }

    // Load paths from ai_config.json
    GRIM::Config::GrimTextPaths paths;
    if (!GRIM::Config::loadGrimTextPaths(paths)) {
        throw std::runtime_error("FATAL: Could not load GRIM-text paths from ai_config.json");
    }
    if (!paths.isValid()) {
        throw std::runtime_error("FATAL: ai_config.json missing required paths (vocab, training_data)");
    }

    paths.printPaths();

    std::string out_training_data = paths.training_data;
    std::string out_vocab = paths.vocab;

    std::cout << "=== GRIM Tokenizer Training ===" << std::endl;
    if (force_rebuild) {
        std::cout << "Mode: FORCE REBUILD (--force)" << std::endl;
    } else {
        std::cout << "Mode: Build if missing or mismatched" << std::endl;
    }
    std::cout << "Vocab path:  " << out_vocab << std::endl;
    std::cout << "GRMT path:   " << out_training_data << std::endl;
    std::cout << std::endl;

    bool ok = GRIM::PrepareTrainingDataFromCache(
        paths,
        out_training_data,
        out_vocab,
        force_rebuild);

    if (!ok) {
        std::cerr << "FATAL: Tokenizer training failed." << std::endl;
        return 1;
    }

    std::cout << "\n=== Tokenizer Training Complete ===" << std::endl;
    std::cout << "Vocab:  " << out_vocab << std::endl;
    std::cout << "GRMT:   " << out_training_data << std::endl;
    return 0;
}
