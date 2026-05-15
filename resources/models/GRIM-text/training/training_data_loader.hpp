#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <random>
#include <filesystem>
#include <unordered_map>
#include "../Shared/GRMT/GrmtFormat.hpp"
#include "../Shared/UnigramByte/UniByte.hpp"  // Tokenizer metadata and AtomTable
#include "../Shared/Execution/ExecutionMetadata.hpp"
#include "../Shared/TokenizerArtifacts/GrmtCorpusIO.hpp"

namespace fs = std::filesystem;

//======================================================//
//  Training Data Structures
//======================================================//

using TrainingSequence = GRIM::TokenizerArtifacts::GrmtSequence;

//======================================================//
//  Training Data Loader
//======================================================//

class GRMTDataLoader {
public:
bool load(const std::string& path) {
        // Check file extension to determine format
        std::string ext = fs::path(path).extension().string();
    if (ext == ".bin") {
        std::cerr << "[DataLoader] Legacy .bin training data is unsupported; regenerate .grmt files." << std::endl;
        return false;
    }
    return loadGRMTFormat(path);
    }
    
    void shuffle(std::mt19937& rng) {
        std::shuffle(sequences_.begin(), sequences_.end(), rng);
    }
    
    const std::vector<TrainingSequence>& getSequences() const { return sequences_; }
    size_t size() const { return sequences_.size(); }
    uint32_t vocabSize() const { return vocab_size_; } // Vocab size from training data file
    
private:
    bool loadGRMTFormat(const std::string& path) {
        GRIM::TokenizerArtifacts::GrmtCorpus corpus;
        try {
            corpus = GRIM::TokenizerArtifacts::loadGrmtCorpus(path);
        } catch (const std::exception& e) {
            std::cerr << "[DataLoader] Failed to load GRMT corpus: " << e.what() << std::endl;
            return false;
        }

                const GRIM::GRMT::Header header = corpus.header;
                const uint32_t version = header.version;
                const uint32_t num_sequences = header.num_sequences;
                const uint32_t vocab_size = header.vocab_size;
        
        vocab_size_ = vocab_size; // Store vocab size from file

        std::cout << "[DataLoader] GRMT version " << version << std::endl;
     std::cout << "[DataLoader] Sequences: " << num_sequences << std::endl;
        std::cout << "[DataLoader] Vocab size: " << vocab_size << std::endl;
        
        sequences_ = std::move(corpus.sequences);
    size_t nonfinite_total = 0;
    size_t nonfinite_sequences = 0;

        if (sequences_.size() != num_sequences) {
            std::cerr << "[DataLoader] GRMT loaded sequence count mismatch: loaded="
                      << sequences_.size() << " header=" << num_sequences << std::endl;
            return false;
        }

        for (auto& seq : sequences_) {
            const uint32_t seq_len = static_cast<uint32_t>(seq.token_ids.size());
                size_t seq_nonfinite = 0;
                for (uint32_t j = 0; j < seq_len; ++j) {
                    if (seq.token_atom_mask[j] && !std::isfinite(seq.token_numeric_values[j])) {
                        seq.token_numeric_values[j] = 0.0f;
                        ++seq_nonfinite;
                    }
                }
                if (seq_nonfinite > 0) {
                    nonfinite_total += seq_nonfinite;
                    nonfinite_sequences++;
                }
}
        if (nonfinite_total > 0) {
            std::cerr << "[DataLoader] Sanitized " << nonfinite_total
                      << " non-finite numeric values across " << nonfinite_sequences
                      << " sequences (mask cleared)" << std::endl;
        }

        // Atom side-channel diagnostics: report what's actually in the GRMT
        size_t total_tokens_loaded = 0;
        size_t atom_tokens_total = 0;
        size_t atom_sequences = 0;
        std::unordered_map<int, size_t> atom_type_counts;
        size_t atom_entries_total = 0;
        for (const auto& seq : sequences_) {
            total_tokens_loaded += seq.token_ids.size();
            bool seq_has_atoms = false;
            for (size_t j = 0; j < seq.token_ids.size(); ++j) {
                if (j < seq.token_atom_mask.size() && seq.token_atom_mask[j]) {
                    atom_tokens_total++;
                    seq_has_atoms = true;
                    atom_type_counts[seq.token_ids[j]]++;
                }
                if (j < seq.atom_entry_ids.size() &&
                    seq.atom_entry_ids[j] != GRIM::Tokenizer::kAtomEntryNone) {
                    atom_entries_total++;
                }
            }
            if (seq_has_atoms) atom_sequences++;
        }
        std::cerr << "[DataLoader] Atom side-channel stats:" << std::endl
                  << "  Total tokens: " << total_tokens_loaded << std::endl
                  << "  Atom tokens: " << atom_tokens_total
                  << " (" << (total_tokens_loaded > 0
                      ? (100.0 * atom_tokens_total / total_tokens_loaded) : 0.0)
                  << "% of tokens)" << std::endl
                  << "  Sequences with atoms: " << atom_sequences
                  << "/" << sequences_.size() << std::endl
                  << "  AtomTable entries reconstructed: " << atom_entries_total << std::endl;
        if (!atom_type_counts.empty()) {
            std::cerr << "  Atom type breakdown:" << std::endl;
            for (const auto& [tid, count] : atom_type_counts) {
                auto type = GRIM::Tokenizer::tokenIdToAtomType(tid);
                std::cerr << "    " << GRIM::Tokenizer::atomTypeName(type)
                          << " (token " << tid << "): " << count << std::endl;
            }
        }
        if (atom_tokens_total == 0) {
            std::cerr << "[DataLoader] WARNING: Zero atom tokens in GRMT! "
                      << "Atom detection may not have been enabled during encoding. "
                      << "Delete .grmt files and regenerate with scratch_block_reasoning.enabled=true"
                      << std::endl;
        }
        return true;
    }

    std::vector<TrainingSequence> sequences_;
    uint32_t vocab_size_ = 0; // Vocab size from GRMT file header
};
