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
#include <optional>
#include <unordered_map>
#include "../Shared/DynaSeqs/DynaSeq_GPU.hpp"
#include "../Shared/UnigramByte/UniByte.hpp"  // For kTextFeatureDim

namespace fs = std::filesystem;

//======================================================//
//  Training Data Structures
//======================================================//

struct TrainingSequence {
    std::vector<int> token_ids;
    std::vector<int> targets;  // Next-token targets
    std::vector<float> token_numeric_values;
    std::vector<uint8_t> token_atom_mask;           // 1 if this position is any atom type
    std::vector<uint16_t> token_text_features;  // [tokens * kTextFeatureDim] FP16
    std::vector<uint32_t> token_atom_flags;          // GRMT v8: per-token AtomTable flags (type-specific metadata)
    std::shared_ptr<GRIM::Tokenizer::AtomTable> atom_table;  // Atom registry (shared across sliding windows)
    std::vector<uint32_t> atom_entry_ids;                    // Per-token index into atom_table (kAtomEntryNone = no atom)
};

//======================================================//
//  Sequence accessor by id (aligned with catalog seq_id)
//======================================================//
struct TrainingSampleView {
    uint32_t seq_id;
    const std::vector<int>* tokens;
    const std::vector<int>* targets;
    const std::vector<float>* token_numeric_values;
    const std::vector<uint8_t>* token_atom_mask;
    const std::vector<uint16_t>* token_text_features;
    const std::vector<uint32_t>* token_atom_flags;
    std::shared_ptr<const GRIM::Tokenizer::AtomTable> atom_table;
    const std::vector<uint32_t>* atom_entry_ids;
};

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
        if (catalog_dirty_) rebuildCatalog();
        // Shuffle sequences and keep catalog aligned by applying the same permutation.
        std::vector<size_t> perm(sequences_.size());
        for (size_t i = 0; i < perm.size(); ++i) perm[i] = i;
        std::shuffle(perm.begin(), perm.end(), rng);

        std::vector<TrainingSequence> shuffled_seq;
        shuffled_seq.reserve(sequences_.size());
        std::vector<GRIM::DynaSeq::SequenceMetadata> shuffled_meta;
        shuffled_meta.reserve(catalog_.entries().size());

        for (size_t new_idx = 0; new_idx < perm.size(); ++new_idx) {
            size_t old_idx = perm[new_idx];
            shuffled_seq.push_back(std::move(sequences_[old_idx]));
            auto meta = catalog_.entries()[old_idx];
            meta.seq_id = static_cast<uint32_t>(new_idx);
            shuffled_meta.push_back(meta);
        }

        sequences_ = std::move(shuffled_seq);
        catalog_override_ = std::move(shuffled_meta);
    }
    
    const std::vector<TrainingSequence>& getSequences() const { return sequences_; }
    size_t size() const { return sequences_.size(); }
    uint32_t vocabSize() const { return vocab_size_; } // Vocab size from training data file

    // Validate sequences against current tokenizer vocab size
    bool validateVocabSize(uint32_t tokenizer_vocab_size, std::ostream& err_stream) const {
        if (vocab_size_ != tokenizer_vocab_size) {
            err_stream << "\n========================================\n";
            err_stream << "FATAL: Vocab size mismatch!\n";
            err_stream << "  Training data (.grmt): " << vocab_size_ << " tokens\n";
            err_stream << "  Current tokenizer:     " << tokenizer_vocab_size << " tokens\n";
            err_stream << "\nThis happens when tokenizer layout or GRMT format changes.\n";
            err_stream << "The .grmt files contain OLD encoded sequences incompatible with NEW tokenizer.\n";
            err_stream << "\n✅ SOLUTION: Delete .grmt files to force regeneration:\n";
            err_stream << "  Remove-Item resources/models/GRIM-text/training/data/*.grmt -Force\n";
            err_stream << "  Then re-run training - auto-prepare will regenerate with new encoding.\n";
            err_stream << "========================================\n";
            return false;
        }
        
        // Validate that all token IDs in sequences are within vocab bounds
        for (size_t i = 0; i < sequences_.size(); ++i) {
            for (int token_id : sequences_[i].token_ids) {
                if (token_id < 0 || static_cast<uint32_t>(token_id) >= vocab_size_) {
                    err_stream << "FATAL: Sequence " << i << " contains out-of-bounds token ID " 
                              << token_id << " (vocab_size=" << vocab_size_ << ")\n";
                    return false;
                }
            }
            // Also validate targets (when not masked)
            for (size_t j = 0; j < sequences_[i].targets.size(); ++j) {
                int target_id = sequences_[i].targets[j];
                if (target_id >= 0 && static_cast<uint32_t>(target_id) >= vocab_size_) {
                    err_stream << "FATAL: Sequence " << i << " position " << j 
                              << " contains out-of-bounds target ID " 
                              << target_id << " (vocab_size=" << vocab_size_ << ")\n";
                    return false;
                }
            }
        }
        return true;
    }

    // Catalog of measured lengths for batching.
    const GRIM::DynaSeq::Catalog& catalog() {
        if (catalog_dirty_) rebuildCatalog();
        return catalog_;
    }

    // Direct sample view by seq_id (after shuffle preserves ordering).
    std::optional<TrainingSampleView> getSample(uint32_t seq_id) {
        if (catalog_dirty_) rebuildCatalog();
        if (seq_id >= sequences_.size()) return std::nullopt;
        return TrainingSampleView{seq_id,
                                  &sequences_[seq_id].token_ids,
                                  &sequences_[seq_id].targets,
                                  &sequences_[seq_id].token_numeric_values,
                                  &sequences_[seq_id].token_atom_mask,
                                  &sequences_[seq_id].token_text_features,
                                  &sequences_[seq_id].token_atom_flags,
                                  sequences_[seq_id].atom_table,
                                  &sequences_[seq_id].atom_entry_ids};
    }
    
private:
    bool loadGRMTFormat(const std::string& path) {
     std::ifstream file(path, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open: " << path << std::endl;
  return false;
   }
        
        uint32_t magic;
   file.read(reinterpret_cast<char*>(&magic), 4);
        if (magic != 0x474D5254) {
    std::cerr << "Invalid GRMT file (magic: 0x" << std::hex << magic << std::dec << ")" << std::endl;
    std::cerr << "If you recently changed tokenizer layout, delete .grmt files and regenerate:" << std::endl;
    std::cerr << "  Remove-Item resources/models/GRIM-text/training/data/*.grmt" << std::endl;
    return false;
        }
      
        uint32_t version, num_sequences, vocab_size;
        file.read(reinterpret_cast<char*>(&version), 4);
      file.read(reinterpret_cast<char*>(&num_sequences), 4);
      file.read(reinterpret_cast<char*>(&vocab_size), 4);
        
        vocab_size_ = vocab_size; // Store vocab size from file
        
        // GRMT v8 required — atom_flags side channel
        if (version != 8) {
            std::cerr << "[DataLoader] FATAL: Unsupported GRMT version " << version
                      << " (required: 8). Delete .grmt files and regenerate training data." << std::endl;
            return false;
        }

        std::cout << "[DataLoader] GRMT version " << version << std::endl;
     std::cout << "[DataLoader] Sequences: " << num_sequences << std::endl;
        std::cout << "[DataLoader] Vocab size: " << vocab_size << std::endl;
        
   sequences_.clear();
      sequences_.reserve(num_sequences);
    size_t nonfinite_total = 0;
    size_t nonfinite_sequences = 0;
        
     for (uint32_t i = 0; i < num_sequences; ++i) {
     uint32_t seq_len;
        file.read(reinterpret_cast<char*>(&seq_len), 4);
            
    TrainingSequence seq;
            seq.token_ids.resize(seq_len);
            seq.targets.resize(seq_len);
            seq.token_numeric_values.resize(seq_len);
            seq.token_atom_mask.resize(seq_len);
            seq.token_text_features.resize(seq_len * GRIM::Tokenizer::kTextFeatureDim);
            seq.token_atom_flags.resize(seq_len);
            // atom_table and atom_entry_ids are built after reading strings below
  
            // Bulk read token_ids (written as int array by DataLoader.cu)
            file.read(reinterpret_cast<char*>(seq.token_ids.data()),
                      seq_len * sizeof(int));
            
            // Read pre-computed targets immediately after token_ids
            file.read(reinterpret_cast<char*>(seq.targets.data()),
                      seq_len * sizeof(int));
            
            if (seq_len > 0) {
                file.read(reinterpret_cast<char*>(seq.token_numeric_values.data()),
                          seq_len * sizeof(float));
                file.read(reinterpret_cast<char*>(seq.token_atom_mask.data()),
                          seq_len * sizeof(uint8_t));
                // GRMT v7: text features (no separate text_mask — atom_mask covers it)
                file.read(reinterpret_cast<char*>(seq.token_text_features.data()),
                          seq_len * GRIM::Tokenizer::kTextFeatureDim * sizeof(uint16_t));
                // GRMT v8: atom_flags (type-specific metadata from AtomTable)
                file.read(reinterpret_cast<char*>(seq.token_atom_flags.data()),
                          seq_len * sizeof(uint32_t));
                // GRMT v6: read atom text strings, then reconstruct AtomTable
                std::vector<std::string> temp_atom_text(seq_len);
                for (uint32_t j = 0; j < seq_len; ++j) {
                    uint16_t slen = 0;
                    file.read(reinterpret_cast<char*>(&slen), sizeof(uint16_t));
                    if (slen > 0) {
                        temp_atom_text[j].resize(slen);
                        file.read(temp_atom_text[j].data(), slen);
                    }
                }
                // Reconstruct AtomTable + atom_entry_ids from stored strings + token IDs
                seq.atom_table = std::make_shared<GRIM::Tokenizer::AtomTable>();
                seq.atom_entry_ids.assign(seq_len, GRIM::Tokenizer::kAtomEntryNone);
                for (uint32_t j = 0; j < seq_len; ++j) {
                    if (!temp_atom_text[j].empty()) {
                        const int tid = seq.token_ids[j];
                        if (tid >= GRIM::Tokenizer::ATOM_TOKEN_OFFSET &&
                            tid < GRIM::Tokenizer::UNIGRAM_VOCAB_OFFSET) {
                            const auto type = GRIM::Tokenizer::tokenIdToAtomType(tid);
                            seq.atom_entry_ids[j] = seq.atom_table->registerAtom(
                                type, temp_atom_text[j]);
                        }
                    }
                }
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

            sequences_.push_back(std::move(seq));
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
        std::cout << "[DataLoader] Atom side-channel stats:\n"
                  << "  Total tokens: " << total_tokens_loaded << "\n"
                  << "  Atom tokens: " << atom_tokens_total
                  << " (" << (total_tokens_loaded > 0
                      ? (100.0 * atom_tokens_total / total_tokens_loaded) : 0.0)
                  << "% of tokens)\n"
                  << "  Sequences with atoms: " << atom_sequences
                  << "/" << sequences_.size() << "\n"
                  << "  AtomTable entries reconstructed: " << atom_entries_total << "\n";
        if (!atom_type_counts.empty()) {
            std::cout << "  Atom type breakdown:\n";
            for (const auto& [tid, count] : atom_type_counts) {
                auto type = GRIM::Tokenizer::tokenIdToAtomType(tid);
                std::cout << "    " << GRIM::Tokenizer::atomTypeName(type)
                          << " (token " << tid << "): " << count << "\n";
            }
        }
        if (atom_tokens_total == 0) {
            std::cerr << "[DataLoader] WARNING: Zero atom tokens in GRMT! "
                      << "Atom detection may not have been enabled during encoding. "
                      << "Delete .grmt files and regenerate with scratch_block_reasoning.enabled=true"
                      << std::endl;
        }
      
        catalog_dirty_ = true;
        return true;
    }

    void rebuildCatalog() {
        catalog_.clear();
        if (!catalog_override_.empty()) {
            // reuse shuffled metadata (seq_ids already re-written)
            for (auto meta : catalog_override_) {
                catalog_.add(meta.seq_length, meta.active_tokens, meta.data_offset, meta.difficulty, meta.source_id);
            }
            catalog_override_.clear();
            catalog_dirty_ = false;
            return;
        }
        for (uint32_t i = 0; i < sequences_.size(); ++i) {
            const auto& seq = sequences_[i];
            const uint32_t len = static_cast<uint32_t>(seq.token_ids.size());
            catalog_.add(len, len, /*data_offset*/0, /*difficulty*/0, /*source_id*/0);
        }
        catalog_dirty_ = false;
    }

    std::vector<TrainingSequence> sequences_;
    GRIM::DynaSeq::Catalog catalog_;
    std::vector<GRIM::DynaSeq::SequenceMetadata> catalog_override_; // used when shuffled
    bool catalog_dirty_ = true;
    uint32_t vocab_size_ = 0; // Vocab size from GRMT file header
};
