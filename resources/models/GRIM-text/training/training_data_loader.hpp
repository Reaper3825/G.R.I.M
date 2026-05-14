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
#include "../Common/grim_model_serialization_version.hpp"
#include "../Shared/UnigramByte/UniByte.hpp"  // Tokenizer metadata and AtomTable
#include "../Shared/Execution/ExecutionMetadata.hpp"

namespace fs = std::filesystem;

//======================================================//
//  Training Data Structures
//======================================================//

struct TrainingSequence {
    std::vector<int> token_ids;
    std::vector<int> targets;  // Next-token targets
    std::vector<float> token_numeric_values;
    std::vector<uint8_t> token_atom_mask;           // 1 if this position is any atom type
    std::vector<uint32_t> token_atom_flags;          // GRMT v8: per-token AtomTable flags (type-specific metadata)
    std::shared_ptr<GRIM::Tokenizer::AtomTable> atom_table;  // Atom registry (shared across sliding windows)
    std::vector<uint32_t> atom_entry_ids;                    // Per-token index into atom_table (kAtomEntryNone = no atom)
    std::vector<int32_t> token_exec_slots;                   // Per-token slot_id for execution: >=0 = valid slot, -1 = non-state-bearing

    // ═══════════════════════════════════════════════════════════════════════════
    // COMPILED STRUCTURED-EXECUTION PAYLOAD
    // Derived from StructuredExecutionRecord by the canonical builder.
    // execution_active is the AUTHORITATIVE activation bit.
    // token_exec_slots (above) is the runtime binding projection.
    // teacher_steps is the supervision projection.
    // Both are paired projections of one canonical record.
    // Runtime D_row is reconstructed from compiled_bootstrap_bindings ∪ teacher_steps.
    // ═══════════════════════════════════════════════════════════════════════════
    bool execution_active = false;
    std::vector<GRIM::Execution::CompiledBootstrapBinding> compiled_bootstrap_bindings;
    std::vector<GRIM::Execution::TeacherStep> teacher_steps;
    std::vector<GRIM::Execution::SlotSelectionTarget> slot_selection_targets;
};

//======================================================//
//  Sequence accessor by id
//======================================================//
struct TrainingSampleView {
    uint32_t seq_id;
    const std::vector<int>* tokens;
    const std::vector<int>* targets;
    const std::vector<float>* token_numeric_values;
    const std::vector<uint8_t>* token_atom_mask;
    const std::vector<uint32_t>* token_atom_flags;
    std::shared_ptr<const GRIM::Tokenizer::AtomTable> atom_table;
    const std::vector<uint32_t>* atom_entry_ids;

    // Compiled structured-execution payload access.
    // execution_active is the authoritative activation bit.
    // Downstream batching/validation consumes these without re-reading the source.
    bool execution_active = false;
    const std::vector<int32_t>* token_exec_slots = nullptr;
    const std::vector<GRIM::Execution::TeacherStep>* teacher_steps = nullptr;
    const std::vector<GRIM::Execution::CompiledBootstrapBinding>* compiled_bootstrap_bindings = nullptr;
    const std::vector<GRIM::Execution::SlotSelectionTarget>* slot_selection_targets = nullptr;
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
        std::shuffle(sequences_.begin(), sequences_.end(), rng);
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

    // Direct sample view by seq_id in the current sequence order.
    std::optional<TrainingSampleView> getSample(uint32_t seq_id) {
        if (seq_id >= sequences_.size()) return std::nullopt;
        const auto& seq = sequences_[seq_id];
        return TrainingSampleView{seq_id,
                                  &seq.token_ids,
                                  &seq.targets,
                                  &seq.token_numeric_values,
                                  &seq.token_atom_mask,
                                  &seq.token_atom_flags,
                                  seq.atom_table,
                                  &seq.atom_entry_ids,
                                  seq.execution_active,
                                  &seq.token_exec_slots,
                                  &seq.teacher_steps,
                                  &seq.compiled_bootstrap_bindings,
                                  &seq.slot_selection_targets};
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
        
        // GRMT format version must match GRMT_FORMAT_VERSION (single source of truth in grim_model_serialization_version.hpp)
        if (version != GRIM::GRMT_FORMAT_VERSION) {
            std::cerr << "[DataLoader] FATAL: Unsupported GRMT version " << version
                      << " (required: " << GRIM::GRMT_FORMAT_VERSION << "). Delete .grmt files and regenerate training data." << std::endl;
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

                // GRMT v11: compiled structured-execution payload
                // Order follows plan: exec_active, token_exec_slots, then bindings/steps/targets
                {
                    uint8_t exec_active = 0;
                    file.read(reinterpret_cast<char*>(&exec_active), sizeof(uint8_t));
                    seq.execution_active = (exec_active != 0);

                    seq.token_exec_slots.resize(seq_len);
                    file.read(reinterpret_cast<char*>(seq.token_exec_slots.data()),
                              seq_len * sizeof(int32_t));

                    // Compiled bootstrap bindings
                    uint32_t cbb_count = 0;
                    file.read(reinterpret_cast<char*>(&cbb_count), sizeof(uint32_t));
                    static_assert(sizeof(GRIM::Execution::CompiledBootstrapBinding) == 12,
                        "CompiledBootstrapBinding must be 12 bytes for bulk GRMT deserialization");
                    if (cbb_count > 0) {
                        seq.compiled_bootstrap_bindings.resize(cbb_count);
                        file.read(reinterpret_cast<char*>(seq.compiled_bootstrap_bindings.data()),
                                  cbb_count * sizeof(GRIM::Execution::CompiledBootstrapBinding));
                    }

                    // Teacher steps
                    uint32_t ts_count = 0;
                    file.read(reinterpret_cast<char*>(&ts_count), sizeof(uint32_t));
                    static_assert(sizeof(GRIM::Execution::TeacherStep) == 20,
                        "TeacherStep must be 20 bytes for bulk GRMT deserialization");
                    if (ts_count > 0) {
                        seq.teacher_steps.resize(ts_count);
                        file.read(reinterpret_cast<char*>(seq.teacher_steps.data()),
                                  ts_count * sizeof(GRIM::Execution::TeacherStep));
                    }

                    // Slot selection targets (field-by-field due to struct padding)
                    uint32_t sst_count = 0;
                    file.read(reinterpret_cast<char*>(&sst_count), sizeof(uint32_t));
                    if (sst_count > 0) {
                        seq.slot_selection_targets.resize(sst_count);
                        for (uint32_t si = 0; si < sst_count; ++si) {
                            uint8_t kind = 0;
                            file.read(reinterpret_cast<char*>(&kind), sizeof(uint8_t));
                            seq.slot_selection_targets[si].kind =
                                static_cast<GRIM::Execution::SlotSelectionTargetKind>(kind);
                            file.read(reinterpret_cast<char*>(&seq.slot_selection_targets[si].slot_id),
                                      sizeof(int32_t));
                        }
                    }
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
