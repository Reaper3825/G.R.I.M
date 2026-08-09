//======================================================//
//  UniByte.hpp
//  Unified Unigram LM + Byte Fallback Tokenizer for GRIM
//  
//  This orchestrator combines:
//  - Unigram LM for high-quality subword tokenization
//  - Byte fallback for 100% coverage
//  - Structural detection for atom injection
//  - Placeholder system for atom reasoning integration
//  
//  Token ID Layout:
//    [0-3]                    = Special tokens (<unk>, <pad>, <s>, </s>)
//    [4-259]                  = Byte tokens (fallback)
//    [ATOM_TOKEN_OFFSET..UNIGRAM_VOCAB_OFFSET-1] = Atom tokens (structural placeholders)
//    [UNIGRAM_VOCAB_OFFSET+]  = Unigram vocabulary (regular pieces only)
//  
//  Author: GRIM Team
//  Date: December 2025
//======================================================//

#pragma once

#include "AtomTable.hpp"
#include "Detectors/DetectorRegistry.hpp"
#include "Detectors/StructuralSpan.hpp"
#include "TokenLayout.hpp"
#include "Unigram.hpp"
#include "../HyperParameters/HyperparameterGroupings.hpp"

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace GRIM {
namespace Tokenizer {

//======================================================//
//  Encoded Result with Metadata
//======================================================//
struct UniByteResult {
    std::vector<int> token_ids;
    std::vector<StructuralSpan> atoms;          // Detected structures
    std::vector<bool> is_byte_fallback;         // Per-token: was byte fallback used?
    std::vector<float> token_numeric_values;    // Per-token packed value from AtomTable (all atom types, 0 if none)
    std::vector<uint32_t> token_atom_flags;     // Per-token type-specific flags from AtomTable (0 if not atom)
    std::vector<uint8_t> token_atom_mask;       // Per-token atom mask (1 if token is any atom type)
    std::shared_ptr<AtomTable> atom_table;       // Per-sequence atom registry (shared across windows)
    std::vector<uint32_t> atom_entry_ids;        // Per-token index into atom_table (kAtomEntryNone = no atom)
    size_t unigram_tokens = 0;
    size_t byte_tokens = 0;
    size_t atom_tokens = 0;
    
    // ═══════════════════════════════════════════════════════════════════════════
    // Pipeline validation: ensures all per-token arrays are consistent before
    // the result enters the batching/tensor pipeline.  Rule 20: crash on mismatch.
    // ═══════════════════════════════════════════════════════════════════════════
    void validate(const char* caller) const {
        const size_t n = token_ids.size();
        if (is_byte_fallback.size() != n) {
            throw std::runtime_error(
                std::string(caller) + ": UniByteResult.is_byte_fallback.size()=" +
                std::to_string(is_byte_fallback.size()) + " != token_ids.size()=" +
                std::to_string(n));
        }
        if (token_numeric_values.size() != n) {
            throw std::runtime_error(
                std::string(caller) + ": UniByteResult.token_numeric_values.size()=" +
                std::to_string(token_numeric_values.size()) + " != token_ids.size()=" +
                std::to_string(n));
        }
        if (token_atom_flags.size() != n) {
            throw std::runtime_error(
                std::string(caller) + ": UniByteResult.token_atom_flags.size()=" +
                std::to_string(token_atom_flags.size()) + " != token_ids.size()=" +
                std::to_string(n));
        }
        if (token_atom_mask.size() != n) {
            throw std::runtime_error(
                std::string(caller) + ": UniByteResult.token_atom_mask.size()=" +
                std::to_string(token_atom_mask.size()) + " != token_ids.size()=" +
                std::to_string(n));
        }
        if (atom_entry_ids.size() != n) {
            throw std::runtime_error(
                std::string(caller) + ": UniByteResult.atom_entry_ids.size()=" +
                std::to_string(atom_entry_ids.size()) + " != token_ids.size()=" +
                std::to_string(n));
        }
        if (unigram_tokens + byte_tokens + atom_tokens != n) {
            throw std::runtime_error(
                std::string(caller) + ": UniByteResult token count mismatch: unigram=" +
                std::to_string(unigram_tokens) + " + byte=" +
                std::to_string(byte_tokens) + " + atom=" +
                std::to_string(atom_tokens) + " != total=" +
                std::to_string(n));
        }
    }
    
    // Total token count
    size_t size() const { return token_ids.size(); }
    bool empty() const { return token_ids.empty(); }
};

//======================================================//
//  Decode Request
//======================================================//
struct DecodeRequest {
    std::vector<int> owned_token_ids;
    const int* token_ids = nullptr;
    size_t token_count = 0;
    const uint32_t* atom_entry_ids = nullptr;
    size_t atom_entry_count = 0;
    const float* token_numeric_values = nullptr;
    size_t token_numeric_count = 0;
    const uint8_t* token_atom_mask = nullptr;
    size_t token_atom_mask_count = 0;
    const AtomTable* atom_table = nullptr;

    // When true, invalid UTF-8 in byte-fallback runs is replaced with the
    // Unicode replacement character (U+FFFD) instead of throwing. Use for
    // inspecting raw/partial model output; strict validation stays the default.
    bool lenient_invalid_utf8 = true;

    explicit DecodeRequest(const std::vector<int>& ids)
        : token_ids(ids.data()), token_count(ids.size()) {}

    explicit DecodeRequest(std::initializer_list<int> ids)
        : owned_token_ids(ids),
          token_ids(owned_token_ids.data()),
          token_count(owned_token_ids.size()) {}

    explicit DecodeRequest(const UniByteResult& result)
        : token_ids(result.token_ids.data()),
          token_count(result.token_ids.size()),
          atom_entry_ids(result.atom_entry_ids.data()),
          atom_entry_count(result.atom_entry_ids.size()),
          token_numeric_values(result.token_numeric_values.data()),
          token_numeric_count(result.token_numeric_values.size()),
          token_atom_mask(result.token_atom_mask.data()),
          token_atom_mask_count(result.token_atom_mask.size()),
          atom_table(result.atom_table.get()) {}

    explicit DecodeRequest(const std::vector<int>& ids,
                           const std::vector<uint32_t>& entry_ids,
                           const AtomTable* table,
                           const std::vector<float>& numeric_values,
                           const std::vector<uint8_t>& atom_mask)
        : token_ids(ids.data()),
          token_count(ids.size()),
          atom_entry_ids(entry_ids.data()),
          atom_entry_count(entry_ids.size()),
          token_numeric_values(numeric_values.data()),
          token_numeric_count(numeric_values.size()),
          token_atom_mask(atom_mask.data()),
          token_atom_mask_count(atom_mask.size()),
          atom_table(table) {}

    DecodeRequest(const DecodeRequest&) = delete;
    DecodeRequest& operator=(const DecodeRequest&) = delete;
};

//  UniByte - Main Orchestrator
//======================================================//
class UniByte {
public:
    explicit UniByte(const ::GRIM::HyperParameters::TokenizerHP& hp);
    ~UniByte();

    // Disable copy
    UniByte(const UniByte&) = delete;
    UniByte& operator=(const UniByte&) = delete;

    // Move support
    UniByte(UniByte&&) noexcept;
    UniByte& operator=(UniByte&&) noexcept;

    //--------------------------------------------------//
    // Initialization
    //--------------------------------------------------//

    // Initialize GPU resources
    bool initGPU();
    const UnigramTrainingRuntimeReport& lastTrainingRuntimeReport() const;

    //--------------------------------------------------//
    // Encoding
    //--------------------------------------------------//
    
    // The single tokenization op. Runs structural/atom detection per config and
    // returns the full result; callers wanting only token IDs read result.token_ids.
    UniByteResult tokenizeWithMetadata(
        const std::string& text,
        size_t forced_segment_boundary = std::string::npos,
        size_t* token_count_at_boundary = nullptr) const;

    // Multi-boundary form used for invisible nested logical delimiters. Byte
    // boundaries must be nondecreasing; duplicate offsets are allowed and
    // receive the same token count. Boundaries force independent segmentation
    // without emitting delimiter tokens.
    UniByteResult tokenizeWithMetadata(
        const std::string& text,
        const std::vector<size_t>& forced_segment_boundaries,
        std::vector<size_t>* token_counts_at_boundaries) const;
    
    //--------------------------------------------------//
    // Decoding
    //--------------------------------------------------//
    
    // Decode token IDs to text
    std::string decode(const DecodeRequest& request) const;

    //--------------------------------------------------//
    // Vocabulary Info
    //--------------------------------------------------//
    
    // Canonical tokenizer vocab size: full token ID space written to GRMT headers.
    int vocabSize() const;

    //--------------------------------------------------//
    // Component Access
    //--------------------------------------------------//

    UnigramLM& unigramLM() { return unigram_; }
    const UnigramLM& unigramLM() const { return unigram_; }

private:
    ::GRIM::HyperParameters::TokenizerHP tokenizer_hp_;
    UnigramLM unigram_;
    
    bool gpu_initialized_ = false;
    Detector::DetectorRegistry detector_registry_;
};

} // namespace Tokenizer
} // namespace GRIM
