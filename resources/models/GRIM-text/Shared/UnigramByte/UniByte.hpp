//======================================================//
//  UniByte.hpp
//  Unified Unigram LM + Byte Fallback Tokenizer for GRIM
//  
//  This orchestrator combines:
//  - Unigram LM for high-quality subword tokenization
//  - Byte fallback for 100% coverage
//  - Structural detection for atom injection
//  - Placeholder system for ScratchBlock integration
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
#include "Byte.hpp"
#include "Detectors/TokenizerDetector.hpp"
#include "TokenLayout.hpp"
#include "Unigram.hpp"
#include "../HyperParameters/HyperparameterGroupings.hpp"

#include <cuda_runtime.h>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace GRIM {
namespace Tokenizer {

//======================================================//
//  Structural Detection Result
//======================================================//
struct StructuralSpan {
    size_t start;           // Start position in text (may include leading whitespace)
    size_t end;             // End position (exclusive)
    AtomType atom_type;     // Type of structure detected
    
    // Zero-copy buffer reference (NO std::string allocation!)
    const char* buffer_ptr; // Pointer to original text buffer
    uint32_t offset;        // Offset in buffer
    uint32_t length;        // Length of span (end - start)
    
    // Content bounds (same as offset/length since no widening)
    uint32_t content_offset; // Offset to atom content
    uint32_t content_length; // Length of atom content
    
    int placeholder_id;     // Token ID of placeholder
    
    // Helper: get string_view of full span (may include leading whitespace)
    std::string_view view() const {
        return std::string_view(buffer_ptr + offset, length);
    }
    
    // Helper: get string_view of just the atom content (no whitespace)
    std::string_view contentView() const {
        return std::string_view(buffer_ptr + content_offset, content_length);
    }
};

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
    const AtomTable* atom_table = nullptr;

    DecodeRequest(const std::vector<int>& ids)
        : token_ids(ids.data()), token_count(ids.size()) {}

    DecodeRequest(std::initializer_list<int> ids)
        : owned_token_ids(ids),
          token_ids(owned_token_ids.data()),
          token_count(owned_token_ids.size()) {}

    DecodeRequest(const UniByteResult& result)
        : token_ids(result.token_ids.data()),
          token_count(result.token_ids.size()),
          atom_entry_ids(result.atom_entry_ids.data()),
          atom_entry_count(result.atom_entry_ids.size()),
          atom_table(result.atom_table.get()) {}

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
    
    // Load vocabulary from file (tries binary first, then text)
    bool load(const std::string& vocab_path);
    
    // Save vocabulary to file (binary primary, text optional)
    bool save(const std::string& vocab_path, bool save_text_format = false, float score_multiplier = 1.0f) const;
    
    // Train from corpus
    bool train(const std::vector<std::string>& texts);
    
    // Train with explicit vocab size
    void trainFromCorpus(const std::vector<std::string>& corpus, int target_vocab_size) {
        tokenizer_hp_.target_vocab_size = target_vocab_size;
        train(corpus);
    }
    
    // Initialize GPU resources
    bool initGPU();

    //--------------------------------------------------//
    // Encoding
    //--------------------------------------------------//
    
    // Standard encode (returns just token IDs)
    // Uses scratch block reasoning if enabled, otherwise falls back to normal UnigramByte
    std::vector<int> encode(const std::string& text) const;
    
    // Full tokenization with metadata (includes atom detection results)
    UniByteResult tokenizeWithMetadata(const std::string& text) const;
    
    //--------------------------------------------------//
    // Scratch Block Reasoning Control
    //--------------------------------------------------//
    
    // Enable/disable scratch block reasoning at runtime
    void setScratchBlockReasoning(bool enabled);
    bool isScratchBlockReasoningEnabled() const { return tokenizer_hp_.enable_scratch_block_reasoning; }
    
    //--------------------------------------------------//
    // Decoding
    //--------------------------------------------------//
    
    // Decode token IDs to text
    std::string decode(const DecodeRequest& request) const;

    //--------------------------------------------------//
    // Structural Detection
    //--------------------------------------------------//

    // Detect raw-text features before tokenization. This operates on source
    // byte offsets only; it does not inspect or classify token IDs.
    std::vector<Detector::RawTextDetection> detectRawText(const std::string& text) const;
    
    // Detect structures in text
    std::vector<StructuralSpan> detectStructures(const std::string& text) const;
    
    // Inject placeholders for detected structures
    std::string injectPlaceholders(const std::string& text,
                                    std::vector<StructuralSpan>& out_spans) const;

    //--------------------------------------------------//
    // Vocabulary Info
    //--------------------------------------------------//
    
    // Canonical tokenizer vocab size: full token ID space written to GRMT headers.
    int vocabSize() const;
    
    // Token layout — runtime-queried from live component sizes
    TokenLayout tokenLayout() const;

    //--------------------------------------------------//
    // Component Access
    //--------------------------------------------------//
    
    ByteEncoder& byteEncoder() { return byte_encoder_; }
    const ByteEncoder& byteEncoder() const { return byte_encoder_; }
    
    UnigramLM& unigramLM() { return unigram_; }
    const UnigramLM& unigramLM() const { return unigram_; }

private:
    ::GRIM::HyperParameters::TokenizerHP tokenizer_hp_;
    ByteEncoder byte_encoder_;
    UnigramLM unigram_;
    
    bool gpu_initialized_ = false;
    
    // Structural detection patterns
    struct DetectorState;
    std::unique_ptr<DetectorState> detector_;
    
    void initDetector();
    
    // Internal encoding with structural awareness
    UniByteResult encodeInternal(const std::string& text,
                                  const std::vector<StructuralSpan>& structures) const;
};

} // namespace Tokenizer
} // namespace GRIM
