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
#include "Unigram.hpp"

#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <functional>

namespace GRIM {
namespace Tokenizer {

// Text feature side-channel (FP16, fixed width).
constexpr int kTextFeatureDim = 16;

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
    std::vector<uint16_t> token_text_features;  // Per-token text features [tokens * kTextFeatureDim] (FP16)
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
        const size_t expected_feat = n * kTextFeatureDim;
        if (token_text_features.size() != expected_feat) {
            throw std::runtime_error(
                std::string(caller) + ": UniByteResult.token_text_features.size()=" +
                std::to_string(token_text_features.size()) + " != token_ids.size()*kTextFeatureDim=" +
                std::to_string(expected_feat));
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
//  UniByte Configuration
//======================================================//
struct UniByteConfig {
    // Vocabulary
    int target_vocab_size = 50000;
    float character_coverage = 0.9995f;
    int min_subword_freq = 3;  // Minimum frequency for subwords to be included
    bool prune_during_mining = false;  // Enable memory pruning during subword mining (disable if you have lots of RAM)
    bool enable_parallel_subword_mining = true;  // Parallelize subword counting during vocab training
    int subword_mining_workers = 0;  // 0 = auto, >0 fixed worker count
    size_t subword_mining_max_bytes = 0;  // 0 = use HyperParameters::UNIGRAM_MAX_SUBWORD_BYTES
    
    // Scratch Block Reasoning (AtomTable-based structured reasoning)
    bool enable_scratch_block_reasoning = true;  // Toggle internal reasoning layer
    
    // Structural detection (only used if scratch block reasoning enabled)
    // Number atoms are the only remaining supported tokenizer-side detection.
    bool detect_numbers = true;
    
    // Byte fallback
    bool enable_byte_fallback = true;
    
    // GPU settings
    bool prefer_gpu = true;
    int gpu_batch_size = 32;
};

//======================================================//
//  TokenLayout — runtime-queried token ID ranges
//
//  Built from live component sizes, NOT hardcoded constants.
//  If you add a special token, grow AtomType, or change byte
//  encoding, this struct automatically reflects it.
//======================================================//
struct TokenLayout {
    // Per-region sizes (queried from components, not hardcoded)
    int num_special  = 0;   // <unk>, <pad>, <s>, </s>, ...
    int num_bytes    = 0;   // raw byte tokens (0x00-0xFF)
    int num_atoms    = 0;   // registered atom type slots
    int num_unigram  = 0;   // learned subword pieces

    // Computed offsets — each region is [offset, offset+count)
    int special_offset() const { return 0; }
    int byte_offset()    const { return num_special; }
    int atom_offset()    const { return num_special + num_bytes; }
    int unigram_offset() const { return num_special + num_bytes + num_atoms; }
    int total_vocab()    const { return num_special + num_bytes + num_atoms + num_unigram; }

    // Classification — is this token id in a given region?
    bool isSpecial(int id) const { return id >= special_offset() && id < byte_offset(); }
    bool isByte(int id)    const { return id >= byte_offset()    && id < atom_offset(); }
    bool isAtom(int id)    const { return id >= atom_offset()    && id < unigram_offset(); }
    bool isUnigram(int id) const { return id >= unigram_offset() && id < total_vocab(); }

    // The masking question: should this token NEVER be a prediction target?
    // UNK (0): encoding failure, never a valid target
    // PAD (1): structural batching artifact, never a valid target
    // BOS (2): always position-0 input, never a mid-sequence target
    // EOS (3): VALID TARGET — model MUST learn to predict end-of-sequence!
    bool isNonContent(int id) const {
        return id == UNK_TOKEN_ID || id == PAD_TOKEN_ID || id == BOS_TOKEN_ID;
    }

    // First token ID that represents actual content (byte tokens and above)
    // Note: EOS (3) is below this but IS a valid target — use isNonContent() for masking.
    int firstContentTokenId() const { return num_special; }
};

//======================================================//
//  UniByte - Main Orchestrator
//======================================================//
class UniByte {
public:
    explicit UniByte(const UniByteConfig& config = UniByteConfig());
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
        config_.target_vocab_size = target_vocab_size;
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
    
    // Full encode with metadata (includes atom detection results)
    UniByteResult encodeWithMetadata(const std::string& text) const;
    
    // Batch encode
    std::vector<std::vector<int>> encodeBatch(const std::vector<std::string>& texts) const;
    
    //--------------------------------------------------//
    // Scratch Block Reasoning Control
    //--------------------------------------------------//
    
    // Enable/disable scratch block reasoning at runtime
    void setScratchBlockReasoning(bool enabled);
    bool isScratchBlockReasoningEnabled() const { return config_.enable_scratch_block_reasoning; }
    
    // GPU encode
    bool encodeGPU(const char* d_text,
                   size_t length,
                   int* d_token_ids,
                   int* d_token_count,
                   int max_tokens);

    //--------------------------------------------------//
    // Decoding
    //--------------------------------------------------//
    
    // Decode token IDs to text
    std::string decode(const std::vector<int>& token_ids) const;
    std::string decode(const int* token_ids, size_t count) const;
    
    // Decode with atom resolution (needs atom values from ScratchBlock).
    //
    // NOTE: this type-only resolver cannot disambiguate repeated atoms of the
    // same type (e.g. two ATOM_INTs in the same sequence map to identical
    // token IDs). Prefer the entry-aware overload below when the caller has
    // access to UniByteResult::atom_entry_ids.
    using AtomResolver = std::function<std::string(int token_id, AtomType type)>;
    std::string decodeWithAtoms(const std::vector<int>& token_ids,
                                 const AtomResolver& resolver) const;

    // Entry-aware decode: resolver receives the per-token atom_entry_id from
    // UniByteResult::atom_entry_ids, so callers can resolve repeated same-type
    // atoms without relying on call-order state. atom_entry_ids must have the
    // same length as token_ids; non-atom positions are ignored by the resolver.
    using AtomEntryResolver =
        std::function<std::string(uint32_t entry_id, AtomType type)>;
    std::string decodeWithAtoms(const std::vector<int>& token_ids,
                                 const std::vector<uint32_t>& atom_entry_ids,
                                 const AtomEntryResolver& resolver) const;

    //--------------------------------------------------//
    // Structural Detection
    //--------------------------------------------------//
    
    // Detect structures in text
    std::vector<StructuralSpan> detectStructures(const std::string& text) const;
    
    // Inject placeholders for detected structures
    std::string injectPlaceholders(const std::string& text,
                                    std::vector<StructuralSpan>& out_spans) const;

    //--------------------------------------------------//
    // Vocabulary Info
    //--------------------------------------------------//
    
    int vocabSize() const;
    int totalVocabSize() const;  // Including bytes and atoms
    
    // Cap vocabulary to top-K most frequent tokens (reduces loss computation time)
    void capVocabSize(int max_vocab);
    
    // Special token IDs
    int padId() const;
    int unkId() const;
    int bosId() const;
    int eosId() const;
    
    // Token layout — runtime-queried from live component sizes
    TokenLayout tokenLayout() const;

    // Token type checking
    bool isSpecialToken(int token_id) const;
    bool isByteToken(int token_id) const;
    bool isAtomToken(int token_id) const;
    bool isUnigramToken(int token_id) const;
    
    // Get token info
    std::string tokenToString(int token_id) const;

    //--------------------------------------------------//
    // Component Access
    //--------------------------------------------------//
    
    ByteEncoder& byteEncoder() { return byte_encoder_; }
    const ByteEncoder& byteEncoder() const { return byte_encoder_; }
    
    UnigramLM& unigramLM() { return unigram_; }
    const UnigramLM& unigramLM() const { return unigram_; }

private:
    UniByteConfig config_;
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

#include "Detectors.hpp"
