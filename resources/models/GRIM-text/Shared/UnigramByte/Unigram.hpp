//======================================================//
//  Unigram.hpp
//  Unigram Language Model tokenizer for GRIM
//  
//  Unigram LM uses a probabilistic model where each token
//  has a log probability (score). Encoding finds the most
//  likely segmentation using Viterbi algorithm.
//  
//  Token ID layout is defined in TokenLayout.hpp.
//  
//  Author: GRIM Team
//  Date: December 2025
//======================================================//

#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

#include "TokenLayout.hpp"  // AtomType, token ID constants, layout helpers

namespace GRIM {
namespace Tokenizer {

//======================================================//
//  Atom Span for Training Pipeline
//  Simple byte-offset pair marking detected atom regions.
//  The tokenizer skips these regions during vocab training
//  so atom internals (://@.com etc.) don't contaminate
//  character counts, subword mining, or EM scoring.
//======================================================//
struct AtomSpan {
    size_t start;  // Start byte offset in text (inclusive)
    size_t end;    // End byte offset in text (exclusive)
};

//======================================================//
//  Unigram Token
//======================================================//
struct UnigramPiece {
    std::string text;
    float score;           // Log probability
    // token_id is NOT stored — it's ALWAYS (UNIGRAM_VOCAB_OFFSET + index_in_pieces_).
    // Storing it caused 1078 collisions during EM prune/backfill (Issue #148).
    bool is_user_defined;  // High priority, never pruned
};

// Durable runtime finalization metadata produced by tokenizer training.
// This is not vocabulary state: it records the Viterbi workspace capacity
// required to encode the training corpus envelope after the final trie upload.
struct UnigramTrainingRuntimeReport {
    size_t required_viterbi_workspace_length = 0;
    std::uint64_t finalized_trie_generation = 0;
    int final_piece_count = 0;
};

// Snapshot of tokenizer runtime GPU state. Raw device pointers remain hidden
// in UnigramGpuMemory; callers only see generation/capacity readiness facts.
struct UnigramRuntimeStateSnapshot {
    bool initialized = false;
    bool ready_for_live_trie = false;
    size_t workspace_max_length = 0;
    std::uint64_t uploaded_trie_generation = 0;
    std::uint64_t live_trie_generation = 0;
};

// Durable CUDA buffer owner for UnigramLM. Kept out of this header so
// tokenizer logic owns vocab/trie semantics while memory files own allocation.
class UnigramGpuMemory;

// Per-segmentation RAII owner for Viterbi dynamic-programming state.
class UnigramViterbiSession;

//======================================================//
//  UnigramLM - Unigram Language Model Tokenizer
//======================================================//
class UnigramLM {
public:
    UnigramLM();
    ~UnigramLM();

    // Disable copy
    UnigramLM(const UnigramLM&) = delete;
    UnigramLM& operator=(const UnigramLM&) = delete;

    // Move support
    UnigramLM(UnigramLM&&) noexcept;
    UnigramLM& operator=(UnigramLM&&) noexcept;

    //--------------------------------------------------//
    // Vocabulary Management
    //--------------------------------------------------//

    // Train vocabulary from corpus
    bool trainFromCorpus(const std::vector<std::string>& texts,
                         int target_vocab_size,
                           float character_coverage = 0.9995f,
                           int min_subword_freq = 3,
                           bool prune_during_mining = false,
                           bool enable_parallel_subword_mining = true,
                           int subword_mining_workers = 0,
                           size_t subword_mining_max_bytes = 0);

    // Train with atom-aware spans: atom regions are SKIPPED during
    // character counting, subword mining, and EM iterations.
    // atom_spans[i] = list of atom spans for texts[i] (parallel arrays).
    bool trainFromCorpus(const std::vector<std::string>& texts,
                         const std::vector<std::vector<AtomSpan>>& atom_spans,
                         int target_vocab_size,
                           float character_coverage = 0.9995f,
                           int min_subword_freq = 3,
                           bool prune_during_mining = false,
                           bool enable_parallel_subword_mining = true,
                           int subword_mining_workers = 0,
                           size_t subword_mining_max_bytes = 0);
    
    // Canonical public vocab write entrypoint. The actual storage mutation is
    // performed by VocabWriteOp.hpp so pieces_ and piece_to_id_ stay synchronized.
    void writePiece(const std::string& text, float score, bool is_user_defined);
    
    // Compute token_id for piece at given index in pieces_
    static int tokenIdForIndex(int index) { return UNIGRAM_VOCAB_OFFSET + index; }
    
    // Get piece by ID
    const UnigramPiece* getPiece(int token_id) const;
    
    // Get ID by piece text
    int getPieceId(const std::string& text) const;
    
    // Check if piece exists
    bool hasPiece(const std::string& text) const;

    //--------------------------------------------------//
    // CPU Encoding/Decoding
    //--------------------------------------------------//
    
    // Encode text using Viterbi (best segmentation)
    // prepend_space=true: prepend ▁ (start of text / first segment)
    // prepend_space=false: skip prepend (mid-text segment after atom)
    std::vector<int> encode(const std::string& text, bool prepend_space = true) const;
    
    // Decode token IDs to text
    std::string decode(const std::vector<int>& token_ids) const;

    //--------------------------------------------------//
    // GPU Encoding/Decoding
    //--------------------------------------------------//
    
    // Initialize GPU resources (call before GPU operations)
    bool initGPU();
    bool initGPUForMaxSequenceLength(size_t required_max_sequence_length);
    bool runtimeReadyForMaxSequenceLength(size_t required_max_sequence_length) const;
    void requireRuntimeReadyForMaxSequenceLength(size_t required_max_sequence_length,
                                                const char* caller) const;
    void requireRuntimeReadyForLastTraining(const char* caller) const;
    UnigramRuntimeStateSnapshot runtimeStateSnapshot() const;
    const UnigramTrainingRuntimeReport& lastTrainingRuntimeReport() const;

    //--------------------------------------------------//
    // Vocabulary Info
    //--------------------------------------------------//
    
    int pieceCount() const { return static_cast<int>(pieces_.size()); }
    
    // SentencePiece-style whitespace normalization (▁ ↔ space)
    static std::string normalizeForTokenization(const std::string& text);
    static std::string denormalizeFromTokenization(const std::string& text);
    
    //--------------------------------------------------//
    // Configuration
    //--------------------------------------------------//
    
    void setByteFallbackEnabled(bool enabled) { enable_byte_fallback_ = enabled; }
    bool byteFallbackEnabled() const { return enable_byte_fallback_; }
    
    // Build trie from vocabulary (must call after adding pieces, before encoding)
    void buildTrie();

private:
    friend class UnigramViterbiSession;

    // Vocabulary storage
    std::vector<UnigramPiece> pieces_;
    std::unordered_map<std::string, int> piece_to_id_;
    
    bool enable_byte_fallback_ = true;
    
    // Trie for fast prefix lookup (GPU-friendly)
    struct TrieNode {
        int token_id = -1;          // -1 if not end of token
        float score = UNKNOWN_SCORE;
        int children[256];          // Child indices (-1 if none)
        
        TrieNode() : token_id(-1), score(UNKNOWN_SCORE) {
            std::fill(std::begin(children), std::end(children), -1);
        }
    };
    std::vector<TrieNode> trie_;
    
    // GPU resources are owned by UnigramGpuMemory in UnigramGpuMemory.*.
    // UnigramLM only requests initialization/upload; it does not own raw CUDA lifetime details here.
    std::unique_ptr<UnigramGpuMemory> gpu_;
    std::uint64_t trie_generation_ = 0;
    UnigramTrainingRuntimeReport last_training_runtime_report_;
    
    // Upload trie to GPU
    bool uploadTrieToGPU(size_t workspace_sequence_length);
};

} // namespace Tokenizer
} // namespace GRIM
