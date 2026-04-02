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

#include <cuda_runtime.h>
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
    bool is_special;
    bool is_user_defined;  // High priority, never pruned
};

//======================================================//
//  Viterbi Node (for decoding/encoding)
//======================================================//
struct ViterbiNode {
    float score;           // Best score to reach this position
    int prev_pos;          // Previous position in best path
    int token_id;          // Token ID of piece ending here
    int piece_length;      // Length of piece in bytes
};

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
    
    // Load vocabulary from file (text format - tab-separated)
    bool load(const std::string& vocab_path);
    
    // Load vocabulary from binary file (KTMG format) - faster
    bool loadBinary(const std::string& vocab_path);
    
    // Save vocabulary to file (binary primary, text optional)
    bool save(const std::string& vocab_path, bool save_text_format = false) const;
    
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
    
    // Add a piece to vocab. Token ID is ALWAYS (UNIGRAM_VOCAB_OFFSET + pieces_.size()).
    // No token_id parameter — the position IS the ID. This prevents collision bugs.
    void addPiece(const std::string& text, float score, bool is_user_defined);
    
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
    std::vector<int> encode(const std::string& text) const;
    
    // Encode with token info
    std::vector<UnigramPiece> encodeWithPieces(const std::string& text) const;
    
    // Decode token IDs to text
    std::string decode(const std::vector<int>& token_ids) const;
    std::string decode(const int* token_ids, size_t count) const;

    //--------------------------------------------------//
    // GPU Encoding/Decoding
    //--------------------------------------------------//
    
    // Initialize GPU resources (call before GPU operations)
    bool initGPU();
    
    // Encode on GPU using parallel Viterbi
    // Returns indices where byte fallback is needed
    bool encodeGPU(const char* d_text,
                   size_t length,
                   int* d_token_ids,
                   int* d_token_count,
                   int max_tokens,
                   bool* d_needs_byte_fallback);  // Per-position flags
    
    // Batch encode on GPU
    bool encodeBatchGPU(const char* const* d_texts,
                        const size_t* lengths,
                        int** d_token_ids,
                        int* d_token_counts,
                        int max_tokens_per_seq,
                        size_t batch_size);
    
    // Decode on GPU
    bool decodeGPU(const int* d_token_ids,
                   size_t count,
                   char* d_output,
                   size_t* d_output_length,
                   size_t max_output_length);

    //--------------------------------------------------//
    // Vocabulary Info
    //--------------------------------------------------//
    
    int vocabSize() const { return static_cast<int>(pieces_.size()); }
    int totalVocabSize() const { return UNIGRAM_VOCAB_OFFSET + vocabSize(); }
    int unkId() const { return unk_id_; }
    
    // Cap vocabulary to top-K pieces by score (keeps most frequent)
    void capVocabSize(int max_size);
    
    // SentencePiece-style whitespace normalization (▁ ↔ space)
    static std::string normalizeForTokenization(const std::string& text);
    static std::string denormalizeFromTokenization(const std::string& text);
    
    // Special token IDs
    int padId() const { return pad_id_; }
    int bosId() const { return bos_id_; }
    int eosId() const { return eos_id_; }

    //--------------------------------------------------//
    // Configuration
    //--------------------------------------------------//
    
    void setUnkId(int id) { unk_id_ = id; }
    void setBosId(int id) { bos_id_ = id; }
    void setEosId(int id) { eos_id_ = id; }
    void setPadId(int id) { pad_id_ = id; }
    void setByteFallbackEnabled(bool enabled) { enable_byte_fallback_ = enabled; }
    bool byteFallbackEnabled() const { return enable_byte_fallback_; }
    
    // Build trie from vocabulary (must call after adding pieces, before encoding)
    void buildTrie();

private:
    // Vocabulary storage
    std::vector<UnigramPiece> pieces_;
    std::unordered_map<std::string, int> piece_to_id_;
    
    // Special token IDs (ABSOLUTE token IDs, not relative to any offset)
    int unk_id_ = UNK_TOKEN_ID;   // 0
    int pad_id_ = PAD_TOKEN_ID;   // 1
    int bos_id_ = BOS_TOKEN_ID;   // 2
    int eos_id_ = EOS_TOKEN_ID;   // 3
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
    
    // GPU resources
    struct GPUData {
        // Trie on device
        int* d_trie_children;       // [num_nodes * 256]
        int* d_trie_token_ids;      // [num_nodes]
        float* d_trie_scores;       // [num_nodes]
        int num_nodes;
        
        // Piece data for decoding
        char* d_piece_data;         // Concatenated piece strings
        int* d_piece_offsets;       // Start offset of each piece
        int* d_piece_lengths;       // Length of each piece
        
        // Viterbi workspace (pre-allocated, fixed capacity)
        float* d_viterbi_scores;
        int* d_viterbi_prev;
        int* d_viterbi_tokens;
        size_t workspace_max_length;  // Maximum sequence length supported
        
        bool initialized = false;
    };
    std::unique_ptr<GPUData> gpu_;
    
    // Upload trie to GPU
    bool uploadTrieToGPU();
    
    // Viterbi implementation
    std::vector<ViterbiNode> viterbi(const std::string& text) const;
    
    // Backtrack Viterbi path
    std::vector<int> backtrack(const std::vector<ViterbiNode>& nodes, int end_pos) const;
};

} // namespace Tokenizer
} // namespace GRIM
