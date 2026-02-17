//======================================================//
//  Unigram.hpp
//  Unigram Language Model tokenizer for GRIM
//  
//  Unigram LM uses a probabilistic model where each token
//  has a log probability (score). Encoding finds the most
//  likely segmentation using Viterbi algorithm.
//  
//  Key advantages over BPE:
//  - Multiple valid segmentations (can sample)
//  - Better handling of rare/OOV words
//  - Naturally integrates with byte fallback
//  - Trains on actual language model objective
//  
//  Token ID layout:
//    [0-3]                    = Special tokens (<unk>, <pad>, <s>, </s>)
//    [4-259]                  = Reserved for byte fallback
//    [ATOM_TOKEN_OFFSET..UNIGRAM_VOCAB_OFFSET-1] = Reserved for atoms (ScratchBlock)
//    [UNIGRAM_VOCAB_OFFSET+]  = Unigram vocabulary (regular pieces only)
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

#include "Byte.hpp"  // For BYTE_TOKEN_OFFSET, BYTE_VOCAB_SIZE, special token IDs

namespace GRIM {
namespace Tokenizer {

//======================================================//
//  Atom Types for ScratchBlock Integration
//======================================================//
enum class AtomType : int {
    // Structural atoms (detected during tokenization)
    ATOM_NONE = 0,
    
    // Boundary markers (model learns these as delimiters)
    ATOM_END,               // End of any atom sequence
    
    // Numbers
    ATOM_INTEGER,           // 42, -17
    ATOM_FLOAT,             // 3.14, -2.5e10
    ATOM_HEX,               // 0xFF, 0x1A2B
    ATOM_BINARY,            // 0b1010
    
    // Code structures  
    ATOM_IDENTIFIER,        // variable_name, functionName
    ATOM_STRING_LITERAL,    // "hello", 'world'
    ATOM_REGEX,             // /pattern/flags
    
    // Special sequences
    ATOM_URL,               // https://...
    ATOM_EMAIL,             // user@domain.com
    ATOM_PATH,              // /usr/bin, C:\Windows
    ATOM_DATE,              // 2025-12-08
    ATOM_TIME,              // 14:30:00
    ATOM_IP_ADDRESS,        // 192.168.1.1
    
    // Math/Logic
    ATOM_EQUATION,          // x + y = z
    ATOM_EXPRESSION,        // (a * b) + c

    // Count of contiguous, in-use atom types (used for token layout)
    ATOM_ACTIVE_COUNT,
    
    // Reserved for future
    ATOM_RESERVED_START = 200,
    ATOM_RESERVED_END = 255,
    
    ATOM_TYPE_COUNT
};

constexpr int kAtomTypeCount = static_cast<int>(AtomType::ATOM_ACTIVE_COUNT);

//======================================================//
//  Constants
//======================================================//
constexpr int ATOM_TOKEN_OFFSET = BYTE_TOKEN_OFFSET + BYTE_VOCAB_SIZE;  // Atoms start after byte tokens (260)
inline int ATOM_VOCAB_SIZE = kAtomTypeCount;  // Atom slots derived from AtomType count
inline int UNIGRAM_VOCAB_OFFSET = ATOM_TOKEN_OFFSET + ATOM_VOCAB_SIZE;
inline uint32_t ATOM_TOKEN_BASE = static_cast<uint32_t>(ATOM_TOKEN_OFFSET);
inline uint32_t ATOM_TOKEN_MAX = static_cast<uint32_t>(UNIGRAM_VOCAB_OFFSET);
inline uint32_t MAX_ATOM_TOKENS = static_cast<uint32_t>(ATOM_VOCAB_SIZE);
constexpr int MAX_PIECE_LENGTH = 32;           // Maximum token length in bytes
constexpr float UNKNOWN_SCORE = -100.0f;       // Score for unknown pieces

inline void configureTokenLayout(int /*atom_vocab_size*/) {
    // Atom tokens are reserved for type-only placeholders; size derived from AtomType.
    ATOM_VOCAB_SIZE = kAtomTypeCount;
    UNIGRAM_VOCAB_OFFSET = ATOM_TOKEN_OFFSET + ATOM_VOCAB_SIZE;
    ATOM_TOKEN_MAX = static_cast<uint32_t>(UNIGRAM_VOCAB_OFFSET);
    MAX_ATOM_TOKENS = static_cast<uint32_t>(ATOM_VOCAB_SIZE);
}

//======================================================//
//  Unigram Token
//======================================================//
struct UnigramPiece {
    std::string text;
    float score;           // Log probability
    int token_id;
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
                           bool prune_during_mining = false);
    
    // DELETED: Auto-ID overload removed per Rule 20 (no silent fallbacks)
    // Callers MUST provide explicit token_id = UNIGRAM_VOCAB_OFFSET + pieces_.size()
    
    // Add a piece with explicit token_id (REQUIRED - no auto-ID)
    void addPiece(const std::string& text, float score, int token_id, bool is_user_defined);
    
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

// Convert atom type to token ID
inline int atomTypeToTokenId(AtomType type) {
    return ATOM_TOKEN_OFFSET + static_cast<int>(type);
}

// Convert token ID to atom type
inline AtomType tokenIdToAtomType(int token_id) {
    if (token_id < ATOM_TOKEN_OFFSET || token_id >= UNIGRAM_VOCAB_OFFSET) {
        return AtomType::ATOM_NONE;
    }
    return static_cast<AtomType>(token_id - ATOM_TOKEN_OFFSET);
}

// Check if token ID is an atom
inline bool isAtomToken(int token_id) {
    return token_id >= ATOM_TOKEN_OFFSET && token_id < UNIGRAM_VOCAB_OFFSET;
}

} // namespace Tokenizer
} // namespace GRIM
