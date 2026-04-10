//======================================================//
//  Unigram.cu
//  CUDA implementation of Unigram Language Model tokenizer
//
//  Inference, vocab I/O, trie building, Viterbi, GPU encoding.
//  Training pipeline is in UnigramTrainer.cu.
//  Text utilities are in TextUtils.cu.
//======================================================//

#include "Unigram.hpp"
#include "TextUtils.hpp"
#include "Byte.hpp"
#include "HyperParameters/HyperParameters_GPU.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <unordered_set>

namespace GRIM {

// Punctuation isolation guard for Viterbi.
// Returns true if the character at the given position is a punctuation character
// that should be tokenized in isolation (never merged with adjacent letters/digits).
// Used in both CPU Viterbi and GPU kernel to enforce punctuation boundary splitting.
// Note: space (0x20) is NOT punctuation — it's a normal character that can appear
// in multi-word vocab tokens like "of the".
// MUST be __host__ __device__ for CUDA kernel use.
__host__ __device__ static inline bool isPunctBoundary(unsigned char c) {
    return (c >= '!' && c <= '/') ||  // !"#$%&'()*+,-./
           (c >= ':' && c <= '@') ||  // :;<=>?@
           (c >= '[' && c <= '`') ||  // [\]^_`
           (c >= '{' && c <= '~');    // {|}~
}

// Public static method wrappers — delegate to TextUtils
std::string Tokenizer::UnigramLM::normalizeForTokenization(const std::string& text) {
    return Tokenizer::normalizeSpaces(text);
}
std::string Tokenizer::UnigramLM::denormalizeFromTokenization(const std::string& text) {
    return Tokenizer::denormalizeSpaces(text);
}

namespace Tokenizer {

//======================================================//
//  CUDA Kernels
//======================================================//

__global__ void kernelViterbiForward(
    const char* __restrict__ text,
    size_t length,
    const int* __restrict__ trie_children,    // [num_nodes * 256]
    const int* __restrict__ trie_token_ids,   // [num_nodes]
    const float* __restrict__ trie_scores,    // [num_nodes]
    int num_trie_nodes,
    float* __restrict__ viterbi_scores,       // [length + 1]
    int* __restrict__ viterbi_prev,           // [length + 1]
    int* __restrict__ viterbi_tokens,         // [length + 1]
    bool* __restrict__ needs_fallback,        // [length]
    int unk_id,
    float unk_score,
    bool enable_byte_fallback
) {
    // Single thread processes positions SEQUENTIALLY to maintain Viterbi invariants
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    // Initialize position 0
    viterbi_scores[0] = 0.0f;
    viterbi_prev[0] = -1;
    viterbi_tokens[0] = -1;
    
    // Process each position sequentially (required for correctness)
    for (size_t pos = 1; pos <= length; ++pos) {
        unsigned char cur_byte = static_cast<unsigned char>(text[pos - 1]);
        bool cur_is_punct = isPunctBoundary(cur_byte);
        
        // PUNCTUATION ISOLATION GUARD:
        // If this position's character is punctuation, force it to be a single
        // byte token. Skip trie matching entirely — punctuation is never merged
        // with adjacent letters/digits.
        if (cur_is_punct) {
            // Emit as byte token: BYTE_TOKEN_OFFSET + byte_value
            viterbi_scores[pos] = viterbi_scores[pos - 1] + unk_score;
            viterbi_prev[pos] = static_cast<int>(pos - 1);
            viterbi_tokens[pos] = static_cast<int>(cur_byte) + 4;  // BYTE_TOKEN_OFFSET = 4
            continue;
        }
        
        float best_score = -1e30f;
        int best_prev = -1;
        int best_token = unk_id;
        bool found_match = false;
        
        // Try all possible pieces ending at position `pos`
        // Walk backwards from pos, traversing trie
        int node = 0;  // Start at trie root
        
        for (size_t start = pos; start > 0 && (pos - start) < MAX_PIECE_LENGTH; --start) {
            size_t idx = start - 1;
            unsigned char c = static_cast<unsigned char>(text[idx]);
            
            // PUNCTUATION ISOLATION GUARD:
            // Stop backward walk if we hit a punctuation character —
            // pieces must not span across a punctuation boundary.
            if (isPunctBoundary(c)) break;
            
            // Navigate trie
            int child = trie_children[node * 256 + c];
            if (child < 0) break;  // No path in trie
            
            node = child;
            
            // Check if this node is end of a token
            int token_id = trie_token_ids[node];
            if (token_id >= 0) {
                float piece_score = trie_scores[node];
                // Safe: viterbi_scores[start - 1] was computed in previous iteration
                float candidate_score = viterbi_scores[start - 1] + piece_score;
                
                if (candidate_score > best_score) {
                    best_score = candidate_score;
                    best_prev = static_cast<int>(start - 1);
                    best_token = token_id;
                    found_match = true;
                }
            }
        }
        
        // If no match found, mark for byte fallback
        if (!found_match) {
            best_prev = static_cast<int>(pos - 1);
            best_token = unk_id;
            best_score = viterbi_scores[pos - 1] + unk_score;

            if (enable_byte_fallback && needs_fallback) {
                needs_fallback[pos - 1] = true;
            }
        }
        
        viterbi_scores[pos] = best_score;
        viterbi_prev[pos] = best_prev;
        viterbi_tokens[pos] = best_token;
    }
}

// Kernel: Backtrack Viterbi path
__global__ void kernelViterbiBacktrack(
    size_t length,
    const int* __restrict__ viterbi_prev,
    const int* __restrict__ viterbi_tokens,
    int* __restrict__ output_tokens,
    int* __restrict__ output_count,
    int max_tokens
) {
    // Single thread does backtracking
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    // Count tokens first
    int count = 0;
    int pos = static_cast<int>(length);
    while (pos > 0) {
        count++;
        pos = viterbi_prev[pos];
    }
    
    // Clamp to max_tokens to prevent buffer overflow
    if (count > max_tokens) {
        count = max_tokens;
    }
    
    // Write tokens in reverse
    *output_count = count;
    pos = static_cast<int>(length);
    int write_idx = count - 1;
    while (pos > 0 && write_idx >= 0) {
        output_tokens[write_idx] = viterbi_tokens[pos];
        pos = viterbi_prev[pos];
        write_idx--;
    }
}

// Kernel: Decode tokens to text
__global__ void kernelUnigramDecode(
    const int* __restrict__ token_ids,
    size_t count,
    const char* __restrict__ piece_data,
    const int* __restrict__ piece_offsets,
    const int* __restrict__ piece_lengths,
    int vocab_offset,
    int vocab_size,
    char* __restrict__ output,
    size_t* __restrict__ output_length,
    size_t max_output
) {
    // Single thread for now (could parallelize with scan)
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    size_t out_pos = 0;
    
    for (size_t i = 0; i < count && out_pos < max_output; ++i) {
        int tid = token_ids[i];
        
        // Check if it's a unigram token
        if (tid >= vocab_offset && tid < vocab_offset + vocab_size) {
            int piece_idx = tid - vocab_offset;
            int offset = piece_offsets[piece_idx];
            int len = piece_lengths[piece_idx];
            
            for (int j = 0; j < len && out_pos < max_output; ++j) {
                output[out_pos++] = piece_data[offset + j];
            }
        }
        // Byte tokens handled by caller
    }
    
    *output_length = out_pos;
}

// Kernel: Trie lookup for batch encoding
__global__ void kernelTrieLookup(
    const char* __restrict__ text,
    size_t length,
    size_t start_pos,
    const int* __restrict__ trie_children,
    const int* __restrict__ trie_token_ids,
    const float* __restrict__ trie_scores,
    int* __restrict__ match_token,
    int* __restrict__ match_length,
    float* __restrict__ match_score
) {
    // Single thread traverses trie from start_pos
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    int node = 0;
    int best_token = -1;
    int best_length = 0;
    float best_score = -1e30f;
    
    for (size_t i = start_pos; i < length && (i - start_pos) < MAX_PIECE_LENGTH; ++i) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        int child = trie_children[node * 256 + c];
        
        if (child < 0) break;
        node = child;
        
        int token_id = trie_token_ids[node];
        if (token_id >= 0) {
            float score = trie_scores[node];
            if (score > best_score) {
                best_token = token_id;
                best_length = static_cast<int>(i - start_pos + 1);
                best_score = score;
            }
        }
    }
    
    *match_token = best_token;
    *match_length = best_length;
    *match_score = best_score;
}

//======================================================//
//  UnigramLM Implementation
//======================================================//

UnigramLM::UnigramLM() 
    : gpu_(std::make_unique<GPUData>())
{
    // Start with empty vocabulary - special tokens added by load() or trainFromCorpus()
}

UnigramLM::~UnigramLM() {
    if (gpu_ && gpu_->initialized) {
        cudaFree(gpu_->d_trie_children);
        cudaFree(gpu_->d_trie_token_ids);
        cudaFree(gpu_->d_trie_scores);
        cudaFree(gpu_->d_piece_data);
        cudaFree(gpu_->d_piece_offsets);
        cudaFree(gpu_->d_piece_lengths);
        cudaFree(gpu_->d_viterbi_scores);
        cudaFree(gpu_->d_viterbi_prev);
        cudaFree(gpu_->d_viterbi_tokens);
    }
}

UnigramLM::UnigramLM(UnigramLM&& other) noexcept
    : pieces_(std::move(other.pieces_))
    , piece_to_id_(std::move(other.piece_to_id_))
    , unk_id_(other.unk_id_)
    , pad_id_(other.pad_id_)
    , bos_id_(other.bos_id_)
    , eos_id_(other.eos_id_)
    , enable_byte_fallback_(other.enable_byte_fallback_)
    , trie_(std::move(other.trie_))
    , gpu_(std::move(other.gpu_))
{
}

UnigramLM& UnigramLM::operator=(UnigramLM&& other) noexcept {
    if (this != &other) {
        pieces_ = std::move(other.pieces_);
        piece_to_id_ = std::move(other.piece_to_id_);
        unk_id_ = other.unk_id_;
        pad_id_ = other.pad_id_;
        bos_id_ = other.bos_id_;
        eos_id_ = other.eos_id_;
        enable_byte_fallback_ = other.enable_byte_fallback_;
        trie_ = std::move(other.trie_);
        gpu_ = std::move(other.gpu_);
    }
    return *this;
}

//--------------------------------------------------//
// Vocabulary Management
//--------------------------------------------------//

void UnigramLM::addPiece(const std::string& text, float score, bool is_user_defined) {
    if (piece_to_id_.count(text)) {
        // Update existing piece's score. Token ID is immutable (= UNIGRAM_VOCAB_OFFSET + index).
        int idx = piece_to_id_[text];
        pieces_[idx].score = score;
        if (!pieces_[idx].is_user_defined) {
            pieces_[idx].is_user_defined = is_user_defined;
        }
        return;
    }
    
    UnigramPiece piece;
    piece.text = text;
    piece.score = score;
    // token_id is NOT stored — it's ALWAYS (UNIGRAM_VOCAB_OFFSET + index).
    piece.is_special = (!text.empty() && text.front() == '<' && text.back() == '>');
    piece.is_user_defined = is_user_defined;
    
    piece_to_id_[text] = static_cast<int>(pieces_.size());
    pieces_.push_back(piece);
}

const UnigramPiece* UnigramLM::getPiece(int token_id) const {
    // Special tokens are not in pieces_ — they have absolute IDs 0-3
    // Callers should check isSpecialToken() separately if they need special token info
    int idx = token_id - UNIGRAM_VOCAB_OFFSET;
    if (idx < 0 || idx >= static_cast<int>(pieces_.size())) {
        return nullptr;
    }
    return &pieces_[idx];
}

int UnigramLM::getPieceId(const std::string& text) const {
    // Check special tokens by name
    if (text == "<unk>") return UNK_TOKEN_ID;
    if (text == "<pad>") return PAD_TOKEN_ID;
    if (text == "<s>")   return BOS_TOKEN_ID;
    if (text == "</s>")  return EOS_TOKEN_ID;
    
    auto it = piece_to_id_.find(text);
    if (it == piece_to_id_.end()) {
        return unk_id_;  // UNK_TOKEN_ID = 0 (absolute)
    }
    return UNIGRAM_VOCAB_OFFSET + it->second;
}

bool UnigramLM::hasPiece(const std::string& text) const {
    return piece_to_id_.count(text) > 0;
}

//--------------------------------------------------//
// Vocab I/O
//--------------------------------------------------//

bool UnigramLM::load(const std::string& vocab_path) {
    std::cout << "[UnigramLM] Opening vocab file: " << vocab_path << std::endl << std::flush;
    std::ifstream file(vocab_path);
    if (!file.is_open()) {
        std::cerr << "[UnigramLM] Failed to open vocab file: " << vocab_path << std::endl;
        return false;
    }
    
    std::cout << "[UnigramLM] Clearing existing vocab..." << std::endl << std::flush;
    pieces_.clear();
    piece_to_id_.clear();
    
    // Special tokens are now at absolute IDs 0-3, NOT stored in pieces_.
    // They are handled by getPieceId() directly.
    
    std::cout << "[UnigramLM] Reading vocab pieces..." << std::endl << std::flush;
    std::string line;
    int line_count = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        line_count++;
        if (line_count == 1) {
            std::cout << "[UnigramLM] First line: " << line.substr(0, std::min(size_t(50), line.size())) << std::endl << std::flush;
        }
        
        // Format: <piece>\t<score>
        size_t tab_pos = line.rfind('\t');
        if (tab_pos == std::string::npos) {
            std::cout << "[UnigramLM] WARNING: Line " << line_count << " has no tab, skipping" << std::endl << std::flush;
            continue;
        }
        
        try {
            std::string piece = line.substr(0, tab_pos);
            std::string score_str = line.substr(tab_pos + 1);
            
            // Skip lines with empty or invalid scores
            if (score_str.empty() || score_str == "\t" || score_str == " ") {
                std::cout << "[UnigramLM] WARNING: Line " << line_count << " has empty score, skipping" << std::endl << std::flush;
                continue;
            }
            
            float score = std::stof(score_str);
            
            // Skip special tokens from file — they're handled as absolute IDs 0-3
            if (piece == "<unk>" || piece == "<pad>" || piece == "<s>" || piece == "</s>") {
                continue;
            }
            
            if (line_count == 1) {
                std::cout << "[UnigramLM] First piece: '" << piece.substr(0, std::min(size_t(30), piece.size())) 
                          << "', score: " << score << std::endl << std::flush;
                std::cout << "[UnigramLM] Calling addPiece..." << std::endl << std::flush;
            }
            
            addPiece(piece, score, false);
            
            if (line_count == 1) {
                std::cout << "[UnigramLM] First addPiece succeeded" << std::endl << std::flush;
            }
        } catch (const std::exception& e) {
            std::cout << "[UnigramLM] ERROR parsing line " << line_count << ": " << e.what() << std::endl << std::flush;
            std::cout << "[UnigramLM]   Line content: " << line.substr(0, std::min(size_t(100), line.size())) << std::endl << std::flush;
            std::cout << "[UnigramLM]   Skipping this line and continuing..." << std::endl << std::flush;
            continue;  // Skip bad lines instead of failing
        }
        
        if (line_count % 10000 == 0) {
            std::cout << "[UnigramLM] Loaded " << line_count << " pieces..." << std::endl << std::flush;
        }
    }
    
    std::cout << "[UnigramLM] Building trie..." << std::endl << std::flush;
    buildTrie();
    std::cout << "[UnigramLM] Trie built" << std::endl << std::flush;
    
    std::cout << "[UnigramLM] Loaded " << pieces_.size() << " pieces from " << vocab_path << std::endl;
    return true;
}

bool UnigramLM::loadBinary(const std::string& vocab_path) {
    std::ifstream bin_file(vocab_path, std::ios::binary);
    if (!bin_file.is_open()) {
        std::cerr << "[UnigramLM] Failed to open binary vocab file: " << vocab_path << std::endl;
        return false;
    }
    
    // Read and verify KTMG magic (4 bytes)
    char magic[4];
    bin_file.read(magic, 4);
    if (magic[0] != 'K' || magic[1] != 'T' || magic[2] != 'M' || magic[3] != 'G') {
        throw std::runtime_error("[UnigramLM] Invalid binary vocab magic header - file corrupted or not a KTMG vocab file");
    }
    
    // Read version (2 bytes) - v4 required (SentencePiece ▁ normalization)
    uint16_t version;
    bin_file.read(reinterpret_cast<char*>(&version), 2);
    if (version != 4) {
        throw std::runtime_error("[UnigramLM] Vocab file version " + std::to_string(version) + 
            " is not supported. Required version 4 (SentencePiece normalization). Retrain tokenizer.");
    }
    
    // Skip checksum (4 bytes)
    uint32_t checksum;
    bin_file.read(reinterpret_cast<char*>(&checksum), 4);
    
    // Read config vocab_size (4 bytes) - number of unigram pieces
    uint32_t config_vocab_size;
    bin_file.read(reinterpret_cast<char*>(&config_vocab_size), 4);
    
    // Skip max_length (4 bytes)
    uint32_t max_length;
    bin_file.read(reinterpret_cast<char*>(&max_length), 4);
    
    // Skip flags (3 bytes)
    char flags[3];
    bin_file.read(flags, 3);
    
    // Read total vocab size (4 bytes) - includes bytes + atoms + unigram
    uint32_t total_vocab_size;
    bin_file.read(reinterpret_cast<char*>(&total_vocab_size), 4);
    
    // Clear existing vocab
    pieces_.clear();
    piece_to_id_.clear();
    pieces_.reserve(config_vocab_size);
    
    // Read pieces: length (4 bytes) + text + score (4 bytes float) + token_id (4 bytes)
    std::vector<char> text_buffer;
    text_buffer.reserve(MAX_PIECE_LENGTH);
    
    for (uint32_t i = 0; i < config_vocab_size; ++i) {
        uint32_t len;
        bin_file.read(reinterpret_cast<char*>(&len), 4);
        
        if (len > MAX_PIECE_LENGTH) {
            throw std::runtime_error("[UnigramLM] Invalid piece length " + std::to_string(len) + 
                " at index " + std::to_string(i) + " - vocab file corrupted");
        }
        
        text_buffer.resize(len);
        bin_file.read(text_buffer.data(), len);
        std::string text(text_buffer.data(), len);
        
        float score;
        bin_file.read(reinterpret_cast<char*>(&score), 4);
        
        int token_id;
        bin_file.read(reinterpret_cast<char*>(&token_id), 4);
        
        // Skip special tokens from binary — they're at absolute IDs 0-3 now
        if (text == "<unk>" || text == "<pad>" || text == "<s>" || text == "</s>") {
            continue;
        }
        
        // Validate stored token_id matches position-derived ID.
        // If mismatch, the vocab file was produced by the buggy tokenizer.
        int expected_id = UNIGRAM_VOCAB_OFFSET + static_cast<int>(pieces_.size());
        if (token_id != expected_id) {
            throw std::runtime_error(
                "[UnigramLM] vocab.bin token_id mismatch at piece " + std::to_string(i) +
                " ('" + text.substr(0, 30) + "'): stored=" + std::to_string(token_id) +
                " expected=" + std::to_string(expected_id) +
                ". Retrain tokenizer to fix (old vocab had EM prune/backfill collision bug).");
        }
        
        addPiece(text, score, false);
    }
    
    if (!bin_file) {
        throw std::runtime_error("[UnigramLM] Error reading binary vocab file - unexpected EOF or read error");
    }
    
    buildTrie();
    
    std::cout << "[UnigramLM] Loaded " << pieces_.size() << " pieces from binary: " << vocab_path << std::endl;
    std::cout << "[UnigramLM] Embedding vocab size: " << total_vocab_size
              << " (" << NUM_SPECIAL_TOKENS << " special + " << BYTE_VOCAB_SIZE << " bytes + "
              << ATOM_VOCAB_SIZE << " atom type placeholders + "
              << pieces_.size() << " unigram pieces)" << std::endl;
    return true;
}

bool UnigramLM::save(const std::string& vocab_path, bool save_text_format, float score_multiplier) const {
    // Primary: Save binary format (.bin)
    // Binary format: KTMG magic + version + checksum + config + vocab_size + pieces
    std::string bin_path = vocab_path;
    size_t dot_pos = bin_path.rfind('.');
    if (dot_pos != std::string::npos) {
        std::string ext = bin_path.substr(dot_pos);
        if (ext != ".bin") {
            bin_path = bin_path.substr(0, dot_pos) + ".bin";
        }
    } else {
        bin_path += ".bin";
    }
    
    std::ofstream bin_file(bin_path, std::ios::binary);
    if (!bin_file.is_open()) {
        std::cerr << "[UnigramLM] Failed to create binary vocab file: " << bin_path << std::endl;
        return false;
    }
    
    // Header: KTMG magic (4 bytes)
    const char magic[4] = {'K', 'T', 'M', 'G'};
    bin_file.write(magic, 4);
    
    // Version (2 bytes) - version 4 (SentencePiece ▁-normalized pieces)
    uint16_t version = 4;
    bin_file.write(reinterpret_cast<const char*>(&version), 2);
    
    // Checksum placeholder (4 bytes) - not used currently
    uint32_t checksum = 0;
    bin_file.write(reinterpret_cast<const char*>(&checksum), 4);
    
    // Config vocab_size (4 bytes) - number of unigram pieces
    uint32_t config_vocab_size = static_cast<uint32_t>(pieces_.size());
    bin_file.write(reinterpret_cast<const char*>(&config_vocab_size), 4);
    
    // Max length (4 bytes)
    uint32_t max_length = MAX_PIECE_LENGTH;
    bin_file.write(reinterpret_cast<const char*>(&max_length), 4);
    
    // 3 bools (3 bytes) - reserved for config flags
    char flags[3] = {0, 0, 0};
    bin_file.write(flags, 3);
    
    // Actual vocab size including special+byte+atom offsets (4 bytes)
    uint32_t total_vocab_size = static_cast<uint32_t>(
        NUM_SPECIAL_TOKENS + BYTE_VOCAB_SIZE + ATOM_VOCAB_SIZE + pieces_.size());
    bin_file.write(reinterpret_cast<const char*>(&total_vocab_size), 4);
    
    if (score_multiplier != 1.0f) {
        std::cout << "[UnigramLM] Applying vocab_score_multiplier=" << score_multiplier << " to all piece scores on save" << std::endl;
    }
    
    // Write pieces: length (4 bytes) + text + score (4 bytes float) + token_id (4 bytes)
    // token_id is position-derived (UNIGRAM_VOCAB_OFFSET + i), written for format compat.
    for (size_t i = 0; i < pieces_.size(); ++i) {
        const auto& piece = pieces_[i];
        uint32_t len = static_cast<uint32_t>(piece.text.size());
        bin_file.write(reinterpret_cast<const char*>(&len), 4);
        bin_file.write(piece.text.data(), len);
        float scaled_score = piece.score * score_multiplier;
        bin_file.write(reinterpret_cast<const char*>(&scaled_score), 4);
        int tid = tokenIdForIndex(static_cast<int>(i));
        bin_file.write(reinterpret_cast<const char*>(&tid), 4);
    }
    
    bin_file.close();
    std::cout << "[UnigramLM] Saved binary vocab (" << total_vocab_size
              << " embedding vocab = " << NUM_SPECIAL_TOKENS << " special + "
              << BYTE_VOCAB_SIZE << " bytes + " << ATOM_VOCAB_SIZE
              << " atom type placeholders + " << pieces_.size()
              << " unigram pieces) to " << bin_path << std::endl;
    
    // Optional: Save text format (.txt) for human readability
    if (save_text_format) {
        std::string txt_path = bin_path.substr(0, bin_path.rfind('.')) + ".txt";
        std::ofstream txt_file(txt_path);
        if (txt_file.is_open()) {
            for (const auto& piece : pieces_) {
                txt_file << piece.text << "\t" << (piece.score * score_multiplier) << "\n";
            }
            txt_file.close();
            std::cout << "[UnigramLM] Saved text vocab (human-readable) to " << txt_path << std::endl;
        } else {
            std::cerr << "[UnigramLM] Warning: Failed to create text vocab file: " << txt_path << std::endl;
        }
    }
    
    return true;
}

//--------------------------------------------------//
// Trie Building
//--------------------------------------------------//

void UnigramLM::buildTrie() {
    trie_.clear();
    trie_.push_back(TrieNode());  // Root node
    
    for (size_t i = 0; i < pieces_.size(); ++i) {
        const auto& piece = pieces_[i];
        int node = 0;
        
        for (unsigned char c : piece.text) {
            if (trie_[node].children[c] < 0) {
                trie_[node].children[c] = static_cast<int>(trie_.size());
                trie_.push_back(TrieNode());
            }
            node = trie_[node].children[c];
        }
        
        // Token ID is ALWAYS position-derived. No stored field.
        trie_[node].token_id = tokenIdForIndex(static_cast<int>(i));
        trie_[node].score = piece.score;
    }
}

//--------------------------------------------------//
// CPU Viterbi Encoding
//--------------------------------------------------//

std::vector<ViterbiNode> UnigramLM::viterbi(const std::string& text) const {
    size_t n = text.size();
    std::vector<ViterbiNode> nodes(n + 1);
    
    // Initialize
    nodes[0].score = 0.0f;
    nodes[0].prev_pos = -1;
    nodes[0].token_id = -1;
    nodes[0].piece_length = 0;
    
    for (size_t i = 1; i <= n; ++i) {
        nodes[i].score = -1e30f;
        nodes[i].prev_pos = -1;
        nodes[i].token_id = unk_id_;  // Absolute UNK_TOKEN_ID = 0
        nodes[i].piece_length = 1;
    }
    
    // Forward pass
    for (size_t pos = 0; pos < n; ++pos) {
        if (nodes[pos].score < -1e20f) continue;  // Unreachable
        
        unsigned char cur_byte = static_cast<unsigned char>(text[pos]);
        bool cur_is_punct = isPunctBoundary(cur_byte);
        
        // PUNCTUATION ISOLATION GUARD:
        // If current position is a punctuation character, force it to be emitted
        // as a single byte token and skip trie matching entirely.
        // This prevents tokens like "however," or "al." from ever being selected.
        if (cur_is_punct) {
            float byte_score = nodes[pos].score + UNKNOWN_SCORE;
            if (byte_score > nodes[pos + 1].score) {
                nodes[pos + 1].score = byte_score;
                nodes[pos + 1].prev_pos = static_cast<int>(pos);
                nodes[pos + 1].token_id = static_cast<int>(cur_byte) + BYTE_TOKEN_OFFSET;
                nodes[pos + 1].piece_length = 1;
            }
            continue;  // Skip trie search — punctuation is always isolated
        }
        
        // Try all pieces starting at pos
        {
            if (trie_.empty()) {
                throw std::runtime_error("viterbi(): trie_ is empty — buildTrie() was never called. "
                                         "Caller MUST build trie before encoding at " + 
                                         std::string(__FILE__) + ":" + std::to_string(__LINE__));
            }
            int node = 0;
            for (size_t len = 1; len <= MAX_PIECE_LENGTH && pos + len <= n; ++len) {
                unsigned char c = static_cast<unsigned char>(text[pos + len - 1]);
                
                // PUNCTUATION ISOLATION GUARD:
                // Stop extending the piece if we hit a punctuation character.
                // This prevents the trie from matching tokens that contain
                // punctuation mixed with letters (e.g., "al.", "et al.,").
                if (isPunctBoundary(c)) break;
                
                if (trie_[node].children[c] < 0) break;
                node = trie_[node].children[c];
                
                if (trie_[node].token_id >= 0) {
                    float score = nodes[pos].score + trie_[node].score;
                    
                    if (score > nodes[pos + len].score) {
                        nodes[pos + len].score = score;
                        nodes[pos + len].prev_pos = static_cast<int>(pos);
                        nodes[pos + len].token_id = trie_[node].token_id;
                        nodes[pos + len].piece_length = static_cast<int>(len);
                    }
                }
            }
        }
        
        if (enable_byte_fallback_) {
            // Byte fallback: allow single byte advance
            float byte_score = nodes[pos].score + UNKNOWN_SCORE;
            if (byte_score > nodes[pos + 1].score) {
                unsigned char byte_val = static_cast<unsigned char>(text[pos]);
                nodes[pos + 1].score = byte_score;
                nodes[pos + 1].prev_pos = static_cast<int>(pos);
                nodes[pos + 1].token_id = static_cast<int>(byte_val) + BYTE_TOKEN_OFFSET;  // Byte token ID (offset by specials)
                nodes[pos + 1].piece_length = 1;
            }
        } else {
            // Byte fallback disabled: advance with <unk> instead of byte tokens.
            std::cout << "[UnigramLM] Warning: byte fallback disabled, using <unk> token for unknown bytes" << std::endl;
            float unk_score = nodes[pos].score + UNKNOWN_SCORE;
            if (unk_score > nodes[pos + 1].score) {
                nodes[pos + 1].score = unk_score;
                nodes[pos + 1].prev_pos = static_cast<int>(pos);
                nodes[pos + 1].token_id = unk_id_;  // Absolute UNK_TOKEN_ID = 0
                nodes[pos + 1].piece_length = 1;
            }
        }
    }
    
    return nodes;
}

std::vector<int> UnigramLM::backtrack(const std::vector<ViterbiNode>& nodes, int end_pos) const {
    std::vector<int> tokens;
    int pos = end_pos;
    
    while (pos > 0) {
        tokens.push_back(nodes[pos].token_id);
        pos = nodes[pos].prev_pos;
    }
    
    std::reverse(tokens.begin(), tokens.end());
    return tokens;
}

std::vector<int> UnigramLM::encode(const std::string& text, bool prepend_space) const {
    if (text.empty()) return {};

    // SentencePiece-style normalization: spaces → ▁
    // prepend_space=true adds leading ▁ (start of text / first segment)
    // prepend_space=false skips prepend (mid-text segment after atom)
    std::string normalized = normalizeSpaces(text, prepend_space);
    auto nodes = viterbi(normalized);
    return backtrack(nodes, static_cast<int>(normalized.size()));
}

std::vector<UnigramPiece> UnigramLM::encodeWithPieces(const std::string& text) const {
    auto token_ids = encode(text);
    std::vector<UnigramPiece> result;
    result.reserve(token_ids.size());
    
    for (int tid : token_ids) {
        if (tid >= SPECIAL_TOKEN_OFFSET && tid < NUM_SPECIAL_TOKENS) {
            // Special token — token_id is NOT stored on UnigramPiece (it's position-derived).
            // Caller should use the token_ids from encode(), not from pieces.
            UnigramPiece piece;
            if (tid == UNK_TOKEN_ID) piece.text = "<unk>";
            else if (tid == PAD_TOKEN_ID) piece.text = "<pad>";
            else if (tid == BOS_TOKEN_ID) piece.text = "<s>";
            else if (tid == EOS_TOKEN_ID) piece.text = "</s>";
            piece.score = -10.0f;
            piece.is_special = true;
            piece.is_user_defined = true;
            result.push_back(piece);
        } else if (tid >= BYTE_TOKEN_OFFSET && tid < ATOM_TOKEN_OFFSET) {
            // Byte token
            UnigramPiece piece;
            piece.text = std::string(1, static_cast<char>(tid - BYTE_TOKEN_OFFSET));
            piece.score = UNKNOWN_SCORE;
            piece.is_special = false;
            piece.is_user_defined = false;
            result.push_back(piece);
        } else if (tid >= UNIGRAM_VOCAB_OFFSET) {
            const UnigramPiece* p = getPiece(tid);
            if (p) {
                result.push_back(*p);
            }
        }
    }
    
    return result;
}

std::string UnigramLM::decode(const std::vector<int>& token_ids) const {
    return decode(token_ids.data(), token_ids.size());
}

std::string UnigramLM::decode(const int* token_ids, size_t count) const {
    std::string result;
    
    for (size_t i = 0; i < count; ++i) {
        int tid = token_ids[i];
        
        if (tid >= SPECIAL_TOKEN_OFFSET && tid < NUM_SPECIAL_TOKENS) {
            // Special token — decode as their text repr
            if (tid == UNK_TOKEN_ID) result += "<unk>";
            else if (tid == PAD_TOKEN_ID) { /* skip padding */ }
            else if (tid == BOS_TOKEN_ID) result += "<s>";
            else if (tid == EOS_TOKEN_ID) result += "</s>";
        } else if (tid >= BYTE_TOKEN_OFFSET && tid < BYTE_TOKEN_OFFSET + BYTE_VOCAB_SIZE) {
            // Byte token (subtract BYTE_TOKEN_OFFSET to get raw byte)
            result.push_back(static_cast<char>(tid - BYTE_TOKEN_OFFSET));
        } else if (tid >= UNIGRAM_VOCAB_OFFSET) {
            // Unigram token
            const UnigramPiece* p = getPiece(tid);
            if (p) {
                result += p->text;
            }
        }
        // Atom tokens are handled by ScratchBlock
    }
    
    return denormalizeSpaces(result);
}

void UnigramLM::capVocabSize(int max_size) {
    if (max_size >= static_cast<int>(pieces_.size())) {
        return;  // Already smaller than cap
    }
    
    if (max_size < 4) {
        throw std::runtime_error("capVocabSize: max_size must be >= 4 to include minimum vocabulary");
    }
    
    // Sort pieces by score (descending) to keep most frequent
    // But always keep user-defined tokens (special tokens) regardless of score
    std::vector<size_t> indices(pieces_.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    std::stable_sort(indices.begin(), indices.end(), [this](size_t a, size_t b) {
        // User-defined tokens always come first
        if (pieces_[a].is_user_defined != pieces_[b].is_user_defined) {
            return pieces_[a].is_user_defined;  // user-defined = true sorts before false
        }
        return pieces_[a].score > pieces_[b].score;  // Higher score = more frequent
    });
    
    // Keep top max_size pieces
    std::vector<UnigramPiece> new_pieces;
    new_pieces.reserve(max_size);
    
    std::unordered_map<std::string, int> new_piece_to_id;
    
    for (int i = 0; i < max_size && i < static_cast<int>(indices.size()); ++i) {
        UnigramPiece piece = pieces_[indices[i]];
        // Token ID is always UNIGRAM_VOCAB_OFFSET + index — no field to reassign.
        new_piece_to_id[piece.text] = static_cast<int>(new_pieces.size());
        new_pieces.push_back(piece);
    }
    
    pieces_ = std::move(new_pieces);
    piece_to_id_ = std::move(new_piece_to_id);
    
    // Rebuild trie for fast encoding (uses new token_ids)
    buildTrie();
    
    std::cout << "[UnigramLM] Capped vocab to " << pieces_.size() << " pieces" << std::endl;
}

//--------------------------------------------------//
// GPU Implementation
//--------------------------------------------------//

bool UnigramLM::initGPU() {
    if (gpu_->initialized) return true;
    
    if (trie_.empty()) {
        buildTrie();
    }
    
    return uploadTrieToGPU();
}

bool UnigramLM::uploadTrieToGPU() {
    cudaError_t err;
    size_t num_nodes = trie_.size();
    
    // Helper lambda for cleanup on failure
    auto cleanup = [this]() {
        if (gpu_->d_trie_children) { cudaFree(gpu_->d_trie_children); gpu_->d_trie_children = nullptr; }
        if (gpu_->d_trie_token_ids) { cudaFree(gpu_->d_trie_token_ids); gpu_->d_trie_token_ids = nullptr; }
        if (gpu_->d_trie_scores) { cudaFree(gpu_->d_trie_scores); gpu_->d_trie_scores = nullptr; }
        if (gpu_->d_piece_data) { cudaFree(gpu_->d_piece_data); gpu_->d_piece_data = nullptr; }
        if (gpu_->d_piece_offsets) { cudaFree(gpu_->d_piece_offsets); gpu_->d_piece_offsets = nullptr; }
        if (gpu_->d_piece_lengths) { cudaFree(gpu_->d_piece_lengths); gpu_->d_piece_lengths = nullptr; }
        if (gpu_->d_viterbi_scores) { cudaFree(gpu_->d_viterbi_scores); gpu_->d_viterbi_scores = nullptr; }
        if (gpu_->d_viterbi_prev) { cudaFree(gpu_->d_viterbi_prev); gpu_->d_viterbi_prev = nullptr; }
        if (gpu_->d_viterbi_tokens) { cudaFree(gpu_->d_viterbi_tokens); gpu_->d_viterbi_tokens = nullptr; }
    };
    
    // Allocate trie arrays
    err = cudaMalloc(&gpu_->d_trie_children, num_nodes * 256 * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "[UnigramLM] Failed to allocate trie_children" << std::endl;
        return false;
    }
    
    err = cudaMalloc(&gpu_->d_trie_token_ids, num_nodes * sizeof(int));
    if (err != cudaSuccess) {
        cleanup();
        return false;
    }
    
    err = cudaMalloc(&gpu_->d_trie_scores, num_nodes * sizeof(float));
    if (err != cudaSuccess) {
        cleanup();
        return false;
    }
    
    // Flatten and upload trie data
    std::vector<int> children_flat(num_nodes * 256);
    std::vector<int> token_ids_flat(num_nodes);
    std::vector<float> scores_flat(num_nodes);
    
    for (size_t i = 0; i < num_nodes; ++i) {
        for (int c = 0; c < 256; ++c) {
            children_flat[i * 256 + c] = trie_[i].children[c];
        }
        token_ids_flat[i] = trie_[i].token_id;
        scores_flat[i] = trie_[i].score;
    }
    
    cudaMemcpy(gpu_->d_trie_children, children_flat.data(), 
               num_nodes * 256 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_->d_trie_token_ids, token_ids_flat.data(),
               num_nodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_->d_trie_scores, scores_flat.data(),
               num_nodes * sizeof(float), cudaMemcpyHostToDevice);
    
    gpu_->num_nodes = static_cast<int>(num_nodes);
    
    // Upload piece data for decoding
    size_t total_piece_length = 0;
    for (const auto& p : pieces_) {
        total_piece_length += p.text.size();
    }
    
    std::vector<char> piece_data(total_piece_length);
    std::vector<int> piece_offsets(pieces_.size());
    std::vector<int> piece_lengths(pieces_.size());
    
    size_t offset = 0;
    for (size_t i = 0; i < pieces_.size(); ++i) {
        piece_offsets[i] = static_cast<int>(offset);
        piece_lengths[i] = static_cast<int>(pieces_[i].text.size());
        std::copy(pieces_[i].text.begin(), pieces_[i].text.end(), 
                  piece_data.begin() + offset);
        offset += pieces_[i].text.size();
    }
    
    err = cudaMalloc(&gpu_->d_piece_data, total_piece_length > 0 ? total_piece_length : 1);
    if (err != cudaSuccess) { cleanup(); return false; }
    
    err = cudaMalloc(&gpu_->d_piece_offsets, pieces_.size() * sizeof(int));
    if (err != cudaSuccess) { cleanup(); return false; }
    
    err = cudaMalloc(&gpu_->d_piece_lengths, pieces_.size() * sizeof(int));
    if (err != cudaSuccess) { cleanup(); return false; }
    
    // Pre-allocate Viterbi workspace with fixed capacity
    constexpr size_t MAX_SEQUENCE_LENGTH = HyperParameters::UNIGRAM_MAX_SEQUENCE_LENGTH;
    gpu_->workspace_max_length = MAX_SEQUENCE_LENGTH;
    
    err = cudaMalloc(&gpu_->d_viterbi_scores, (MAX_SEQUENCE_LENGTH + 1) * sizeof(float));
    if (err != cudaSuccess) { cleanup(); return false; }
    
    err = cudaMalloc(&gpu_->d_viterbi_prev, (MAX_SEQUENCE_LENGTH + 1) * sizeof(int));
    if (err != cudaSuccess) { cleanup(); return false; }
    
    err = cudaMalloc(&gpu_->d_viterbi_tokens, (MAX_SEQUENCE_LENGTH + 1) * sizeof(int));
    if (err != cudaSuccess) { cleanup(); return false; }
    
    cudaMemcpy(gpu_->d_piece_data, piece_data.data(), 
               total_piece_length, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_->d_piece_offsets, piece_offsets.data(),
               pieces_.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_->d_piece_lengths, piece_lengths.data(),
               pieces_.size() * sizeof(int), cudaMemcpyHostToDevice);
    
    gpu_->initialized = true;
    std::cout << "[UnigramLM] GPU initialized with " << num_nodes << " trie nodes" << std::endl;
    return true;
}

bool UnigramLM::encodeGPU(const char* d_text,
                          size_t length,
                          int* d_token_ids,
                          int* d_token_count,
                          int max_tokens,
                          bool* d_needs_byte_fallback) {
    if (!gpu_->initialized) {
        if (!initGPU()) return false;
    }
    
    // Validate length against pre-allocated workspace capacity
    if (length > gpu_->workspace_max_length) {
        std::cerr << "[UnigramLM] Input length " << length 
                  << " exceeds workspace capacity " << gpu_->workspace_max_length << std::endl;
        return false;
    }
    
    const bool enable_fallback = enable_byte_fallback_ && d_needs_byte_fallback;
    if (enable_fallback) {
        // Initialize fallback flags to false
        // NOTE: Using default stream (nullptr) is intentional here - Unigram operates
        // in the data loading path which may run on separate thread from training
        cudaMemsetAsync(d_needs_byte_fallback, 0, length * sizeof(bool), nullptr);
    }
    
    // Forward pass (single-threaded kernel due to Viterbi sequential dependency)
    bool* fallback_ptr = enable_fallback ? d_needs_byte_fallback : nullptr;
    // INTENTIONAL: Stream 0 for synchronous Viterbi forward pass (cudaMemcpy follows)
    // NOTE: Kernel runs single-threaded because Viterbi has O(n) sequential dependency
    kernelViterbiForward<<<1, 1, 0, 0>>>(
        d_text, length,
        gpu_->d_trie_children, gpu_->d_trie_token_ids, gpu_->d_trie_scores,
        gpu_->num_nodes,
        gpu_->d_viterbi_scores, gpu_->d_viterbi_prev, gpu_->d_viterbi_tokens,
        fallback_ptr,
        unk_id_,  // Absolute UNK_TOKEN_ID = 0
        UNKNOWN_SCORE,
        enable_fallback
    );
    
    // INTENTIONAL: Stream 0 for synchronous Viterbi backtrack (result copied back immediately)
    // Backtrack with max_tokens to prevent buffer overflow
    kernelViterbiBacktrack<<<1, 1, 0, 0>>>(
        length,
        gpu_->d_viterbi_prev, gpu_->d_viterbi_tokens,
        d_token_ids, d_token_count,
        max_tokens
    );
    
    return cudaGetLastError() == cudaSuccess;
}

bool UnigramLM::encodeBatchGPU(const char* const* d_texts,
                                const size_t* lengths,
                                int** d_token_ids,
                                int* d_token_counts,
                                int max_tokens_per_seq,
                                size_t batch_size) {
    if (batch_size == 0) return true;
    
    // NOTE: Sequences are processed sequentially because:
    // 1. Viterbi algorithm has inherent sequential dependency (each position depends on all previous)
    // 2. The pre-allocated workspace (d_viterbi_scores/prev/tokens) is shared across all sequences
    // True batch parallelization would require per-sequence workspace allocation, which trades
    // memory for parallelism. For typical batch sizes (8-32), sequential processing is adequate
    // since the bottleneck is usually elsewhere (embedding lookup, attention).
    
    // Find max length to allocate fallback buffer once (avoid per-iteration malloc!)
    size_t max_len = 0;
    for (size_t i = 0; i < batch_size; ++i) {
        max_len = std::max(max_len, lengths[i]);
    }
    
    // Single allocation for entire batch
    bool* d_fallback = nullptr;
    cudaError_t err = cudaMalloc(&d_fallback, max_len > 0 ? max_len * sizeof(bool) : sizeof(bool));
    if (err != cudaSuccess) {
        std::cerr << "[UnigramLM] Failed to allocate batch fallback buffer" << std::endl;
        return false;
    }
    
    // Process each sequence reusing the same buffer
    bool success = true;
    for (size_t i = 0; i < batch_size && success; ++i) {
        success = encodeGPU(d_texts[i], lengths[i], d_token_ids[i], 
                            &d_token_counts[i], max_tokens_per_seq, d_fallback);
    }
    
    cudaFree(d_fallback);
    return success;
}

bool UnigramLM::decodeGPU(const int* d_token_ids,
                          size_t count,
                          char* d_output,
                          size_t* d_output_length,
                          size_t max_output_length) {
    if (!gpu_->initialized) {
        if (!initGPU()) return false;
    }
    
    // INTENTIONAL: Stream 0 for synchronous decode (result copied back immediately)
    kernelUnigramDecode<<<1, 1, 0, 0>>>(
        d_token_ids, count,
        gpu_->d_piece_data, gpu_->d_piece_offsets, gpu_->d_piece_lengths,
        UNIGRAM_VOCAB_OFFSET, static_cast<int>(pieces_.size()),
        d_output, d_output_length, max_output_length
    );
    
    return cudaGetLastError() == cudaSuccess;
}

} // namespace Tokenizer
} // namespace GRIM
