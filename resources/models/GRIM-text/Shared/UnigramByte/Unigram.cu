//======================================================//
//  Unigram.cu
//  CUDA implementation of Unigram Language Model tokenizer
//======================================================//

#include "Unigram.hpp"
#include "Byte.hpp"
#include "HyperParameters/HyperParameters_GPU.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <unordered_set>
#include <cctype>

namespace GRIM {

//======================================================//
//  Case Normalization
//  Enforces case-insensitive vocabulary to prevent
//  near-duplicate tokens like 'The' vs 'the' vs ' the'
//======================================================//

static std::string toLowerASCII(const std::string& s) {
    std::string result = s;
    for (char& c : result) {
        if (c >= 'A' && c <= 'Z') c = c - 'A' + 'a';
    }
    return result;
}

// Check if character is punctuation (ASCII subset for speed)
static inline bool isPunct(char c) {
    return (c >= '!' && c <= '/') ||  // !"#$%&'()*+,-./
           (c >= ':' && c <= '@') ||  // :;<=>?@
           (c >= '[' && c <= '`') ||  // [\]^_`
           (c >= '{' && c <= '~');    // {|}~
}

static inline bool isWhitespaceASCII(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

// Normalize for near-duplicate detection: lowercase + strip leading/trailing whitespace
// Used to detect pairs like " the"/"the" and repeated variants like "soooo"/"soo"
static std::string normalizeForDedup(const std::string& s) {
    std::string lower = toLowerASCII(s);
    size_t start = 0;
    while (start < lower.size() && isWhitespaceASCII(static_cast<unsigned char>(lower[start]))) {
        ++start;
    }
    if (start >= lower.size()) return "";

    size_t end = lower.size();
    while (end > start && isWhitespaceASCII(static_cast<unsigned char>(lower[end - 1]))) {
        --end;
    }

    std::string normalized;
    normalized.reserve(end - start);

    char prev = '\0';
    int run = 0;
    bool last_was_space = false;

    for (size_t i = start; i < end; ++i) {
        unsigned char uc = static_cast<unsigned char>(lower[i]);
        char c = static_cast<char>(uc);

        if (isWhitespaceASCII(uc)) {
            if (!last_was_space) {
                normalized.push_back(' ');
                last_was_space = true;
            }
            prev = '\0';
            run = 0;
            continue;
        }

        last_was_space = false;
        if (c == prev) {
            run++;
        } else {
            prev = c;
            run = 1;
        }

        int keep_limit = std::numeric_limits<int>::max();
        if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) {
            keep_limit = 2;
        } else if (c < 128 && isPunct(c)) {
            keep_limit = (c == '.') ? 3 : 2;
        }

        if (run <= keep_limit) {
            normalized.push_back(c);
        }
    }

    return normalized;
}

//======================================================//
//  Subword Quality Filter
//  Rejects linguistically nonsensical patterns like "P.," 
//======================================================//

// Check if character is alphanumeric
static inline bool isAlnum(char c) {
    return (c >= 'A' && c <= 'Z') || 
           (c >= 'a' && c <= 'z') || 
           (c >= '0' && c <= '9');
}

// Reject long repeated runs (e.g., "aaaaaa", "!!!!!!", "word  word")
static bool hasExcessiveRunLength(const std::string& s) {
    if (s.empty()) return false;

    char prev = s[0];
    int run = 1;
    for (size_t i = 1; i < s.size(); ++i) {
        char c = s[i];
        if (c == prev) {
            run++;
        } else {
            prev = c;
            run = 1;
        }

        unsigned char uc = static_cast<unsigned char>(c);
        const bool is_alpha = (uc >= 'A' && uc <= 'Z') || (uc >= 'a' && uc <= 'z');
        const bool is_digit = (uc >= '0' && uc <= '9');

        if (is_alpha && run > 3) return true;
        if (is_digit && run > 6) return true;
        if (uc < 128 && isPunct(c) && run > 4) return true;
        if (isWhitespaceASCII(uc) && run > 1) return true;
    }

    return false;
}

// Reject repeated whole-pattern tokens like "hahaha", "abcabcabc", "121212"
static bool isRepeatedPatternNoise(const std::string& s) {
    if (s.size() < 6) return false;

    for (unsigned char c : s) {
        if (isWhitespaceASCII(c)) return false;
    }

    const size_t max_pattern_len = std::min<size_t>(4, s.size() / 3);
    for (size_t pattern_len = 1; pattern_len <= max_pattern_len; ++pattern_len) {
        if (s.size() % pattern_len != 0) continue;
        const size_t repeats = s.size() / pattern_len;
        if (repeats < 3) continue;

        bool repeated = true;
        for (size_t i = pattern_len; i < s.size(); ++i) {
            if (s[i] != s[i % pattern_len]) {
                repeated = false;
                break;
            }
        }
        if (repeated) return true;
    }

    return false;
}

// Reject stutter-like tokens where the same word repeats 3+ times ("i i i")
static bool isWordLevelStutter(const std::string& s) {
    size_t i = 0;
    while (i < s.size() && isWhitespaceASCII(static_cast<unsigned char>(s[i]))) ++i;
    if (i >= s.size()) return false;

    size_t first_start = i;
    while (i < s.size() && !isWhitespaceASCII(static_cast<unsigned char>(s[i]))) ++i;
    size_t first_len = i - first_start;
    if (first_len == 0) return false;

    int word_count = 1;
    while (i < s.size()) {
        while (i < s.size() && isWhitespaceASCII(static_cast<unsigned char>(s[i]))) ++i;
        if (i >= s.size()) break;

        size_t word_start = i;
        while (i < s.size() && !isWhitespaceASCII(static_cast<unsigned char>(s[i]))) ++i;
        size_t word_len = i - word_start;

        if (word_len != first_len) return false;
        for (size_t k = 0; k < word_len; ++k) {
            if (s[word_start + k] != s[first_start + k]) return false;
        }

        word_count++;
    }

    return word_count >= 3;
}

static bool isRepetitionNoise(const std::string& s) {
    return hasExcessiveRunLength(s) || isRepeatedPatternNoise(s) || isWordLevelStutter(s);
}

// Check if subword is linguistically valid
// Rejects garbage patterns that shouldn't be vocabulary tokens
static bool isValidSubword(const std::string& s) {
    if (s.empty()) return false;
    if (s.size() == 1) return true;  // Single chars always OK

    // Repetition stripping: reject obvious run-length / stutter artifacts
    if (isRepetitionNoise(s)) return false;
    
    // Count character types
    int alpha_count = 0;
    int digit_count = 0;
    int punct_count = 0;
    int space_count = 0;
    int upper_count = 0;
    int lower_count = 0;
    int control_count = 0;
    
    for (unsigned char c : s) {
        if (c >= 'A' && c <= 'Z') { alpha_count++; upper_count++; }
        else if (c >= 'a' && c <= 'z') { alpha_count++; lower_count++; }
        else if (c >= '0' && c <= '9') digit_count++;
        else if (isWhitespaceASCII(c)) space_count++;
        else if (c < 128 && isPunct(static_cast<char>(c))) punct_count++;
        else if (c < 32 || c == 127) control_count++;
    }
    
    int total = static_cast<int>(s.size());

    // Reject control bytes and pure multi-whitespace fragments.
    if (control_count > 0) return false;
    if (space_count == total) return false;
    
    // === REJECT PATTERNS ===
    
    // 1. Pure punctuation combinations (except common ones like "..." or "--")
    if (punct_count == total) {
        // Allow only: "...", "--", "***", etc. (repeated single char)
        bool all_same = true;
        for (size_t i = 1; i < s.size(); ++i) {
            if (s[i] != s[0]) { all_same = false; break; }
        }
        if (!all_same) return false;
        // Allow repeated punctuation only up to length 4
        if (s.size() > 4) return false;
    }
    
    // 2. Single letter + punctuation garbage (like "P.,", "A:", "B;")
    //    Exception: Common patterns like "I'" (contractions), "U.S", etc.
    if (alpha_count == 1 && punct_count >= 1 && s.size() <= 4) {
        // Find the letter position
        size_t letter_pos = 0;
        for (size_t i = 0; i < s.size(); ++i) {
            unsigned char c = static_cast<unsigned char>(s[i]);
            if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) {
                letter_pos = i;
                break;
            }
        }
        
        // Reject if letter is NOT at start or end (weird position)
        if (letter_pos != 0 && letter_pos != s.size() - 1) {
            return false;
        }
        
        // Reject single uppercase + multiple punct (like "P.,")
        if (upper_count == 1 && punct_count >= 2) {
            return false;
        }
        
        // Reject single letter + comma/semicolon/colon at end (like "P,", "A;")
        char last = s.back();
        char first = s.front();
        if (alpha_count == 1 && (last == ',' || last == ';' || last == ':')) {
            // Exception: common abbreviations with periods before comma
            if (!(s.size() >= 2 && s[s.size()-2] == '.')) {
                return false;
            }
        }
        if (alpha_count == 1 && (first == ',' || first == ';' || first == ':')) {
            return false;
        }
    }
    
    // 3. Mixed punctuation chaos (multiple different punctuation types)
    if (punct_count >= 2 && alpha_count <= 1) {
        std::unordered_set<char> punct_types;
        for (char c : s) {
            if (isPunct(c)) punct_types.insert(c);
        }
        // More than 2 different punctuation types is garbage
        if (punct_types.size() > 2) return false;
        
        // Certain combos are always bad
        bool has_comma = punct_types.count(',') > 0;
        bool has_period = punct_types.count('.') > 0;
        bool has_semicolon = punct_types.count(';') > 0;
        bool has_colon = punct_types.count(':') > 0;
        
        // ".,", ",.", ".;", etc. are garbage
        if (has_comma && has_period && alpha_count == 0) return false;
        if (has_comma && has_semicolon) return false;
        if (has_period && has_semicolon && alpha_count <= 1) return false;
    }
    
    // 4. Starts/ends with space + punctuation (like " ,", "; ")
    if (s.size() >= 2) {
        if (s[0] == ' ' && isPunct(s[1])) return false;
        if (s.back() == ' ' && s.size() >= 2 && isPunct(s[s.size()-2])) {
            // Exception: ". " and ", " are OK (sentence/clause boundaries)
            if (!(s[s.size()-2] == '.' || s[s.size()-2] == ',' || 
                  s[s.size()-2] == '!' || s[s.size()-2] == '?')) {
                return false;
            }
        }
    }
    
    // 5. Reject tokens that are mostly punctuation with scattered letters
    //    (like ".a.", "a.b", etc. unless it's a known pattern)
    if (punct_count > alpha_count && alpha_count >= 1 && alpha_count <= 2) {
        // Allow: contractions like "'s", "'t", "'ll", "'re", "'ve"
        if (s.size() == 2 && s[0] == '\'') return true;
        if (s.size() == 3 && s[0] == '\'') return true;
        
        // Allow: possessives/quotes like "s'"
        if (s.size() == 2 && s[1] == '\'') return true;
        
        // Allow: abbreviation patterns like "U.S" or "e.g"
        if (alpha_count == 2 && punct_count == 1 && s.find('.') != std::string::npos) {
            // Check if it's letter.letter pattern
            bool valid_abbrev = false;
            for (size_t i = 1; i < s.size() - 1; ++i) {
                if (s[i] == '.' && isAlnum(s[i-1]) && isAlnum(s[i+1])) {
                    valid_abbrev = true;
                    break;
                }
            }
            if (valid_abbrev) return true;
        }
        
        // Otherwise reject
        return false;
    }
    
    // 6. Tokens starting with lowercase + immediate uppercase (like "aB")
    //    These are almost never valid (except camelCase which is rare in natural text)
    if (s.size() >= 2 && lower_count > 0 && upper_count > 0) {
        if (s[0] >= 'a' && s[0] <= 'z' && s[1] >= 'A' && s[1] <= 'Z') {
            // Reject short camelCase fragments
            if (s.size() <= 3) return false;
        }
    }
    
    // === ACCEPT ===
    return true;
}

//======================================================//
namespace Tokenizer {

//======================================================//
//  CUDA Kernels
//======================================================//

// Kernel: Sequential Viterbi forward pass
// CRITICAL: Viterbi has O(n) sequential dependency - each position depends on ALL previous.
// Parallel execution would cause data races (reading viterbi_scores before they're computed).
// We parallelize the TRIE SEARCH within each position, not across positions.
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

// DELETED: Auto-ID overload removed per Rule 20 (no silent fallbacks)
// Callers MUST provide explicit token_id = UNIGRAM_VOCAB_OFFSET + pieces_.size()

void UnigramLM::addPiece(const std::string& text, float score, int token_id, bool is_user_defined) {
    if (piece_to_id_.count(text)) {
        // Update existing piece - but preserve token_id for user-defined tokens!
        // User-defined tokens (special tokens like <unk>, <s>) have manually assigned IDs.
        // If we overwrite their token_id, we create gaps in the ID sequence and duplicates.
        int id = piece_to_id_[text];
        pieces_[id].score = score;
        // Only update token_id if the existing piece is NOT user-defined
        if (!pieces_[id].is_user_defined) {
            pieces_[id].token_id = token_id;
            pieces_[id].is_user_defined = is_user_defined;
        }
        return;
    }
    
    UnigramPiece piece;
    piece.text = text;
    piece.score = score;
    piece.token_id = token_id;
    piece.is_special = (text.front() == '<' && text.back() == '>');
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
            
            addPiece(piece, score, UNIGRAM_VOCAB_OFFSET + static_cast<int>(pieces_.size()), false);
            
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
    
    // Read version (2 bytes) - v3 required (includes token_id)
    uint16_t version;
    bin_file.read(reinterpret_cast<char*>(&version), 2);
    if (version != 3) {
        throw std::runtime_error("[UnigramLM] Vocab file version " + std::to_string(version) + 
            " is not supported. Required version 3. Re-save vocab with current tooling.");
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
        
        // Assign new token_id based on current layout
        int new_token_id = UNIGRAM_VOCAB_OFFSET + static_cast<int>(pieces_.size());
        
        addPiece(text, score, new_token_id, false);
    }
    
    if (!bin_file) {
        throw std::runtime_error("[UnigramLM] Error reading binary vocab file - unexpected EOF or read error");
    }
    
    buildTrie();
    
    std::cout << "[UnigramLM] Loaded " << pieces_.size() << " pieces from binary: " << vocab_path << std::endl;
    std::cout << "[UnigramLM] Total vocab size (with bytes+atoms): " << total_vocab_size << std::endl;
    return true;
}

bool UnigramLM::save(const std::string& vocab_path, bool save_text_format) const {
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
    
    // Version (2 bytes) - version 3 (includes token_id per piece)
    uint16_t version = 3;
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
    
    // Write pieces: length (4 bytes) + text + score (4 bytes float) + token_id (4 bytes)
    for (const auto& piece : pieces_) {
        uint32_t len = static_cast<uint32_t>(piece.text.size());
        bin_file.write(reinterpret_cast<const char*>(&len), 4);
        bin_file.write(piece.text.data(), len);
        bin_file.write(reinterpret_cast<const char*>(&piece.score), 4);
        bin_file.write(reinterpret_cast<const char*>(&piece.token_id), 4);
    }
    
    bin_file.close();
    std::cout << "[UnigramLM] Saved binary vocab (" << total_vocab_size 
              << " total tokens) to " << bin_path << std::endl;
    
    // Optional: Save text format (.txt) for human readability
    if (save_text_format) {
        std::string txt_path = bin_path.substr(0, bin_path.rfind('.')) + ".txt";
        std::ofstream txt_file(txt_path);
        if (txt_file.is_open()) {
            for (const auto& piece : pieces_) {
                txt_file << piece.text << "\t" << piece.score << "\n";
            }
            txt_file.close();
            std::cout << "[UnigramLM] Saved text vocab (human-readable) to " << txt_path << std::endl;
        } else {
            std::cerr << "[UnigramLM] Warning: Failed to create text vocab file: " << txt_path << std::endl;
        }
    }
    
    return true;
}

bool UnigramLM::trainFromCorpus(const std::vector<std::string>& texts,
                                 int target_vocab_size,
                                 float character_coverage,
                                 int min_subword_freq,
                                 bool prune_during_mining) {
    std::cout << "[UnigramLM] Training vocabulary from " << texts.size() 
              << " texts (target_vocab_size=" << target_vocab_size << ")" << std::endl;
    std::cout << "[UnigramLM] min_subword_freq=" << min_subword_freq 
              << ", prune_during_mining=" << (prune_during_mining ? "true" : "false") << std::endl;
    
    // CRITICAL: Segment documents into sentences BEFORE subword mining.
    // This prevents learning garbage tokens that cross sentence boundaries
    // (e.g., ". The", "., B", "). This" which are meaningless).
    std::vector<std::string> sentences;
    sentences.reserve(texts.size() * 5);  // Estimate ~5 sentences per document
    
    // Simple sentence segmentation: split on [.!?] followed by whitespace and capital
    // Also split on newlines (paragraph boundaries)
    for (const auto& text : texts) {
        if (text.empty()) continue;
        
        size_t start = 0;
        for (size_t i = 0; i < text.size(); ++i) {
            char c = text[i];
            bool is_sentence_end = false;
            
            // Check for sentence-ending punctuation followed by whitespace + capital
            if ((c == '.' || c == '!' || c == '?') && i + 2 < text.size()) {
                char next = text[i + 1];
                char after = text[i + 2];
                // Pattern: [.!?] + space + uppercase letter
                if ((next == ' ' || next == '\n' || next == '\t') && 
                    (after >= 'A' && after <= 'Z')) {
                    is_sentence_end = true;
                }
            }
            // Also split on newlines (paragraph boundaries)
            else if (c == '\n' && i > start) {
                is_sentence_end = true;
            }
            
            if (is_sentence_end) {
                // Include the punctuation in this sentence
                size_t end = (c == '\n') ? i : i + 1;
                std::string sentence = text.substr(start, end - start);
                // Trim whitespace
                size_t first = sentence.find_first_not_of(" \t\n\r");
                if (first != std::string::npos) {
                    size_t last = sentence.find_last_not_of(" \t\n\r");
                    sentence = sentence.substr(first, last - first + 1);
                    if (sentence.length() >= 3) {  // Skip very short fragments
                        sentences.push_back(std::move(sentence));
                    }
                }
                start = (c == '\n') ? i + 1 : i + 2;  // Skip past whitespace
            }
        }
        // Add remaining text as final sentence
        if (start < text.size()) {
            std::string sentence = text.substr(start);
            size_t first = sentence.find_first_not_of(" \t\n\r");
            if (first != std::string::npos) {
                size_t last = sentence.find_last_not_of(" \t\n\r");
                sentence = sentence.substr(first, last - first + 1);
                if (sentence.length() >= 3) {
                    sentences.push_back(std::move(sentence));
                }
            }
        }
    }
    
    std::cout << "[UnigramLM] Segmented " << texts.size() << " documents into " 
              << sentences.size() << " sentences for subword mining" << std::endl;
    
    // === CASE NORMALIZATION ===
    // Lowercase ALL training text for case-insensitive vocabulary.
    // Uppercase characters are still encodable via byte fallback (tokens 0-255).
    // This prevents near-duplicate tokens: 'The'/'the', ' And'/' and', etc.
    std::cout << "[UnigramLM] Normalizing to lowercase (case-insensitive vocab)" << std::endl;
    for (auto& sent : sentences) {
        sent = toLowerASCII(sent);
    }
    
    // Also create lowercased copy of full texts for char counting + EM iterations
    std::vector<std::string> lowercase_texts;
    lowercase_texts.reserve(texts.size());
    for (const auto& text : texts) {
        lowercase_texts.push_back(toLowerASCII(text));
    }
    
    // Use sentences instead of full documents for the rest of training
    const std::vector<std::string>& training_units = sentences;
    
    // Minimum frequency: subwords appearing fewer times than this are not included
    // This prevents noise from rare subwords while keeping everything that matters
    const int MIN_SUBWORD_FREQ = min_subword_freq;
    
    // Calculate total corpus size (for logging)
    size_t total_corpus_bytes = 0;
    for (const auto& text : texts) {
        total_corpus_bytes += text.size();
    }
    std::cout << "[UnigramLM] Total corpus size: " << (total_corpus_bytes / (1024*1024)) << " MB" << std::endl;
    
    // Calculate sentence corpus size for sampling
    size_t total_sentence_bytes = 0;
    for (const auto& sent : training_units) {
        total_sentence_bytes += sent.size();
    }
    
    // For large corpora, sample SENTENCES to avoid OOM during subword generation
    constexpr size_t MAX_SUBWORD_MINING_BYTES = HyperParameters::UNIGRAM_MAX_SUBWORD_BYTES;
    bool use_sampling = total_sentence_bytes > MAX_SUBWORD_MINING_BYTES;
    std::vector<size_t> sample_indices;
    
    if (use_sampling) {
        std::mt19937 rng(42);
        std::vector<size_t> all_indices(training_units.size());
        std::iota(all_indices.begin(), all_indices.end(), 0);
        std::shuffle(all_indices.begin(), all_indices.end(), rng);
        
        size_t sampled_bytes = 0;
        for (size_t idx : all_indices) {
            if (sampled_bytes >= MAX_SUBWORD_MINING_BYTES) break;
            sample_indices.push_back(idx);
            sampled_bytes += training_units[idx].size();
        }
        std::cout << "[UnigramLM] Sampling " << sample_indices.size() << " sentences (" 
                  << (sampled_bytes / (1024*1024)) << " MB) for subword mining" << std::endl;
    }
    
    // Step 1: Count character frequencies (use ALL texts, lowercased)
    std::unordered_map<std::string, int> char_counts;
    size_t total_chars = 0;
    
    for (const auto& text : lowercase_texts) {
        for (size_t i = 0; i < text.size(); ) {
            int seq_len = 1;
            unsigned char c = static_cast<unsigned char>(text[i]);
            
            // Handle UTF-8
            if ((c & 0x80) == 0x00) seq_len = 1;
            else if ((c & 0xE0) == 0xC0) seq_len = 2;
            else if ((c & 0xF0) == 0xE0) seq_len = 3;
            else if ((c & 0xF8) == 0xF0) seq_len = 4;
            
            if (i + seq_len <= text.size()) {
                std::string ch = text.substr(i, seq_len);
                char_counts[ch]++;
                total_chars++;
            }
            i += seq_len;
        }
    }
    
    // Step 2: Build initial vocabulary (all characters meeting coverage)
    std::vector<std::pair<std::string, int>> sorted_chars(char_counts.begin(), char_counts.end());
    std::sort(sorted_chars.begin(), sorted_chars.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    pieces_.clear();
    piece_to_id_.clear();
    
    // Special tokens are at absolute IDs 0-3, NOT stored in pieces_.
    // pieces_ contains only regular unigram vocabulary.
    
    size_t covered = 0;
    size_t coverage_target = static_cast<size_t>(total_chars * character_coverage);
    
    for (const auto& [ch, count] : sorted_chars) {
        if (covered >= coverage_target && pieces_.size() >= 256) break;
        
        float score = std::log(static_cast<float>(count) / total_chars);
        addPiece(ch, score, UNIGRAM_VOCAB_OFFSET + static_cast<int>(pieces_.size()), false);
        covered += count;
    }
    
    std::cout << "[UnigramLM] Initial vocab: " << pieces_.size() 
              << " characters, coverage: " << (100.0f * covered / total_chars) << "%" << std::endl;
    
    // Step 3: Generate candidate subwords from SENTENCES (never cross sentence boundaries)
    std::unordered_map<std::string, int> subword_counts;
    subword_counts.reserve(1000000);  // Pre-allocate reasonable amount
    
    const auto& texts_for_subwords = use_sampling ? sample_indices : std::vector<size_t>{};
    size_t num_texts_to_process = use_sampling ? sample_indices.size() : training_units.size();
    size_t progress_interval = std::max(size_t(1), num_texts_to_process / 20);
    
    std::cout << "[UnigramLM] Mining subwords from " << num_texts_to_process << " sentences..." << std::endl;
    
    for (size_t ti = 0; ti < num_texts_to_process; ++ti) {
        const std::string& text = use_sampling ? training_units[sample_indices[ti]] : training_units[ti];
        
        // Progress reporting
        if (ti % progress_interval == 0) {
            std::cout << "[UnigramLM] Subword mining: " << ti << "/" << num_texts_to_process 
                      << " (" << (100 * ti / num_texts_to_process) << "%), "
                      << subword_counts.size() << " unique subwords" << std::endl;
        }
        
        // Limit subword length based on corpus size to control memory
        size_t max_len = use_sampling ? MAX_PIECE_LENGTH : std::min(static_cast<size_t>(MAX_PIECE_LENGTH), size_t(16));
        
        // Build list of UTF-8 character boundary positions
        std::vector<size_t> char_positions;
        char_positions.reserve(text.size());
        for (size_t i = 0; i < text.size(); ) {
            char_positions.push_back(i);
            unsigned char c = static_cast<unsigned char>(text[i]);
            size_t char_len = 1;
            if ((c & 0x80) == 0x00) char_len = 1;
            else if ((c & 0xE0) == 0xC0) char_len = 2;
            else if ((c & 0xF0) == 0xE0) char_len = 3;
            else if ((c & 0xF8) == 0xF0) char_len = 4;
            i += char_len;
        }
        char_positions.push_back(text.size());  // End sentinel
        
        // Extract subwords at character boundaries only
        size_t num_chars = char_positions.size() - 1;
        for (size_t ci = 0; ci < num_chars; ++ci) {
            // Extract subwords of 2 to max_len characters (not bytes)
            for (size_t char_count = 2; char_count <= max_len && ci + char_count <= num_chars; ++char_count) {
                size_t byte_start = char_positions[ci];
                size_t byte_end = char_positions[ci + char_count];
                // Skip very long byte sequences (likely binary garbage)
                if (byte_end - byte_start > 64) continue;
                std::string subword = text.substr(byte_start, byte_end - byte_start);
                
                // CRITICAL: Filter out linguistically nonsensical patterns
                // This prevents garbage tokens like "P.," from entering vocabulary
                if (!isValidSubword(subword)) continue;
                
                subword_counts[subword]++;
            }
        }
        
        // Safety valve: if enabled and we've accumulated too many unique subwords, prune low-frequency ones
        if (prune_during_mining && subword_counts.size() > 50000000) {  // 50M entries ~= 2GB memory
            std::cout << "[UnigramLM] Pruning low-frequency subwords to control memory..." << std::endl;
            for (auto it = subword_counts.begin(); it != subword_counts.end(); ) {
                if (it->second < 3) {
                    it = subword_counts.erase(it);
                } else {
                    ++it;
                }
            }
            std::cout << "[UnigramLM] After pruning: " << subword_counts.size() << " subwords" << std::endl;
        }
    }
    
    std::cout << "[UnigramLM] Subword mining complete: " << subword_counts.size() << " unique subwords" << std::endl;
    
    // Step 4: Add top-K most frequent subwords up to target_vocab_size
    // Sort by frequency descending, add until we hit target OR run out of candidates meeting min_freq
    std::vector<std::pair<std::string, int>> sorted_subwords(subword_counts.begin(), subword_counts.end());
    std::sort(sorted_subwords.begin(), sorted_subwords.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    int added = 0;
    int filtered = 0;
    int repetition_filtered = 0;
    int near_duplicates = 0;
    const int max_to_add = target_vocab_size > 0 ? target_vocab_size : std::numeric_limits<int>::max();
    
    // Track normalized forms to detect near-duplicates that differ only by
    // whitespace/case/repetition artifacts (e.g., " the" vs "the", "soooo" vs "soo").
    // Keep the highest-count surviving variant.
    std::unordered_set<std::string> normalized_added;
    
    for (const auto& [subword, count] : sorted_subwords) {
        if (count < MIN_SUBWORD_FREQ) break;  // Stop when frequency too low
        if (added >= max_to_add) break;        // Stop when target reached
        if (hasPiece(subword)) continue;

        // Track repetition stripping separately for better visibility.
        if (isRepetitionNoise(subword)) {
            repetition_filtered++;
            continue;
        }
        
        // Double-check quality filter (should already be filtered during mining, but be safe)
        if (!isValidSubword(subword)) {
            filtered++;
            continue;
        }
        
        // Near-duplicate detection: reject subwords whose normalized form
        // (lowercase + whitespace normalization + repeat compression) matches
        // an already-added piece.
        // Since sorted_subwords is sorted by count DESC, the highest-count
        // variant wins. E.g., " the" (50K) added first, then "the" (3K) rejected.
        std::string norm = normalizeForDedup(subword);
        if (norm.empty()) {
            filtered++;
            continue;
        }
        if (normalized_added.count(norm)) {
            near_duplicates++;
            continue;
        }
        normalized_added.insert(norm);
        
        float score = std::log(static_cast<float>(count) / total_chars);
        addPiece(subword, score, UNIGRAM_VOCAB_OFFSET + static_cast<int>(pieces_.size()), false);
        added++;
    }
    
    if (filtered > 0) {
        std::cout << "[UnigramLM] Filtered " << filtered << " invalid subword patterns" << std::endl;
    }
    if (repetition_filtered > 0) {
        std::cout << "[UnigramLM] Filtered " << repetition_filtered
                  << " repetition/stutter subword patterns" << std::endl;
    }
    if (near_duplicates > 0) {
        std::cout << "[UnigramLM] Rejected " << near_duplicates 
                  << " near-duplicate subwords (whitespace/case/repetition variants)" << std::endl;
    }
    std::cout << "[UnigramLM] Added " << added << " subwords (min_freq=" << MIN_SUBWORD_FREQ 
              << ", target=" << target_vocab_size << "), total vocab: " << pieces_.size() << std::endl;
    
    // Step 5: EM iterations to refine scores ONLY (no pruning!)
    // Tokens that exist in the corpus should NEVER be removed
    constexpr int EM_ITERATIONS = HyperParameters::UNIGRAM_EM_ITERATIONS;
    
    // EM iterations: ONLY refine scores, NO pruning, NO re-adding
    // The vocabulary is fixed after initial subword mining - we just improve probability estimates
    for (int em_iter = 0; em_iter < EM_ITERATIONS; ++em_iter) {
        std::cout << "[UnigramLM] EM iteration " << (em_iter + 1) << "/" << EM_ITERATIONS << std::endl;
        
        // Build trie for current vocabulary
        buildTrie();
        
        // E-step: Count token usage via Viterbi segmentation
        std::unordered_map<int, double> token_counts;  // token_id -> count
        double total_tokens = 0.0;
        
        for (const auto& text : lowercase_texts) {
            if (text.empty()) continue;
            
            // Run Viterbi to get optimal segmentation (on lowercased text)
            auto nodes = viterbi(text);
            auto tokens = backtrack(nodes, static_cast<int>(text.size()));
            
            // Count each token used in this segmentation
            for (int token_id : tokens) {
                token_counts[token_id] += 1.0;
                total_tokens += 1.0;
            }
        }
        
        // M-step: Re-estimate probabilities for ALL tokens (no pruning!)
        // Tokens with zero count get a small smoothed probability
        constexpr double SMOOTHING = 0.1;  // Add-k smoothing
        double smoothed_total = total_tokens + SMOOTHING * static_cast<double>(pieces_.size());
        
        int zero_count_tokens = 0;
        for (auto& piece : pieces_) {
            if (piece.is_user_defined) continue;  // Don't modify special tokens
            
            double count = token_counts[piece.token_id] + SMOOTHING;
            // score = log(smoothed_count / smoothed_total)
            piece.score = static_cast<float>(std::log(count / smoothed_total));
            
            if (token_counts[piece.token_id] == 0) {
                zero_count_tokens++;
            }
        }
        
        std::cout << "[UnigramLM] Iteration " << (em_iter + 1) << ": " 
                  << static_cast<int>(total_tokens) << " tokens used, "
                  << zero_count_tokens << " tokens unused (kept with smoothed prob)" << std::endl;
    }
    
    // Final trie build with refined scores
    buildTrie();
    
    std::cout << "[UnigramLM] Training complete. Final vocab size: " << pieces_.size() << std::endl;
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
        
        trie_[node].token_id = piece.token_id;
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
        
        // Try all pieces starting at pos
        int node = 0;
        for (size_t len = 1; len <= MAX_PIECE_LENGTH && pos + len <= n; ++len) {
            unsigned char c = static_cast<unsigned char>(text[pos + len - 1]);
            
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

std::vector<int> UnigramLM::encode(const std::string& text) const {
    if (text.empty()) return {};
    
    auto nodes = viterbi(text);
    return backtrack(nodes, static_cast<int>(text.size()));
}

std::vector<UnigramPiece> UnigramLM::encodeWithPieces(const std::string& text) const {
    auto token_ids = encode(text);
    std::vector<UnigramPiece> result;
    result.reserve(token_ids.size());
    
    for (int tid : token_ids) {
        if (tid >= SPECIAL_TOKEN_OFFSET && tid < NUM_SPECIAL_TOKENS) {
            // Special token
            UnigramPiece piece;
            piece.token_id = tid;
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
            piece.token_id = tid;
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
    
    return result;
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
        
        // CRITICAL: Reassign token_id to maintain contiguous ID space
        // Without this, getPiece(token_id) returns wrong pieces after pruning
        piece.token_id = UNIGRAM_VOCAB_OFFSET + static_cast<int>(new_pieces.size());
        
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
