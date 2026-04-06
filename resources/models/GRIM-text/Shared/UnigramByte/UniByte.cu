//======================================================//
//  UniByte.cu
//  CUDA implementation of unified tokenizer orchestrator
//  
//  STRUCTURAL DETECTION: Uses Aho-Corasick DFA for O(n) 
//  prefix matching (URLs, emails, hex/binary numbers).
//  
//  See AhoCorasick.hpp for DFA implementation.
//======================================================//

#include "UniByte.hpp"
#include "AhoCorasick.hpp"
#include "AtomTable.hpp"
#include "Detectors.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

namespace GRIM {
namespace Tokenizer {

//======================================================//
//  CUDA Kernels
//======================================================//

// Kernel: Classify token types in parallel
__global__ void kernelClassifyTokens(
    const int* __restrict__ token_ids,
    size_t count,
    int* __restrict__ token_types,  // 0=byte, 1=atom, 2=unigram
    int byte_end,
    int atom_end
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        int tid = token_ids[idx];
        if (tid < byte_end) {
            token_types[idx] = 0;  // Byte
        } else if (tid < atom_end) {
            token_types[idx] = 1;  // Atom
        } else {
            token_types[idx] = 2;  // Unigram
        }
    }
}

// Kernel: Detect number patterns (simplified GPU version)
__global__ void kernelDetectNumbers(
    const char* __restrict__ text,
    size_t length,
    bool* __restrict__ is_number_start,
    bool* __restrict__ is_number_part
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length) return;
    
    char c = text[idx];
    bool is_digit = (c >= '0' && c <= '9');
    bool is_sign = (c == '+' || c == '-');
    bool is_dot = (c == '.');
    bool is_exp = (c == 'e' || c == 'E');
    bool is_hex_prefix = false;
    
    // Check for hex prefix
    if (idx + 1 < length && c == '0' && (text[idx + 1] == 'x' || text[idx + 1] == 'X')) {
        is_hex_prefix = true;
    }
    
    // Mark potential number starts
    is_number_start[idx] = is_digit || 
                           (is_sign && idx + 1 < length && 
                            (text[idx + 1] >= '0' && text[idx + 1] <= '9')) ||
                           is_hex_prefix;
    
    // Mark parts of numbers
    is_number_part[idx] = is_digit || is_dot || is_exp || is_sign ||
                          (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F') ||
                          c == 'x' || c == 'X';
}

// Kernel: Mark structural boundaries
__global__ void kernelMarkStructuralBoundaries(
    const char* __restrict__ text,
    size_t length,
    int* __restrict__ boundary_type  // 0=none, 1=number, 2=url, etc.
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length) return;
    
    boundary_type[idx] = 0;
    
    char c = text[idx];
    
    // Simple heuristics for structural boundaries
    // URL detection: http:// or https://
    if (idx + 7 < length) {
        bool is_http = (text[idx] == 'h' && text[idx+1] == 't' && 
                        text[idx+2] == 't' && text[idx+3] == 'p');
        if (is_http) {
            bool is_https = (idx + 8 < length && text[idx+4] == 's');
            int colon_pos = is_https ? 5 : 4;
            if (text[idx + colon_pos] == ':' && 
                text[idx + colon_pos + 1] == '/' && 
                text[idx + colon_pos + 2] == '/') {
                boundary_type[idx] = 2;  // URL
            }
        }
    }
    
    // Email detection: look for @ preceded by alphanum
    if (c == '@' && idx > 0) {
        char prev = text[idx - 1];
        bool valid_prev = (prev >= 'a' && prev <= 'z') || 
                         (prev >= 'A' && prev <= 'Z') ||
                         (prev >= '0' && prev <= '9') ||
                         prev == '.' || prev == '_' || prev == '-';
        if (valid_prev) {
            boundary_type[idx] = 3;  // Email marker (need to find bounds)
        }
    }
}

//======================================================//
//  Detector State - Aho-Corasick Based (10-100x faster)
//======================================================//

    struct UniByte::DetectorState {
        AhoCorasick url_prefixes;       // kept empty (non-numeric atoms removed)
        AhoCorasick email_indicator;    // kept empty
        AhoCorasick number_prefixes;    // 0x, 0b for hex/binary -> ATOM_NUM
        
        DetectorState() {
            url_prefixes.build();
            email_indicator.build();
            
            number_prefixes.addPattern("0x", AtomType::ATOM_NUM);
            number_prefixes.addPattern("0X", AtomType::ATOM_NUM);
            number_prefixes.addPattern("0b", AtomType::ATOM_NUM);
            number_prefixes.addPattern("0B", AtomType::ATOM_NUM);
            number_prefixes.build();
        }
    };

//======================================================//
//  UniByte Implementation
//======================================================//

UniByte::UniByte(const UniByteConfig& config)
    : config_(config)
    , detector_(nullptr)
{
    unigram_.setByteFallbackEnabled(config_.enable_byte_fallback);
    detector_ = std::make_unique<DetectorState>();
    initDetector();
}

UniByte::~UniByte() = default;

UniByte::UniByte(UniByte&& other) noexcept
    : config_(other.config_)
    , byte_encoder_(std::move(other.byte_encoder_))
    , unigram_(std::move(other.unigram_))
    , gpu_initialized_(other.gpu_initialized_)
    , detector_(std::move(other.detector_))
{
    other.gpu_initialized_ = false;
}

UniByte& UniByte::operator=(UniByte&& other) noexcept {
    if (this != &other) {
        config_ = other.config_;
        byte_encoder_ = std::move(other.byte_encoder_);
        unigram_ = std::move(other.unigram_);
        gpu_initialized_ = other.gpu_initialized_;
        detector_ = std::move(other.detector_);
        other.gpu_initialized_ = false;
    }
    return *this;
}

void UniByte::initDetector() {
    if (!detector_) {
        detector_ = std::make_unique<DetectorState>();
    }
}

//--------------------------------------------------//
// Initialization
//--------------------------------------------------//

bool UniByte::load(const std::string& vocab_path) {
    std::string bin_path = vocab_path;
    size_t dot_pos = bin_path.rfind('.');
    if (dot_pos == std::string::npos) {
        bin_path += ".bin";
    } else {
        std::string ext = bin_path.substr(dot_pos);
        if (ext == ".txt") {
            throw std::runtime_error(
                "[UniByte] Text vocab loading is forbidden. Caller must provide a .bin vocab path, got: " +
                vocab_path);
        }
        if (ext != ".bin") {
            throw std::runtime_error(
                "[UniByte] Unsupported vocab extension '" + ext +
                "'. Caller must provide a .bin vocab path: " + vocab_path);
        }
    }

    std::ifstream test(bin_path, std::ios::binary);
    if (!test.good()) {
        throw std::runtime_error(
            "[UniByte] Required binary vocab file does not exist or is not readable: " + bin_path);
    }
    test.close();

    return unigram_.loadBinary(bin_path);
}

bool UniByte::save(const std::string& vocab_path, bool save_text_format) const {
    return unigram_.save(vocab_path, save_text_format);
}

bool UniByte::train(const std::vector<std::string>& texts) {
    // Detect atom spans in each text BEFORE training.
    // Atom regions (URLs, emails, numbers, dates, paths, etc.) are SKIPPED
    // during character counting and subword mining so their internal chars
    // (://@.com etc.) don't contaminate the vocabulary.
    std::vector<std::vector<AtomSpan>> all_atom_spans;
    all_atom_spans.reserve(texts.size());
    
    size_t total_atoms = 0;
    for (const auto& text : texts) {
        auto structures = detectStructures(text);
        std::vector<AtomSpan> spans;
        spans.reserve(structures.size());
        for (const auto& s : structures) {
            spans.push_back({s.start, s.end});
        }
        total_atoms += spans.size();
        all_atom_spans.push_back(std::move(spans));
    }
    
    std::cout << "[UniByte] Detected " << total_atoms << " atoms across "
              << texts.size() << " texts (will skip during vocab training)" << std::endl;
    
    return unigram_.trainFromCorpus(texts, all_atom_spans,
                                     config_.target_vocab_size, 
                                     config_.character_coverage,
                                     config_.min_subword_freq,
                                     config_.prune_during_mining,
                                     config_.enable_parallel_subword_mining,
                                     config_.subword_mining_workers,
                                     config_.subword_mining_max_bytes);
}

bool UniByte::initGPU() {
    if (gpu_initialized_) return true;
    
    if (!unigram_.initGPU()) {
        std::cerr << "[UniByte] Failed to initialize Unigram GPU" << std::endl;
        return false;
    }
    
    gpu_initialized_ = true;
    return true;
}

//--------------------------------------------------//
// Structural Detection
//--------------------------------------------------//

std::vector<StructuralSpan> UniByte::detectStructures(const std::string& text) const {
    std::vector<StructuralSpan> spans;
    spans.reserve(32);  // Pre-allocate for typical case
    
    if (!detector_) return spans;
    
    // Lambda to create span with zero-copy buffer reference
    auto makeSpan = [&text](size_t start, size_t end, AtomType type) -> StructuralSpan {
        StructuralSpan span;
        span.start = start;
        span.end = end;
        span.atom_type = type;
        span.buffer_ptr = text.data();
        span.offset = static_cast<uint32_t>(start);
        span.length = static_cast<uint32_t>(end - start);
        span.content_offset = static_cast<uint32_t>(start);
        span.content_length = static_cast<uint32_t>(end - start);
        span.placeholder_id = atomTypeToTokenId(type);
        return span;
    };
    
    //--------------------------------------------------//
    // Phase 1: Aho-Corasick prefix detection (O(n) single pass)
    //--------------------------------------------------//
    
    // Non-numeric atom types removed: URLs, emails, dates, times, IPs, paths
    // all fall through to regular byte/unigram tokenization now.
    
    //--------------------------------------------------//
    // Number detection with Aho-Corasick prefix hints
    //--------------------------------------------------//
    
    if (config_.detect_numbers) {
        auto num_prefix_hits = detector_->number_prefixes.search(text);
        
        for (const auto& hit : num_prefix_hits) {
            size_t end;
            if (Detector::detectHex(text, hit.start, end)) {
                spans.push_back(makeSpan(hit.start, end, AtomType::ATOM_NUM));
            } else if (Detector::detectBinary(text, hit.start, end)) {
                spans.push_back(makeSpan(hit.start, end, AtomType::ATOM_NUM));
            }
        }
        
        for (size_t i = 0; i < text.size(); ) {
            if (i + 1 < text.size() && text[i] == '0' && 
                (text[i+1] == 'x' || text[i+1] == 'X' || text[i+1] == 'b' || text[i+1] == 'B')) {
                ++i;
                continue;
            }
            
            size_t end;
            
            if (Detector::detectFloat(text, i, end)) {
                spans.push_back(makeSpan(i, end, AtomType::ATOM_NUM));
                i = end;
                continue;
            }
            
            if (Detector::detectInteger(text, i, end)) {
                spans.push_back(makeSpan(i, end, AtomType::ATOM_NUM));
                i = end;
                continue;
            }
            
            ++i;
        }
    }
    
    // Sort by position and remove overlaps
    std::sort(spans.begin(), spans.end(), 
              [](const auto& a, const auto& b) { return a.start < b.start; });
    
    // Remove overlapping spans (keep first/longest)
    std::vector<StructuralSpan> non_overlapping;
    size_t last_end = 0;
    for (const auto& span : spans) {
        if (span.start >= last_end) {
            non_overlapping.push_back(span);
            last_end = span.end;
        }
    }

    return non_overlapping;
}

std::string UniByte::injectPlaceholders(const std::string& text,
                                         std::vector<StructuralSpan>& out_spans) const {
    out_spans = detectStructures(text);
    
    if (out_spans.empty()) {
        return text;
    }
    
    // Build result with placeholders
    std::string result;
    result.reserve(text.size());
    
    size_t pos = 0;
    for (const auto& span : out_spans) {
        // Add text before span
        if (span.start > pos) {
            result += text.substr(pos, span.start - pos);
        }
        
        // Add placeholder character (will be tokenized as atom token)
        // Use private-use Unicode area: U+E000 + atom_type
        // For simplicity, we use a control character sequence
        result += '\x1F';  // Unit separator
        result += static_cast<char>(static_cast<int>(span.atom_type));
        result += '\x1F';
        
        pos = span.end;
    }
    
    // Add remaining text
    if (pos < text.size()) {
        result += text.substr(pos);
    }
    
    return result;
}

//--------------------------------------------------//
// Scratch Block Reasoning Control
//--------------------------------------------------//

void UniByte::setScratchBlockReasoning(bool enabled) {
    config_.enable_scratch_block_reasoning = enabled;
}

//--------------------------------------------------//
// Encoding
//--------------------------------------------------//

std::vector<int> UniByte::encode(const std::string& text) const {
    // If scratch block reasoning is disabled, use fast path (normal UnigramByte)
    if (!config_.enable_scratch_block_reasoning) {
        // FAST PATH: No structural detection, no AtomTable, just pure tokenization
        return unigram_.encode(text);
    }
    
    // SCRATCH BLOCK REASONING PATH: Use AtomTable for structural reasoning
    auto result = encodeWithMetadata(text);
    return result.token_ids;
}

UniByteResult UniByte::encodeWithMetadata(const std::string& text) const {
    // Detect structures first
    auto structures = detectStructures(text);
    auto result = encodeInternal(text, structures);
    
    // Pipeline contract: validate all per-token arrays are consistent
    // before this result can enter BatchPayload / GPU tensor pipeline.
    result.validate("UniByte::encodeWithMetadata");
    
    return result;
}

//======================================================//
//  Text Feature Encoding (16-dim FP16 per atom token)
//======================================================//
//
//  Feature layout (16 dimensions):
//    [0-3]:   Atom type one-hot (4 bits for type category)
//    [4-7]:   Atom sub-type encoding (specific type within category)
//    [8-11]:  Length/magnitude features (log-scaled)
//    [12-15]: Structure-specific semantic features
//
//  Categories:
//    NUMERIC   = 0: Integer, Float, Hex, Binary
//    TEMPORAL  = 1: Date, Time
//    STRUCTURAL= 2: URL, Email, Path, IP
//    STRING    = 3: StringLiteral, Identifier
//======================================================//

namespace {

// FP16 conversion helper (C++ side)
inline uint16_t floatToFp16(float value) {
    __half h = __float2half(value);
    return *reinterpret_cast<uint16_t*>(&h);
}

// Encode text features for an atom token into 16-dim FP16 vector
// With single ATOM_NUM type, features focus on numeric magnitude/structure.
void encodeAtomTextFeatures(
    AtomType atom_type,
    const std::string_view raw_text,
    const AtomValue* parsed,
    uint16_t* out_features  // [kTextFeatureDim]
) {
    for (int i = 0; i < kTextFeatureDim; ++i) {
        out_features[i] = floatToFp16(0.0f);
    }
    
    if (atom_type != AtomType::ATOM_NUM) return;
    
    // Dim 0: numeric category indicator
    out_features[0] = floatToFp16(1.0f);
    
    // Dims [8-11]: Length/magnitude features
    float len_f = static_cast<float>(raw_text.size());
    out_features[8] = floatToFp16(std::min(len_f / 100.0f, 1.0f));
    out_features[9] = floatToFp16(std::log2f(len_f + 1.0f) / 10.0f);
    
    if (parsed) {
        if (auto* int_val = std::get_if<AtomInteger>(parsed)) {
            float log_mag = std::log2f(std::abs(static_cast<float>(int_val->value)) + 1.0f);
            out_features[10] = floatToFp16(std::min(log_mag / 32.0f, 1.0f));
            out_features[11] = floatToFp16(int_val->value < 0 ? 1.0f : 0.0f);
        } else if (auto* float_val = std::get_if<AtomFloat>(parsed)) {
            float log_mag = std::log2f(std::abs(static_cast<float>(float_val->value)) + 1.0f);
            out_features[10] = floatToFp16(std::min(log_mag / 32.0f, 1.0f));
            out_features[11] = floatToFp16(float_val->has_exponent ? 1.0f : 0.0f);
        }
    }
    
    // Dims [12-14]: Character composition
    int digit_count = 0, alpha_count = 0, special_count = 0;
    for (char c : raw_text) {
        if (std::isdigit(static_cast<unsigned char>(c))) ++digit_count;
        else if (std::isalpha(static_cast<unsigned char>(c))) ++alpha_count;
        else ++special_count;
    }
    float total = static_cast<float>(raw_text.size()) + 1e-6f;
    out_features[12] = floatToFp16(static_cast<float>(digit_count) / total);
    out_features[13] = floatToFp16(static_cast<float>(alpha_count) / total);
    out_features[14] = floatToFp16(static_cast<float>(special_count) / total);
    
    // Dim 15: has decimal point
    bool has_dot = raw_text.find('.') != std::string_view::npos;
    out_features[15] = floatToFp16(has_dot ? 1.0f : 0.0f);
}

}  // anonymous namespace

UniByteResult UniByte::encodeInternal(const std::string& text,
                                       const std::vector<StructuralSpan>& structures) const {
    UniByteResult result;
    
    if (text.empty()) {
        return result;
    }
    
    // Initialize atom registry for this encoding pass
    result.atom_table = std::make_shared<AtomTable>();

    // Pre-allocate based on heuristic: ~1 token per 3-4 bytes + atoms
    // This avoids repeated vector reallocation during push_back
    const size_t estimated_tokens = (text.size() / 3) + structures.size() + 8;
    result.token_ids.reserve(estimated_tokens);
    result.is_byte_fallback.reserve(estimated_tokens);
    result.token_numeric_values.reserve(estimated_tokens);
    result.token_atom_flags.reserve(estimated_tokens);
    result.atom_entry_ids.reserve(estimated_tokens);
    result.token_text_features.reserve(estimated_tokens * kTextFeatureDim);
    result.token_atom_mask.reserve(estimated_tokens);
    result.atoms.reserve(structures.size());
    
    // Pre-computed zero text features (FP16) — avoid recomputing floatToFp16(0.0f) per token
    static const uint16_t kZeroFp16 = floatToFp16(0.0f);
    
    // Helper: append zeros for non-atom tokens (no text features)
    auto appendZeroSideChannels = [&]() {
        result.token_text_features.insert(
            result.token_text_features.end(), kTextFeatureDim, kZeroFp16);
        result.token_atom_mask.push_back(0);
    };
    
    // Process text in segments between structures
    size_t pos = 0;
    size_t struct_idx = 0;
    
    bool is_first_segment = true;
    auto appendSegmentTokens = [&](size_t start, size_t end) {
        if (end <= start) {
            return;
        }
        std::string segment = text.substr(start, end - start);
        // encode() returns token IDs directly — no need for encodeWithPieces
        // Note: UnigramLM::encode() normalizes to lowercase internally.
        // Only prepend ▁ for the first text segment; subsequent segments
        // after atoms already have their leading space as a regular character.
        auto segment_ids = unigram_.encode(segment, is_first_segment);
        is_first_segment = false;
                        
        for (int tid : segment_ids) {
            result.token_ids.push_back(tid);
            result.token_numeric_values.push_back(0.0f);
            result.token_atom_flags.push_back(0);
            result.atom_entry_ids.push_back(kAtomEntryNone);  // no atom at this position
            appendZeroSideChannels();  // Non-atom tokens get zero features + mask=0
            
            if (tid >= BYTE_TOKEN_OFFSET && tid < BYTE_TOKEN_OFFSET + BYTE_VOCAB_SIZE) {
                result.is_byte_fallback.push_back(true);
                result.byte_tokens++;
            } else {
                result.is_byte_fallback.push_back(false);
                result.unigram_tokens++;
            }
        }
    };
    
    while (pos < text.size()) {
        // Check if we're at a structure
        if (struct_idx < structures.size() && pos == structures[struct_idx].start) {
            StructuralSpan span = structures[struct_idx];

            int atom_token_id = span.placeholder_id;
            float numeric_value = 0.0f;
            AtomValue parsed_value;
            bool has_parsed = false;
            
            // Parse atom for both numeric and text features
            auto parsed = AtomTable::parseAtom(span.atom_type, std::string(span.contentView()));
            if (parsed.success) {
                has_parsed = true;
                parsed_value = parsed.value;
                
                if (Tokenizer::isNumericAtom(span.atom_type)) {
                    if (auto* int_val = std::get_if<AtomInteger>(&parsed.value)) {
                        numeric_value = static_cast<float>(int_val->value);
                    } else if (auto* float_val = std::get_if<AtomFloat>(&parsed.value)) {
                        float numeric = static_cast<float>(float_val->value);
                        if (std::isfinite(numeric)) {
                            numeric_value = numeric;
                        }
                    }
                }
            }

            const bool numeric_atom = Tokenizer::isNumericAtom(span.atom_type);
            // Only bail out if the parser actually FAILED — NOT if the value happens to be 0.
            // The literal integer "0" is a valid atom (numeric_value=0.0f, has_parsed=true).
            const bool numeric_parse_failed = numeric_atom && !has_parsed;
            if (numeric_parse_failed) {
                appendSegmentTokens(span.start, span.end);
                pos = span.end;
                struct_idx++;
                continue;
            }

            // Emit a single atom token with full AtomTable-backed side channels
            uint32_t entry_id = result.atom_table->registerSpan(span);
            
            // Read back AtomTable's packed values — covers ALL atom types
            // (dates, times, IPs, etc. get meaningful numeric_value and flags
            // via packNumericValue(), not just INT/FLOAT/HEX/BIN)
            const auto* entry = result.atom_table->getAtom(entry_id);
            float packed_numeric = numeric_value;  // fallback to parser result
            uint32_t packed_flags = 0;
            if (entry) {
                packed_numeric = entry->numeric_value;
                packed_flags = entry->flags;
                // For numeric atoms, prefer the direct parser result if AtomTable
                // re-packed it differently (float precision preservation)
                if (Tokenizer::isNumericAtom(span.atom_type) && has_parsed) {
                    packed_numeric = numeric_value;
                }
            }
            
            result.token_ids.push_back(atom_token_id);
            result.is_byte_fallback.push_back(false);
            result.token_numeric_values.push_back(packed_numeric);
            result.token_atom_flags.push_back(packed_flags);
            result.atom_entry_ids.push_back(entry_id);
            
            // Encode text features for atom token
            uint16_t text_features[kTextFeatureDim];
            encodeAtomTextFeatures(
                span.atom_type, 
                span.contentView(), 
                has_parsed ? &parsed_value : nullptr,
                text_features
            );
            for (int i = 0; i < kTextFeatureDim; ++i) {
                result.token_text_features.push_back(text_features[i]);
            }
            result.token_atom_mask.push_back(1);  // Atom tokens: unified mask
            
            result.atoms.push_back(span);
            result.atom_tokens++;

            pos = span.end;
            struct_idx++;
            continue;
        }
        
        // Find end of current segment (either next structure or end of text)
        size_t segment_end = text.size();
        if (struct_idx < structures.size()) {
            segment_end = structures[struct_idx].start;
        }
        
        // Encode segment with Unigram
        appendSegmentTokens(pos, segment_end);
        
        pos = segment_end;
    }
    
    return result;
}

std::vector<std::vector<int>> UniByte::encodeBatch(const std::vector<std::string>& texts) const {
    std::vector<std::vector<int>> results;
    results.reserve(texts.size());
    
    for (const auto& text : texts) {
        results.push_back(encode(text));
    }
    
    return results;
}

bool UniByte::encodeGPU(const char* d_text,
                        size_t length,
                        int* d_token_ids,
                        int* d_token_count,
                        int max_tokens) {
    if (!gpu_initialized_) {
        if (!initGPU()) return false;
    }
    
    // Note: Byte fallback flags not needed - fallback tokens are indistinguishable from regular tokens
    // Position embeddings work identically for all token IDs (byte fallback or unigram)
    return unigram_.encodeGPU(d_text, length, d_token_ids, 
                              d_token_count, max_tokens, nullptr);
}

//--------------------------------------------------//
// Decoding
//--------------------------------------------------//

std::string UniByte::decode(const std::vector<int>& token_ids) const {
    return decode(token_ids.data(), token_ids.size());
}

std::string UniByte::decode(const int* token_ids, size_t count) const {
    std::string result;
    
    for (size_t i = 0; i < count; ++i) {
        int tid = token_ids[i];
        
        if (tid >= SPECIAL_TOKEN_OFFSET && tid < NUM_SPECIAL_TOKENS) {
            // Special token
            if (tid == UNK_TOKEN_ID) result += "<unk>";
            else if (tid == PAD_TOKEN_ID) { /* skip padding */ }
            else if (tid == BOS_TOKEN_ID) result += "<s>";
            else if (tid == EOS_TOKEN_ID) result += "</s>";
        } else if (isByteToken(tid)) {
            // Byte token - direct character output
            result.push_back(static_cast<char>(tid - BYTE_TOKEN_OFFSET));
        } else if (isAtomToken(tid)) {
            // Atom tokens map to AtomTable entries; raw values are resolved separately.
            continue;
        } else if (isUnigramToken(tid)) {
            // Unigram token
            const UnigramPiece* piece = unigram_.getPiece(tid);
            if (piece) {
                result += piece->text;
            }
        }
    }
    
    return UnigramLM::denormalizeFromTokenization(result);
}

std::string UniByte::decodeWithAtoms(const std::vector<int>& token_ids,
                                      const AtomResolver& resolver) const {
    std::string result;
    
    for (int tid : token_ids) {
        if (isSpecialToken(tid)) {
            if (tid == UNK_TOKEN_ID) result += "<unk>";
            else if (tid == PAD_TOKEN_ID) { /* skip padding */ }
            else if (tid == BOS_TOKEN_ID) result += "<s>";
            else if (tid == EOS_TOKEN_ID) result += "</s>";
        } else if (isByteToken(tid)) {
            result.push_back(static_cast<char>(tid - BYTE_TOKEN_OFFSET));
        } else if (isAtomToken(tid)) {
            AtomType type = tokenIdToAtomType(tid);
            result += resolver(tid, type);
        } else if (isUnigramToken(tid)) {
            const UnigramPiece* piece = unigram_.getPiece(tid);
            if (piece) {
                result += piece->text;
            }
        }
    }
    
    return UnigramLM::denormalizeFromTokenization(result);
}

//--------------------------------------------------//
// Vocabulary Info
//--------------------------------------------------//

int UniByte::vocabSize() const {
    return unigram_.vocabSize();
}

int UniByte::totalVocabSize() const {
    return NUM_SPECIAL_TOKENS + BYTE_VOCAB_SIZE + ATOM_VOCAB_SIZE + unigram_.vocabSize();
}

TokenLayout UniByte::tokenLayout() const {
    TokenLayout layout;
    layout.num_special = NUM_SPECIAL_TOKENS;          // from Byte.hpp (live constexpr)
    layout.num_bytes   = BYTE_VOCAB_SIZE;             // from Byte.hpp (live constexpr)
    layout.num_atoms   = ATOM_VOCAB_SIZE;             // from Unigram.hpp (inline, set by configureTokenLayout)
    layout.num_unigram = unigram_.vocabSize();        // from UnigramLM (actual loaded piece count)
    return layout;
}

void UniByte::capVocabSize(int max_vocab) {
    // max_vocab = max number of UNIGRAM PIECES (same semantics as tokenizer.vocab_size
    // in ai_config.json).  Specials, bytes, and atoms are always included on top.
    // The old code subtracted the fixed offset here, but that made max_vocab_size=10000
    // cap to 9723 pieces when vocab_size=10000 trained exactly 10000 — creating a
    // mismatch between the GRMT file (encoded with 10000 pieces) and the Phase1
    // tokenizer (capped to 9723).  Now the semantics match: if you train 10000 pieces
    // and set max_vocab_size=10000, no capping occurs.
    if (max_vocab <= 0) {
        throw std::runtime_error("max_vocab must be > 0");
    }
    unigram_.capVocabSize(max_vocab);
}

int UniByte::padId() const {
    return PAD_TOKEN_ID;  // Absolute ID = 1
}

int UniByte::unkId() const {
    return UNK_TOKEN_ID;  // Absolute ID = 0
}

int UniByte::bosId() const {
    return BOS_TOKEN_ID;  // Absolute ID = 2
}

int UniByte::eosId() const {
    return EOS_TOKEN_ID;  // Absolute ID = 3
}

bool UniByte::isSpecialToken(int token_id) const {
    return token_id >= SPECIAL_TOKEN_OFFSET && token_id < NUM_SPECIAL_TOKENS;
}

bool UniByte::isByteToken(int token_id) const {
    return token_id >= BYTE_TOKEN_OFFSET && token_id < ATOM_TOKEN_OFFSET;
}

bool UniByte::isAtomToken(int token_id) const {
    return token_id >= ATOM_TOKEN_OFFSET && token_id < UNIGRAM_VOCAB_OFFSET;
}

bool UniByte::isUnigramToken(int token_id) const {
    return token_id >= UNIGRAM_VOCAB_OFFSET;
}

std::string UniByte::tokenToString(int token_id) const {
    // Handle special tokens explicitly (before byte/atom/unigram checks)
    if (token_id == UNK_TOKEN_ID) return "<UNK>";
    if (token_id == PAD_TOKEN_ID) return "<PAD>";
    if (token_id == BOS_TOKEN_ID) return "<BOS>";
    if (token_id == EOS_TOKEN_ID) return "<EOS>";

    if (isByteToken(token_id)) {
        return byte_encoder_.tokenToString(token_id);
    } else if (isAtomToken(token_id)) {
        AtomType type = tokenIdToAtomType(token_id);
        if (type == AtomType::ATOM_NUM) return "<NUM>";
        return "<ATOM>";
    } else if (isUnigramToken(token_id)) {
        const UnigramPiece* piece = unigram_.getPiece(token_id);
        if (piece) {
            return piece->text;
        }
    }
    return "<UNK>";
}


} // namespace Tokenizer
} // namespace GRIM
