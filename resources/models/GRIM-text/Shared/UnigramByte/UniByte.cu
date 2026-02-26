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
        // Pre-built Aho-Corasick automata for O(n) prefix detection
        AhoCorasick url_prefixes;       // http://, https://, ftp://, etc.
        AhoCorasick email_indicator;    // @ symbol as email hint
        AhoCorasick number_prefixes;    // 0x, 0b for hex/binary
        
        DetectorState() {
            // URL prefixes (case insensitive)
            url_prefixes.setCaseInsensitive(true);
            url_prefixes.addPattern("http://", AtomType::ATOM_URL);
            url_prefixes.addPattern("https://", AtomType::ATOM_URL);
            url_prefixes.addPattern("ftp://", AtomType::ATOM_URL);
            url_prefixes.addPattern("ftps://", AtomType::ATOM_URL);
            url_prefixes.addPattern("ws://", AtomType::ATOM_URL);
            url_prefixes.addPattern("wss://", AtomType::ATOM_URL);
            url_prefixes.addPattern("file://", AtomType::ATOM_URL);
            url_prefixes.build();
            
            // Email indicator (@ symbol triggers full email validation)
            email_indicator.addPattern("@", AtomType::ATOM_EMAIL);
            email_indicator.build();
            
            // Number prefixes for hex/binary
            number_prefixes.addPattern("0x", AtomType::ATOM_HEX);
            number_prefixes.addPattern("0X", AtomType::ATOM_HEX);
            number_prefixes.addPattern("0b", AtomType::ATOM_BINARY);
            number_prefixes.addPattern("0B", AtomType::ATOM_BINARY);
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
    // Try binary first (faster), fall back to text
    std::string bin_path = vocab_path;
    size_t dot_pos = bin_path.rfind('.');
    if (dot_pos != std::string::npos) {
        std::string ext = bin_path.substr(dot_pos);
        if (ext == ".txt") {
            bin_path = bin_path.substr(0, dot_pos) + ".bin";
        } else if (ext != ".bin") {
            bin_path = bin_path.substr(0, dot_pos) + ".bin";
        }
    } else {
        bin_path += ".bin";
    }
    
    // Try binary format first
    std::ifstream test(bin_path, std::ios::binary);
    if (test.good()) {
        test.close();
        if (unigram_.loadBinary(bin_path)) {
            return true;
        }
        std::cerr << "[UniByte] Binary load failed, trying text format..." << std::endl;
    }
    
    // Fall back to text format
    std::string txt_path = bin_path.substr(0, bin_path.rfind('.')) + ".txt";
    return unigram_.load(txt_path);
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
        // Content bounds initially same as span bounds (before whitespace widening)
        span.content_offset = static_cast<uint32_t>(start);
        span.content_length = static_cast<uint32_t>(end - start);
        span.placeholder_id = atomTypeToTokenId(type);
        return span;
    };
    
    //--------------------------------------------------//
    // Phase 1: Aho-Corasick prefix detection (O(n) single pass)
    //--------------------------------------------------//
    
    // Detect URLs via prefix matching + manual extent finding
    if (config_.detect_urls) {
        auto url_hits = detector_->url_prefixes.search(text);
        for (const auto& hit : url_hits) {
            // Extend from prefix to find full URL
            size_t end = hit.end;
            while (end < text.size()) {
                char c = text[end];
                // URL continues until whitespace or certain punctuation
                if (std::isspace(c) || c == '<' || c == '>' || c == '"' || c == '\'' || c == ')') {
                    break;
                }
                ++end;
            }
            // Trim trailing punctuation
            while (end > hit.end && (text[end-1] == '.' || text[end-1] == ',' || text[end-1] == ';')) {
                --end;
            }
            if (end > hit.end) {
                spans.push_back(makeSpan(hit.start, end, AtomType::ATOM_URL));
            }
        }
    }
    
    // Detect emails via @ indicator + bidirectional extent finding
    if (config_.detect_emails) {
        auto email_hits = detector_->email_indicator.search(text);
        for (const auto& hit : email_hits) {
            // hit.start is position of '@'
            size_t at_pos = hit.start;
            
            // Find local part (before @)
            size_t local_start = at_pos;
            while (local_start > 0) {
                char c = text[local_start - 1];
                if (std::isalnum(c) || c == '.' || c == '_' || c == '-' || c == '+') {
                    --local_start;
                } else {
                    break;
                }
            }
            
            // Find domain part (after @)
            size_t domain_end = at_pos + 1;
            bool has_dot = false;
            while (domain_end < text.size()) {
                char c = text[domain_end];
                if (std::isalnum(c) || c == '-') {
                    ++domain_end;
                } else if (c == '.') {
                    has_dot = true;
                    ++domain_end;
                } else {
                    break;
                }
            }
            
            // Validate: must have local part, domain with dot, and reasonable length
            if (local_start < at_pos && has_dot && domain_end > at_pos + 2) {
                spans.push_back(makeSpan(local_start, domain_end, AtomType::ATOM_EMAIL));
            }
        }
    }
    
    //--------------------------------------------------//
    // Phase 2: Linear scan for patterns without good prefixes
    //--------------------------------------------------//
    
    // Detect dates (YYYY-MM-DD or MM/DD/YYYY formats)
    if (config_.detect_dates) {
        for (size_t i = 0; i < text.size(); ) {
            size_t end;
            if (Detector::detectDate(text, i, end)) {
                spans.push_back(makeSpan(i, end, AtomType::ATOM_DATE));
                i = end;
            } else {
                ++i;
            }
        }
    }
    
    // Detect times (HH:MM:SS or H:MM am/pm)
    if (config_.detect_dates) {  // Reuse date flag for times
        for (size_t i = 0; i < text.size(); ) {
            size_t end;
            if (Detector::detectTime(text, i, end)) {
                spans.push_back(makeSpan(i, end, AtomType::ATOM_TIME));
                i = end;
            } else {
                ++i;
            }
        }
    }
    
    // Detect IP addresses (N.N.N.N where 0 <= N <= 255)
    if (config_.detect_urls) {  // Reuse URL flag for IPs
        for (size_t i = 0; i < text.size(); ) {
            size_t end;
            if (Detector::detectIPAddress(text, i, end)) {
                spans.push_back(makeSpan(i, end, AtomType::ATOM_IP_ADDRESS));
                i = end;
            } else {
                ++i;
            }
        }
    }
    
    // Detect file paths (/unix/path or C:\windows\path)
    if (config_.detect_paths) {
        for (size_t i = 0; i < text.size(); ) {
            size_t end;
            if (Detector::detectPath(text, i, end)) {
                spans.push_back(makeSpan(i, end, AtomType::ATOM_PATH));
                i = end;
            } else {
                ++i;
            }
        }
    }
    
    //--------------------------------------------------//
    // Phase 3: Number detection with Aho-Corasick prefix hints
    //--------------------------------------------------//
    
    if (config_.detect_numbers) {
        // Use Aho-Corasick to find hex/binary prefixes (0x, 0b) in O(n)
        auto num_prefix_hits = detector_->number_prefixes.search(text);
        
        // First pass: process hex/binary from prefix hits
        // NOTE: Overlaps handled by final deduplication pass (no need for processed[] array)
        for (const auto& hit : num_prefix_hits) {
            size_t end;
            if (hit.atom_type == AtomType::ATOM_HEX && Detector::detectHex(text, hit.start, end)) {
                spans.push_back(makeSpan(hit.start, end, AtomType::ATOM_HEX));
            } else if (hit.atom_type == AtomType::ATOM_BINARY && Detector::detectBinary(text, hit.start, end)) {
                spans.push_back(makeSpan(hit.start, end, AtomType::ATOM_BINARY));
            }
        }
        
        // Second pass: scan for integers/floats (no good prefix for these)
        // Skip positions that could overlap with hex/binary hits (0x..., 0b...)
        for (size_t i = 0; i < text.size(); ) {
            // Skip hex/binary prefixes (these were handled in first pass)
            if (i + 1 < text.size() && text[i] == '0' && 
                (text[i+1] == 'x' || text[i+1] == 'X' || text[i+1] == 'b' || text[i+1] == 'B')) {
                ++i;
                continue;
            }
            
            size_t end;
            
            // Try float first (must come before integer to avoid partial match)
            if (Detector::detectFloat(text, i, end)) {
                spans.push_back(makeSpan(i, end, AtomType::ATOM_FLOAT));
                i = end;
                continue;
            }
            
            // Try integer
            if (Detector::detectInteger(text, i, end)) {
                spans.push_back(makeSpan(i, end, AtomType::ATOM_INTEGER));
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

    // Widen spans to include leading whitespace (avoids glued atoms)
    // NOTE: content_offset/content_length preserve original atom bounds
    size_t prev_end = 0;
    for (auto& span : non_overlapping) {
        size_t new_start = span.start;
        while (new_start > prev_end &&
               std::isspace(static_cast<unsigned char>(text[new_start - 1]))) {
            --new_start;
        }
        if (new_start != span.start) {
            // Widen tokenization bounds but preserve content bounds
            span.start = new_start;
            span.offset = static_cast<uint32_t>(new_start);
            span.length = static_cast<uint32_t>(span.end - new_start);
            // content_offset/content_length remain unchanged (original atom text)
        }
        prev_end = span.end;
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
void encodeAtomTextFeatures(
    AtomType atom_type,
    const std::string_view raw_text,
    const AtomValue* parsed,
    uint16_t* out_features  // [kTextFeatureDim]
) {
    // Zero-initialize
    for (int i = 0; i < kTextFeatureDim; ++i) {
        out_features[i] = floatToFp16(0.0f);
    }
    
    // Compute category and sub-type
    int category = 0;
    int subtype = 0;
    
    switch (atom_type) {
        // NUMERIC category (0)
        case AtomType::ATOM_INTEGER:
            category = 0; subtype = 0; break;
        case AtomType::ATOM_FLOAT:
            category = 0; subtype = 1; break;
        case AtomType::ATOM_HEX:
            category = 0; subtype = 2; break;
        case AtomType::ATOM_BINARY:
            category = 0; subtype = 3; break;
        
        // TEMPORAL category (1)
        case AtomType::ATOM_DATE:
            category = 1; subtype = 0; break;
        case AtomType::ATOM_TIME:
            category = 1; subtype = 1; break;
        
        // STRUCTURAL category (2)
        case AtomType::ATOM_URL:
            category = 2; subtype = 0; break;
        case AtomType::ATOM_EMAIL:
            category = 2; subtype = 1; break;
        case AtomType::ATOM_PATH:
            category = 2; subtype = 2; break;
        case AtomType::ATOM_IP_ADDRESS:
            category = 2; subtype = 3; break;
        
        // STRING category (3)
        case AtomType::ATOM_STRING_LITERAL:
            category = 3; subtype = 0; break;
        case AtomType::ATOM_IDENTIFIER:
            category = 3; subtype = 1; break;
        case AtomType::ATOM_REGEX:
            category = 3; subtype = 2; break;
        case AtomType::ATOM_EQUATION:
            category = 3; subtype = 3; break;
        case AtomType::ATOM_EXPRESSION:
            category = 3; subtype = 4; break;
        
        default:
            category = 3; subtype = 7; break;
    }
    
    // Dims [0-3]: Category one-hot encoding
    out_features[category] = floatToFp16(1.0f);
    
    // Dims [4-7]: Sub-type encoding (soft encoding for related types)
    out_features[4 + (subtype % 4)] = floatToFp16(0.8f);
    
    // Dims [8-11]: Length/magnitude features
    float len_f = static_cast<float>(raw_text.size());
    out_features[8] = floatToFp16(std::min(len_f / 100.0f, 1.0f));  // Normalized length
    out_features[9] = floatToFp16(std::log2f(len_f + 1.0f) / 10.0f);  // Log length
    
    // Add category-specific magnitude features
    if (parsed) {
        if (auto* int_val = std::get_if<AtomInteger>(parsed)) {
            float log_mag = std::log2f(std::abs(static_cast<float>(int_val->value)) + 1.0f);
            out_features[10] = floatToFp16(std::min(log_mag / 32.0f, 1.0f));  // Log magnitude
            out_features[11] = floatToFp16(int_val->value < 0 ? 1.0f : 0.0f);  // Sign bit
        } else if (auto* float_val = std::get_if<AtomFloat>(parsed)) {
            float log_mag = std::log2f(std::abs(static_cast<float>(float_val->value)) + 1.0f);
            out_features[10] = floatToFp16(std::min(log_mag / 32.0f, 1.0f));
            out_features[11] = floatToFp16(float_val->has_exponent ? 1.0f : 0.0f);
        } else if (auto* url_val = std::get_if<AtomURL>(parsed)) {
            out_features[10] = floatToFp16(url_val->scheme == "https" ? 1.0f : 0.5f);
            out_features[11] = floatToFp16(!url_val->query.empty() ? 1.0f : 0.0f);
        } else if (auto* date_val = std::get_if<AtomDate>(parsed)) {
            out_features[10] = floatToFp16(static_cast<float>(date_val->year - 1900) / 200.0f);
            out_features[11] = floatToFp16(static_cast<float>(date_val->month) / 12.0f);
        } else if (auto* time_val = std::get_if<AtomTime>(parsed)) {
            out_features[10] = floatToFp16(static_cast<float>(time_val->hour) / 24.0f);
            out_features[11] = floatToFp16(static_cast<float>(time_val->minute) / 60.0f);
        } else if (auto* path_val = std::get_if<AtomPath>(parsed)) {
            out_features[10] = floatToFp16(path_val->is_absolute ? 1.0f : 0.0f);
            out_features[11] = floatToFp16(path_val->is_windows ? 1.0f : 0.0f);
        } else if (auto* ip_val = std::get_if<AtomIP>(parsed)) {
            out_features[10] = floatToFp16(static_cast<float>(ip_val->octets[0]) / 255.0f);
            out_features[11] = floatToFp16(ip_val->is_valid ? 1.0f : 0.0f);
        } else if (auto* id_val = std::get_if<AtomIdentifier>(parsed)) {
            out_features[10] = floatToFp16(static_cast<float>(id_val->style) / 4.0f);
            out_features[11] = floatToFp16(0.5f);
        }
    }
    
    // Dims [12-15]: Structure-specific semantic features
    // Character composition analysis
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
    
    // Has common separator characters
    bool has_slash = raw_text.find('/') != std::string_view::npos;
    bool has_colon = raw_text.find(':') != std::string_view::npos;
    bool has_dot = raw_text.find('.') != std::string_view::npos;
    bool has_at = raw_text.find('@') != std::string_view::npos;
    out_features[15] = floatToFp16(
        (has_slash ? 0.25f : 0.0f) + 
        (has_colon ? 0.25f : 0.0f) + 
        (has_dot ? 0.25f : 0.0f) + 
        (has_at ? 0.25f : 0.0f)
    );
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
    
    auto appendSegmentTokens = [&](size_t start, size_t end) {
        if (end <= start) {
            return;
        }
        std::string segment = text.substr(start, end - start);
        // encode() returns token IDs directly — no need for encodeWithPieces
        // Note: UnigramLM::encode() normalizes to lowercase internally.
        auto segment_ids = unigram_.encode(segment);
                        
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
            auto parsed = AtomTable::parseAtom(span.atom_type, std::string(span.view()));
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

            // BUG FIX: Emit leading whitespace BEFORE the atom token
            // The span was widened to include leading whitespace (detectStructures),
            // but that whitespace must be tokenized separately to preserve it in output.
            // span.start = widened start (may include whitespace)
            // span.content_offset = original atom content start
            if (span.content_offset > span.offset) {
                size_t whitespace_start = span.offset;
                size_t whitespace_end = span.content_offset;
                appendSegmentTokens(whitespace_start, whitespace_end);
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
                if (Tokenizer::isNumericAtom(span.atom_type) && numeric_value != 0.0f) {
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
                span.view(), 
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
    
    return result;
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
    
    return result;
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
        switch (type) {
            case AtomType::ATOM_INTEGER: return "<INT>";
            case AtomType::ATOM_FLOAT: return "<FLOAT>";
            case AtomType::ATOM_HEX: return "<HEX>";
            case AtomType::ATOM_BINARY: return "<BIN>";
            case AtomType::ATOM_URL: return "<URL>";
            case AtomType::ATOM_EMAIL: return "<EMAIL>";
            case AtomType::ATOM_PATH: return "<PATH>";
            case AtomType::ATOM_DATE: return "<DATE>";
            case AtomType::ATOM_TIME: return "<TIME>";
            case AtomType::ATOM_IP_ADDRESS: return "<IP>";
            case AtomType::ATOM_STRING_LITERAL: return "<STR>";
            case AtomType::ATOM_IDENTIFIER: return "<ID>";
            default: return "<ATOM>";
        }
    } else if (isUnigramToken(token_id)) {
        const UnigramPiece* piece = unigram_.getPiece(token_id);
        if (piece) {
            return piece->text;
        }
    }
    return "<UNK>";
}

//======================================================//
//  Detector Functions
//======================================================//

namespace Detector {

bool detectInteger(const std::string& text, size_t pos, size_t& end) {
    if (pos >= text.size()) return false;
    
    size_t i = pos;
    
    // Optional sign
    if (text[i] == '+' || text[i] == '-') {
        if (i + 1 >= text.size() || !std::isdigit(text[i + 1])) {
            return false;
        }
        ++i;
    }
    
    // Must have at least one digit
    if (!std::isdigit(text[i])) return false;
    
    // Consume digits
    while (i < text.size() && std::isdigit(text[i])) {
        ++i;
    }
    
    // Reject if followed by '.' (could be float like 5.3)
    if (i < text.size() && text[i] == '.') {
        return false;
    }
    
    // Reject if followed by e/E + (digit or sign) — scientific notation like 5e3, 5E+2
    // Do NOT reject "5english", "5em", etc. — detectFloat already failed for those,
    // so deferring would create a gap where BOTH detectors reject the digit.
    if (i < text.size() && (text[i] == 'e' || text[i] == 'E')) {
        if (i + 1 < text.size()) {
            char next = text[i + 1];
            if (std::isdigit(next) || next == '+' || next == '-') {
                return false;  // Genuine scientific notation — defer to detectFloat
            }
        }
        // "5english", "5em", "5E_something" — not scientific notation, accept as integer
    }
    
    // NOTE: We intentionally do NOT reject digits followed by alpha (e.g. "5th", "100ms", "3D").
    // The digit run is the integer atom; the alpha suffix gets tokenized separately via Viterbi.
    // The old guard `if (isalpha(text[i])) return false` caused digits in ordinals, units, and
    // version strings to bypass atom detection entirely, leaking raw byte tokens into training.
    
    end = i;
    return i > pos;
}

bool detectFloat(const std::string& text, size_t pos, size_t& end) {
    if (pos >= text.size()) return false;
    
    size_t i = pos;
    bool has_dot = false;
    bool has_exp = false;
    bool has_digit = false;
    
    // Optional sign
    if (text[i] == '+' || text[i] == '-') {
        ++i;
    }
    
    // Integer part or leading dot
    while (i < text.size() && std::isdigit(text[i])) {
        has_digit = true;
        ++i;
    }
    
    // Decimal point
    if (i < text.size() && text[i] == '.') {
        has_dot = true;
        ++i;
        
        // Fractional part
        while (i < text.size() && std::isdigit(text[i])) {
            has_digit = true;
            ++i;
        }
    }
    
    // Exponent
    if (i < text.size() && (text[i] == 'e' || text[i] == 'E')) {
        has_exp = true;
        ++i;
        
        // Optional exponent sign
        if (i < text.size() && (text[i] == '+' || text[i] == '-')) {
            ++i;
        }
        
        // Exponent digits
        bool has_exp_digit = false;
        while (i < text.size() && std::isdigit(text[i])) {
            has_exp_digit = true;
            ++i;
        }
        
        if (!has_exp_digit) return false;
    }
    
    // Must have dot or exponent to be float (not just integer)
    if (!has_digit || (!has_dot && !has_exp)) return false;
    
    end = i;
    return true;
}

bool detectHex(const std::string& text, size_t pos, size_t& end) {
    if (pos + 2 >= text.size()) return false;
    
    if (text[pos] != '0' || (text[pos + 1] != 'x' && text[pos + 1] != 'X')) {
        return false;
    }
    
    size_t i = pos + 2;
    
    // Must have at least one hex digit
    auto isHexDigit = [](char c) {
        return std::isdigit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
    };
    
    if (!isHexDigit(text[i])) return false;
    
    while (i < text.size() && isHexDigit(text[i])) {
        ++i;
    }
    
    end = i;
    return true;
}

bool detectBinary(const std::string& text, size_t pos, size_t& end) {
    if (pos + 2 >= text.size()) return false;
    
    if (text[pos] != '0' || (text[pos + 1] != 'b' && text[pos + 1] != 'B')) {
        return false;
    }
    
    size_t i = pos + 2;
    
    // Must have at least one binary digit
    if (text[i] != '0' && text[i] != '1') return false;
    
    while (i < text.size() && (text[i] == '0' || text[i] == '1')) {
        ++i;
    }
    
    end = i;
    return true;
}

// NOTE: detectURL and detectEmail removed - Aho-Corasick in detectStructures() handles these

bool detectPath(const std::string& text, size_t pos, size_t& end) {
    if (pos >= text.size()) return false;
    
    // Unix path: starts with /
    // Windows path: starts with C:\ or similar
    
    size_t i = pos;
    
    if (text[i] == '/') {
        // Unix absolute path
        ++i;
        while (i < text.size()) {
            char c = text[i];
            if (std::isalnum(c) || c == '/' || c == '.' || c == '_' || c == '-') {
                ++i;
            } else {
                break;
            }
        }
    } else if (std::isalpha(text[i]) && i + 2 < text.size() && 
               text[i + 1] == ':' && (text[i + 2] == '\\' || text[i + 2] == '/')) {
        // Windows absolute path
        i += 3;
        while (i < text.size()) {
            char c = text[i];
            if (std::isalnum(c) || c == '\\' || c == '/' || c == '.' || c == '_' || c == '-' || c == ' ') {
                ++i;
            } else {
                break;
            }
        }
    } else {
        return false;
    }
    
    end = i;
    return i > pos + 1;
}

bool detectDate(const std::string& text, size_t pos, size_t& end) {
    if (pos + 7 > text.size()) return false;  // Minimum: 1/1/25
    
    size_t i = pos;
    
    // Try YYYY-MM-DD format
    if (i + 10 <= text.size() &&
        std::isdigit(text[i]) && std::isdigit(text[i+1]) && 
        std::isdigit(text[i+2]) && std::isdigit(text[i+3]) &&
        text[i+4] == '-' &&
        std::isdigit(text[i+5]) && std::isdigit(text[i+6]) &&
        text[i+7] == '-' &&
        std::isdigit(text[i+8]) && std::isdigit(text[i+9])) {
        end = i + 10;
        return true;
    }
    
    // Try MM/DD/YYYY or M/D/YY format
    int num1 = 0, num2 = 0, num3 = 0;
    size_t start = i;
    
    while (i < text.size() && std::isdigit(text[i]) && i - start < 4) {
        num1 = num1 * 10 + (text[i] - '0');
        ++i;
    }
    
    if (i >= text.size() || text[i] != '/') return false;
    ++i;
    start = i;
    
    while (i < text.size() && std::isdigit(text[i]) && i - start < 4) {
        num2 = num2 * 10 + (text[i] - '0');
        ++i;
    }
    
    if (i >= text.size() || text[i] != '/') return false;
    ++i;
    start = i;
    
    while (i < text.size() && std::isdigit(text[i]) && i - start < 4) {
        num3 = num3 * 10 + (text[i] - '0');
        ++i;
    }
    
    // Basic validation
    if (num1 < 1 || num1 > 12) return false;
    if (num2 < 1 || num2 > 31) return false;
    
    end = i;
    return true;
}

bool detectTime(const std::string& text, size_t pos, size_t& end) {
    if (pos + 4 > text.size()) return false;  // Minimum: 1:23
    
    // BUG FIX: Require whitespace before time to avoid false positives (e.g., arXiv:1234.5425v2)
    if (pos > 0 && !std::isspace(text[pos - 1])) return false;
    
    size_t i = pos;
    
    // Hours
    int hours = 0;
    while (i < text.size() && std::isdigit(text[i]) && i - pos < 2) {
        hours = hours * 10 + (text[i] - '0');
        ++i;
    }
    
    if (i >= text.size() || text[i] != ':') return false;
    ++i;
    
    // Minutes
    if (i + 2 > text.size() || !std::isdigit(text[i]) || !std::isdigit(text[i+1])) {
        return false;
    }
    i += 2;
    
    // Optional seconds
    if (i + 3 <= text.size() && text[i] == ':' && 
        std::isdigit(text[i+1]) && std::isdigit(text[i+2])) {
        i += 3;
    }
    
    // Optional AM/PM
    if (i + 2 <= text.size()) {
        if ((text[i] == 'a' || text[i] == 'A' || text[i] == 'p' || text[i] == 'P') &&
            (text[i+1] == 'm' || text[i+1] == 'M')) {
            i += 2;
        } else if (i + 3 <= text.size() && text[i] == ' ' &&
                   (text[i+1] == 'a' || text[i+1] == 'A' || text[i+1] == 'p' || text[i+1] == 'P') &&
                   (text[i+2] == 'm' || text[i+2] == 'M')) {
            i += 3;
        }
    }
    
    // BUG FIX: Require whitespace after time to avoid false positives
    if (i < text.size() && !std::isspace(text[i]) && text[i] != ',' && text[i] != '.' && text[i] != ')' && text[i] != ']') {
        return false;
    }
    
    end = i;
    return true;
}

bool detectIPAddress(const std::string& text, size_t pos, size_t& end) {
    if (pos + 7 > text.size()) return false;  // Minimum: 1.1.1.1
    
    size_t i = pos;
    int octets = 0;
    
    for (int oct = 0; oct < 4; ++oct) {
        int value = 0;
        size_t start = i;
        
        while (i < text.size() && std::isdigit(text[i]) && i - start < 3) {
            value = value * 10 + (text[i] - '0');
            ++i;
        }
        
        if (i == start || value > 255) return false;
        octets++;
        
        if (oct < 3) {
            if (i >= text.size() || text[i] != '.') return false;
            ++i;
        }
    }
    
    if (octets != 4) return false;
    
    end = i;
    return true;
}

bool detectStringLiteral(const std::string& text, size_t pos, size_t& end) {
    if (pos >= text.size()) return false;
    
    char quote = text[pos];
    if (quote != '"' && quote != '\'') return false;
    
    size_t i = pos + 1;
    
    while (i < text.size()) {
        if (text[i] == '\\' && i + 1 < text.size()) {
            // Escape sequence
            i += 2;
        } else if (text[i] == quote) {
            end = i + 1;
            return true;
        } else {
            ++i;
        }
    }
    
    return false;  // Unclosed string
}

bool detectIdentifier(const std::string& text, size_t pos, size_t& end) {
    if (pos >= text.size()) return false;
    
    char c = text[pos];
    if (!std::isalpha(c) && c != '_') return false;
    
    size_t i = pos + 1;
    
    while (i < text.size()) {
        c = text[i];
        if (std::isalnum(c) || c == '_') {
            ++i;
        } else {
            break;
        }
    }
    
    // Must be at least 2 characters to be interesting
    if (i - pos < 2) return false;
    
    end = i;
    return true;
}

} // namespace Detector

} // namespace Tokenizer
} // namespace GRIM
