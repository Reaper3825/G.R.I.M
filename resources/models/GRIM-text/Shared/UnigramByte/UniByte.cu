//======================================================//
//  UniByte.cu
//  CUDA implementation of unified tokenizer orchestrator
//  
//  STRUCTURAL DETECTION: Uses the raw-text detector registry to scan source
//  byte offsets before tokenization. Detector output never classifies token IDs.
//======================================================//

#include "UniByte.hpp"
#include "AtomTable.hpp"
#include "Detectors/DetectorRegistry.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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
//  Detector State - Raw Text Registry
//======================================================//

    struct UniByte::DetectorState {
        Detector::DetectorRegistry registry;

        DetectorState()
            : registry(Detector::makeDefaultRawTextDetectorRegistry())
        {
            if (registry.empty()) {
                throw std::runtime_error("UniByte::DetectorState registry initialized empty");
            }
        }
    };

//======================================================//
//  UniByte Implementation
//======================================================//

UniByte::UniByte(const ::GRIM::HyperParameters::TokenizerHP& hp)
    : tokenizer_hp_(hp)
    , detector_(nullptr)
{
    unigram_.setByteFallbackEnabled(tokenizer_hp_.enable_byte_fallback);
    detector_ = std::make_unique<DetectorState>();
    initDetector();
}

UniByte::~UniByte() = default;

UniByte::UniByte(UniByte&& other) noexcept
    : tokenizer_hp_(std::move(other.tokenizer_hp_))
    , byte_encoder_(std::move(other.byte_encoder_))
    , unigram_(std::move(other.unigram_))
    , gpu_initialized_(other.gpu_initialized_)
    , detector_(std::move(other.detector_))
{
    other.gpu_initialized_ = false;
}

UniByte& UniByte::operator=(UniByte&& other) noexcept {
    if (this != &other) {
        tokenizer_hp_ = std::move(other.tokenizer_hp_);
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

bool UniByte::save(const std::string& vocab_path, bool save_text_format, float score_multiplier) const {
    return unigram_.save(vocab_path, save_text_format, score_multiplier);
}

bool UniByte::train(const std::vector<std::string>& texts) {
    // Detect atom spans in each text BEFORE training.
    // Atom regions (URLs, emails, numbers, dates, paths, etc.) are SKIPPED
    // during character counting and subword mining so their internal chars
    // (://@.com etc.) don't contaminate the vocabulary.
    //
    // When scratch block reasoning is disabled, atom detection is bypassed
    // entirely so vocab mining matches the runtime tokenization path
    // (encode() -> unigram_.encode() with no atom emission).
    //
    // We must also use the SAME parse-success predicate as encodeInternal():
    // a span whose AtomTable::parseAtom() fails will be encoded as regular
    // text at runtime, so it must NOT be skipped during vocab mining.
    std::vector<std::vector<AtomSpan>> all_atom_spans;
    all_atom_spans.reserve(texts.size());

    size_t total_atoms = 0;
    size_t total_skipped_unparseable = 0;
    const bool detect = tokenizer_hp_.enable_scratch_block_reasoning;
    for (const auto& text : texts) {
        std::vector<AtomSpan> spans;
        if (detect) {
            auto structures = detectStructures(text);
            spans.reserve(structures.size());
            for (const auto& s : structures) {
                // Mirror encodeInternal()'s fallback: if the parser fails,
                // the runtime encoder emits raw text for this span, so vocab
                // mining must see those characters too.
                auto parsed = AtomTable::parseAtom(
                    s.atom_type, std::string(s.contentView()));
                if (!parsed.success) {
                    ++total_skipped_unparseable;
                    continue;
                }
                spans.push_back({s.start, s.end});
            }
            total_atoms += spans.size();
        }
        all_atom_spans.push_back(std::move(spans));
    }

    std::cout << "[UniByte] Detected " << total_atoms << " atoms across "
              << texts.size() << " texts (will skip during vocab training); "
              << "unparseable spans treated as text: " << total_skipped_unparseable
              << "; scratch_block_reasoning=" << (detect ? "on" : "off") << std::endl;
    
    return unigram_.trainFromCorpus(texts, all_atom_spans,
                                     tokenizer_hp_.target_vocab_size, 
                                     tokenizer_hp_.character_coverage,
                                     tokenizer_hp_.min_subword_freq,
                                     tokenizer_hp_.prune_during_mining,
                                     tokenizer_hp_.enable_parallel_subword_mining,
                                     tokenizer_hp_.subword_mining_workers,
                                     tokenizer_hp_.subword_mining_max_bytes);
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

std::vector<Detector::RawTextDetection> UniByte::detectRawText(const std::string& text) const {
    if (!detector_) {
        throw std::runtime_error("UniByte::detectRawText requires initialized detector registry");
    }

    const Detector::RawTextDetectorOptions options(
        tokenizer_hp_.detect_numbers,
        true,
        true);
    return detector_->registry.scan(text, options);
}

std::vector<StructuralSpan> UniByte::detectStructures(const std::string& text) const {
    std::vector<StructuralSpan> spans;
    spans.reserve(32);  // Pre-allocate for typical case
    
    if (!detector_) {
        throw std::runtime_error("UniByte::detectStructures requires initialized detector registry");
    }
    
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
    
    const auto detections = detectRawText(text);
    for (const auto& detection : detections) {
        if (!detection.emitsAtom()) {
            continue;
        }
        spans.push_back(makeSpan(detection.start, detection.end, detection.atom_type));
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
    out_spans.clear();

    // Honor scratch_block_reasoning: when disabled, no atoms exist as far as
    // the rest of the pipeline is concerned, so the input must pass through
    // verbatim with no placeholders injected.
    if (!tokenizer_hp_.enable_scratch_block_reasoning) {
        return text;
    }

    // Apply the SAME parse-success filter that train() and encodeInternal()
    // use, so an injected placeholder never represents a span that the
    // encoder would have emitted as raw text instead.
    auto detected = detectStructures(text);
    out_spans.reserve(detected.size());
    for (const auto& s : detected) {
        auto parsed = AtomTable::parseAtom(
            s.atom_type, std::string(s.contentView()));
        if (!parsed.success) {
            continue;
        }
        out_spans.push_back(s);
    }

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
    tokenizer_hp_.enable_scratch_block_reasoning = enabled;
}

//--------------------------------------------------//
// Encoding
//--------------------------------------------------//

std::vector<int> UniByte::encode(const std::string& text) const {
    // If scratch block reasoning is disabled, use fast path (normal UnigramByte)
    if (!tokenizer_hp_.enable_scratch_block_reasoning) {
        // FAST PATH: No structural detection, no AtomTable, just pure tokenization
        return unigram_.encode(text);
    }
    
    // SCRATCH BLOCK REASONING PATH: Use AtomTable for structural reasoning
    auto result = tokenizeWithMetadata(text);
    return result.token_ids;
}

UniByteResult UniByte::tokenizeWithMetadata(const std::string& text) const {
    // Honor scratch block reasoning toggle: when disabled, skip atom detection
    // entirely so callers on the metadata path see the same plain unigram
    // tokenization as encode(). Atom side-channels remain zero-filled.
    std::vector<StructuralSpan> structures;
    if (tokenizer_hp_.enable_scratch_block_reasoning) {
        structures = detectStructures(text);
    }
    auto result = encodeInternal(text, structures);
    
    // Pipeline contract: validate all per-token arrays are consistent
    // before this result can enter BatchPayload / GPU tensor pipeline.
    result.validate("UniByte::tokenizeWithMetadata");
    
    return result;
}

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
    result.token_atom_mask.reserve(estimated_tokens);
    result.atoms.reserve(structures.size());

    // Helper: append atom mask for non-atom tokens.
    auto appendNonAtomSideChannels = [&]() {
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
        // encode() returns token IDs directly; metadata side channels are already assembled above.
        // Note: UnigramLM::encode() normalizes to lowercase internally.
        // Only prepend ▁ when this is genuinely the first emitted token of
        // the sequence: start-of-text AND nothing has been emitted yet (no
        // prior atom, no prior segment). Tracking via a flag that was only
        // cleared on segment emission caused atom-first inputs ("42apples")
        // to be tokenized like "42 apples".
        const bool prepend_word_boundary =
            (start == 0) && result.token_ids.empty();
        auto segment_ids = unigram_.encode(segment, prepend_word_boundary);
                        
        for (int tid : segment_ids) {
            result.token_ids.push_back(tid);
            result.token_numeric_values.push_back(0.0f);
            result.token_atom_flags.push_back(0);
            result.atom_entry_ids.push_back(kAtomEntryNone);  // no atom at this position
            appendNonAtomSideChannels();  // Non-atom tokens get mask=0
            
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
            bool has_parsed = false;
            
            // Parse atom for numeric side channels.
            auto parsed = AtomTable::parseAtom(span.atom_type, std::string(span.contentView()));
            if (parsed.success) {
                has_parsed = true;
                
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

//--------------------------------------------------//
// Decoding
//--------------------------------------------------//

namespace {

void appendDecodedLayoutToken(std::string& result, int token_id) {
    if (token_id == PAD_TOKEN_ID) return;
    result += specialTokenText(token_id);
}

void appendDecodedByteToken(std::string& result, int token_id) {
    result.push_back(static_cast<char>(token_id - BYTE_TOKEN_OFFSET));
}

void appendDecodedUnigramToken(std::string& result, const UnigramLM& unigram, int token_id) {
    const UnigramPiece* piece = unigram.getPiece(token_id);
    if (!piece) {
        throw std::runtime_error("UniByte::decode: unigram token_id=" + std::to_string(token_id) +
                                 " has no backing UnigramPiece");
    }
    result += piece->text;
}

void appendDecodedAtomToken(std::string& result, const DecodeRequest& request, size_t index, int token_id) {
    if (!request.atom_entry_ids || !request.atom_table) {
        throw std::runtime_error("UniByte::decode: atom token_id=" + std::to_string(token_id) +
                                 " requires DecodeRequest built from UniByteResult");
    }
    if (request.atom_entry_count != request.token_count) {
        throw std::runtime_error("UniByte::decode: atom_entry_count=" +
                                 std::to_string(request.atom_entry_count) +
                                 " != token_count=" + std::to_string(request.token_count));
    }

    const uint32_t entry_id = request.atom_entry_ids[index];
    if (entry_id == kAtomEntryNone) {
        throw std::runtime_error("UniByte::decode: atom token_id=" + std::to_string(token_id) +
                                 " has kAtomEntryNone at index=" + std::to_string(index));
    }

    const AtomEntry* entry = request.atom_table->getAtom(entry_id);
    if (!entry) {
        throw std::runtime_error("UniByte::decode: atom_entry_id=" + std::to_string(entry_id) +
                                 " has no backing AtomEntry");
    }
    result += request.atom_table->atomToString(*entry);
}

} // namespace

std::string UniByte::decode(const DecodeRequest& request) const {
    if (!request.token_ids && request.token_count != 0) {
        throw std::runtime_error("UniByte::decode: token_ids is NULL while token_count=" +
                                 std::to_string(request.token_count));
    }

    std::string result;
    const TokenLayout layout = tokenLayout();
    for (size_t i = 0; i < request.token_count; ++i) {
        const int tid = request.token_ids[i];
        if (layout.isSpecial(tid)) {
            appendDecodedLayoutToken(result, tid);
        } else if (layout.isByte(tid)) {
            appendDecodedByteToken(result, tid);
        } else if (layout.isAtom(tid)) {
            appendDecodedAtomToken(result, request, i, tid);
        } else if (layout.isUnigram(tid)) {
            appendDecodedUnigramToken(result, unigram_, tid);
        } else {
            throw std::runtime_error("UniByte::decode: token_id=" + std::to_string(tid) +
                                     " is outside all known token ranges");
        }
    }

    return UnigramLM::denormalizeFromTokenization(result);
}

//--------------------------------------------------//
// Vocabulary Info
//--------------------------------------------------//

int UniByte::vocabSize() const {
    return UNIGRAM_VOCAB_OFFSET + unigram_.pieceCount();
}

TokenLayout UniByte::tokenLayout() const {
    TokenLayout layout;
    layout.num_special = NUM_SPECIAL_TOKENS;          // from Byte.hpp (live constexpr)
    layout.num_bytes   = BYTE_VOCAB_SIZE;             // from Byte.hpp (live constexpr)
    layout.num_atoms   = ATOM_VOCAB_SIZE;             // from Unigram.hpp (inline, set by configureTokenLayout)
    layout.num_unigram = unigram_.pieceCount();      // learned subword piece count
    return layout;
}


} // namespace Tokenizer
} // namespace GRIM
