//======================================================//
//  UniByte.cu
//  Implementation of unified tokenizer orchestrator
//  
//  STRUCTURAL DETECTION: Uses the raw-text detector registry to scan source
//  byte offsets before tokenization. Detector output never classifies token IDs.
//======================================================//

#include "UniByte.hpp"
#include "AtomTable.hpp"
#include "Detectors/DetectorRegistry.hpp"
#include "TextUtils.hpp"
#include "UnigramViterbi.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>

namespace GRIM {
namespace Tokenizer {

//======================================================//
//  UniByte Implementation
//======================================================//

UniByte::UniByte(const ::GRIM::HyperParameters::TokenizerHP& hp)
    : tokenizer_hp_(hp)
    , detector_registry_(Detector::makeDefaultRawTextDetectorRegistry())
{
    unigram_.setByteFallbackEnabled(tokenizer_hp_.enable_byte_fallback);
    if (detector_registry_.empty()) {
        throw std::runtime_error("UniByte detector registry initialized empty");
    }
}

UniByte::~UniByte() = default;

UniByte::UniByte(UniByte&& other) noexcept
    : tokenizer_hp_(std::move(other.tokenizer_hp_))
    , byte_encoder_(std::move(other.byte_encoder_))
    , unigram_(std::move(other.unigram_))
    , gpu_initialized_(other.gpu_initialized_)
    , detector_registry_(std::move(other.detector_registry_))
{
    other.gpu_initialized_ = false;
}

UniByte& UniByte::operator=(UniByte&& other) noexcept {
    if (this != &other) {
        tokenizer_hp_ = std::move(other.tokenizer_hp_);
        byte_encoder_ = std::move(other.byte_encoder_);
        unigram_ = std::move(other.unigram_);
        gpu_initialized_ = other.gpu_initialized_;
        detector_registry_ = std::move(other.detector_registry_);
        other.gpu_initialized_ = false;
    }
    return *this;
}

//--------------------------------------------------//
// Initialization
//--------------------------------------------------//

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
            const Detector::RawTextDetectorOptions detector_options(
                tokenizer_hp_.detect_numbers,
                true,
                true);
            auto structures = detector_registry_.detectStructures(text, detector_options);
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
        const Detector::RawTextDetectorOptions detector_options(
            tokenizer_hp_.detect_numbers,
            true,
            true);
        structures = detector_registry_.detectStructures(text, detector_options);
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
        const std::string normalized_segment = normalizeSpaces(segment, prepend_word_boundary);
        UnigramViterbiSession segment_session(unigram_, normalized_segment, "UniByte::encodeInternal");
        const auto& segment_ids = segment_session.tokens();
        const auto& segment_fallback_flags = segment_session.fallbackFlags();
        if (segment_fallback_flags.size() != segment_ids.size()) {
            throw std::runtime_error("UniByte::encodeInternal: Viterbi fallback flag count=" +
                                     std::to_string(segment_fallback_flags.size()) +
                                     " != token count=" + std::to_string(segment_ids.size()));
        }
                        
        for (size_t i = 0; i < segment_ids.size(); ++i) {
            const int tid = segment_ids[i];
            const bool fallback_transition = segment_fallback_flags[i];
            const bool byte_fallback_used = fallback_transition && unigram_.byteFallbackEnabled();
            const bool token_is_byte_id = tid >= BYTE_TOKEN_OFFSET && tid < BYTE_TOKEN_OFFSET + BYTE_VOCAB_SIZE;
            if (token_is_byte_id != byte_fallback_used) {
                throw std::runtime_error("UniByte::encodeInternal: Viterbi fallback flag/token-id mismatch at segment token index=" +
                                         std::to_string(i) + ": token_id=" + std::to_string(tid) +
                                         ", fallback_transition=" + (fallback_transition ? std::string("true") : std::string("false")) +
                                         ", byte_fallback_enabled=" + (unigram_.byteFallbackEnabled() ? std::string("true") : std::string("false")));
            }

            result.token_ids.push_back(tid);
            result.token_numeric_values.push_back(0.0f);
            result.token_atom_flags.push_back(0);
            result.atom_entry_ids.push_back(kAtomEntryNone);  // no atom at this position
            appendNonAtomSideChannels();  // Non-atom tokens get mask=0
            
            if (byte_fallback_used) {
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
