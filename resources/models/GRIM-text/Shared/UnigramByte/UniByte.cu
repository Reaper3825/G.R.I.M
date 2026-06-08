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
    , unigram_(hp.enable_byte_fallback)
    , detector_registry_(Detector::makeDefaultRawTextDetectorRegistry())
{
    if (detector_registry_.empty()) {
        throw std::runtime_error("UniByte detector registry initialized empty");
    }
}

UniByte::~UniByte() = default;

UniByte::UniByte(UniByte&& other) noexcept
    : tokenizer_hp_(std::move(other.tokenizer_hp_))
    , unigram_(std::move(other.unigram_))
    , gpu_initialized_(other.gpu_initialized_)
    , detector_registry_(std::move(other.detector_registry_))
{
    other.gpu_initialized_ = false;
}

UniByte& UniByte::operator=(UniByte&& other) noexcept {
    if (this != &other) {
        tokenizer_hp_ = std::move(other.tokenizer_hp_);
        unigram_ = std::move(other.unigram_);
        gpu_initialized_ = other.gpu_initialized_;
        detector_registry_ = std::move(other.detector_registry_);
        other.gpu_initialized_ = false;
    }
    return *this;
}

bool UniByte::initGPU() {
    if (!unigram_.initGPU()) {
        std::cerr << "[UniByte] Failed to initialize Unigram GPU" << std::endl;
        return false;
    }
    
    gpu_initialized_ = true;
    return true;
}

const UnigramTrainingRuntimeReport& UniByte::lastTrainingRuntimeReport() const {
    return unigram_.lastTrainingRuntimeReport();
}

//--------------------------------------------------//
// Encoding
//--------------------------------------------------//

UniByteResult UniByte::tokenizeWithMetadata(const std::string& text) const {
    UniByteResult result;

    if (text.empty()) {
        return result;
    }

    // Honor atom reasoning toggle: when disabled, skip atom detection
    // entirely so callers still get plain unigram tokenization while
    // atom side-channels remain zero-filled.
    std::vector<Detector::RawTextDetection> detections;
    if (tokenizer_hp_.enable_atom_reasoning) {
        const Detector::RawTextDetectorOptions detector_options(
            tokenizer_hp_.detect_numbers,
            true,
            true);
        detections = detector_registry_.scan(text, detector_options);
    }

    // Initialize and finalize atom registry before any unigram segmentation.
    // Detector-emitted atom spans are registered in the per-sequence AtomTable;
    // the later merge loop only emits placeholders for these finalized spans.
    AtomTableFromDetectionsResult atom_table_build = createAtomTableFromRawTextDetections(
        std::string_view(text.data(), text.size()),
        detections,
        "UniByte::tokenizeWithMetadata");
    result.atom_table = std::move(atom_table_build.atom_table);
    std::vector<AtomTokenizationPayload> atom_tokens = std::move(atom_table_build.atom_tokens);

    // Pre-allocate based on heuristic: ~1 token per 3-4 bytes + atoms.
    const size_t estimated_tokens = (text.size() / 3) + atom_tokens.size() + 8;
    result.token_ids.reserve(estimated_tokens);
    result.is_byte_fallback.reserve(estimated_tokens);
    result.token_numeric_values.reserve(estimated_tokens);
    result.token_atom_flags.reserve(estimated_tokens);
    result.atom_entry_ids.reserve(estimated_tokens);
    result.token_atom_mask.reserve(estimated_tokens);
    result.atoms.reserve(atom_tokens.size());

    auto appendNonAtomSideChannels = [&]() {
        result.token_atom_mask.push_back(0);
    };

    size_t pos = 0;
    size_t struct_idx = 0;

    auto appendSegmentTokens = [&](size_t start, size_t end) {
        if (end <= start) {
            return;
        }
        std::string segment = text.substr(start, end - start);
        const bool prepend_word_boundary =
            (start == 0) && result.token_ids.empty();
        const std::string normalized_segment = normalizeSpaces(segment, prepend_word_boundary);
        UnigramViterbiSession segment_session(unigram_, normalized_segment, "UniByte::tokenizeWithMetadata");
        const auto& segment_ids = segment_session.tokens();
        const auto& segment_fallback_flags = segment_session.fallbackFlags();
        if (segment_fallback_flags.size() != segment_ids.size()) {
            throw std::runtime_error("UniByte::tokenizeWithMetadata: Viterbi fallback flag count=" +
                                     std::to_string(segment_fallback_flags.size()) +
                                     " != token count=" + std::to_string(segment_ids.size()));
        }

        for (size_t i = 0; i < segment_ids.size(); ++i) {
            const int tid = segment_ids[i];
            const bool fallback_transition = segment_fallback_flags[i];
            const bool byte_fallback_used = fallback_transition && tokenizer_hp_.enable_byte_fallback;
            const bool token_is_byte_id = tid >= BYTE_TOKEN_OFFSET && tid < BYTE_TOKEN_OFFSET + BYTE_VOCAB_SIZE;
            if (token_is_byte_id != byte_fallback_used) {
                throw std::runtime_error("UniByte::tokenizeWithMetadata: Viterbi fallback flag/token-id mismatch at segment token index=" +
                                         std::to_string(i) + ": token_id=" + std::to_string(tid) +
                                         ", fallback_transition=" + (fallback_transition ? std::string("true") : std::string("false")) +
                                         ", byte_fallback_enabled=" + (tokenizer_hp_.enable_byte_fallback ? std::string("true") : std::string("false")));
            }

            result.token_ids.push_back(tid);
            result.token_numeric_values.push_back(0.0f);
            result.token_atom_flags.push_back(0);
            result.atom_entry_ids.push_back(kAtomEntryNone);
            appendNonAtomSideChannels();

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
        if (struct_idx < atom_tokens.size() && pos == atom_tokens[struct_idx].span.start) {
            const AtomTokenizationPayload& atom_payload = atom_tokens[struct_idx];
            const StructuralSpan& span = atom_payload.span;

            result.token_ids.push_back(atom_payload.token_id);
            result.is_byte_fallback.push_back(atom_payload.is_byte_fallback);
            result.token_numeric_values.push_back(atom_payload.token_numeric_value);
            result.token_atom_flags.push_back(atom_payload.token_atom_flags);
            result.atom_entry_ids.push_back(atom_payload.atom_entry_id);
            result.token_atom_mask.push_back(atom_payload.token_atom_mask);

            result.atoms.push_back(span);
            result.atom_tokens++;

            pos = span.end;
            struct_idx++;
            continue;
        }

        size_t segment_end = text.size();
        if (struct_idx < atom_tokens.size()) {
            segment_end = atom_tokens[struct_idx].span.start;
        }

        appendSegmentTokens(pos, segment_end);

        pos = segment_end;
    }
    
    // Pipeline contract: validate all per-token arrays are consistent
    // before this result can enter BatchPayload / GPU tensor pipeline.
    result.validate("UniByte::tokenizeWithMetadata");
    
    return result;
}

//--------------------------------------------------//
// Decoding
//--------------------------------------------------//

std::string UniByte::decode(const DecodeRequest& request) const {
    if (!request.token_ids && request.token_count != 0) {
        throw std::runtime_error("UniByte::decode: token_ids is NULL while token_count=" +
                                 std::to_string(request.token_count));
    }
    if (request.atom_entry_ids && request.atom_entry_count != request.token_count) {
        throw std::runtime_error("UniByte::decode: atom_entry_count=" +
                                 std::to_string(request.atom_entry_count) +
                                 " != token_count=" + std::to_string(request.token_count));
    }
    if (request.token_numeric_values && request.token_numeric_count != request.token_count) {
        throw std::runtime_error("UniByte::decode: token_numeric_count=" +
                                 std::to_string(request.token_numeric_count) +
                                 " != token_count=" + std::to_string(request.token_count));
    }
    if (request.token_atom_mask && request.token_atom_mask_count != request.token_count) {
        throw std::runtime_error("UniByte::decode: token_atom_mask_count=" +
                                 std::to_string(request.token_atom_mask_count) +
                                 " != token_count=" + std::to_string(request.token_count));
    }

    std::string result;
    const TokenLayout layout = tokenLayoutFromActualVocabOrThrow(
        static_cast<std::uint32_t>(vocabSize()),
        "UniByte::decode");
    for (size_t i = 0; i < request.token_count; ++i) {
        const int tid = request.token_ids[i];
        if (layout.isSpecial(tid)) {
            if (tid != PAD_TOKEN_ID) {
                result += specialTokenText(tid);
            }
        } else if (layout.isByte(tid)) {
            result.push_back(static_cast<char>(tid - BYTE_TOKEN_OFFSET));
        } else if (layout.isAtom(tid)) {
            bool decoded_atom = false;
            if (request.atom_entry_ids && request.atom_table) {
                const uint32_t entry_id = request.atom_entry_ids[i];
                if (entry_id != kAtomEntryNone) {
                    const auto entry = request.atom_table->getAtom(entry_id);
                    if (!entry) {
                        throw std::runtime_error("UniByte::decode: atom_entry_id=" + std::to_string(entry_id) +
                                                 " has no backing AtomEntry");
                    }
                    result += request.atom_table->atomToString(*entry);
                    decoded_atom = true;
                }
            }
            if (!decoded_atom && request.token_atom_mask && request.token_numeric_values &&
                request.token_atom_mask[i] != 0) {
                result += formatNumericValue(request.token_numeric_values[i]);
                decoded_atom = true;
            }
            if (!decoded_atom) {
                result += "<";
                result += atomTypeName(tokenIdToAtomType(tid));
                result += ">";
            }
        } else if (layout.isUnigram(tid)) {
            const UnigramPiece* piece = unigram_.getPiece(tid);
            if (!piece) {
                throw std::runtime_error("UniByte::decode: unigram token_id=" + std::to_string(tid) +
                                         " has no backing UnigramPiece");
            }
            result += piece->text;
        } else {
            throw std::runtime_error("UniByte::decode: token_id=" + std::to_string(tid) +
                                     " is outside all known token ranges");
        }
    }

    return denormalizeSpaces(result);
}

//--------------------------------------------------//
// Vocabulary Info
//--------------------------------------------------//

int UniByte::vocabSize() const {
    return UNIGRAM_VOCAB_OFFSET + unigram_.pieceCount();
}


} // namespace Tokenizer
} // namespace GRIM
