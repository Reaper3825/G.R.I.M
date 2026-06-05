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

namespace {

[[noreturn]] void throwUnparseableDetectedAtom(const char* caller,
                                               const char* detector_name,
                                               AtomType atom_type,
                                               size_t start,
                                               size_t end,
                                               std::string_view atom_text,
                                               std::string_view parse_error) {
    throw std::runtime_error(std::string(caller) +
                             ": detector-emitted atom span is not parseable; upstream detector/data pipeline bug: detector='" +
                             (detector_name ? std::string(detector_name) : std::string("<unknown>")) +
                             "', atom_type=" + atomTypeName(atom_type) +
                             ", span=[" + std::to_string(start) + ", " + std::to_string(end) +
                             "), text='" + std::string(atom_text) +
                             "', parse_error='" + std::string(parse_error) + "'");
}

} // namespace

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

    // Initialize atom registry before any unigram segmentation. Detector-emitted
    // atom spans are finalized into the per-sequence AtomTable first; the later
    // merge loop only emits placeholders for those pre-registered spans.
    result.atom_table = std::make_shared<AtomTable>();

    // Honor atom reasoning toggle: when disabled, skip atom detection
    // entirely so callers still get plain unigram tokenization while
    // atom side-channels remain zero-filled.
    std::vector<StructuralSpan> structures;
    if (tokenizer_hp_.enable_atom_reasoning) {
        const Detector::RawTextDetectorOptions detector_options(
            tokenizer_hp_.detect_numbers,
            true,
            true);
        const auto detections = detector_registry_.scan(text, detector_options);
        structures.reserve(detections.size());
        for (const auto& detection : detections) {
            if (!detection.emitsAtom()) {
                continue;
            }

            const std::string_view atom_text(text.data() + detection.start,
                                             detection.end - detection.start);
            const auto parsed = AtomTable::parseAtom(detection.atom_type, std::string(atom_text));
            if (!parsed.success) {
                throwUnparseableDetectedAtom("UniByte::tokenizeWithMetadata",
                                             detection.detector_name,
                                             detection.atom_type,
                                             detection.start,
                                             detection.end,
                                             atom_text,
                                             parsed.error_message);
            }

            StructuralSpan span;
            span.start = detection.start;
            span.end = detection.end;
            span.atom_type = detection.atom_type;
            span.buffer_ptr = text.data();
            span.offset = static_cast<uint32_t>(detection.start);
            span.length = static_cast<uint32_t>(detection.end - detection.start);
            span.content_offset = static_cast<uint32_t>(detection.start);
            span.content_length = static_cast<uint32_t>(detection.end - detection.start);
            span.placeholder_id = atomTypeToTokenId(detection.atom_type);
            span.atom_entry_id = result.atom_table->registerSpan(span);
            if (span.atom_entry_id == kAtomEntryNone) {
                throw std::runtime_error("UniByte::tokenizeWithMetadata: registerSpan returned kAtomEntryNone for detector-emitted atom span");
            }
            structures.push_back(span);
        }
    }

    // Pre-allocate based on heuristic: ~1 token per 3-4 bytes + atoms.
    const size_t estimated_tokens = (text.size() / 3) + structures.size() + 8;
    result.token_ids.reserve(estimated_tokens);
    result.is_byte_fallback.reserve(estimated_tokens);
    result.token_numeric_values.reserve(estimated_tokens);
    result.token_atom_flags.reserve(estimated_tokens);
    result.atom_entry_ids.reserve(estimated_tokens);
    result.token_atom_mask.reserve(estimated_tokens);
    result.atoms.reserve(structures.size());

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
        if (struct_idx < structures.size() && pos == structures[struct_idx].start) {
            const StructuralSpan& span = structures[struct_idx];

            int atom_token_id = span.placeholder_id;
            uint32_t entry_id = span.atom_entry_id;
            if (entry_id == kAtomEntryNone) {
                throw std::runtime_error("UniByte::tokenizeWithMetadata: pre-registered atom span is missing atom_entry_id at struct_idx=" +
                                         std::to_string(struct_idx));
            }

            const auto entry = result.atom_table->getAtom(entry_id);
            if (!entry) {
                throw std::runtime_error("UniByte::tokenizeWithMetadata: pre-registered atom_entry_id=" +
                                         std::to_string(entry_id) +
                                         " has no backing AtomEntry at struct_idx=" +
                                         std::to_string(struct_idx));
            }
            float packed_numeric = 0.0f;
            uint32_t packed_flags = 0;
            packed_numeric = entry->numeric_value;
            packed_flags = entry->flags;

            result.token_ids.push_back(atom_token_id);
            result.is_byte_fallback.push_back(false);
            result.token_numeric_values.push_back(packed_numeric);
            result.token_atom_flags.push_back(packed_flags);
            result.atom_entry_ids.push_back(entry_id);
            result.token_atom_mask.push_back(1);

            result.atoms.push_back(span);
            result.atom_tokens++;

            pos = span.end;
            struct_idx++;
            continue;
        }

        size_t segment_end = text.size();
        if (struct_idx < structures.size()) {
            segment_end = structures[struct_idx].start;
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
