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
#include "NumericTokens.hpp"
#include "TextUtils.hpp"
#include "UnigramViterbi.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>

namespace GRIM {
namespace Tokenizer {

namespace {

std::string formatHexByte(unsigned char byte) {
    static constexpr char kHexDigits[] = "0123456789ABCDEF";
    std::string result = "0x";
    result.push_back(kHexDigits[(byte >> 4) & 0x0F]);
    result.push_back(kHexDigits[byte & 0x0F]);
    return result;
}

std::string summarizeByteRun(const std::string& byte_run) {
    constexpr size_t kMaxPreviewBytes = 16;
    std::string summary;
    const size_t preview_count = std::min(byte_run.size(), kMaxPreviewBytes);
    for (size_t i = 0; i < preview_count; ++i) {
        if (!summary.empty()) {
            summary.push_back(' ');
        }
        summary += formatHexByte(static_cast<unsigned char>(byte_run[i]));
    }
    if (byte_run.size() > preview_count) {
        summary += " ...";
    }
    return summary;
}

std::string formatTokenPrefix(const int* token_ids, size_t token_count) {
    std::string result = "[";
    for (size_t i = 0; i < token_count; ++i) {
        if (i != 0) {
            result += ", ";
        }
        result += std::to_string(token_ids[i]);
    }
    result.push_back(']');
    return result;
}

const char* boolText(bool value) {
    if (value) {
        return "true";
    }
    return "false";
}

// U+FFFD REPLACEMENT CHARACTER encoded as UTF-8.
constexpr char kUtf8ReplacementChar[] = "\xEF\xBF\xBD";

void appendValidatedUtf8ByteRun(std::string& result,
                                const std::string& byte_run,
                                const int* token_ids,
                                size_t token_start_index,
                                bool lenient_invalid_utf8) {
    if (byte_run.empty()) {
        return;
    }

    size_t byte_offset = 0;
    while (byte_offset < byte_run.size()) {
        uint32_t codepoint = 0;
        size_t codepoint_len = 0;
        if (!utf8DecodeAt(byte_run, byte_offset, &codepoint, &codepoint_len)) {
            if (lenient_invalid_utf8) {
                // Emit one replacement char and advance a single byte so the
                // rest of the run still gets a chance to decode.
                result += kUtf8ReplacementChar;
                byte_offset += 1;
                continue;
            }
            const unsigned char bad_byte = static_cast<unsigned char>(byte_run[byte_offset]);
            throw std::runtime_error(
                "UniByte::decode: byte-token run starting at token index=" +
                std::to_string(token_start_index) +
                " after prior_token_count=" + std::to_string(token_start_index) +
                " prior_token_ids=" + formatTokenPrefix(token_ids, token_start_index) +
                " produced invalid UTF-8 at run byte offset=" +
                std::to_string(byte_offset) +
                " (token_id=" + std::to_string(token_ids[token_start_index + byte_offset]) +
                ", byte=" + formatHexByte(bad_byte) +
                ", run_bytes=[" + summarizeByteRun(byte_run) + "])"
            );
        }
        result.append(byte_run, byte_offset, codepoint_len);
        byte_offset += codepoint_len;
    }
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

UniByteResult UniByte::tokenizeWithMetadata(
    const std::string& text,
    size_t forced_segment_boundary,
    size_t* token_count_at_boundary) const {
    if (!token_count_at_boundary) {
        return tokenizeWithMetadata(
            text, std::vector<size_t>{}, nullptr);
    }

    std::vector<size_t> token_counts;
    UniByteResult result = tokenizeWithMetadata(
        text,
        std::vector<size_t>{forced_segment_boundary},
        &token_counts);
    *token_count_at_boundary = token_counts.front();
    return result;
}

UniByteResult UniByte::tokenizeWithMetadata(
    const std::string& text,
    const std::vector<size_t>& forced_segment_boundaries,
    std::vector<size_t>* token_counts_at_boundaries) const {
    UniByteResult result;

    const bool track_boundaries = !forced_segment_boundaries.empty();
    std::vector<size_t> ignored_token_counts;
    std::vector<size_t>& token_counts = token_counts_at_boundaries
        ? *token_counts_at_boundaries
        : ignored_token_counts;
    token_counts.assign(forced_segment_boundaries.size(), 0);
    for (size_t index = 0; index < forced_segment_boundaries.size(); ++index) {
        if (forced_segment_boundaries[index] > text.size()) {
            throw std::runtime_error(
                "UniByte::tokenizeWithMetadata: forced segment boundary exceeds text length");
        }
        if (index > 0 &&
            forced_segment_boundaries[index] <
                forced_segment_boundaries[index - 1]) {
            throw std::runtime_error(
                "UniByte::tokenizeWithMetadata: forced segment boundaries must be nondecreasing");
        }
    }
    size_t boundary_index = 0;
    auto recordBoundary = [&](size_t byte_pos) {
        while (boundary_index < forced_segment_boundaries.size() &&
               forced_segment_boundaries[boundary_index] == byte_pos) {
            token_counts[boundary_index] = result.token_ids.size();
            ++boundary_index;
        }
    };

    if (text.empty()) {
        // Keep the contract uniform: every result carries an allocated (possibly
        // empty) per-sequence AtomTable, never a null pointer.
        result.atom_table = std::make_shared<AtomTable>();
        result.local_atom_table = std::make_shared<SequenceLocalAtomTable>();
        recordBoundary(0);
        result.validate("UniByte::tokenizeWithMetadata");
        return result;
    }

    // Honor atom reasoning toggle: when disabled, skip atom detection
    // entirely so callers still get plain unigram tokenization while
    // atom side-channels remain zero-filled.
    std::vector<Detector::RawTextDetection> detections;
    if (tokenizer_hp_.enable_atom_reasoning) {
        const Detector::RawTextDetectorOptions detector_options(
            true,
            true);
        detections = detector_registry_.scan(text, detector_options);
    }

    // Initialize and finalize atom registry before any unigram segmentation.
    // Detector-emitted atom spans are registered in the per-sequence AtomTable;
    // the later merge loop consumes the finalized typed-boundary payloads.
    AtomTableFromDetectionsResult atom_table_build = createAtomTableFromRawTextDetections(
        std::string_view(text.data(), text.size()),
        detections,
        "UniByte::tokenizeWithMetadata");
    result.atom_table = std::move(atom_table_build.atom_table);
    result.local_atom_table = std::move(atom_table_build.local_atom_table);
    std::vector<AtomTokenizationPayload> atom_tokens = std::move(atom_table_build.atom_tokens);

    // Pre-allocate based on heuristic: ~1 token per 3-4 bytes plus two typed
    // boundary tokens per detected atom occurrence.
    const size_t estimated_tokens = (text.size() / 3) + (atom_tokens.size() * 2) + 8;
    result.token_ids.reserve(estimated_tokens);
    result.is_byte_fallback.reserve(estimated_tokens);
    result.token_numeric_values.reserve(estimated_tokens);
    result.token_atom_flags.reserve(estimated_tokens);
    result.atom_entry_ids.reserve(estimated_tokens);
    result.token_local_atom_indices.reserve(estimated_tokens);
    result.token_atom_mask.reserve(estimated_tokens);
    result.atoms.reserve(atom_tokens.size());

    auto appendNonAtomSideChannels = [&]() {
        result.token_atom_mask.push_back(0);
    };

    size_t pos = 0;
    size_t struct_idx = 0;

    auto appendSegmentTokens = [&](size_t start, size_t end, bool encode_numeric_tokens) {
        if (end <= start) {
            return;
        }
        std::string segment = text.substr(start, end - start);
        const bool prepend_word_boundary =
            (start == 0) && result.token_ids.empty();
        const std::string normalized_segment = normalizeSpaces(segment, prepend_word_boundary);
        auto appendViterbiText = [&](std::string_view viterbi_text) {
            if (viterbi_text.empty()) {
                return;
            }
            UnigramViterbiSession segment_session(
                unigram_, std::string(viterbi_text), "UniByte::tokenizeWithMetadata");
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
                                             ", fallback_transition=" + boolText(fallback_transition) +
                                             ", byte_fallback_enabled=" + boolText(tokenizer_hp_.enable_byte_fallback));
                }

                result.token_ids.push_back(tid);
                result.token_numeric_values.push_back(0.0f);
                result.token_atom_flags.push_back(0);
                result.atom_entry_ids.push_back(kAtomEntryNone);
                result.token_local_atom_indices.push_back(kLocalAtomIndexNone);
                appendNonAtomSideChannels();

                result.is_byte_fallback.push_back(false);
                if (byte_fallback_used) {
                    result.is_byte_fallback.back() = true;
                    result.byte_tokens++;
                } else {
                    result.unigram_tokens++;
                }
            }
        };

        if (!encode_numeric_tokens) {
            appendViterbiText(normalized_segment);
            return;
        }

        const std::vector<NumericTokenSpan> numeric_spans =
            findNumericTokenSpans(normalized_segment);
        size_t normalized_pos = 0;
        for (const NumericTokenSpan& numeric_span : numeric_spans) {
            appendViterbiText(std::string_view(normalized_segment).substr(
                normalized_pos, numeric_span.start - normalized_pos));

            const size_t first_numeric_token = result.token_ids.size();
            appendNumericLiteralTokenIds(
                std::string_view(normalized_segment).substr(
                    numeric_span.start, numeric_span.end - numeric_span.start),
                result.token_ids);
            const size_t emitted_numeric_tokens = result.token_ids.size() - first_numeric_token;
            for (size_t i = 0; i < emitted_numeric_tokens; ++i) {
                result.is_byte_fallback.push_back(false);
                result.token_numeric_values.push_back(0.0f);
                result.token_atom_flags.push_back(0);
                result.atom_entry_ids.push_back(kAtomEntryNone);
                result.token_local_atom_indices.push_back(kLocalAtomIndexNone);
                appendNonAtomSideChannels();
            }
            result.numeric_tokens += emitted_numeric_tokens;
            normalized_pos = numeric_span.end;
        }
        appendViterbiText(std::string_view(normalized_segment).substr(normalized_pos));
    };

    while (pos < text.size()) {
        recordBoundary(pos);
        if (struct_idx < atom_tokens.size() && pos == atom_tokens[struct_idx].span.start) {
            const AtomTokenizationPayload& atom_payload = atom_tokens[struct_idx];
            const StructuralSpan& span = atom_payload.span;

            if (!isAtomOpenTokenId(span.open_token_id) ||
                !isAtomCloseTokenId(span.close_token_id) ||
                tokenIdToAtomType(span.open_token_id) != span.atom_type ||
                tokenIdToAtomType(span.close_token_id) != span.atom_type) {
                throw std::runtime_error(
                    "UniByte::tokenizeWithMetadata: atom payload has invalid or mismatched typed boundaries");
            }

            if (boundary_index < forced_segment_boundaries.size() &&
                forced_segment_boundaries[boundary_index] > span.start &&
                forced_segment_boundaries[boundary_index] < span.end) {
                throw std::runtime_error(
                    "UniByte::tokenizeWithMetadata: forced segment boundary falls inside an atom span");
            }

            // The opening boundary is the metadata anchor for the complete
            // typed span. Its numeric target remains out-of-band in the
            // per-token side channels and per-sequence AtomTable.
            result.token_ids.push_back(span.open_token_id);
            result.is_byte_fallback.push_back(false);
            result.token_numeric_values.push_back(atom_payload.token_numeric_value);
            result.token_atom_flags.push_back(atom_payload.token_atom_flags);
            result.atom_entry_ids.push_back(span.atom_entry_id);
            result.token_local_atom_indices.push_back(span.local_atom_index);
            result.token_atom_mask.push_back(1);
            result.atom_tokens++;

            // Keep the detected value model-visible. It follows the ordinary
            // unigram/byte fallback path, but detection is not re-entered, so
            // the content cannot recursively create another atom span.
            appendSegmentTokens(
                static_cast<size_t>(span.content_offset),
                static_cast<size_t>(span.content_offset) +
                    static_cast<size_t>(span.content_length),
                false);

            // Closing boundaries are structural model tokens. The auxiliary
            // atom target is anchored only at the opening boundary.
            result.token_ids.push_back(span.close_token_id);
            result.is_byte_fallback.push_back(false);
            result.token_numeric_values.push_back(0.0f);
            result.token_atom_flags.push_back(0);
            result.atom_entry_ids.push_back(kAtomEntryNone);
            result.token_local_atom_indices.push_back(kLocalAtomIndexNone);
            result.token_atom_mask.push_back(0);
            result.atom_tokens++;

            result.atoms.push_back(span);

            pos = span.end;
            struct_idx++;
            continue;
        }

        size_t segment_end = text.size();
        if (struct_idx < atom_tokens.size()) {
            segment_end = atom_tokens[struct_idx].span.start;
        }
        if (boundary_index < forced_segment_boundaries.size() &&
            forced_segment_boundaries[boundary_index] > pos &&
            forced_segment_boundaries[boundary_index] < segment_end) {
            segment_end = forced_segment_boundaries[boundary_index];
        }

        appendSegmentTokens(pos, segment_end, true);

        pos = segment_end;
    }
    recordBoundary(pos);
    if (track_boundaries &&
        boundary_index != forced_segment_boundaries.size()) {
        throw std::runtime_error(
            "UniByte::tokenizeWithMetadata: failed to materialize all forced segment boundaries");
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
    std::string pending_byte_run;
    size_t pending_byte_run_start = 0;
    const TokenLayout layout = tokenLayoutFromActualVocabOrThrow(
        static_cast<std::uint32_t>(vocabSize()),
        "UniByte::decode");
    auto flushPendingByteRun = [&]() {
        appendValidatedUtf8ByteRun(result,
                                   pending_byte_run,
                                   request.token_ids,
                                   pending_byte_run_start,
                                   request.lenient_invalid_utf8);
        pending_byte_run.clear();
    };
    for (size_t i = 0; i < request.token_count; ++i) {
        const int tid = request.token_ids[i];
        if (layout.isByte(tid)) {
            if (pending_byte_run.empty()) {
                pending_byte_run_start = i;
            }
            pending_byte_run.push_back(static_cast<char>(tid - BYTE_TOKEN_OFFSET));
            continue;
        }

        flushPendingByteRun();
        if (layout.isSpecial(tid)) {
            if (tid != PAD_TOKEN_ID) {
                result += specialTokenText(tid);
            }
        } else if (layout.isNumeric(tid)) {
            result += numericTokenTextOrThrow(tid, "UniByte::decode");
        } else if (layout.isAtom(tid)) {
            // Atom values are model-visible tokens between typed boundaries.
            // Metadata remains available to downstream auxiliary objectives,
            // but decode never substitutes it for in-band span content.
            result += atomTokenText(tid);
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

    flushPendingByteRun();

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
