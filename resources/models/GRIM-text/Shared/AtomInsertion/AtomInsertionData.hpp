//======================================================//
//  AtomInsertionData.hpp
//  Byte-gap supervision for the inference atom inserter
//======================================================//

#pragma once

#include "../UnigramByte/TokenLayout.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace GRIM::AtomInsertion {

// The insertion task reuses the existing opening/closing atom token IDs as
// output classes. This is a task-head dimension, not a new tokenizer vocab.
inline constexpr std::size_t kDelimiterClassCount =
    static_cast<std::size_t>(Tokenizer::ATOM_VOCAB_SIZE);

using GapDelimiterTargets =
    std::array<std::uint8_t, kDelimiterClassCount>;

// Coordinates address gaps around the exact de-annotated source bytes. For N
// bytes, both begin_gap and end_gap are in [0, N]. End is exclusive, so an
// empty string atom may have begin_gap == end_gap.
struct AtomInsertionSpanLabel {
    std::size_t begin_gap = 0;
    std::size_t end_gap = 0;
    Tokenizer::AtomType type = Tokenizer::AtomType::ATOM_INT;
};

struct AtomInsertionExample {
    // Exact source bytes presented to the insertion model. Only authored atom
    // delimiter bytes are removed; whitespace and atom content are preserved.
    std::string plain_text_bytes;

    // One existing byte-token ID per byte in plain_text_bytes. No normalization,
    // unigram segmentation, BOS, EOS, or numeric-token encoding occurs here.
    std::vector<int> byte_token_ids;

    // Dense multi-label supervision over the N + 1 byte gaps. Columns map to
    // [ATOM_TOKEN_OFFSET, UNIGRAM_VOCAB_OFFSET), allowing a gap to contain both
    // a close and an open event for adjacent spans.
    std::vector<GapDelimiterTargets> gap_delimiter_targets;

    // One value per gap. A value of 1 means inserting ASCII delimiter bytes at
    // this gap preserves well-formed UTF-8; 0 marks a gap inside a multibyte
    // code point and must be excluded from training and inference decisions.
    std::vector<std::uint8_t> valid_utf8_gaps;

    // Canonical span view used to validate and decode the dense targets. These
    // are occurrence labels only; AtomTable entry IDs deliberately do not
    // participate in insertion-model supervision.
    std::vector<AtomInsertionSpanLabel> spans;

    std::size_t byteSize() const noexcept { return byte_token_ids.size(); }
    std::size_t gapSize() const noexcept { return gap_delimiter_targets.size(); }
    bool empty() const noexcept { return byte_token_ids.empty(); }

    bool hasDelimiterTarget(std::size_t gap, int delimiter_token_id) const;
    void validate(const char* caller) const;
};

// Converts an existing atom delimiter token ID into its dense task-head column.
// Throws for non-atom IDs.
std::size_t delimiterClassIndexOrThrow(int delimiter_token_id,
                                       const char* caller);

// Inverse mapping for diagnostics and loss-head slicing.
int delimiterTokenIdForClassOrThrow(std::size_t delimiter_class,
                                    const char* caller);

// Builds insertion supervision from authored text such as:
//
//   "A <INT>42</INT> B"
//
// The returned model input is the exact byte string "A 42 B". Opening and
// closing targets are attached to gaps around "42". For non-string atoms,
// whitespace inside authored delimiters remains in plain_text_bytes but stays
// outside the labeled atom span, matching the tokenizer's trimmed content
// contract without deleting source bytes.
AtomInsertionExample buildAtomInsertionExample(
    std::string_view annotated_source,
    const char* caller);

} // namespace GRIM::AtomInsertion

