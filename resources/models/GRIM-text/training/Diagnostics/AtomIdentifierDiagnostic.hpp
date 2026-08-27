//======================================================//
//  AtomIdentifierDiagnostic.hpp
//  Per-batch atom delimiter classification diagnostic.
//======================================================//

#pragma once

#include "../../Shared/Batching/BatchPayload.hpp"
#include "../../Shared/UnigramByte/TokenLayout.hpp"

#include <array>
#include <cstdint>
#include <vector>

#include <cuda_runtime_api.h>

namespace GRIM {
struct Tensor;
}

namespace GRIM::Diagnostics {

struct AtomIdentifierClassDiagnostic {
    std::uint64_t target_positive = 0;
    std::uint64_t target_negative = 0;
    std::uint64_t predicted_positive = 0;
    std::uint64_t true_positive = 0;
    std::uint64_t false_positive = 0;
    std::uint64_t false_negative = 0;
    std::uint64_t true_negative = 0;
    double positive_logit_sum = 0.0;
    double negative_logit_sum = 0.0;
    float positive_logit_min = 0.0f;
    float positive_logit_max = 0.0f;
    float negative_logit_min = 0.0f;
    float negative_logit_max = 0.0f;

    double precision() const noexcept;
    double recall() const noexcept;
    double f1() const noexcept;
    double mean_positive_logit() const noexcept;
    double mean_negative_logit() const noexcept;
};

struct AtomIdentifierBatchDiagnostic {
    int batch_number = 0;
    int sequence_count = 0;
    std::uint64_t valid_gap_count = 0;
    std::uint64_t valid_label_count = 0;
    std::uint64_t target_positive = 0;
    std::uint64_t target_negative = 0;
    std::uint64_t predicted_positive = 0;
    std::uint64_t true_positive = 0;
    std::uint64_t false_positive = 0;
    std::uint64_t false_negative = 0;
    std::uint64_t true_negative = 0;
    std::uint64_t sequences_with_positive_targets = 0;
    std::uint64_t sequences_without_positive_targets = 0;
    float decision_logit = 0.0f;
    std::array<AtomIdentifierClassDiagnostic,
               GRIM::Tokenizer::ATOM_VOCAB_SIZE> by_delimiter{};

    double positive_fraction() const noexcept;
    double negatives_per_positive() const noexcept;
    double precision() const noexcept;
    double recall() const noexcept;
    double f1() const noexcept;
    double macro_f1() const noexcept;
};

// Pure host-side reducer used by the live diagnostic and focused tests.
// delimiter_logits must use the compact layout
// [payload.atomInsertionGapRowCount(), ATOM_VOCAB_SIZE].
AtomIdentifierBatchDiagnostic computeAtomIdentifierBatchDiagnostic(
    const GRIM::Batching::BatchPayload& payload,
    const std::vector<float>& delimiter_logits,
    int batch_idx,
    float decision_logit = 0.0f);

// Per-batch live entry point. The caller owns cadence and supplies the active
// CUDA stream and global step explicitly. It copies only the eight atom
// delimiter columns to the host, computes target/prediction confusion metrics,
// and emits one aggregate plus one per-type module log line.
void runAtomIdentifierDiagnostic(
    const GRIM::Batching::BatchPayload& payload,
    const GRIM::Tensor& full_gap_vocab_logits,
    cudaStream_t stream,
    int batch_idx,
    std::uint64_t global_step,
    float decision_logit = 0.0f);

} // namespace GRIM::Diagnostics
