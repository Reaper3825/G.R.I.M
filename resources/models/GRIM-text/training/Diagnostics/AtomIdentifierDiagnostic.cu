//======================================================//
//  AtomIdentifierDiagnostic.cu
//  Per-batch atom delimiter classification diagnostic.
//======================================================//

#include "AtomIdentifierDiagnostic.hpp"

#include "../../Shared/LogRecorder/LogRecorder.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace GRIM::Diagnostics {
namespace {

constexpr int kTypeCount = GRIM::Tokenizer::kAtomTypeCount;
constexpr int kDelimiterCount = GRIM::Tokenizer::ATOM_VOCAB_SIZE;

static_assert(kDelimiterCount == kTypeCount * 2,
              "atom delimiter layout must contain one open and close class per type");

double safeRatio(std::uint64_t numerator, std::uint64_t denominator) noexcept {
    return denominator == 0
        ? 0.0
        : static_cast<double>(numerator) / static_cast<double>(denominator);
}

void initializeLogitRanges(AtomIdentifierClassDiagnostic& stats) {
    stats.positive_logit_min = std::numeric_limits<float>::infinity();
    stats.positive_logit_max = -std::numeric_limits<float>::infinity();
    stats.negative_logit_min = std::numeric_limits<float>::infinity();
    stats.negative_logit_max = -std::numeric_limits<float>::infinity();
}

void finalizeLogitRanges(AtomIdentifierClassDiagnostic& stats) {
    if (stats.target_positive == 0) {
        stats.positive_logit_min = 0.0f;
        stats.positive_logit_max = 0.0f;
    }
    if (stats.target_negative == 0) {
        stats.negative_logit_min = 0.0f;
        stats.negative_logit_max = 0.0f;
    }
}

std::string delimiterName(int delimiter_class) {
    return GRIM::Tokenizer::atomTokenText(
        GRIM::Tokenizer::ATOM_TOKEN_OFFSET + delimiter_class);
}

void checkCuda(cudaError_t error, const char* caller) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(caller) + ": " + cudaGetErrorString(error));
    }
}

void appendClassMetrics(std::ostringstream& line,
                        const char* boundary_name,
                        int class_index,
                        const AtomIdentifierClassDiagnostic& stats) {
    line << " " << boundary_name << "=" << delimiterName(class_index)
         << "{target=" << stats.target_positive
         << ",pred=" << stats.predicted_positive
         << ",TP=" << stats.true_positive
         << ",FP=" << stats.false_positive
         << ",FN=" << stats.false_negative
         << ",P=" << stats.precision()
         << ",R=" << stats.recall()
         << ",F1=" << stats.f1()
         << ",pos_logit_mean=" << stats.mean_positive_logit()
         << ",pos_logit_min=" << stats.positive_logit_min
         << ",pos_logit_max=" << stats.positive_logit_max
         << ",neg_logit_mean=" << stats.mean_negative_logit()
         << "}";
}

} // namespace

double AtomIdentifierClassDiagnostic::precision() const noexcept {
    return safeRatio(true_positive, true_positive + false_positive);
}

double AtomIdentifierClassDiagnostic::recall() const noexcept {
    return safeRatio(true_positive, true_positive + false_negative);
}

double AtomIdentifierClassDiagnostic::f1() const noexcept {
    const double p = precision();
    const double r = recall();
    return (p + r) == 0.0 ? 0.0 : (2.0 * p * r) / (p + r);
}

double AtomIdentifierClassDiagnostic::mean_positive_logit() const noexcept {
    return target_positive == 0
        ? 0.0
        : positive_logit_sum / static_cast<double>(target_positive);
}

double AtomIdentifierClassDiagnostic::mean_negative_logit() const noexcept {
    return target_negative == 0
        ? 0.0
        : negative_logit_sum / static_cast<double>(target_negative);
}

double AtomIdentifierBatchDiagnostic::positive_fraction() const noexcept {
    return safeRatio(target_positive, valid_label_count);
}

double AtomIdentifierBatchDiagnostic::negatives_per_positive() const noexcept {
    return target_positive == 0
        ? std::numeric_limits<double>::infinity()
        : static_cast<double>(target_negative) /
              static_cast<double>(target_positive);
}

double AtomIdentifierBatchDiagnostic::precision() const noexcept {
    return safeRatio(true_positive, true_positive + false_positive);
}

double AtomIdentifierBatchDiagnostic::recall() const noexcept {
    return safeRatio(true_positive, true_positive + false_negative);
}

double AtomIdentifierBatchDiagnostic::f1() const noexcept {
    const double p = precision();
    const double r = recall();
    return (p + r) == 0.0 ? 0.0 : (2.0 * p * r) / (p + r);
}

double AtomIdentifierBatchDiagnostic::macro_f1() const noexcept {
    double total = 0.0;
    int supported_classes = 0;
    for (const auto& stats : by_delimiter) {
        if (stats.target_positive == 0) {
            continue;
        }
        total += stats.f1();
        ++supported_classes;
    }
    return supported_classes == 0
        ? 0.0
        : total / static_cast<double>(supported_classes);
}

AtomIdentifierBatchDiagnostic computeAtomIdentifierBatchDiagnostic(
    const GRIM::Batching::BatchPayload& payload,
    const std::vector<float>& delimiter_logits,
    int batch_idx,
    float decision_logit) {
    constexpr const char* caller = "computeAtomIdentifierBatchDiagnostic";
    payload.validate(caller);
    if (!payload.EnableAtomIdentification || !payload.isTraining()) {
        throw std::invalid_argument(
            "computeAtomIdentifierBatchDiagnostic: authored atom training payload required");
    }
    if (!std::isfinite(decision_logit)) {
        throw std::invalid_argument(
            "computeAtomIdentifierBatchDiagnostic: decision logit is not finite");
    }

    const std::size_t gap_rows = static_cast<std::size_t>(
        payload.atomInsertionGapRowCount());
    const std::size_t expected_logits =
        gap_rows * static_cast<std::size_t>(kDelimiterCount);
    if (delimiter_logits.size() != expected_logits) {
        throw std::invalid_argument(
            "computeAtomIdentifierBatchDiagnostic: compact delimiter-logit size=" +
            std::to_string(delimiter_logits.size()) +
            " expected=" + std::to_string(expected_logits));
    }

    AtomIdentifierBatchDiagnostic result{};
    result.batch_number = batch_idx + 1;
    result.sequence_count = payload.batch_size;
    result.decision_logit = decision_logit;
    for (auto& stats : result.by_delimiter) {
        initializeLogitRanges(stats);
    }

    const int gaps_per_sequence =
        payload.atom_insertion_gap_rows_per_sequence;
    for (int row = 0; row < payload.batch_size; ++row) {
        std::uint64_t sequence_positive_targets = 0;
        for (int gap = 0; gap < gaps_per_sequence; ++gap) {
            const std::size_t flat_gap =
                static_cast<std::size_t>(row) * gaps_per_sequence + gap;
            if (payload.atom_insertion_valid_gap_mask[flat_gap] == 0) {
                continue;
            }
            ++result.valid_gap_count;

            for (int delimiter_class = 0;
                 delimiter_class < kDelimiterCount;
                 ++delimiter_class) {
                const std::size_t compact_index =
                    flat_gap * kDelimiterCount + delimiter_class;
                const bool target_positive =
                    payload.atom_insertion_gap_targets[compact_index] != 0;
                const float logit = delimiter_logits[compact_index];
                if (!std::isfinite(logit)) {
                    throw std::runtime_error(
                        "computeAtomIdentifierBatchDiagnostic: non-finite logit at gap=" +
                        std::to_string(flat_gap) + " class=" +
                        std::to_string(delimiter_class));
                }
                const bool predicted_positive = logit >= decision_logit;
                auto& stats = result.by_delimiter[delimiter_class];

                if (target_positive) {
                    ++sequence_positive_targets;
                    ++stats.target_positive;
                    stats.positive_logit_sum += static_cast<double>(logit);
                    stats.positive_logit_min =
                        std::min(stats.positive_logit_min, logit);
                    stats.positive_logit_max =
                        std::max(stats.positive_logit_max, logit);
                } else {
                    ++stats.target_negative;
                    stats.negative_logit_sum += static_cast<double>(logit);
                    stats.negative_logit_min =
                        std::min(stats.negative_logit_min, logit);
                    stats.negative_logit_max =
                        std::max(stats.negative_logit_max, logit);
                }
                if (predicted_positive) {
                    ++stats.predicted_positive;
                }

                if (target_positive && predicted_positive) {
                    ++stats.true_positive;
                } else if (!target_positive && predicted_positive) {
                    ++stats.false_positive;
                } else if (target_positive) {
                    ++stats.false_negative;
                } else {
                    ++stats.true_negative;
                }
            }
        }

        if (sequence_positive_targets == 0) {
            ++result.sequences_without_positive_targets;
        } else {
            ++result.sequences_with_positive_targets;
        }
    }

    result.valid_label_count =
        result.valid_gap_count * static_cast<std::uint64_t>(kDelimiterCount);
    for (auto& stats : result.by_delimiter) {
        finalizeLogitRanges(stats);
        result.target_positive += stats.target_positive;
        result.target_negative += stats.target_negative;
        result.predicted_positive += stats.predicted_positive;
        result.true_positive += stats.true_positive;
        result.false_positive += stats.false_positive;
        result.false_negative += stats.false_negative;
        result.true_negative += stats.true_negative;
    }

    if (result.valid_gap_count !=
        static_cast<std::uint64_t>(payload.atom_insertion_valid_gap_count) ||
        result.target_positive !=
        static_cast<std::uint64_t>(payload.atom_insertion_positive_label_count) ||
        result.target_positive + result.target_negative !=
        result.valid_label_count) {
        throw std::runtime_error(
            "computeAtomIdentifierBatchDiagnostic: recounted payload totals disagree with authored telemetry");
    }
    return result;
}

void runAtomIdentifierDiagnostic(
    const GRIM::Batching::BatchPayload& payload,
    const GRIM::Tensor& full_gap_vocab_logits,
    cudaStream_t stream,
    int batch_idx,
    std::uint64_t global_step,
    float decision_logit) {
    if (!payload.EnableAtomIdentification) {
        throw std::invalid_argument(
            "runAtomIdentifierDiagnostic: atom-identification payload required");
    }
    if (!full_gap_vocab_logits.data) {
        throw std::runtime_error(
            "runAtomIdentifierDiagnostic: live gap logits are NULL");
    }

    const auto& shape = full_gap_vocab_logits.shape.require(
        "runAtomIdentifierDiagnostic full_gap_vocab_logits");
    if (!shape.is_2d_layout()) {
        throw std::runtime_error(
            "runAtomIdentifierDiagnostic: gap logits must be a 2D tensor");
    }
    const auto matrix = shape.as_2d();
    const int gap_rows = payload.atomInsertionGapRowCount();
    if (matrix.rows != gap_rows || matrix.cols != payload.vocab_size) {
        throw std::runtime_error(
            "runAtomIdentifierDiagnostic: full-vocabulary gap-logit shape mismatch");
    }

    std::vector<float> compact_logits(
        static_cast<std::size_t>(gap_rows) * kDelimiterCount);
    if (!stream) {
        throw std::invalid_argument(
            "runAtomIdentifierDiagnostic: CUDA stream is NULL");
    }
    checkCuda(
        cudaMemcpy2DAsync(
            compact_logits.data(),
            static_cast<std::size_t>(kDelimiterCount) * sizeof(float),
            full_gap_vocab_logits.data + GRIM::Tokenizer::ATOM_TOKEN_OFFSET,
            static_cast<std::size_t>(payload.vocab_size) * sizeof(float),
            static_cast<std::size_t>(kDelimiterCount) * sizeof(float),
            static_cast<std::size_t>(gap_rows),
            cudaMemcpyDeviceToHost,
            stream),
        "runAtomIdentifierDiagnostic cudaMemcpy2DAsync");
    checkCuda(cudaStreamSynchronize(stream),
              "runAtomIdentifierDiagnostic cudaStreamSynchronize");

    const auto report = computeAtomIdentifierBatchDiagnostic(
        payload, compact_logits, batch_idx, decision_logit);

    std::ostringstream aggregate;
    aggregate << std::fixed << std::setprecision(6)
              << "[AtomIdentifierDiagnostic] batch=" << report.batch_number
              << " threshold_logit=" << report.decision_logit
              << " sequences=" << report.sequence_count
              << " sequences_with_targets="
              << report.sequences_with_positive_targets
              << " sequences_without_targets="
              << report.sequences_without_positive_targets
              << " valid_gaps=" << report.valid_gap_count
              << " labels=" << report.valid_label_count
              << " target_pos=" << report.target_positive
              << " target_neg=" << report.target_negative
              << " pos_rate=" << (100.0 * report.positive_fraction()) << "%"
              << " neg_per_pos=";
    if (report.target_positive == 0) {
        aggregate << "undefined";
    } else {
        aggregate << report.negatives_per_positive();
    }
    aggregate << " predicted_pos=" << report.predicted_positive
              << " TP=" << report.true_positive
              << " FP=" << report.false_positive
              << " FN=" << report.false_negative
              << " TN=" << report.true_negative
              << " precision=" << report.precision()
              << " recall=" << report.recall()
              << " micro_f1=" << report.f1()
              << " macro_f1=" << report.macro_f1();
    GRIM::Logging::EmitModuleInfo(
        GRIM::Logging::ModuleId::ForwardPass,
        aggregate.str(),
        global_step);

    for (int type_index = 0; type_index < kTypeCount; ++type_index) {
        const int open_index = type_index;
        const int close_index = kTypeCount + type_index;
        std::ostringstream type_line;
        type_line << std::fixed << std::setprecision(6)
                  << "[AtomIdentifierDiagnostic] batch=" << report.batch_number
                  << " type=" << GRIM::Tokenizer::atomTypeName(
                         static_cast<GRIM::Tokenizer::AtomType>(type_index));
        appendClassMetrics(
            type_line, "open", open_index, report.by_delimiter[open_index]);
        appendClassMetrics(
            type_line, "close", close_index, report.by_delimiter[close_index]);
        GRIM::Logging::EmitModuleInfo(
            GRIM::Logging::ModuleId::ForwardPass,
            type_line.str(),
            global_step);
    }
}

} // namespace GRIM::Diagnostics
