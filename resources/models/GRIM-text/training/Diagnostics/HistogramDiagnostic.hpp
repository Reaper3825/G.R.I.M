#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace GRIMText::Training {
struct TrainingContext;
}

namespace GRIM::Diagnostics {

struct Histogram {
    float min_value = 0.0f;
    float max_value = 0.0f;
    std::vector<std::uint64_t> bins;
    std::uint64_t underflow_count = 0;
    std::uint64_t overflow_count = 0;
    std::uint64_t non_finite_count = 0;
};

// Bins finite values over the inclusive range [min_value, max_value]. Values
// equal to max_value enter the final bin; values outside the range and NaN/Inf
// are reported by the separate counters.
Histogram computeHistogram(
    const float* device_values,
    std::size_t value_count,
    int bin_count,
    float min_value,
    float max_value,
    cudaStream_t stream);

void logHistogram(
    GRIMText::Training::TrainingContext& ctx,
    const Histogram& histogram,
    std::string_view value_name,
    std::string_view phase,
    int batch_idx);

} // namespace GRIM::Diagnostics
