#include "HistogramDiagnostic.hpp"

#include "../Phases/Phase1_Startup.hpp"
#include "../../Shared/LogRecorder/BatchLogTape.hpp"
#include "../../Shared/LogRecorder/LogRecorder.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace GRIM::Diagnostics {

namespace {

constexpr int kBlockSize = 256;
constexpr int kMaxBlocks = 65535;

__global__ void histogramKernel(
    const float* __restrict__ values,
    std::size_t value_count,
    unsigned long long* __restrict__ counts,
    int bin_count,
    float min_value,
    float max_value)
{
    const std::size_t first_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * gridDim.x;
    const float bin_scale =
        static_cast<float>(bin_count) / (max_value - min_value);

    for (std::size_t index = first_index;
         index < value_count;
         index += stride) {
        const float value = values[index];
        if (!isfinite(value)) {
            atomicAdd(&counts[bin_count + 2], 1ULL);
            continue;
        }
        if (value < min_value) {
            atomicAdd(&counts[bin_count], 1ULL);
            continue;
        }
        if (value > max_value) {
            atomicAdd(&counts[bin_count + 1], 1ULL);
            continue;
        }

        int bin = static_cast<int>((value - min_value) * bin_scale);
        if (bin >= bin_count) {
            bin = bin_count - 1;
        }
        atomicAdd(&counts[bin], 1ULL);
    }
}

void checkCuda(cudaError_t error, const char* caller) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(caller) + ": " + cudaGetErrorString(error));
    }
}

} // namespace

Histogram computeHistogram(
    const float* device_values,
    std::size_t value_count,
    int bin_count,
    float min_value,
    float max_value,
    cudaStream_t stream)
{
    if (!device_values) {
        throw std::runtime_error("computeHistogram: device_values is NULL");
    }
    if (value_count == 0) {
        throw std::runtime_error("computeHistogram: value_count must be greater than zero");
    }
    if (bin_count <= 0) {
        throw std::runtime_error("computeHistogram: bin_count must be greater than zero");
    }
    if (bin_count > std::numeric_limits<int>::max() - 3) {
        throw std::runtime_error("computeHistogram: bin_count is too large");
    }
    if (!std::isfinite(min_value) || !std::isfinite(max_value) ||
        max_value <= min_value) {
        throw std::runtime_error(
            "computeHistogram: range must be finite and max_value must be greater than min_value");
    }
    if (!stream) {
        throw std::runtime_error("computeHistogram: stream is NULL");
    }

    const std::size_t counter_count = static_cast<std::size_t>(bin_count) + 3;
    const std::size_t counter_bytes =
        counter_count * sizeof(unsigned long long);
    unsigned long long* device_counts = nullptr;
    checkCuda(
        cudaMalloc(reinterpret_cast<void**>(&device_counts), counter_bytes),
        "computeHistogram cudaMalloc");

    try {
        checkCuda(
            cudaMemsetAsync(device_counts, 0, counter_bytes, stream),
            "computeHistogram cudaMemsetAsync");

        const std::size_t required_blocks =
            ((value_count - 1) / kBlockSize) + 1;
        const int blocks = static_cast<int>(
            std::min<std::size_t>(required_blocks, kMaxBlocks));
        histogramKernel<<<blocks, kBlockSize, 0, stream>>>(
            device_values,
            value_count,
            device_counts,
            bin_count,
            min_value,
            max_value);
        checkCuda(cudaGetLastError(), "computeHistogram histogramKernel");

        std::vector<unsigned long long> host_counts(counter_count, 0);
        checkCuda(
            cudaMemcpyAsync(
                host_counts.data(),
                device_counts,
                counter_bytes,
                cudaMemcpyDeviceToHost,
                stream),
            "computeHistogram cudaMemcpyAsync");
        checkCuda(
            cudaStreamSynchronize(stream),
            "computeHistogram cudaStreamSynchronize");

        Histogram histogram;
        histogram.min_value = min_value;
        histogram.max_value = max_value;
        histogram.bins.assign(host_counts.begin(), host_counts.begin() + bin_count);
        histogram.underflow_count = host_counts[bin_count];
        histogram.overflow_count = host_counts[bin_count + 1];
        histogram.non_finite_count = host_counts[bin_count + 2];

        unsigned long long* released_counts = device_counts;
        device_counts = nullptr;
        checkCuda(cudaFree(released_counts), "computeHistogram cudaFree");
        return histogram;
    } catch (...) {
        if (device_counts) {
            cudaFree(device_counts);
        }
        throw;
    }
}

void logHistogram(
    GRIMText::Training::TrainingContext& ctx,
    const Histogram& histogram,
    std::string_view value_name,
    std::string_view phase,
    int batch_idx)
{
    constexpr const char* kEquationTag = "TENSOR_HISTOGRAM";
    if (histogram.bins.empty()) {
        throw std::runtime_error("logHistogram: histogram has no bins");
    }

    const double bin_width =
        (static_cast<double>(histogram.max_value) - histogram.min_value) /
        static_cast<double>(histogram.bins.size());

    std::ostringstream log;
    log << std::fixed << std::setprecision(6)
        << '[' << kEquationTag << ']'
        << " value=" << value_name
        << " phase=" << phase
        << " batch=" << (batch_idx + 1)
        << " range=[" << histogram.min_value << ',' << histogram.max_value << ']'
        << " bins=" << histogram.bins.size()
        << " underflow=" << histogram.underflow_count
        << " overflow=" << histogram.overflow_count
        << " non_finite=" << histogram.non_finite_count;

    for (std::size_t bin = 0; bin < histogram.bins.size(); ++bin) {
        const double lower =
            static_cast<double>(histogram.min_value) + bin * bin_width;
        const double upper = lower + bin_width;
        log << "\n  bin[" << bin << "]=[" << lower << ',' << upper
            << (bin + 1 == histogram.bins.size() ? "]" : ")")
            << " count=" << histogram.bins[bin];
    }

    const std::string record = log.str();
    ctx.logging.logger->log(record);
    EQ_LOG(
        ctx.logging.tape.get(),
        GRIM::Logging::LogGroup::Telemetry,
        GRIM::Logging::LogPhase::DIAGNOSTICS,
        batch_idx,
        kEquationTag,
        record.c_str());
}

} // namespace GRIM::Diagnostics
