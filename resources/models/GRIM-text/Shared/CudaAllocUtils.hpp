#pragma once

#include <cuda_runtime.h>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <cstdio>
#include "VerboseLogging.hpp"

namespace GRIM {
namespace CudaAlloc {

namespace detail {

constexpr double kBytesPerMiB = 1024.0 * 1024.0;

struct GpuMemSnapshot {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cudaError_t status = cudaSuccess;
};

inline GpuMemSnapshot captureGpuMemSnapshot() {
    GpuMemSnapshot snapshot{};
    snapshot.status = cudaMemGetInfo(&snapshot.free_bytes, &snapshot.total_bytes);
    return snapshot;
}

inline std::string formatMiB(size_t bytes) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << (static_cast<double>(bytes) / kBytesPerMiB);
    return oss.str();
}

inline void maybeLogGpuAllocationRequest(const char* label, size_t bytes) {
    if constexpr (!GRIM::VerboseLogging::ENABLE_GPU_ALLOCATION_LOGS) {
        return;
    }

    const GpuMemSnapshot snapshot = captureGpuMemSnapshot();
    std::ostringstream oss;
    oss << "[GPU_ALLOC] request label=" << (label ? label : "unnamed")
        << " bytes=" << bytes << " (" << formatMiB(bytes) << " MiB)";
    if (snapshot.status == cudaSuccess) {
        oss << " free=" << formatMiB(snapshot.free_bytes) << " MiB"
            << " total=" << formatMiB(snapshot.total_bytes) << " MiB";
    } else {
        oss << " mem_info_error=" << cudaGetErrorString(snapshot.status);
    }
    fprintf(stderr, "%s\n", oss.str().c_str());
}

inline std::string buildCudaAllocFailureMessage(const char* api, const char* label, size_t bytes, cudaError_t err) {
    const GpuMemSnapshot snapshot = captureGpuMemSnapshot();

    std::ostringstream oss;
    oss << api << " failed for '" << (label ? label : "unnamed") << "': "
        << cudaGetErrorString(err)
        << " | request=" << bytes << " bytes (" << formatMiB(bytes) << " MiB)";

    if (snapshot.status == cudaSuccess) {
        oss << " | free=" << snapshot.free_bytes << " bytes (" << formatMiB(snapshot.free_bytes) << " MiB)"
            << " | total=" << snapshot.total_bytes << " bytes (" << formatMiB(snapshot.total_bytes) << " MiB)";
    } else {
        oss << " | cudaMemGetInfo failed: " << cudaGetErrorString(snapshot.status);
    }

    return oss.str();
}

} // namespace detail

inline void cudaMallocOrThrow(void** ptr, size_t bytes, const char* label) {
    detail::maybeLogGpuAllocationRequest(label, bytes);
    cudaError_t err = cudaMalloc(ptr, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(detail::buildCudaAllocFailureMessage("cudaMalloc", label, bytes, err));
    }
}

} // namespace CudaAlloc
} // namespace GRIM
