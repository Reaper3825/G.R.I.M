#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "VerboseLogging.hpp"

namespace GRIM {
namespace CudaAlloc {

namespace Ledger {

struct AllocationStats {
    std::uint64_t count = 0;
    std::uint64_t bytes = 0;
    std::uint64_t max_request_bytes = 0;
};

struct ScopeState {
    std::string tag;
    std::uint64_t total_count = 0;
    std::uint64_t total_bytes = 0;
    std::unordered_map<std::string, AllocationStats> by_label;
};

inline std::mutex& scopeMutex() {
    static std::mutex mutex;
    return mutex;
}

inline std::atomic<bool>& scopeActive() {
    static std::atomic<bool> active{false};
    return active;
}

inline ScopeState& scopeState() {
    static ScopeState state;
    return state;
}

inline void beginScope(const char* tag) {
    if constexpr (!GRIM::VerboseLogging::ENABLE_GPU_ALLOCATION_LEDGER) {
        return;
    }

    std::lock_guard<std::mutex> lock(scopeMutex());
    scopeActive().store(false, std::memory_order_release);
    ScopeState& state = scopeState();
    state.tag = tag ? tag : "unnamed";
    state.total_count = 0;
    state.total_bytes = 0;
    state.by_label.clear();
    scopeActive().store(true, std::memory_order_release);
}

inline void recordAllocation(const char* label, size_t bytes) {
    if constexpr (!GRIM::VerboseLogging::ENABLE_GPU_ALLOCATION_LEDGER) {
        return;
    }

    if (!scopeActive().load(std::memory_order_acquire)) {
        return;
    }

    std::lock_guard<std::mutex> lock(scopeMutex());
    if (!scopeActive().load(std::memory_order_relaxed)) {
        return;
    }

    ScopeState& state = scopeState();
    const std::string key = label ? label : "unnamed";
    AllocationStats& stats = state.by_label[key];
    ++stats.count;
    stats.bytes += bytes;
    stats.max_request_bytes = std::max<std::uint64_t>(stats.max_request_bytes, bytes);
    ++state.total_count;
    state.total_bytes += bytes;
}

inline std::string endScopeSummary(size_t top_n = 48) {
    if constexpr (!GRIM::VerboseLogging::ENABLE_GPU_ALLOCATION_LEDGER) {
        return std::string();
    }

    std::vector<std::pair<std::string, AllocationStats>> entries;
    std::string tag;
    std::uint64_t total_count = 0;
    std::uint64_t total_bytes = 0;
    {
        std::lock_guard<std::mutex> lock(scopeMutex());
        scopeActive().store(false, std::memory_order_release);
        ScopeState& state = scopeState();
        tag = state.tag;
        total_count = state.total_count;
        total_bytes = state.total_bytes;
        entries.reserve(state.by_label.size());
        for (const auto& item : state.by_label) {
            entries.emplace_back(item.first, item.second);
        }
        state.tag.clear();
        state.total_count = 0;
        state.total_bytes = 0;
        state.by_label.clear();
    }

    std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
        if (a.second.bytes != b.second.bytes) {
            return a.second.bytes > b.second.bytes;
        }
        return a.first < b.first;
    });

    auto formatMiB = [](std::uint64_t bytes) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2)
            << (static_cast<double>(bytes) / (1024.0 * 1024.0));
        return oss.str();
    };

    const std::string display_tag = tag.empty() ? "unnamed" : tag;
    std::string header = "GPU_ALLOC_LEDGER";
    std::string header_payload = display_tag;
    const std::string forward_header = "ForwardAllocationSizes ";
    if (display_tag.rfind(forward_header, 0) == 0) {
        header = "ForwardAllocationSizes";
        header_payload = display_tag.substr(forward_header.size());
    }

    std::ostringstream out;
    out << "[" << header << "] " << header_payload
        << " allocations=" << total_count
        << " labels=" << entries.size()
        << " total_bytes=" << total_bytes
        << " total_MiB=" << formatMiB(total_bytes)
        << " top_labels=" << std::min(top_n, entries.size());

    const size_t limit = std::min(top_n, entries.size());
    for (size_t i = 0; i < limit; ++i) {
        const AllocationStats& stats = entries[i].second;
        out << "\n  " << entries[i].first
            << " count=" << stats.count
            << " bytes=" << stats.bytes
            << " MiB=" << formatMiB(stats.bytes)
            << " max_request_MiB=" << formatMiB(stats.max_request_bytes);
    }

    if (entries.size() > limit) {
        std::uint64_t rest_bytes = 0;
        std::uint64_t rest_count = 0;
        for (size_t i = limit; i < entries.size(); ++i) {
            rest_bytes += entries[i].second.bytes;
            rest_count += entries[i].second.count;
        }
        out << "\n  <remaining_labels> count=" << rest_count
            << " labels=" << (entries.size() - limit)
            << " bytes=" << rest_bytes
            << " MiB=" << formatMiB(rest_bytes);
    }

    return out.str();
}

} // namespace Ledger

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
    Ledger::recordAllocation(label, bytes);
}

} // namespace CudaAlloc
} // namespace GRIM
