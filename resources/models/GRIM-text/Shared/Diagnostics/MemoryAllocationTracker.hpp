#pragma once

#include "../CudaAllocUtils.hpp"
#include <map>

// Allocation hooks shared by TensorContract and the training memory measurer.
// No ownership, resets at phase boundaries, or CUDA synchronization is added.
namespace GRIM::MemoryAccounting {

inline constexpr bool Enabled = false;

enum class Kind { Storage, Gradient, EngineGradient, LeafGradient, Saved, Attention };

inline const char* kindName(Kind kind) noexcept {
    switch (kind) {
        case Kind::Gradient: return "graph_gradient";
        case Kind::EngineGradient: return "engine_gradient";
        case Kind::LeafGradient: return "leaf_gradient";
        case Kind::Saved: return "saved_for_backward";
        case Kind::Attention: return "attention_buffer";
        default: return "storage";
    }
}

struct Allocation {
    std::uint64_t id = 0;
    std::uint64_t bytes = 0;
    int device = 0;
    Kind kind = Kind::Storage;
    std::string label;
    cudaEvent_t pending_free = nullptr;
};

struct Totals {
    std::uint64_t bytes = 0;
    std::uint64_t count = 0;
    std::uint64_t pending_free_bytes = 0;
};

struct Snapshot : Totals {
    // An observation failure makes the inventory incomplete, never fatal.
    std::uint64_t observation_errors = 0;
    std::map<std::pair<Kind, std::string>, Totals> by_owner;
};

class Registry {
    std::mutex mutex_;
    std::unordered_map<void*, Allocation> live_;
    std::uint64_t next_id_ = 0;
    std::atomic<std::uint64_t> errors_{0};

public:
    void error() noexcept { ++errors_; }

    void allocated(void* ptr, size_t bytes, int device, const char* label,
                   Kind kind = Kind::Storage) noexcept {
        if (!ptr) return;
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            auto previous = live_.find(ptr);
            if (previous != live_.end()) {
                // Address reuse can precede polling a completed async free.
                if (previous->second.pending_free) {
                    if (cudaEventDestroy(previous->second.pending_free) != cudaSuccess) error();
                }
                // A successful allocator return also proves address reuse
                // when a synchronous free's bookkeeping is still in flight.
                live_.erase(previous);
            }
            live_.emplace(ptr, Allocation{++next_id_, bytes, device, kind,
                label ? label : "unnamed", nullptr});
        } catch (...) { error(); }
    }

    std::uint64_t identity(void* ptr) noexcept {
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto it = live_.find(ptr);
            return it == live_.end() ? 0 : it->second.id;
        } catch (...) { error(); return 0; }
    }

    void classify(void* ptr, Kind kind) noexcept {
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto it = live_.find(ptr);
            if (it != live_.end()) it->second.kind = kind;
        } catch (...) { error(); }
    }

    void released(void* ptr, std::uint64_t id, cudaEvent_t pending = nullptr) noexcept {
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto it = live_.find(ptr);
            // Never erase a newer allocation that reused this address.
            if (id && it != live_.end() && it->second.id == id) {
                if (pending) { it->second.pending_free = pending; return; }
                live_.erase(it);
            }
            if (pending && cudaEventDestroy(pending) != cudaSuccess) error();
        } catch (...) {
            if (pending) cudaEventDestroy(pending);
            error();
        }
    }

    Snapshot snapshot(int device) {
        std::lock_guard<std::mutex> lock(mutex_);
        Snapshot result;
        for (auto it = live_.begin(); it != live_.end();) {
            auto& allocation = it->second;
            if (allocation.device != device) { ++it; continue; }
            if (allocation.pending_free) {
                const auto status = cudaEventQuery(allocation.pending_free);
                if (status == cudaSuccess) {
                    if (cudaEventDestroy(allocation.pending_free) != cudaSuccess) error();
                    it = live_.erase(it);
                    continue;
                }
                if (status != cudaErrorNotReady) error();
            }
            auto& owner = result.by_owner[{allocation.kind, allocation.label}];
            owner.bytes += allocation.bytes;
            ++owner.count;
            result.bytes += allocation.bytes;
            ++result.count;
            if (allocation.pending_free) {
                owner.pending_free_bytes += allocation.bytes;
                result.pending_free_bytes += allocation.bytes;
            }
            ++it;
        }
        result.observation_errors = errors_.load();
        return result;
    }
};

inline Registry& registry() {
    // Intentionally process-lived: static Tensor destructors may call the hooks.
    static Registry* state = new Registry;
    return *state;
}

inline void cudaMallocOrThrow(void** ptr, size_t bytes, const char* label,
                              Kind kind = Kind::Storage) {
    CudaAlloc::cudaMallocOrThrow(ptr, bytes, label);
    if constexpr (Enabled) {
        // Diagnostics must not turn a successful allocation into an exception.
        try {
            int device = 0;
            if (cudaGetDevice(&device) == cudaSuccess)
                registry().allocated(*ptr, bytes, device, label, kind);
            else registry().error();
        } catch (...) {}
    }
}

inline void classify(void* ptr, Kind kind) noexcept {
    if constexpr (Enabled) {
        try { registry().classify(ptr, kind); } catch (...) {}
    }
}

inline cudaError_t free(void* ptr) noexcept {
    std::uint64_t id = 0;
    if constexpr (Enabled) {
        try { id = registry().identity(ptr); } catch (...) {}
    }
    const auto status = cudaFree(ptr);
    if constexpr (Enabled) {
        try {
            if (status == cudaSuccess) registry().released(ptr, id);
            else registry().error();
        } catch (...) {}
    }
    return status;
}

inline cudaError_t freeAsync(void* ptr, cudaStream_t stream) noexcept {
    std::uint64_t id = 0;
    if constexpr (Enabled) {
        try { id = registry().identity(ptr); } catch (...) {}
    }
    const auto status = cudaFreeAsync(ptr, stream);
    if constexpr (Enabled) {
        if (id) {
            cudaEvent_t event = nullptr;
            try {
                if (status == cudaSuccess &&
                    cudaEventCreateWithFlags(&event, cudaEventDisableTiming) == cudaSuccess &&
                    cudaEventRecord(event, stream) == cudaSuccess) {
                    registry().released(ptr, id, event);
                } else {
                    if (event) cudaEventDestroy(event);
                    // Conservatively retain bytes when completion cannot be observed.
                    registry().error();
                }
            } catch (...) { if (event) cudaEventDestroy(event); }
        }
    }
    return status;
}

} // namespace GRIM::MemoryAccounting
