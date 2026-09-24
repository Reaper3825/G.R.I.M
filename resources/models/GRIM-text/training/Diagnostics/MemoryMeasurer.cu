//======================================================//
//  MemoryMeasurer.cu
//  Immediate, owner-labelled Phase 2 GPU-memory logging.
//======================================================//

#include "MemoryMeasurer.hpp"

#include "../Phases/Phase1_Startup.hpp"
#include "../../Shared/Forward/ModelForwardOutputs.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

namespace GRIMText::Training::Memory {
namespace {

struct GpuMemorySnapshot {
    std::uint64_t device_used = 0;
    std::uint64_t device_free = 0;
    std::uint64_t device_total = 0;
    std::uint64_t pool_reserved_current = 0;
    std::uint64_t pool_reserved_high = 0;
    std::uint64_t pool_used_current = 0;
    std::uint64_t pool_used_high = 0;
    bool pool_available = false;
};

bool queryGpuMemory(GpuMemorySnapshot& snapshot) noexcept {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        return false;
    }

    snapshot.device_free = static_cast<std::uint64_t>(free_bytes);
    snapshot.device_total = static_cast<std::uint64_t>(total_bytes);
    snapshot.device_used = snapshot.device_total - snapshot.device_free;

    int device = 0;
    cudaMemPool_t pool = nullptr;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetDefaultMemPool(&pool, device) != cudaSuccess ||
        pool == nullptr) {
        return true;
    }

    auto readPoolAttribute = [pool](cudaMemPoolAttr attribute, std::uint64_t& destination) {
        unsigned long long value = 0;
        if (cudaMemPoolGetAttribute(pool, attribute, &value) != cudaSuccess) {
            return false;
        }
        destination = static_cast<std::uint64_t>(value);
        return true;
    };

    snapshot.pool_available =
        readPoolAttribute(cudaMemPoolAttrReservedMemCurrent, snapshot.pool_reserved_current) &&
        readPoolAttribute(cudaMemPoolAttrReservedMemHigh, snapshot.pool_reserved_high) &&
        readPoolAttribute(cudaMemPoolAttrUsedMemCurrent, snapshot.pool_used_current) &&
        readPoolAttribute(cudaMemPoolAttrUsedMemHigh, snapshot.pool_used_high);
    return true;
}

double toMiB(std::uint64_t bytes) {
    constexpr double kBytesPerMiB = 1024.0 * 1024.0;
    return static_cast<double>(bytes) / kBytesPerMiB;
}

void logAllocationMemory(TrainingContext& ctx, const std::string& owner,
                         int batch_idx, int accumulation_slot,
                         const GpuMemorySnapshot& device, int device_id,
                         const GRIM::MemoryAccounting::Snapshot& allocations) {
    using namespace GRIM::MemoryAccounting;
    if (owner == "microbatch.persistent_floor") {
        ctx.gpu_memory_baseline_valid = true;
        ctx.gpu_memory_baseline_device = device_id;
        ctx.gpu_memory_baseline_device_bytes = device.device_used;
        ctx.gpu_memory_baseline_tracked_bytes = allocations.bytes;
    }

    std::map<Kind, std::uint64_t> categories;
    for (const auto& entry : allocations.by_owner)
        categories[entry.first.first] += entry.second.bytes;

    std::ostringstream line;
    line << std::fixed << std::setprecision(2)
         << "[GRAPH_MEMORY] sample=" << ctx.gpu_memory_measurement_count
         << " owner=" << owner << " batch=" << (batch_idx + 1)
         << " accumulation_slot=" << accumulation_slot
         << " device=" << device_id << " coverage=tensor_contract"
         << " live_allocations=" << allocations.count
         << " tracked_bytes=" << allocations.bytes
         << " tracked_MiB=" << toMiB(allocations.bytes)
         << " pending_free_bytes=" << allocations.pending_free_bytes
         << " pending_free_MiB=" << toMiB(allocations.pending_free_bytes)
         << " observation_errors=" << allocations.observation_errors
         << " inventory_complete=" << (allocations.observation_errors ? "false" : "true");
    for (const auto kind : {Kind::Storage, Kind::Gradient, Kind::EngineGradient, Kind::LeafGradient,
                            Kind::Saved, Kind::Attention}) {
        line << " " << kindName(kind) << "_bytes=" << categories[kind]
             << " " << kindName(kind) << "_MiB=" << toMiB(categories[kind]);
    }
    if (ctx.gpu_memory_baseline_valid && ctx.gpu_memory_baseline_device == device_id) {
        const auto device_delta = static_cast<std::int64_t>(device.device_used) -
            static_cast<std::int64_t>(ctx.gpu_memory_baseline_device_bytes);
        const auto tracked_delta = static_cast<std::int64_t>(allocations.bytes) -
            static_cast<std::int64_t>(ctx.gpu_memory_baseline_tracked_bytes);
        const auto residual = device_delta - tracked_delta;
        // Signed: a negative residual is useful evidence, not clamped away.
        line << " baseline_device_bytes=" << ctx.gpu_memory_baseline_device_bytes
             << " baseline_tracked_bytes=" << ctx.gpu_memory_baseline_tracked_bytes
             << " device_delta_bytes=" << device_delta
             << " device_delta_MiB=" << (static_cast<double>(device_delta) / (1024.0 * 1024.0))
             << " tracked_delta_bytes=" << tracked_delta
             << " tracked_delta_MiB=" << (static_cast<double>(tracked_delta) / (1024.0 * 1024.0))
             << " residual_bytes=" << residual
             << " residual_MiB=" << (static_cast<double>(residual) / (1024.0 * 1024.0));
    } else {
        line << " baseline=unavailable";
    }
    ctx.logging.logger->log(line.str());

    // Detailed ownership only at the three graph lifetime boundaries. All
    // existing checkpoints receive the compact category/reconciliation line.
    if (owner != "forward.outputs_live" && owner != "backward.complete_graph_live" &&
        owner != "microbatch.graph_cleared") return;
    std::ostringstream inventory;
    inventory << std::fixed << std::setprecision(2)
              << "[GraphAllocationSizes] sample=" << ctx.gpu_memory_measurement_count
              << " owner=" << owner << " batch=" << (batch_idx + 1)
              << " accumulation_slot=" << accumulation_slot
              << " total_bytes=" << allocations.bytes
              << " total_MiB=" << toMiB(allocations.bytes);
    for (const auto& entry : allocations.by_owner) {
        std::string label = entry.first.second;
        for (char& c : label) if (c == ' ' || c == '\n' || c == '\r' || c == '\t') c = '_';
        inventory << "\n  " << label << " category=" << kindName(entry.first.first)
                  << " allocations=" << entry.second.count
                  << " bytes=" << entry.second.bytes
                  << " MiB=" << toMiB(entry.second.bytes)
                  << " pending_free_bytes=" << entry.second.pending_free_bytes;
    }
    ctx.logging.logger->log(inventory.str());
}

} // namespace

void measurePeakMemory(
    TrainingContext& ctx,
    const char* owner,
    int batch_idx,
    int accumulation_slot) noexcept
{
    if constexpr (!EnablePeakMemoryMeasurer) {
        return;
    }

    try {
        const std::string current_owner =
            (owner && owner[0] != '\0') ? owner : "unknown";

        int device_id = -1;
        GRIM::MemoryAccounting::Snapshot allocations;
        bool allocation_query_ok = false;
        try {
            if (cudaGetDevice(&device_id) == cudaSuccess) {
                allocations = GRIM::MemoryAccounting::registry().snapshot(device_id);
                allocation_query_ok = true;
            }
        } catch (...) {
            // Preserve the original device-wide sample if inventory fails.
        }

        GpuMemorySnapshot snapshot;
        if (!queryGpuMemory(snapshot)) {
            ctx.gpu_memory_baseline_valid = false;
            if (ctx.logging.logger) {
                ctx.logging.logger->log(
                    "[PEAK_MEMORY] owner=" + current_owner +
                    " batch=" + std::to_string(batch_idx + 1) +
                    " accumulation_slot=" + std::to_string(accumulation_slot) +
                    " status=query_failed");
            }
            return;
        }

        const bool has_previous_sample = ctx.gpu_memory_measurement_count > 0;
        const std::uint64_t previous_used = ctx.last_gpu_used_bytes;
        const std::string previous_owner = ctx.last_gpu_memory_owner.empty()
            ? "none"
            : ctx.last_gpu_memory_owner;
        const std::int64_t delta = !has_previous_sample
            ? 0
            : (snapshot.device_used >= previous_used
                ? static_cast<std::int64_t>(snapshot.device_used - previous_used)
                : -static_cast<std::int64_t>(previous_used - snapshot.device_used));

        const bool is_new_peak = snapshot.device_used > ctx.peak_gpu_used_bytes;
        if (is_new_peak) {
            ctx.peak_gpu_used_bytes = snapshot.device_used;
            ctx.peak_gpu_used_owner = current_owner;
            ctx.peak_gpu_used_batch = batch_idx;
            ctx.peak_gpu_used_accumulation_slot = accumulation_slot;
        }
        ctx.gpu_total_bytes = snapshot.device_total;
        ctx.last_gpu_used_bytes = snapshot.device_used;
        ctx.last_gpu_memory_owner = current_owner;
        ++ctx.gpu_memory_measurement_count;

        if (!ctx.logging.logger) {
            return;
        }

        const std::uint64_t pool_retained =
            snapshot.pool_reserved_current > snapshot.pool_used_current
                ? snapshot.pool_reserved_current - snapshot.pool_used_current
                : 0;
        const std::uint64_t non_pool_used =
            snapshot.device_used > snapshot.pool_reserved_current
                ? snapshot.device_used - snapshot.pool_reserved_current
                : 0;

        std::ostringstream line;
        line << std::fixed << std::setprecision(2)
             << "[PEAK_MEMORY] sample=" << ctx.gpu_memory_measurement_count
             << " owner=" << current_owner
             << " batch=" << (batch_idx + 1)
             << " accumulation_slot=" << accumulation_slot
             << " current_bytes=" << snapshot.device_used
             << " current_MiB=" << toMiB(snapshot.device_used)
             << " delta_from_previous_bytes=" << delta
             << " delta_from_previous_MiB="
             << (static_cast<double>(delta) / (1024.0 * 1024.0))
             << " previous_owner=" << previous_owner
             << " running_peak_bytes=" << ctx.peak_gpu_used_bytes
             << " running_peak_MiB=" << toMiB(ctx.peak_gpu_used_bytes)
             << " running_peak_owner="
             << (ctx.peak_gpu_used_owner.empty() ? "none" : ctx.peak_gpu_used_owner)
             << " running_peak_batch=" << (ctx.peak_gpu_used_batch + 1)
             << " running_peak_accumulation_slot="
             << ctx.peak_gpu_used_accumulation_slot
             << " new_peak=" << (is_new_peak ? "true" : "false")
             << " device_free_MiB=" << toMiB(snapshot.device_free)
             << " device_total_MiB=" << toMiB(snapshot.device_total);

        if (snapshot.pool_available) {
            line << " async_pool_reserved_MiB=" << toMiB(snapshot.pool_reserved_current)
                 << " async_pool_reserved_high_MiB=" << toMiB(snapshot.pool_reserved_high)
                 << " async_pool_in_use_MiB=" << toMiB(snapshot.pool_used_current)
                 << " async_pool_in_use_high_MiB=" << toMiB(snapshot.pool_used_high)
                 << " async_pool_retained_MiB=" << toMiB(pool_retained)
                 << " non_pool_used_MiB=" << toMiB(non_pool_used);
        } else {
            line << " async_pool=unavailable";
        }

        ctx.logging.logger->log(line.str());
        if (allocation_query_ok) {
            logAllocationMemory(ctx, current_owner, batch_idx, accumulation_slot,
                                snapshot, device_id, allocations);
        } else {
            ctx.gpu_memory_baseline_valid = false;
            ctx.logging.logger->log("[GRAPH_MEMORY] owner=" + current_owner +
                                    " status=query_failed");
        }
    } catch (...) {
        // Memory diagnostics must never mask or replace a training failure.
    }
}

void measureForwardMemory(
    TrainingContext& ctx,
    const GRIM::Forward::ModelForwardOutputs& forward_outputs,
    int batch_idx,
    int accumulation_slot) noexcept
{
    if constexpr (!EnablePeakMemoryMeasurer) {
        return;
    }

    measurePeakMemory(
        ctx, "forward.outputs_live", batch_idx, accumulation_slot);

    try {
        if (ctx.logging.logger) {
            ctx.logging.logger->log(
                forward_outputs.describeRetainedSizes(
                    "batch=" + std::to_string(batch_idx + 1) +
                    " accumulation_slot=" + std::to_string(accumulation_slot)));
        }
    } catch (...) {
        // Memory diagnostics must never mask or replace a training failure.
    }
}

} // namespace GRIMText::Training::Memory
