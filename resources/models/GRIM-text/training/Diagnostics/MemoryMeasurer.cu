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

        GpuMemorySnapshot snapshot;
        if (!queryGpuMemory(snapshot)) {
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
