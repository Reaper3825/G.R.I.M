#include "GradStatsCollector.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

namespace GRIM::GradStats {
namespace {

// Use explicit alignment to ensure GPU and CPU see the same struct layout.
// std::size_t differs between platforms, so use uint64_t for consistency.
struct alignas(8) GradStatsResult {
	double sum_sq;      // offset 0, 8 bytes
	float max_abs;      // offset 8, 4 bytes
	float min_val;      // offset 12, 4 bytes
	float max_val;      // offset 16, 4 bytes
	int has_nan;        // offset 20, 4 bytes
	int has_inf;        // offset 24, 4 bytes
	int _pad;           // offset 28, 4 bytes (explicit padding)
	uint64_t size;      // offset 32, 8 bytes
};  // Total: 40 bytes
static_assert(sizeof(GradStatsResult) == 40, "GradStatsResult size mismatch");

struct GradStatsEntry {
	std::string name;
	int layer = -1;
	float explosion_threshold = 0.0f;
	std::size_t size = 0;
};

std::vector<GradStatsEntry> g_entries;
std::vector<GradStatsResult> g_host_results;
GradStatsResult* g_device_results = nullptr;
std::size_t g_capacity = 0;

// BUG FIX: Helper functions for atomic float min/max operations.
// The old approach of atomicMax(reinterpret_cast<int*>(&f), __float_as_int(v))
// doesn't work correctly because IEEE-754 float bit representation doesn't map
// to integer ordering for negative floats. Use CAS loop with proper float comparison.
__device__ __forceinline__ void atomicMaxFloat(float* addr, float value) {
    int* addr_as_int = reinterpret_cast<int*>(addr);
    int old = *addr_as_int;
    int assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
                        __float_as_int(fmaxf(value, __int_as_float(assumed))));
    } while (assumed != old);
}

__device__ __forceinline__ void atomicMinFloat(float* addr, float value) {
    int* addr_as_int = reinterpret_cast<int*>(addr);
    int old = *addr_as_int;
    int assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
                        __float_as_int(fminf(value, __int_as_float(assumed))));
    } while (assumed != old);
}

__global__ void initGradStatsKernel(GradStatsResult* output, uint64_t size) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		output->sum_sq = 0.0;
		output->max_abs = 0.0f;
		output->min_val = INFINITY;
		output->max_val = -INFINITY;
		output->has_nan = 0;
		output->has_inf = 0;
		output->_pad = 0;
		output->size = size;
	}
}

__global__ void computeGradStatsKernel(
	const float* __restrict__ data,
	uint64_t size,
	GradStatsResult* __restrict__ output) {

	__shared__ double shared_sum_sq[256];
	__shared__ float shared_max_abs[256];
	__shared__ float shared_min[256];
	__shared__ float shared_max[256];
	__shared__ int shared_nan[256];
	__shared__ int shared_inf[256];

	const int tid = threadIdx.x;
	const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const uint64_t stride = blockDim.x * gridDim.x;

	double local_sum_sq = 0.0;
	float local_max_abs = 0.0f;
	float local_min = INFINITY;
	float local_max = -INFINITY;
	int local_nan = 0;
	int local_inf = 0;

	for (uint64_t i = idx; i < size; i += stride) {
		const float val = data[i];
		if (isnan(val)) {
			local_nan = 1;
			continue;
		}
		if (isinf(val)) {
			local_inf = 1;
			continue;
		}

		const float abs_val = fabsf(val);
		local_sum_sq += static_cast<double>(val) * static_cast<double>(val);
		local_max_abs = fmaxf(local_max_abs, abs_val);
		local_min = fminf(local_min, val);
		local_max = fmaxf(local_max, val);
		
		// DEBUG: Log first few values to see what max_abs is based on
		if (i < 10 && blockIdx.x == 0) {
			printf("[GradStats][DEBUG] thread=%d i=%llu val=%.6e abs_val=%.6e local_max_abs=%.6e\n",
			       tid, i, val, abs_val, local_max_abs);
		}
	}

	shared_sum_sq[tid] = local_sum_sq;
	shared_max_abs[tid] = local_max_abs;
	shared_min[tid] = local_min;
	shared_max[tid] = local_max;
	shared_nan[tid] = local_nan;
	shared_inf[tid] = local_inf;
	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			shared_sum_sq[tid] += shared_sum_sq[tid + s];
			shared_max_abs[tid] = fmaxf(shared_max_abs[tid], shared_max_abs[tid + s]);
			shared_min[tid] = fminf(shared_min[tid], shared_min[tid + s]);
			shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
			shared_nan[tid] |= shared_nan[tid + s];
			shared_inf[tid] |= shared_inf[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		atomicAdd(&output->sum_sq, shared_sum_sq[0]);
		
		// DEBUG: Log what values are being sent to atomic operations
		if (blockIdx.x < 2) {
			printf("[GradStats][ATOMIC] block=%d max_abs_before_atomic=%.6e min_val=%.6e max_val=%.6e has_nan=%d has_inf=%d\n",
			       blockIdx.x, shared_max_abs[0], shared_min[0], shared_max[0], shared_nan[0], shared_inf[0]);
		}
		
		// Use proper atomic float operations (CAS-based) instead of broken atomicMax
		atomicMaxFloat(&output->max_abs, shared_max_abs[0]);
		
		if (shared_min[0] != INFINITY) {
			atomicMinFloat(&output->min_val, shared_min[0]);
		}
		if (shared_max[0] != -INFINITY) {
			atomicMaxFloat(&output->max_val, shared_max[0]);
		}

		atomicOr(&output->has_nan, shared_nan[0]);
		atomicOr(&output->has_inf, shared_inf[0]);
	}
}

// BUG FIX: Pre-allocate large capacity to avoid reallocation during batch.
// The old code had a critical bug: when capacity grew from N to N+1:
// 1. Slots 0..N-1 were initialized into buffer at address A
// 2. ensureCapacity freed A and allocated new buffer at address B
// 3. Only slot N was initialized at B
// 4. Slots 0..N-1 at address B contained garbage!
// This caused has_inf=1041408 (garbage) and size=0 (never written).
static constexpr std::size_t INITIAL_CAPACITY = 128;  // More than enough for any batch

bool ensureCapacity(std::size_t required, cudaStream_t stream) {
	if (required <= g_capacity) {
		return true;
	}
	if (!stream) {
		GRIM::Logging::EmitModuleError(
			GRIM::Logging::ModuleId::BackwardPass,
			"[GradStats] Missing CUDA stream for allocation");
		return false;
	}

	// CRITICAL: Free old buffer SYNCHRONOUSLY before reallocating.
	// This ensures the old init/compute kernels have finished writing
	// before we invalidate their output buffer.
	if (g_device_results) {
		cudaStreamSynchronize(stream);  // Wait for pending work to finish
		cudaFree(g_device_results);     // Synchronous free (not async)
		g_device_results = nullptr;
		g_capacity = 0;
	}

	// Allocate with plenty of headroom to avoid mid-batch reallocation
	const std::size_t alloc_size = std::max(required, INITIAL_CAPACITY);
	
	// Use synchronous cudaMalloc for reliability - this only happens once per batch
	const cudaError_t err = cudaMalloc(
		&g_device_results,
		alloc_size * sizeof(GradStatsResult));
	if (err != cudaSuccess) {
		std::ostringstream oss;
		oss << "[GradStats] cudaMalloc failed: " << cudaGetErrorString(err);
		GRIM::Logging::EmitModuleError(GRIM::Logging::ModuleId::BackwardPass, oss.str());
		return false;
	}
	
	// Zero-initialize ALL slots to catch any missed init kernels.
	// This sets has_nan=0, has_inf=0, size=0 which makes uninit detection easier.
	// NOTE: min_val/max_val will be wrong (0 instead of ±INF) but the init kernel
	// will overwrite them anyway. If init kernel doesn't run, we'll see size=0.
	const cudaError_t memset_err = cudaMemset(
		g_device_results,
		0,
		alloc_size * sizeof(GradStatsResult));
	if (memset_err != cudaSuccess) {
		std::ostringstream oss;
		oss << "[GradStats] cudaMemset failed: " << cudaGetErrorString(memset_err);
		GRIM::Logging::EmitModuleError(GRIM::Logging::ModuleId::BackwardPass, oss.str());
		cudaFree(g_device_results);
		g_device_results = nullptr;
		return false;
	}

	g_capacity = alloc_size;
	g_host_results.resize(alloc_size);
	return true;
}

void emitResultLog(const GradStatsEntry& entry,
	const GradStatsResult& result,
	float rms,
	std::uint64_t step,
	GRIM::Logging::ModuleId module,
	bool explosion,
	bool is_error) {
	std::ostringstream oss;
	oss << "[GradCheck] " << entry.name
	    << ": rms=" << rms
	    << ", max_abs=" << result.max_abs;
	if (entry.layer >= 0) {
		oss << " layer=" << entry.layer;
	}
	oss << " min=" << result.min_val
	    << " max=" << result.max_val
	    << " step=" << step;

	if (result.has_nan) {
		oss << " [NAN]";
	}
	if (result.has_inf) {
		oss << " [INF]";
	}
	if (explosion) {
		oss << " [EXPLOSION]";
	}

	if (is_error) {
		GRIM::Logging::EmitModuleError(module, oss.str(), step);
		return;
	}

	GRIM::Logging::EmitModuleInfo(module, oss.str(), step);
}

} // namespace

void beginBatch() {
	g_entries.clear();
}

void enqueue(const char* name,
	int layer,
	const float* data,
	std::size_t size,
	float explosion_threshold,
	cudaStream_t stream) {
	if (!name || !data || size == 0) {
		return;
	}
	if (!stream) {
		GRIM::Logging::EmitModuleError(
			GRIM::Logging::ModuleId::BackwardPass,
			"[GradStats] Missing CUDA stream for enqueue");
		return;
	}

	const std::size_t index = g_entries.size();
	if (!ensureCapacity(index + 1, stream)) {
		return;
	}

	GradStatsEntry entry{};
	entry.name = name;
	entry.layer = layer;
	entry.explosion_threshold = explosion_threshold;
	entry.size = size;
	g_entries.emplace_back(std::move(entry));

	initGradStatsKernel<<<1, 1, 0, stream>>>(g_device_results + index, size);
	
	// BUG FIX: Check for kernel launch errors immediately.
	// If initGradStatsKernel fails, the GradStatsResult struct contains garbage,
	// which can manifest as has_inf with random large values like 1041408.
	{
		cudaError_t launch_err = cudaGetLastError();
		if (launch_err != cudaSuccess) {
			std::ostringstream oss;
			oss << "[GradStats] initGradStatsKernel launch failed for '" << name 
			    << "': " << cudaGetErrorString(launch_err);
			GRIM::Logging::EmitModuleError(GRIM::Logging::ModuleId::BackwardPass, oss.str());
		}
	}

	const int threads = 256;
	const int blocks = std::min(
		static_cast<int>((size + threads - 1) / threads),
		1024);
	computeGradStatsKernel<<<blocks, threads, 0, stream>>>(
		data, size, g_device_results + index);
	
	// BUG FIX: Check for kernel launch errors.
	{
		cudaError_t launch_err = cudaGetLastError();
		if (launch_err != cudaSuccess) {
			std::ostringstream oss;
			oss << "[GradStats] computeGradStatsKernel launch failed for '" << name 
			    << "' with " << blocks << " blocks, " << threads << " threads, size=" << size
			    << ": " << cudaGetErrorString(launch_err);
			GRIM::Logging::EmitModuleError(GRIM::Logging::ModuleId::BackwardPass, oss.str());
		}
	}
}

FlushResult flushAndLog(cudaStream_t stream,
	std::uint64_t step,
	GRIM::Logging::ModuleId module) {
	FlushResult result{};
	if (g_entries.empty()) {
		return result;
	}
	if (!stream || !g_device_results) {
		GRIM::Logging::EmitModuleError(
			module,
			"[GradStats] flushAndLog missing stream or device buffer");
		return result;
	}

	const std::size_t count = g_entries.size();
	if (g_host_results.size() < count) {
		g_host_results.resize(count);
	}

	const cudaError_t copy_err = cudaMemcpyAsync(
		g_host_results.data(),
		g_device_results,
		count * sizeof(GradStatsResult),
		cudaMemcpyDeviceToHost,
		stream);
	if (copy_err != cudaSuccess) {
		std::ostringstream oss;
		oss << "[GradStats] cudaMemcpyAsync failed: " << cudaGetErrorString(copy_err);
		GRIM::Logging::EmitModuleError(module, oss.str());
		return result;
	}

	cudaStreamSynchronize(stream);

	for (std::size_t i = 0; i < count; ++i) {
		const auto& entry = g_entries[i];
		const auto& stats = g_host_results[i];
		
		// DEBUG: Log raw struct values from GPU to see if data corruption
		if (i < 5) {
			fprintf(stderr, "[GradStats][RAW] entry='%s' layer=%d sum_sq=%.6e max_abs_raw=%.6e min_val_raw=%.6e max_val_raw=%.6e has_nan=%d has_inf=%d size=%llu\n",
			        entry.name.c_str(), entry.layer, stats.sum_sq, stats.max_abs, 
			        stats.min_val, stats.max_val, stats.has_nan, stats.has_inf, 
			        (unsigned long long)stats.size);
		}
		
		const std::size_t denom = stats.size > 0 ? stats.size : entry.size;
		const float rms = (denom > 0)
			? static_cast<float>(std::sqrt(stats.sum_sq / static_cast<double>(denom)))
			: 0.0f;
		const bool has_nan = stats.has_nan != 0;
		const bool has_inf = stats.has_inf != 0;
		const bool explosion = entry.explosion_threshold > 0.0f &&
			(rms > entry.explosion_threshold);

		result.has_explosion |= explosion;
		result.has_nan |= has_nan;
		result.has_inf |= has_inf;

		const bool is_error = explosion || has_nan || has_inf;
		
		// CRITICAL DIAGNOSTIC: Detect uninitialized gradient buffers
		// Pattern: min=inf max=-inf indicates buffer never written (all values unchanged from init)
		const bool buffer_uninitialized = 
			(stats.min_val == INFINITY) && (stats.max_val == -INFINITY);
		
		// Log which specific gradient has NaN/Inf/uninitialized buffer for debugging
		if (is_error) {
			char msg_buffer[768];
			if (buffer_uninitialized) {
				snprintf(msg_buffer, sizeof(msg_buffer),
					"[GradStats] UNINITIALIZED BUFFER DETECTED in '%s' layer=%d size=%zu rms=%f min=inf max=-inf (buffer never written!) "
					"CAUSES: (1) Buffer not registered with GradAccumulationController, (2) cudaMemset failed silently, "
					"(3) Backward kernel skipped atomicAdd due to condition check, (4) Wrong buffer pointer queued for diagnostics",
					entry.name.c_str(), entry.layer, entry.size, rms);
				GRIM::Logging::EmitModuleError(module, msg_buffer, step);
			} else {
				snprintf(msg_buffer, sizeof(msg_buffer),
					"[GradStats] NaN/Inf DETECTED in '%s' layer=%d size=%zu rms=%f has_nan=%d has_inf=%d",
					entry.name.c_str(), entry.layer, entry.size, rms, has_nan, has_inf);
				GRIM::Logging::EmitModuleWarning(module, msg_buffer, step);
			}
		}
		
		emitResultLog(entry, stats, rms, step, module, explosion, is_error);
	}

	result.count = count;
	g_entries.clear();
	return result;
}

} // namespace GRIM::GradStats
