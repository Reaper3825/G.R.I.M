//======================================================//
//  LMHeadWeightStats.cu
//  Full-vocab on-device reduction over LM-head weight rows
//======================================================//
//
//  Replaces the per-batch host loop that issued 500 synchronous
//  cudaMemcpy(D2H) calls + a scalar CPU dot-product per sampled row.
//  Now: one kernel, one tiny D2H, one stream sync. Result is EXACT
//  over the full vocabulary (no sampling).
//
//  See LMHeadWeightStats.hpp for math + ownership contract.
//======================================================//

#include "LMHeadWeightStats.hpp"
#include "../../Shared/TensorContract/TokenTypeGate.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <cstring>
#include <cmath>

namespace GRIM::Diagnostics {

namespace {

// IEEE-754 sanity: positive floats compare monotonically as their uint32
// bit patterns. row_rms is sqrt(non-negative) so it is always finite ≥ 0
// (NaN is filtered via the std::isfinite check on the host below). This
// lets us pack (rms_bits << 32 | tok) and use atomicMax<unsigned long long>
// to get exact (max, argmax) in a single fused atomic.
constexpr int kWarpSize = 32;
constexpr int kMaxWarpsPerBlock = 32;  // up to blockDim.x = 1024

#ifdef __CUDACC__
__device__ __forceinline__ float warpReduceSum(float v) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

// One block per vocab row. Threads in the block stripe across the active
// subspace for that row, optionally center it, then accumulate the resulting
// W_eff row RMS over the full d_model denominator (inactive dimensions are 0).
// Per-block thread 0 finalizes row_rms and atomically merges it into the
// three global accumulators.
__global__ void kernelLMHeadRowStats(
    const float* __restrict__ W,           // [vocab_size, d_model]
    int                       vocab_size,
    int                       d_model,
    bool                      center_rows,
    bool                      token_type_gate,
    float                     inv_d_model, // 1.0f / d_model precomputed on host
    float*              __restrict__ d_sum_rms,
    float*              __restrict__ d_sum_rms_sq,
    unsigned long long* __restrict__ d_max_packed)
{
    const int row = blockIdx.x;
    if (row >= vocab_size) return;

    const int tid = threadIdx.x;
    const int bsz = blockDim.x;
    const float* __restrict__ row_ptr = W + static_cast<size_t>(row) * d_model;

    int active_start = 0;
    int active_end = d_model;
    int active_width = d_model;
    if (token_type_gate) {
        const auto gate_range = ::TensorContract::tokenTypeGateRangeForTokenId(
            row, d_model, vocab_size);
        active_start = gate_range.start;
        active_end = gate_range.end;
        active_width = gate_range.width;
    }
    if (active_width <= 0 || active_start < 0 || active_end > d_model || active_start >= active_end) {
        return;
    }

    // 1. Per-thread strided active-subspace reductions.
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int d = active_start + tid; d < active_end; d += bsz) {
        const float v = row_ptr[d];
        sum += v;
        sum_sq += v * v;
    }

    // 2. Warp-level reduction (Hopper + Ampere both support __shfl_down_sync).
    sum = warpReduceSum(sum);
    sum_sq = warpReduceSum(sum_sq);

    // 3. Inter-warp reduction via shared memory. One slot per warp.
    __shared__ float warp_sums_active[kMaxWarpsPerBlock];
    __shared__ float warp_sums[kMaxWarpsPerBlock];
    const int lane  = tid & (kWarpSize - 1);
    const int wid   = tid >> 5;
    if (lane == 0) {
        warp_sums_active[wid] = sum;
        warp_sums[wid] = sum_sq;
    }
    __syncthreads();

    // 4. First warp reduces the per-warp partials.
    if (wid == 0) {
        const int n_warps = (bsz + kWarpSize - 1) / kWarpSize;
        float active_sum = (lane < n_warps) ? warp_sums_active[lane] : 0.0f;
        float v = (lane < n_warps) ? warp_sums[lane] : 0.0f;
        active_sum = warpReduceSum(active_sum);
        v = warpReduceSum(v);

        __shared__ float shared_row_mean;
        if (lane == 0) {
            shared_row_mean = center_rows
                ? active_sum / static_cast<float>(active_width)
                : 0.0f;
            warp_sums[0] = v;
        }
        __syncthreads();

        if (center_rows) {
            float centered_sum_sq = 0.0f;
            for (int d = active_start + tid; d < active_end; d += bsz) {
                const float centered = row_ptr[d] - shared_row_mean;
                centered_sum_sq += centered * centered;
            }
            centered_sum_sq = warpReduceSum(centered_sum_sq);
            if (lane == 0) {
                warp_sums[wid] = centered_sum_sq;
            }
            __syncthreads();

            float centered_total = (lane < n_warps) ? warp_sums[lane] : 0.0f;
            centered_total = warpReduceSum(centered_total);
            if (lane == 0) {
                warp_sums[0] = centered_total;
            }
            __syncthreads();
            v = warp_sums[0];
        } else {
            __syncthreads();
            v = warp_sums[0];
        }

        if (lane == 0) {
            const float row_rms = sqrtf(v * inv_d_model);

            // Stream the per-row scalar into three global accumulators.
            // float atomicAdd works on sm_60+, which covers RTX 3080 (sm_86)
            // and H100 (sm_90).
            atomicAdd(d_sum_rms,    row_rms);
            atomicAdd(d_sum_rms_sq, row_rms * row_rms);

            // Pack (row_rms_bits << 32) | row into uint64 for fused max+argmax.
            // Because row_rms ≥ 0, IEEE-754 bit ordering ⇒ uint32 ordering,
            // so atomicMax over the packed value yields the row with the
            // largest row_rms. Ties broken by larger row index (harmless).
            const unsigned int rms_bits = __float_as_uint(row_rms);
            const unsigned long long packed =
                (static_cast<unsigned long long>(rms_bits) << 32) |
                static_cast<unsigned long long>(static_cast<unsigned int>(row));
            atomicMax(d_max_packed, packed);
        }
    }
}
#endif

inline void cudaCheck(cudaError_t err, const char* where) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("[LMHeadWeightStats] CUDA error at ") + where + ": " +
            cudaGetErrorString(err) + " — at " + __FILE__);
    }
}

} // anonymous namespace

LMHeadWeightStats computeLMHeadWeightStats(
    const float* weights,
    int vocab_size,
    int d_model,
    cudaStream_t stream,
    bool center_rows,
    bool token_type_gate)
{
    // ── Rule 20: validate inputs, fail loud. ──
    if (!weights) {
        throw std::runtime_error(
            "[LMHeadWeightStats] weights pointer is NULL — at " __FILE__);
    }
    if (vocab_size <= 0) {
        throw std::runtime_error(
            "[LMHeadWeightStats] vocab_size must be > 0, got " +
            std::to_string(vocab_size));
    }
    if (d_model <= 0) {
        throw std::runtime_error(
            "[LMHeadWeightStats] d_model must be > 0, got " +
            std::to_string(d_model));
    }
    if (token_type_gate && d_model < 4) {
        throw std::runtime_error(
            "[LMHeadWeightStats] token_type_gate requires d_model >= 4, got " +
            std::to_string(d_model));
    }
    if (!stream) {
        throw std::runtime_error(
            "[LMHeadWeightStats] stream is NULL — caller MUST pass a real stream "
            "(default-stream forbidden under StreamController policy)");
    }

    // ── Block sizing.
    // 256 threads is a good sweet spot for d_model values in [256, 4096]
    // on both sm_86 and sm_90: enough parallelism to saturate memory
    // without wasted lanes when d_model is small.
    constexpr int kThreadsPerBlock = 256;
    static_assert(kThreadsPerBlock <= kMaxWarpsPerBlock * kWarpSize,
                  "kThreadsPerBlock exceeds shared warp_sums slots");

    // ── Stream-ordered scratch: 2 floats + 1 uint64 = 16 bytes.
    // Single contiguous allocation keeps the alloc/free/memcpy cheap.
    struct ScratchLayout {
        float              sum_rms;
        float              sum_rms_sq;
        unsigned long long max_packed;
    };
    ScratchLayout* d_scratch = nullptr;
    cudaCheck(cudaMallocAsync(reinterpret_cast<void**>(&d_scratch),
                              sizeof(ScratchLayout), stream),
              "cudaMallocAsync(scratch)");

    // Zero-init: float 0.0 == bit pattern 0; uint64 0 packs (rms=0, tok=0).
    cudaCheck(cudaMemsetAsync(d_scratch, 0, sizeof(ScratchLayout), stream),
              "cudaMemsetAsync(scratch)");

    // ── Launch reduction kernel.
    const float inv_d_model = 1.0f / static_cast<float>(d_model);
#ifdef __CUDACC__
    kernelLMHeadRowStats<<<vocab_size, kThreadsPerBlock, 0, stream>>>(
        weights, vocab_size, d_model, center_rows, token_type_gate, inv_d_model,
        &d_scratch->sum_rms,
        &d_scratch->sum_rms_sq,
        &d_scratch->max_packed);
    cudaCheck(cudaGetLastError(), "kernelLMHeadRowStats launch");
#else
    (void)inv_d_model;
    throw std::runtime_error(
        "[LMHeadWeightStats] computeLMHeadWeightStats requires CUDA compilation (__CUDACC__ not defined)");
#endif

    // ── Pull 16 bytes back to host (sync on caller's stream).
    ScratchLayout host_scratch{};
    cudaCheck(cudaMemcpyAsync(&host_scratch, d_scratch, sizeof(ScratchLayout),
                              cudaMemcpyDeviceToHost, stream),
              "cudaMemcpyAsync(scratch D2H)");

    cudaCheck(cudaFreeAsync(d_scratch, stream), "cudaFreeAsync(scratch)");
    cudaCheck(cudaStreamSynchronize(stream),    "cudaStreamSynchronize");

    // ── Finalize on host.
    const float V = static_cast<float>(vocab_size);
    LMHeadWeightStats out;
    out.w_rms_mean     = host_scratch.sum_rms    / V;
    out.w_rms_quadmean = std::sqrt(host_scratch.sum_rms_sq / V);

    const unsigned int max_bits = static_cast<unsigned int>(host_scratch.max_packed >> 32);
    const unsigned int max_tok  = static_cast<unsigned int>(host_scratch.max_packed & 0xFFFFFFFFull);
    // Reinterpret the 32 high bits as a float. Safe: row_rms was written via
    // __float_as_uint on a non-negative finite value (sqrtf of a finite sum).
    float max_val;
    static_assert(sizeof(float) == sizeof(unsigned int), "float != uint32");
    std::memcpy(&max_val, &max_bits, sizeof(float));
    out.w_rms_max     = max_val;
    out.w_rms_max_tok = static_cast<int>(max_tok);

    // ── Rule 20: validate finiteness before returning.
    if (!std::isfinite(out.w_rms_mean) ||
        !std::isfinite(out.w_rms_quadmean) ||
        !std::isfinite(out.w_rms_max))
    {
        throw std::runtime_error(
            "[LMHeadWeightStats] non-finite result (mean=" +
            std::to_string(out.w_rms_mean) + " quadmean=" +
            std::to_string(out.w_rms_quadmean) + " max=" +
            std::to_string(out.w_rms_max) +
            ") — likely NaN/Inf in LM-head weights, fail loud at " __FILE__);
    }

    return out;
}

} // namespace GRIM::Diagnostics
