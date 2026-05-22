#pragma once
//======================================================//
//  LMHeadWeightStats.hpp
//  Full-vocab on-device reduction over LM-head effective weight rows
//======================================================//
//
//  PURPOSE
//  =======
//  Replaces the host-side 500-row sampled CPU loop in the
//  [LOGIT_SCALE_EQUATION] diagnostic with a single CUDA kernel
//  that computes EXACT statistics over every vocab row of the
//  effective LM-head weight matrix W_eff executed by the forward path.
//
//  Per row v ∈ [0, vocab_size):
//      row_rms[v] = sqrt( (1/d_model) · Σ_d W_eff[v,d]² )
//
//  W_eff semantics are controlled by the two boolean options below:
//    - token_type_gate=false, center_rows=false: W_eff = W_lm
//    - token_type_gate=true,  center_rows=false: W_eff = type_gate_rows_by_token_type(W_lm)
//    - token_type_gate=false, center_rows=true:  W_eff = center_rows(W_lm)
//    - token_type_gate=true,  center_rows=true:  W_eff = center_rows_by_token_type_gate(W_lm)
//
//  Returned aggregates (over the full vocab, not a sample):
//      w_rms_mean     = (1/V) · Σ_v row_rms[v]
//      w_rms_quadmean = sqrt( (1/V) · Σ_v row_rms[v]² )
//                     = the W_rms term in
//                       logit_std ≈ sqrt(d_model) · h_rms · w_rms_quadmean
//      w_rms_max      = max_v row_rms[v]      (exact — no sampling miss)
//      w_rms_max_tok  = argmax_v row_rms[v]
//
//  COMPUTE MODEL
//  =============
//  - One CUDA block per vocab row.
//  - Block does an in-row sum-of-squares: warp shuffle-down + shared-mem
//    inter-warp reduction. No host-side per-row loop.
//  - Per-block thread 0 atomically accumulates row_rms / row_rms² and
//    runs an atomicMax over (row_rms_bits<<32 | tok) to obtain the
//    exact max and its argmax in one fused step.
//
//  COMPUTE CAPABILITY
//  ==================
//  Requires sm_60+ (atomicAdd<float>, atomicMax<unsigned long long>).
//  Verified for sm_86 (RTX 3080) and sm_90 (H100). No sm_90-only intrinsics.
//
//  OWNERSHIP
//  =========
//  Diagnostic-only. Allocates a 16-byte device scratch via cudaMallocAsync
//  on the caller's stream and frees it via cudaFreeAsync before returning.
//  Synchronizes the stream once at the end to read the 4 result scalars.
//
//  Author: Austin Wadkins
//  Date: April 2026
//======================================================//

#include <cuda_runtime.h>
#include <cstdint>

namespace GRIM::Diagnostics {

struct LMHeadWeightStats {
    float w_rms_mean      = 0.0f;  // arithmetic mean of per-row RMS
    float w_rms_quadmean  = 0.0f;  // sqrt(E[row_rms²]) — used in logit-scale equation
    float w_rms_max       = 0.0f;  // exact max over all vocab rows
    int   w_rms_max_tok   = -1;    // argmax row index
};

// Computes exact per-row RMS aggregates over the full LM-head effective weight
// matrix executed by the forward path.
//
// Args:
//   weights      Device pointer to W, shape [vocab_size, d_model], row-major.
//                MUST be valid for the entire kernel duration.
//   vocab_size   Number of rows in W. Must be > 0.
//   d_model      Number of columns in W. Must be > 0 and ≤ 8192 (block sizing).
//   stream       CUDA stream the kernel + alloc + memcpy run on. NULL forbidden.
//   center_rows  Whether to subtract the row mean inside the active subspace.
//   token_type_gate Whether to zero out inactive token-type subspace dims.
//
// Returns:
//   Filled LMHeadWeightStats. Throws std::runtime_error on any CUDA error.
//
// Cost:
//   1 kernel launch + 1 stream-ordered alloc + 1 stream-ordered free + 1
//   D2H of 16 bytes + 1 stream sync. O(vocab_size · d_model) FLOPs on-device.
LMHeadWeightStats computeLMHeadWeightStats(
    const float* weights,
    int vocab_size,
    int d_model,
    cudaStream_t stream,
    bool center_rows = false,
    bool token_type_gate = false);

} // namespace GRIM::Diagnostics
