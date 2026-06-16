#pragma once
//======================================================//
//  AblationFlags.hpp
//  Compile-time sublayer-ablation toggles for diagnosing
//  residual-stream mode collapse (RHO_BUILDUP analysis).
//
//  Flip a flag to true, rebuild, run. No CLI args, no env
//  vars (HPC-friendly: the behavior is baked into the binary
//  for a given job submission).
//
//  Effect: zeroes the named sublayer's CONTRIBUTION to the
//  residual stream while leaving the full forward compute and
//  the autograd graph intact (tensor shapes and grad wiring
//  are unchanged; the branch simply adds nothing, and its
//  parameters receive zero gradient so the sublayer is frozen).
//
//  Use to isolate which sublayer injects the shared/common-mode
//  direction that drives rho up at L1:
//    - kZeroAttnResidual = true  -> embedding + FFN-only stack
//    - kZeroFfnResidual  = true  -> embedding + attention-only stack
//  Run each in isolation and re-check the RHO_BUILDUP depth profile.
//
//  FINE-GRAINED ATTENTION PROBES (sub-attention ablations):
//  These only take effect when kZeroAttnResidual = false (i.e. the
//  attention branch still feeds the residual). They zero a specific
//  internal attention signal to isolate WHICH part of attention drives
//  the shared-direction buildup, while leaving every tensor shape and
//  the autograd graph intact (the zeroed tensor's producers receive
//  zero gradient and are frozen):
//    - kZeroAttnV       = true -> value vectors zeroed before SDPA.
//        QK scores + softmax still compute, but the weighted sum is of
//        zeros, so attn_out == 0. Isolates whether the value content
//        (vs the attention routing) carries the collapse direction.
//    - kZeroAttnOProj   = true -> output projection (W_o · attn_out)
//        zeroed AFTER the matmul. attn_out is still produced from V, but
//        its projection into the residual is suppressed and W_o/b_o are
//        frozen. Isolates the O-projection's contribution specifically.
//    - kZeroAttnQKScores = true -> Q zeroed after RoPE, so the content
//        score QKᵀ collapses to 0 and softmax is driven by the ALiBi
//        positional bias ONLY (content-independent positional pooling).
//        Isolates content-based routing from positional averaging.
//        NOTE: this zeroes the QKᵀ CONTENT term only; the ALiBi bias is
//        added inside the FlashAttention kernel and is NOT removed here.
//======================================================//

namespace GRIM { namespace Ablation {

// When true, attention sublayer contributes 0 to the residual:
//   residual1 = input  (+ optional centering)
inline constexpr bool kZeroAttnResidual = false;

// When true, FFN sublayer contributes 0 to the residual:
//   output = residual1
inline constexpr bool kZeroFfnResidual = false;

// When true, zero the attention VALUE vectors before SDPA.
// Effect: attn_out == 0 (softmax-weighted sum of zeros), QK/softmax
// still computed, V-producing params frozen.
inline constexpr bool kZeroAttnV = false;

// When true, zero the attention OUTPUT PROJECTION after W_o · attn_out.
// Effect: proj_out == 0; attn_out still computed from V; W_o/b_o frozen.
inline constexpr bool kZeroAttnOProj = true;

// When true, zero Q (after RoPE) so the QKᵀ CONTENT score is 0.
// Effect: attention weights come from the ALiBi positional bias only;
// Q-producing params frozen. ALiBi bias itself is NOT zeroed here.
inline constexpr bool kZeroAttnQKScores = false;

// Derived: does the attention branch still deliver a gradient signal to its
// own QKV projection (W_qkv)? Used by the backward gradient-connectivity
// verifier to decide whether to require W_qkv signal or fall back to the FFN
// path. Any ablation that forces attn_out (or its residual contribution) to
// zero also zeroes W_qkv's gradient:
//   - kZeroAttnResidual : whole attention contribution zeroed at the residual.
//   - kZeroAttnV        : attn_out = softmax · 0 = 0  -> no Q/K/V gradient.
//   - kZeroAttnOProj    : proj_out = 0               -> no attn_out gradient.
// kZeroAttnQKScores is intentionally NOT included: zeroing Q still leaves a
// live value path (attn_out = softmax(ALiBi) · V), so W_qkv keeps signal.
inline constexpr bool kAttnDeliversParamGradient =
    !kZeroAttnResidual && !kZeroAttnV && !kZeroAttnOProj;

} } // namespace GRIM::Ablation
