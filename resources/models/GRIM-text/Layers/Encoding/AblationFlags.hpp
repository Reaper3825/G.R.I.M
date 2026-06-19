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
//    - kZeroAlibiBias  = true -> disable the ALiBi positional bias inside
//        the vendored Dao FlashAttention kernel by passing a NULL slope
//        pointer (the kernel's HasAlibi switch then selects the no-bias
//        path for BOTH forward and backward). Attention then routes on the
//        QKᵀ content score + RoPE only, with no distance penalty. Complements
//        kZeroAttnQKScores: that keeps ALiBi and drops content; this keeps
//        content and drops ALiBi. Q/K/V projections still train normally.
//    - kZeroRope       = true -> skip the RoPE rotation of Q/K entirely, so
//        Q/K reach attention unrotated (no rotary positional encoding). The
//        autograd graph stays connected (Q/K flow from split_and_reshape_qkv
//        straight into SDPA), so W_qkv still receives full gradient. Combine
//        with kZeroAlibiBias to remove ALL positional information and measure
//        whether positional encoding drives the shared-direction buildup.
//    - kDisableCausalMask = true -> drop the causal mask so every query attends
//        to ALL keys (full bidirectional self-attention) instead of only past
//        positions. Isolates whether the autoregressive mask (e.g. the BOS
//        attention-sink / early-token over-attention) drives the shared
//        direction. The autograd graph and all attention gradients are intact;
//        only the kernel's masking mode changes.
//        IMPORTANT: this flag CANNOT be toggled from the header alone. The
//        causal-mask behavior is baked into which FlashAttention kernel
//        templates compile, so it is controlled by the CMake option
//        GRIM_ABLATE_DISABLE_CAUSAL_MASK. That option (ON) drops CAUSAL_ONLY
//        (compiling the non-causal kernels) AND defines the macro below so this
//        flag flips to true in lockstep with the build.
//        CURRENT DEFAULT: the option defaults to ON, so the causal mask is
//        DISABLED by default for the collapse investigation. Restore the causal
//        (autoregressive) mask by configuring -DGRIM_ABLATE_DISABLE_CAUSAL_MASK=OFF.
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
inline constexpr bool kZeroAttnV = true;

// When true, zero the attention OUTPUT PROJECTION after W_o · attn_out.
// Effect: proj_out == 0; attn_out still computed from V; W_o/b_o frozen.
inline constexpr bool kZeroAttnOProj = false;

// When true, zero Q (after RoPE) so the QKᵀ CONTENT score is 0.
// Effect: attention weights come from the ALiBi positional bias only;
// Q-producing params frozen. ALiBi bias itself is NOT zeroed here.
inline constexpr bool kZeroAttnQKScores = false;

// When true, disable the ALiBi positional bias in the FlashAttention kernel
// by passing a NULL slope pointer (kernel runs its no-alibi path for both
// forward and backward). Attention routes on QKᵀ content + RoPE only; no
// distance penalty. Q/K/V projections still receive normal gradients.
inline constexpr bool kZeroAlibiBias = false;

// When true, skip the RoPE rotation so Q/K reach attention unrotated (no
// rotary positional encoding). Graph stays connected (Q/K flow from
// split_and_reshape_qkv into SDPA), so W_qkv keeps full gradient. Combine
// with kZeroAlibiBias to strip ALL positional information.
inline constexpr bool kZeroRope = false;

// When true, disable the causal mask so attention is fully bidirectional (every
// query sees every key) instead of autoregressive. Driven by the CMake option
// GRIM_ABLATE_DISABLE_CAUSAL_MASK so the constexpr cannot drift from the kernel
// build: that option both compiles the non-causal FlashAttention kernels (drops
// GRIM_FLASHATTN_CAUSAL_ONLY) and defines the macro below. Flipping this header
// alone has NO effect (and would throw at runtime in a causal-only build).
// The option DEFAULTS TO ON, so this is true (causal mask disabled) by default.
#if defined(GRIM_ABLATE_DISABLE_CAUSAL_MASK)
inline constexpr bool kDisableCausalMask = true;
#else
inline constexpr bool kDisableCausalMask = false;
#endif

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
