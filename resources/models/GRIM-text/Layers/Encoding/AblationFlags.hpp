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
//======================================================//

namespace GRIM { namespace Ablation {

// When true, attention sublayer contributes 0 to the residual:
//   residual1 = input  (+ optional centering)
inline constexpr bool kZeroAttnResidual = false;

// When true, FFN sublayer contributes 0 to the residual:
//   output = residual1
inline constexpr bool kZeroFfnResidual = true;

} } // namespace GRIM::Ablation
