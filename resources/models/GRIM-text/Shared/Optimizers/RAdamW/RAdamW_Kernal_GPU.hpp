//======================================================//
//  RAdamW_Kernal_GPU.hpp
//  Rectified AdamW (Liu et al. 2019 + Loshchilov & Hutter 2019
//  decoupled weight decay): kernel + orchestration.
//
//  Always applies DECOUPLED weight decay (the "W" in RAdamW / AdamW)
//  AND the ρ_∞ / ρ_t variance-rectification math (the "R"). There is
//  no toggle: rectification IS RAdamW — disabling it would just be
//  AdamW, which lives in its own kernel.
//
//  Per-step branching:
//    - ρ_t > 4 → rectified adaptive update (with r_t)
//    - ρ_t ≤ 4 → un-adapted first-moment (SGD-with-momentum) warmup
//
//  Hyperparameter convention:
//    - HyperParameters_GPU.hpp is the single source of truth for
//      DEFAULT constants (RADAMW_BETA1/2/EPSILON).
//    - RUNTIME values are passed into launch* by signature so the
//      kernel never reads global hyperparameter symbols.
//
//  Moment buffers (m, v) live on `ParameterGroup` and are SHARED
//  with AdamW — checkpoint format does not change.
//======================================================//

#pragma once

#include <cstddef>
#include <vector>
#include <cuda_runtime_api.h>

namespace GRIM {

struct ParameterGroup;  // Forward declaration

//------------------------------------------------------//
//  Low-level kernel (single ParameterGroup)
//------------------------------------------------------//

/// RAdamW update for one ParameterGroup.
/// All hyperparameters explicit — no globals read inside the kernel.
/// Rule 20: throws on any invalid argument or NULL buffer.
void launchRAdamWKernel(ParameterGroup& group,
                       float learning_rate,
                       float weight_decay,
                       int   step,
                       float beta1,
                       float beta2,
                       float epsilon,
                       cudaStream_t stream);

//------------------------------------------------------//
//  Orchestration (all ParameterGroups)
//------------------------------------------------------//

/// Run one RAdamW optimizer step across all parameter groups.
/// Mirrors launchAdamWStep() semantics:
///   - Applies depth-aware upsilon × per-group weight_decay_multiplier × wd
///   - Applies per-group lr_multiplier
///   - Skips embedding groups when frozen (same convention as AdamW)
/// Rule 20: throws if any group has missing optimizer state.
void launchRAdamWStep(std::vector<ParameterGroup>& groups,
                     float learning_rate,
                     float weight_decay,
                     int   step,
                     float beta1,
                     float beta2,
                     float epsilon,
                     cudaStream_t stream,
                     int   embedding_freeze_after_step = -1);

} // namespace GRIM
