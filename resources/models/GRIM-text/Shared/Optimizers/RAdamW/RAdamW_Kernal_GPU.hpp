//======================================================//
//  RAdam_Kernal_GPU.hpp
//  Rectified Adam (Liu et al. 2019) optimizer:
//      kernel + orchestration.
//
//  Hyperparameter convention (per training-loop request):
//    - HyperParameters_GPU.hpp is the single source of truth
//      for DEFAULT constants (RADAM_BETA1/2/EPSILON,
//      RADAM_USE_RECTIFICATION_DEFAULT).
//    - The RUNTIME values are passed into launch* by signature
//      so the kernel never reads global hyperparameter symbols.
//
//  Variance-rectification toggle (`use_rectification`):
//    - false (DEFAULT): SKIP the ρ_∞ / ρ_t variance-rectification
//      math entirely. Kernel becomes plain bias-corrected Adam with
//      decoupled weight decay (standard Adam math, NOT RAdam).
//    - true:  Run full RAdam (Liu et al. 2019) — when ρ_t > 4 use the
//      rectified adaptive update (with `r_t`); otherwise use un-adapted
//      first-moment (SGD-with-momentum) warmup step.
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

/// RAdam update for one ParameterGroup.
/// All hyperparameters explicit — no globals read inside the kernel.
/// Rule 20: throws on any invalid argument or NULL buffer.
void launchRAdamKernel(ParameterGroup& group,
                       float learning_rate,
                       float weight_decay,
                       int   step,
                       float beta1,
                       float beta2,
                       float epsilon,
                       bool  use_rectification,
                       cudaStream_t stream);

//------------------------------------------------------//
//  Orchestration (all ParameterGroups)
//------------------------------------------------------//

/// Run one RAdam optimizer step across all parameter groups.
/// Mirrors launchAdamWStep() semantics:
///   - Applies depth-aware upsilon × per-group weight_decay_multiplier × wd
///   - Applies per-group lr_multiplier
///   - Skips embedding groups when frozen (same convention as AdamW)
/// Rule 20: throws if any group has missing optimizer state.
void launchRAdamStep(std::vector<ParameterGroup>& groups,
                     float learning_rate,
                     float weight_decay,
                     int   step,
                     float beta1,
                     float beta2,
                     float epsilon,
                     bool  use_rectification,
                     cudaStream_t stream,
                     int   embedding_freeze_after_step = -1);

} // namespace GRIM
