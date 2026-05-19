//======================================================//
//  AdamW_Kernal_GPU.hpp
//  Production AdamW optimizer: kernel + orchestration
//  Hardcoded hyperparameters: β₁=0.9, β₂=0.999, ε=1e-8
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

/// AdamW update operating directly on a ParameterGroup.
/// Reads weights/grads/m/v from the group's Tensors — no cached raw pointers.
void launchAdamWKernel(ParameterGroup& group,
					   float learning_rate,
					   float weight_decay,
					   int step,
					   cudaStream_t stream);

//------------------------------------------------------//
//  Orchestration (all ParameterGroups)
//------------------------------------------------------//
//
//  These free functions contain the AdamW optimizer logic that
//  was previously on LanguageModel. They operate directly on
//  parameter groups — no model dependency.
//
//  Callers are responsible for:
//    1. Building parameter groups through Phase1 ParameterGroupRegistration
//    2. Providing a valid stream
//    3. Incrementing step AFTER calling launchAdamWStep()

/// Run one AdamW optimizer step across all parameter groups.
/// Applies depth-aware upsilon regularization to weight decay.
/// When embedding_freeze_after_step >= 0 and step >= that threshold,
/// groups with ParamStatsBucket::EMBEDDING are skipped.
/// Rule 20: Throws if any group has missing optimizer state.
void launchAdamWStep(std::vector<ParameterGroup>& groups,
                     float learning_rate,
                     float weight_decay,
                     int step,
                     cudaStream_t stream,
                     int embedding_freeze_after_step = -1);

/// Zero all first-moment (m) and second-moment (v) buffers.
/// Used by soft restart to reset optimizer state.
void resetAdamWMoments(std::vector<ParameterGroup>& groups,
                       cudaStream_t stream);

/// Scale all first-moment (m) and second-moment (v) buffers by a factor.
/// Used by soft restart to dampen momentum without full reset.
/// Rule 20: Throws if scale <= 0.
void scaleAdamWMoments(std::vector<ParameterGroup>& groups,
                       float scale,
                       cudaStream_t stream);

} // namespace GRIM

