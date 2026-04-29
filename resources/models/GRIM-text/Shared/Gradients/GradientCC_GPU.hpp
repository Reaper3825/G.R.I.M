//======================================================//
//  GradientCC_GPU.hpp
//  Gradient Clamp + Clip — kernel launchers + registry-level API
//
//  Two layers:
//    1. Raw kernel launchers (extern "C") — operate on float* buffers
//    2. Registry API (GRIM::GradClip) — operates on ParameterGroup tensors
//       via GradNorm measurement + global RMS clipping
//======================================================//

#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <string>

// Forward declarations — avoid pulling full headers into every translation unit
namespace GRIM {
    struct ParameterGroup;
    enum class ParamGroupType : uint8_t;
    namespace GradNorm {
        struct GradNormScratch;
    }
}

//======================================================//
//  Layer 1: Raw kernel launchers
//======================================================//

// Clamp gradients element-wise into [min_val, max_val] (handles inf/NaN).
void launchClampGradients(
	float* gradients,
	int n,
	float min_val,
	float max_val,
	cudaStream_t stream);

// Scale all gradient elements by a constant factor: gradients[i] *= scale_factor.
void launchScaleGradients(
	float* gradients,
	int n,
	float scale_factor,
	cudaStream_t stream);

//======================================================//
//  Layer 2: Registry-level gradient clipping
//
//  Operates on the ParameterGroup tensor registry:
//    1. Measures per-type gradient norms via GradNormGPU
//    2. Computes one global RMS over all registered gradient tensors
//    3. Scales all gradients in-place through the tensor registry
//======================================================//

namespace GRIM::GradClip {

/// Configuration for registry-level clipping (passed by caller)
struct ClipConfig {
    float max_rms = 0.0f;       ///< Global RMS threshold — gradients above this get scaled down
};

/// Result of a clipGradientNorms() call — all values valid immediately on return
struct ClipResult {
    float global_rms_pre = 0.0f;
    float global_rms_post = 0.0f;

    bool clipped = false;
    bool any_clipped() const { return clipped; }
};

/**
 * Measure gradient norms and clip globally through the tensor registry.
 *
 * 1. Calls measureGradientNorms() on the ParameterGroup array
 * 2. Aggregates all finite per-group sum_sq values into one global RMS
 * 3. If global RMS > config.max_rms, scales all gradients in-place
 *    via launchScaleGradients on each ParameterGroup's grad tensor
 * 4. Syncs stream internally — ClipResult is valid on return
 *
 * @param groups     ParameterGroup array (from model->parameterGroups())
 * @param num_groups Number of groups in the array
 * @param scratch    Pre-allocated GradNormScratch (from allocateGradNormScratch)
 * @param config     Clip threshold
 * @param stream     CUDA stream for all GPU work
 * @return           Pre/post clip metrics
 *
 * @throws std::runtime_error on GradNorm measurement failure
 */
ClipResult clipGradientNorms(
    ParameterGroup* groups,
    size_t num_groups,
    GradNorm::GradNormScratch* scratch,
    const ClipConfig& config,
    cudaStream_t stream
);

} // namespace GRIM::GradClip
