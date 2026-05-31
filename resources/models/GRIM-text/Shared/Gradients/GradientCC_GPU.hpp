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
#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <memory>
#include <string>

#include "../GradNorm/GradNormGPU.hpp"

// Forward declarations — avoid pulling full headers into every translation unit
namespace GRIM {
    struct ParameterGroup;
    enum class ParamGroupType : uint8_t;
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
    struct TopGroup {
        size_t index = 0;
        float rms = 0.0f;
        float sum_sq = 0.0f;
        uint64_t count = 0;
        int type = -1;
        int layer_index = -1;
        bool valid = false;
    };

    static constexpr size_t kTopGroupCount = 5;

    size_t measured_group_count = 0;  ///< Number of groups measured by clipGradientNorms()
    float global_rms_pre = 0.0f;
    float global_rms_post = 0.0f;
    float encoder_rms_pre = 0.0f;
    float scratchblock_rms_pre = 0.0f;
    GradNorm::GradMetrics metrics{};
    std::array<TopGroup, kTopGroupCount> top_groups{};

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
 * @param groups     ParameterGroup array (from StartupParameterRegistry::parameterGroups())
 * @param num_groups Number of groups in the array
 * @param scratch    TrainingState-owned GradNormScratch pointer; allocated here if null
 * @param config     Clip threshold
 * @param stream     CUDA stream for all GPU work
 * @return           Pre/post clip metrics plus the clipping input group count
 *
 * @throws std::runtime_error on GradNorm measurement failure
 */
ClipResult clipGradientNorms(
    ParameterGroup* groups,
    size_t num_groups,
    std::unique_ptr<GradNorm::GradNormScratch>& scratch,
    const ClipConfig& config,
    cudaStream_t stream
);

} // namespace GRIM::GradClip
