//======================================================//
//  GradientCC_GPU.hpp
//  Gradient Clamp + Clip — kernel launchers + registry-level API
//
//  Two layers:
//    1. Raw kernel launchers (extern "C") — operate on float* buffers
//    2. Registry API (GRIM::GradClip) — operates on ParameterGroup tensors
//       via GradNorm measurement + per-component RMS clipping
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
//    2. Computes per-component RMS (emb bucket vs enc bucket)
//    3. Scales gradients in-place through the tensor registry
//
//  Two independent clip buckets (Issue #139):
//    emb: LM_HEAD (+ EMBEDDING if untied)
//    enc: ATTENTION + FFN + RMSNORM + SCRATCHBLOCK +
//         NUMERIC_HEAD + MTP + REASONING_HEAD + EXECUTION_BLOCK
//======================================================//

namespace GRIM::GradClip {

/// Configuration for registry-level clipping (passed by caller)
struct ClipConfig {
    float max_rms = 0.0f;       ///< RMS threshold — components exceeding this get scaled down
    bool tie_embeddings = false; ///< When true, EMBEDDING bucket is empty (shares LM_HEAD grad)
};

/// Result of a clipGradientNorms() call — all values valid immediately on return
struct ClipResult {
    // Per-component RMS BEFORE clipping
    float emb_rms = 0.0f;
    float enc_rms = 0.0f;

    // Per-component RMS AFTER clipping (clamped to max_rms)
    float emb_rms_post = 0.0f;
    float enc_rms_post = 0.0f;

    // Combined RMS: sqrt(emb² + enc²)
    float total_rms_pre = 0.0f;
    float total_rms_post = 0.0f;

    bool emb_clipped = false;
    bool enc_clipped = false;
    bool any_clipped() const { return emb_clipped || enc_clipped; }
};

/**
 * Measure gradient norms and clip per-component through the tensor registry.
 *
 * 1. Calls measureGradientNorms() on the ParameterGroup array
 * 2. Aggregates per-type sum_sq into emb/enc buckets
 * 3. If bucket RMS > config.max_rms, scales that bucket's gradients in-place
 *    via launchScaleGradients on each ParameterGroup's grad tensor
 * 4. Syncs stream internally — ClipResult is valid on return
 *
 * @param groups     ParameterGroup array (from model->parameterGroups())
 * @param num_groups Number of groups in the array
 * @param scratch    Pre-allocated GradNormScratch (from allocateGradNormScratch)
 * @param config     Clip threshold + tie_embeddings flag
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
