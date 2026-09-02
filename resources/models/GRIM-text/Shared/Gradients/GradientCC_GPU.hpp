//======================================================//
//  GradientCC_GPU.hpp
//  GRIM::GradClip — accumulation normalization + global RMS clipping
//  over the registered ParameterGroup tensor array.
//======================================================//

#pragma once

#include <cuda_runtime.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "../HyperParameters/HyperparameterGroupings.hpp"

namespace GRIM {
    struct ParameterGroup;
    enum class ParamGroupType : uint8_t;
}

namespace GRIM::GradClip {

struct alignas(64) ClipMetrics {
    double embedding_sum_sq = 0.0;
    double lm_head_sum_sq = 0.0;
    double attention_sum_sq = 0.0;
    double ffn_sum_sq = 0.0;
    double rmsnorm_sum_sq = 0.0;
    double arg_selector_sum_sq = 0.0;

    uint64_t embedding_count = 0;
    uint64_t lm_head_count = 0;
    uint64_t attention_count = 0;
    uint64_t ffn_count = 0;
    uint64_t rmsnorm_count = 0;
    uint64_t arg_selector_count = 0;

    uint32_t has_nan = 0;
    uint32_t has_inf = 0;
    uint32_t groups_processed = 0;
    uint32_t _pad = 0;

    int32_t first_nan_group = -1;
    int32_t first_inf_group = -1;
    float first_nan_value = 0.0f;
    float first_inf_value = 0.0f;
};

struct ClipScratch {
    float* d_partial_sums = nullptr;
    float* h_partial_sums = nullptr;
    ClipMetrics* h_metrics = nullptr;
    size_t max_groups = 0;

    ClipScratch() = default;
    ~ClipScratch();
    ClipScratch(const ClipScratch&) = delete;
    ClipScratch& operator=(const ClipScratch&) = delete;
    ClipScratch(ClipScratch&&) = delete;
    ClipScratch& operator=(ClipScratch&&) = delete;
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
    ClipMetrics metrics{};
    std::array<TopGroup, kTopGroupCount> top_groups{};

    bool clipped = false;
    bool any_clipped() const { return clipped; }
};

/**
 * Normalize and optionally clip gradients globally through the tensor registry.
 *
 * 1. Applies schedule_hp.accumulation_normalization_scale to all groups
 * 2. Measures per-group gradient norms
 * 3. Aggregates all finite per-group sum_sq values into one global RMS
 * 4. If clipping_hp.enabled and global RMS > clipping_hp.effective_per_token_limit,
 *    scales all gradients in-place
 * 5. Syncs stream internally — ClipResult is valid on return
 *
 * @param groups      ParameterGroup array (from StartupParameterRegistry::parameterGroups())
 * @param num_groups  Number of groups in the array
 * @param scratch     OptimizerState-owned ClipScratch pointer; allocated here if null
 * @param clipping_hp Gradient clipping HP (enabled flag + effective_per_token_limit)
 * @param schedule_hp Training schedule HP (accumulation_normalization_scale)
 * @param stream      CUDA stream for all GPU work
 * @return            Pre/post clip metrics plus the clipping input group count
 *
 * @throws std::runtime_error on invalid HP values or GradNorm measurement failure
 */
ClipResult clipGradientNorms(
    ParameterGroup* groups,
    size_t num_groups,
    std::unique_ptr<ClipScratch>& scratch,
    const HyperParameters::GradientClippingHP& clipping_hp,
    const HyperParameters::TrainingScheduleHP& schedule_hp,
    cudaStream_t stream
);

} // namespace GRIM::GradClip
