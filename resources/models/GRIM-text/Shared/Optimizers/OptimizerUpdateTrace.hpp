//======================================================//
//  OptimizerUpdateTrace.hpp
//  Optimizer-boundary adaptive update magnitude diagnostics.
//
//  This is deliberately an optimizer operation, not a per-batch/autograd
//  operation: it runs after launchOptimizerUpdate() and reads live
//  ParameterGroup optimizer moment buffers. It owns no cached state.
//======================================================//

#pragma once

#include <string>
#include <vector>
#include <cuda_runtime_api.h>

#include "../HyperParameters/HyperparameterGroupings.hpp"
#include "../TensorContract/TensorContract_GPU.hpp"

namespace GRIM {

constexpr int kOptimizerUpdateTraceSampleSize = 64;
constexpr int kOptimizerUpdateTraceTypeCount = static_cast<int>(ParamGroupType::COUNT);

struct OptimizerUpdateTraceMetrics {
    bool valid = false;

    // Per ParamGroupType enum value.
    float adaptive_update_rms[kOptimizerUpdateTraceTypeCount] = {};
    float param_rms[kOptimizerUpdateTraceTypeCount] = {};
    float update_over_param[kOptimizerUpdateTraceTypeCount] = {};
    int element_count[kOptimizerUpdateTraceTypeCount] = {};
    bool has_data[kOptimizerUpdateTraceTypeCount] = {};

    static const char* typeName(int type_idx);
};

/// Compute per-component optimizer adaptive-update RMS by sampling live
/// ParameterGroup parameter and moment buffers after launchOptimizerUpdate().
/// This does not cache pre-step tensors and intentionally excludes decoupled
/// weight decay, so it answers how the optimizer's adaptive term scales each
/// component type.
OptimizerUpdateTraceMetrics computeOptimizerUpdateTrace(
    const std::vector<ParameterGroup>& groups,
    const HyperParameters::OptimizerUpdateHP& hp,
    float learning_rate,
    int optimizer_step,
    cudaStream_t stream
);

/// Format timestamp-safe log lines. The caller should log each returned string
/// independently; do not emit a multi-line string through the training logger.
std::vector<std::string> formatOptimizerUpdateTraceLines(
    const OptimizerUpdateTraceMetrics& metrics,
    int optimizer_step,
    bool tied_embeddings
);

} // namespace GRIM
