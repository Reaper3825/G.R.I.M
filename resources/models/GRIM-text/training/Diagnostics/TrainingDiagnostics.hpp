//======================================================//
//  TrainingDiagnostics.hpp
//  Diagnostic structs and functions extracted from Phase2_TrainingLoop.cu
//
//  Contains: WeightSample
//
//  Author: Extracted Feb 2026
//======================================================//

#pragma once

#include <string>
#include <cuda_runtime.h>

#include "../../GRIM/grim_language_model_cuda.hpp"

namespace GRIM::Diagnostics {

//======================================================//
//  WeightSample — small weight snapshot for optimizer diagnostics
//======================================================//

constexpr int kWeightSampleSize = 10;

struct WeightSample {
    bool valid = false;
    float values[kWeightSampleSize] = {0.0f};
    float rms = 0.0f;
};

WeightSample sampleWeightStats(const GRIM::LMHeadLayer* lm_head, const GRIM::TrainingState& ts, bool sync_for_host = false);
std::string formatWeightSample(const WeightSample& sample);
float computeUpdateRms(const WeightSample& before, const WeightSample& after);

} // namespace GRIM::Diagnostics
