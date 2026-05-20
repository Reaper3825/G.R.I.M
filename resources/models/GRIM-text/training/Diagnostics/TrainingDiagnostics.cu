//======================================================//
//  TrainingDiagnostics.cu
//  Implementation of diagnostic functions extracted from Phase2_TrainingLoop.cu
//
//  Contains: WeightSample
//
//  Author: Extracted Feb 2026
//======================================================//

#include "TrainingDiagnostics.hpp"

#include "../../Layers/LMHead/lm_head_GPU.hpp"
#include "../../Shared/TrainingState/TrainingState_GPU.hpp"

#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>

namespace GRIM::Diagnostics {

//======================================================//
//  WeightSample
//======================================================//

WeightSample sampleWeightStats(const GRIM::LMHeadLayer* lm_head, const GRIM::TrainingState& ts, bool sync_for_host) {
    WeightSample sample{};
    if (!lm_head || !lm_head->weights().data) {
        return sample;
    }

    if (!sync_for_host) {
        return sample;
    }

    cudaStream_t stream = ts.stream_ctrl.getPrimaryStream();
    cudaMemcpyAsync(sample.values, lm_head->weights().data,
                    kWeightSampleSize * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    // Sync primary stream only (not full device) so we can read the values
    cudaStreamSynchronize(stream);
    
    float sum_sq = 0.0f;
    for (int i = 0; i < kWeightSampleSize; ++i) {
        sum_sq += sample.values[i] * sample.values[i];
    }
    sample.rms = std::sqrt(sum_sq / kWeightSampleSize);
    sample.valid = true;
    return sample;
}

std::string formatWeightSample(const WeightSample& sample) {
    if (!sample.valid) {
        return "lm_head_weights=nullptr";
    }
    
    std::ostringstream oss;
    oss << "lm_w[0:10]=[";
    for (int i = 0; i < kWeightSampleSize; ++i) {
        oss << std::fixed << std::setprecision(6) << sample.values[i];
        if (i + 1 < kWeightSampleSize) oss << ",";
    }
    oss << "] rms=" << std::scientific << std::setprecision(4) << sample.rms;
    return oss.str();
}

float computeUpdateRms(const WeightSample& before, const WeightSample& after) {
    if (!before.valid || !after.valid) {
        return 0.0f;
    }
    
    float sum_sq = 0.0f;
    for (int i = 0; i < kWeightSampleSize; ++i) {
        const float delta = after.values[i] - before.values[i];
        sum_sq += delta * delta;
    }
    return std::sqrt(sum_sq / kWeightSampleSize);
}

} // namespace GRIM::Diagnostics
