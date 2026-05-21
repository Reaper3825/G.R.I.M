//======================================================//
//  MTPDiagnostics.hpp
//  Host-side Multi-Token Prediction diagnostic payload
//======================================================//

#pragma once

#include <vector>

namespace GRIM::MTP {

struct MTPDiagnostics {
    std::vector<float> head_loss;  // Raw per-head CE/NLL loss, not alpha/K weighted
    std::vector<float> head_acc;
    float L0_main = 0.0f;       // Main (next-token) loss before adding MTP terms
    float alpha_effective = 0.0f;
    float L_total = 0.0f;
    bool valid = false;

    void clear() {
        head_loss.clear();
        head_acc.clear();
        L0_main = 0.0f;
        alpha_effective = 0.0f;
        L_total = 0.0f;
        valid = false;
    }
};

} // namespace GRIM::MTP