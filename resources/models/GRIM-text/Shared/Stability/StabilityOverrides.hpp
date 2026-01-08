#pragma once

#include <cstdint>
#include <nlohmann/json_fwd.hpp>

namespace GRIM::Stability {

struct Overrides {
    int batch_size = 0;
    int max_seq_len = 0;
    int max_tokens_per_batch = 0;
    float clip_abs = 0.0f;
    float clip_per_token = 0.0f;
    float lr_min = 0.0f;

    bool any() const {
        return batch_size > 0 || max_seq_len > 0 || max_tokens_per_batch > 0 ||
               clip_abs > 0.0f || clip_per_token > 0.0f || lr_min > 0.0f;
    }
};

// Load overrides from either training.config.stability_overrides or
// training.config.loss.stability_overrides. Returns true if any value parsed.
bool loadStabilityOverrides(const nlohmann::json& training_cfg, Overrides& out);

} // namespace GRIM::Stability
