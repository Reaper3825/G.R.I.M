#include "StabilityOverrides.hpp"
#include <nlohmann/json.hpp>

namespace GRIM::Stability {

namespace {

bool applyOverridesFromNode(const nlohmann::json& node, Overrides& out) {
    if (!node.is_object()) {
        return false;
    }
    out.batch_size = node.value("batch_size", out.batch_size);
    out.max_seq_len = node.value("max_seq_len", out.max_seq_len);
    out.max_tokens_per_batch = node.value("max_tokens_per_batch", out.max_tokens_per_batch);
    out.clip_abs = node.value("clip_abs", out.clip_abs);
    out.clip_per_token = node.value("clip_per_token", out.clip_per_token);
    out.lr_min = node.value("lr_min", out.lr_min);
    return true;
}

} // namespace

bool loadStabilityOverrides(const nlohmann::json& training_cfg, Overrides& out) {
    bool parsed = false;

    if (auto it = training_cfg.find("stability_overrides"); it != training_cfg.end()) {
        parsed |= applyOverridesFromNode(*it, out);
    }

    if (auto loss_it = training_cfg.find("loss"); loss_it != training_cfg.end() && loss_it->is_object()) {
        if (auto loss_override = loss_it->find("stability_overrides");
            loss_override != loss_it->end()) {
            parsed |= applyOverridesFromNode(*loss_override, out);
        }
    }

    return parsed && out.any();
}

} // namespace GRIM::Stability
