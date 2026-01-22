#include "LossContext.hpp"

#include <sstream>

namespace GRIM::LossContext {
namespace {
constexpr const char* kLossModule = "Loss";
}  // namespace

Loss::LossConfig BuildLossConfig(const LossOptions& opts, bool emit_log) {
    Loss::LossConfig cfg{};

    cfg.label_smoothing.enabled = opts.label_smoothing_enabled;
    cfg.label_smoothing.epsilon = opts.label_smoothing_epsilon;

    cfg.focal.enabled = opts.focal_enabled;
    cfg.focal.gamma = opts.focal_gamma;
    cfg.focal.alpha = opts.focal_alpha;

    cfg.preference.enabled = opts.preference_enabled;
    cfg.preference.beta = opts.preference_beta;

    cfg.distillation.enabled = opts.distillation_enabled;
    cfg.distillation.temperature = opts.distillation_temperature;
    cfg.distillation.lambda = opts.distillation_lambda;

    cfg.masking.enabled = opts.masking_enabled;
    cfg.masking.tag = opts.masking_tag;
    
    // Issue #44 FIX: Entropy regularization to prevent mode collapse
    cfg.entropy_reg.enabled = opts.entropy_reg_enabled;
    cfg.entropy_reg.lambda = opts.entropy_reg_lambda;

    if (emit_log) {
        std::ostringstream oss;
        oss << "[LossConfig] ls=" << (cfg.label_smoothing.enabled ? "on" : "off")
            << " eps=" << cfg.label_smoothing.epsilon
            << " focal=" << (cfg.focal.enabled ? "on" : "off")
            << " gamma=" << cfg.focal.gamma
            << " alpha=" << cfg.focal.alpha
            << " pref=" << (cfg.preference.enabled ? "on" : "off")
            << " beta=" << cfg.preference.beta
            << " distill=" << (cfg.distillation.enabled ? "on" : "off")
            << " T=" << cfg.distillation.temperature
            << " lambda=" << cfg.distillation.lambda
            << " mask=" << (cfg.masking.enabled ? cfg.masking.tag : "off")
            << " entropy_reg=" << (cfg.entropy_reg.enabled ? "on" : "off")  // Issue #44
            << " ent_lambda=" << cfg.entropy_reg.lambda;
        GRIM::Logging::EmitModuleInfo(kLossModule, oss.str());
    }

    return cfg;
}

Loss::LossContext MakeContext(const TensorViews& views) {
    Loss::LossContext ctx{};
    ctx.logits = views.logits;
    ctx.targets = views.targets;
    ctx.teacher_logits = views.teacher_logits;
    ctx.reference_logits = views.reference_logits;
    ctx.token_mask = views.token_mask;
    ctx.sequence_weights = views.sequence_weights;
    ctx.sequence_weight_count = views.sequence_weight_count;
    ctx.position_byte_lengths = views.position_byte_lengths;  // GRMT v6
    ctx.valid_tokens = views.valid_tokens;
    ctx.batch_size = views.batch_size;
    ctx.seq_len = views.seq_len;
    ctx.vocab_size = views.vocab_size;
    ctx.stream = views.stream;
    return ctx;
}

}  // namespace GRIM::LossContext
