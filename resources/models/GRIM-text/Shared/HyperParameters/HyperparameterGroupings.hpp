#pragma once

#include "HyperParameters_GPU.hpp"
#include "../Dynamic_LR/LRSchedule.hpp"

namespace GRIM::HyperParameters {

struct CoreRunHP {
    int epochs = 0;
    int64_t seed = 0;
    int gradient_accumulation_steps = 0;
    bool single_batch_overfit_enabled = false;
    int single_batch_overfit_max_steps = 0;
};

struct CapacityHP {
    int batch_size = 0;
    int max_seq_len = 0;
    int gradient_accumulation_steps = 0;
};

struct DataLoadingHP {
    int min_seq_valid_tokens = 0;
    int sliding_window_stride = 0;
};

struct LearningRateScheduleInputs {
    float learning_rate = 0.0f;
    float cosine_decay_min_lr = 0.0f;
    int warmup_steps = 0;
    bool cosine_decay_enabled = false;
    bool cosine_warm_restarts = false;
};

struct GradientClippingHP {
    bool enabled = false;
    float configured_clip_norm = 0.0f;
    float effective_per_token_limit = EPSILON_GRADIENT_CLIP;
};

inline CoreRunHP coreRunHP(const StartupConfig& config) {
    const auto& hp = config.hyperparameters;
    CoreRunHP view;
    view.epochs = hp.epochs;
    view.seed = hp.seed;
    view.gradient_accumulation_steps = hp.gradient_accumulation_steps;
    view.single_batch_overfit_enabled = hp.single_batch_overfit_enabled;
    view.single_batch_overfit_max_steps = hp.single_batch_overfit_max_steps;
    return view;
}

inline CapacityHP capacityHP(const StartupConfig& config) {
    CapacityHP view;
    view.batch_size = config.hyperparameters.batch_size;
    view.max_seq_len = config.max_seq_len;
    view.gradient_accumulation_steps = config.hyperparameters.gradient_accumulation_steps;
    return view;
}

inline DataLoadingHP dataLoadingHP(const StartupConfig& config) {
    DataLoadingHP view;
    view.min_seq_valid_tokens = config.hyperparameters.min_seq_valid_tokens;
    view.sliding_window_stride = config.sliding_window_stride;
    return view;
}

inline LearningRateScheduleInputs learningRateScheduleInputs(
    const ::GRIM::Config::TrainingHyperparameters& hp)
{
    LearningRateScheduleInputs inputs;
    inputs.learning_rate = hp.learning_rate;
    inputs.cosine_decay_min_lr = hp.cosine_decay_min_lr;
    inputs.warmup_steps = hp.warmup_steps;
    inputs.cosine_decay_enabled = hp.cosine_decay_enabled;
    inputs.cosine_warm_restarts = hp.cosine_warm_restarts;
    return inputs;
}

inline GradientClippingHP gradientClippingHP(
    const ::GRIM::Config::TrainingHyperparameters& hp)
{
    GradientClippingHP view;
    view.configured_clip_norm = hp.grad_clip_norm;
    view.enabled = hp.grad_clip_norm > 0.0f;
    view.effective_per_token_limit = std::max(
        hp.grad_clip_norm, EPSILON_GRADIENT_CLIP);
    return view;
}

inline ::GRIM::LR::LRScheduleConfig makeLRScheduleConfig(
    const LearningRateScheduleInputs& inputs,
    int total_steps,
    int steps_per_epoch)
{
    ::GRIM::LR::LRScheduleConfig cfg;
    cfg.base_lr = inputs.learning_rate;
    cfg.cosine_decay_min_lr = inputs.cosine_decay_min_lr;
    cfg.warmup_steps = inputs.warmup_steps;
    cfg.total_steps = total_steps;
    cfg.steps_per_epoch = steps_per_epoch;
    cfg.cosine_decay_enabled = inputs.cosine_decay_enabled;
    cfg.warm_restarts = inputs.cosine_warm_restarts;
    return cfg;
}

} // namespace GRIM::HyperParameters

