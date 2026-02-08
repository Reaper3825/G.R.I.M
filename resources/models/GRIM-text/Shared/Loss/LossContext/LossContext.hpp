#pragma once

#include <string>

#include "../Loss.hpp"
#include "../../LogRecorder/LogRecorder.hpp"
#include "../../HyperParameters/HyperParameters_GPU.hpp"

namespace GRIM::LossContext {

// User-friendly options for wiring LossConfig.
// All defaults sourced from HyperParameters for single source of truth.
struct LossOptions {
    bool label_smoothing_enabled = HyperParameters::DEFAULT_LOSS_LABEL_SMOOTHING_ENABLED;
    float label_smoothing_epsilon = HyperParameters::DEFAULT_LOSS_LABEL_SMOOTHING_EPSILON;

    bool focal_enabled = HyperParameters::DEFAULT_LOSS_FOCAL_ENABLED;
    float focal_gamma = HyperParameters::DEFAULT_LOSS_FOCAL_GAMMA;
    float focal_alpha = HyperParameters::DEFAULT_LOSS_FOCAL_ALPHA;

    bool preference_enabled = HyperParameters::DEFAULT_LOSS_PREFERENCE_ENABLED;
    float preference_beta = HyperParameters::DEFAULT_LOSS_PREFERENCE_BETA;

    bool distillation_enabled = HyperParameters::DEFAULT_LOSS_DISTILLATION_ENABLED;
    float distillation_temperature = HyperParameters::DEFAULT_LOSS_DISTILLATION_TEMPERATURE;
    float distillation_lambda = HyperParameters::DEFAULT_LOSS_DISTILLATION_LAMBDA;

    bool masking_enabled = HyperParameters::DEFAULT_LOSS_MASKING_ENABLED;
    std::string masking_tag{};
    
    // Issue #44 FIX: Entropy regularization to prevent mode collapse
    // reg = λ * Σ_v p_v² (penalizes logit concentration)
    bool entropy_reg_enabled = false;
    float entropy_reg_lambda = 0.0f;
};

struct TensorViews {
    const float* logits = nullptr;
    const int* targets = nullptr;
    const float* teacher_logits = nullptr;
    const float* reference_logits = nullptr;
    const float* token_mask = nullptr;
    const float* sequence_weights = nullptr;
    int sequence_weight_count = 0;
    int valid_tokens = 0;  // optional override
    int batch_size = 0;
    int seq_len = 0;
    int vocab_size = 0;
    cudaStream_t stream = nullptr;
};

}  // namespace GRIM::LossContext
