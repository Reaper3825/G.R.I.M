#pragma once
//======================================================//
//  LossContext.hpp
//  User-friendly loss configuration options.
//
//  Rule 26: TensorViews DELETED — dead code. The production loss path
//  uses autograd::unified_loss() with raw parameters derived from
//  AutogradContext.payload (BatchPayload). No intermediate struct needed.
//
//  Rule 20: #include "../Loss.hpp" REMOVED — LossOptions does not depend
//  on any type in Loss.hpp. The old Loss.hpp structs (LossContext,
//  LossBreakdown, DeviceBuffers, etc.) are dead code from the pre-autograd
//  loss system.
//======================================================//

#include <string>

#include "../../HyperParameters/HyperParameters_GPU.hpp"

namespace GRIM::LossContext {

// User-friendly options for wiring autograd::LossConfig.
// All defaults sourced from HyperParameters for single source of truth.
// Flow: Phase1_Startup → LossOptions → model.setLossOptions() → buildLossConfig() → autograd::LossConfig
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

    // Entropy regularization: reg = -λ * H(p) = λ * Σ p*log(p)
    bool entropy_reg_enabled = false;
    float entropy_reg_lambda = 0.0f;
};

}  // namespace GRIM::LossContext
