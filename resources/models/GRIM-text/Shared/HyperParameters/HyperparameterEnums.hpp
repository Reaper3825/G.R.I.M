#pragma once
#ifndef GRIM_SHARED_HYPERPARAMETER_ENUMS_HPP
#define GRIM_SHARED_HYPERPARAMETER_ENUMS_HPP

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace GRIM::HyperParameters {

inline std::string normalizeHyperparameterEnumToken(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

//======================================================//
// Training stage
// Determines whether a run may initialize model parameters from scratch.
//======================================================//
enum class TrainingStage : uint8_t {
    UNSPECIFIED,
    PT,    // Pre-training; may start from a fresh random initialization.
    SFT,   // Supervised fine-tuning; requires an existing model checkpoint.
    DPO,   // Direct preference optimization; requires an existing checkpoint.
    RLHF   // Reinforcement learning from human feedback; requires a checkpoint.
};

inline const char* trainingStageToString(TrainingStage stage) {
    switch (stage) {
        case TrainingStage::UNSPECIFIED: return "UNSPECIFIED";
        case TrainingStage::PT: return "PT";
        case TrainingStage::SFT: return "SFT";
        case TrainingStage::DPO: return "DPO";
        case TrainingStage::RLHF: return "RLHF";
    }
    throw std::runtime_error("trainingStageToString: unknown TrainingStage enum value");
}

inline const char* trainingStageToJsonString(TrainingStage stage) {
    switch (stage) {
        case TrainingStage::UNSPECIFIED: return "unspecified";
        case TrainingStage::PT: return "pt";
        case TrainingStage::SFT: return "sft";
        case TrainingStage::DPO: return "dpo";
        case TrainingStage::RLHF: return "rlhf";
    }
    throw std::runtime_error("trainingStageToJsonString: unknown TrainingStage enum value");
}

inline TrainingStage parseTrainingStage(const std::string& value) {
    const std::string normalized = normalizeHyperparameterEnumToken(value);
    if (normalized == "pt" || normalized == "pretraining" || normalized == "pre_training") {
        return TrainingStage::PT;
    }
    if (normalized == "sft" || normalized == "supervised_fine_tuning") {
        return TrainingStage::SFT;
    }
    if (normalized == "dpo" || normalized == "direct_preference_optimization") {
        return TrainingStage::DPO;
    }
    if (normalized == "rlhf" || normalized == "reinforcement_learning_from_human_feedback") {
        return TrainingStage::RLHF;
    }
    throw std::runtime_error(
        "ai_config.json: training.config.training_stage has unknown value '" + value +
        "' (valid: pt, sft, dpo, rlhf)");
}

//======================================================//
// Positional Encoding Configuration
// Supports ALiBi, RoPE, and Hybrid (ALiBi+RoPE)
//======================================================//
enum class PositionalEncodingType {
    UNSPECIFIED,// Fail-loud sentinel; JSON must author a real encoding type
    NONE,       // Learned/additive position embeddings (no ALiBi/RoPE inside attention)
    ALIBI,      // Attention with Linear Biases (bias-based, good for long-range)
    ROPE,       // Rotary Position Embedding (rotation-based, good for local patterns)
    ALIBI_ROPE  // Hybrid: ALiBi for long-range + RoPE for local patterns (recommended)
};

inline const char* positionalEncodingTypeToString(PositionalEncodingType type) {
    switch (type) {
        case PositionalEncodingType::UNSPECIFIED: return "UNSPECIFIED";
        case PositionalEncodingType::NONE: return "NONE";
        case PositionalEncodingType::ALIBI: return "ALIBI";
        case PositionalEncodingType::ROPE: return "ROPE";
        case PositionalEncodingType::ALIBI_ROPE: return "ALIBI_ROPE";
    }
    throw std::runtime_error("positionalEncodingTypeToString: unknown PositionalEncodingType enum value");
}

inline const char* positionalEncodingTypeToJsonString(PositionalEncodingType type) {
    switch (type) {
        case PositionalEncodingType::UNSPECIFIED: return "unspecified";
        case PositionalEncodingType::NONE: return "none";
        case PositionalEncodingType::ALIBI: return "alibi";
        case PositionalEncodingType::ROPE: return "rope";
        case PositionalEncodingType::ALIBI_ROPE: return "alibi_rope";
    }
    throw std::runtime_error("positionalEncodingTypeToJsonString: unknown PositionalEncodingType enum value");
}

inline PositionalEncodingType parsePositionalEncodingType(const std::string& value) {
    const std::string normalized = normalizeHyperparameterEnumToken(value);
    if (normalized == "none") {
        return PositionalEncodingType::NONE;
    }
    if (normalized == "alibi") {
        return PositionalEncodingType::ALIBI;
    }
    if (normalized == "rope") {
        return PositionalEncodingType::ROPE;
    }
    if (normalized == "alibi_rope") {
        return PositionalEncodingType::ALIBI_ROPE;
    }
    throw std::runtime_error(
        "ai_config.json: training.config.positional_encoding has unknown value '" + value + "'");
}

inline PositionalEncodingType parsePositionalEncodingFlags(bool use_rope, bool use_alibi) {
    if (use_rope && use_alibi) {
        return PositionalEncodingType::ALIBI_ROPE;
    }
    if (use_rope) {
        return PositionalEncodingType::ROPE;
    }
    if (use_alibi) {
        return PositionalEncodingType::ALIBI;
    }
    throw std::runtime_error(
        "ai_config.json: training.config must enable at least one of use_rope or use_alibi");
}

inline bool usesALiBi(PositionalEncodingType type) {
    return type == PositionalEncodingType::ALIBI || type == PositionalEncodingType::ALIBI_ROPE;
}

inline bool usesRoPE(PositionalEncodingType type) {
    return type == PositionalEncodingType::ROPE || type == PositionalEncodingType::ALIBI_ROPE;
}

//======================================================//
// Generation / Sampling strategy enum
//======================================================//
enum class SamplingStrategy {
    UNSPECIFIED = -1,
    GREEDY = 0,
    TOP_K = 1,
    TOP_P = 2,
    MIN_P = 3,
    TYPICAL = 4,
    TOP_K_TOP_P = 5,
    BEAM_SEARCH = 6
};

inline const char* samplingStrategyToJsonString(SamplingStrategy strategy) {
    switch (strategy) {
        case SamplingStrategy::UNSPECIFIED: return "unspecified";
        case SamplingStrategy::GREEDY: return "greedy";
        case SamplingStrategy::TOP_K: return "top_k";
        case SamplingStrategy::TOP_P: return "top_p";
        case SamplingStrategy::MIN_P: return "min_p";
        case SamplingStrategy::TYPICAL: return "typical";
        case SamplingStrategy::TOP_K_TOP_P: return "top_k_top_p";
        case SamplingStrategy::BEAM_SEARCH: return "beam_search";
    }
    throw std::runtime_error("samplingStrategyToJsonString: unknown SamplingStrategy enum value");
}

inline SamplingStrategy parseGenerationSamplingStrategy(const std::string& strategy) {
    const std::string normalized = normalizeHyperparameterEnumToken(strategy);
    if (normalized == "greedy") return SamplingStrategy::GREEDY;
    if (normalized == "top_k") return SamplingStrategy::TOP_K;
    if (normalized == "top_p") return SamplingStrategy::TOP_P;
    if (normalized == "min_p") return SamplingStrategy::MIN_P;
    if (normalized == "typical") return SamplingStrategy::TYPICAL;
    if (normalized == "top_k_top_p") return SamplingStrategy::TOP_K_TOP_P;
    if (normalized == "beam_search") return SamplingStrategy::BEAM_SEARCH;

    throw std::runtime_error(
        "ai_config.json: training.config.generation_strategy has unknown value '" + strategy + "'");
}

enum class OptimizerKind {
    UNSPECIFIED = -1,
    ADAMW = 0,
    RADAMW = 1
};

inline const char* optimizerKindToString(OptimizerKind kind) {
    switch (kind) {
        case OptimizerKind::UNSPECIFIED: return "UNSPECIFIED";
        case OptimizerKind::ADAMW: return "adamw";
        case OptimizerKind::RADAMW: return "radamw";
    }
    throw std::runtime_error("optimizerKindToString: unknown OptimizerKind enum value");
}

inline OptimizerKind parseOptimizerKind(const std::string& value) {
    const std::string normalized = normalizeHyperparameterEnumToken(value);
    if (normalized == "adamw") {
        return OptimizerKind::ADAMW;
    }
    if (normalized == "radamw") {
        return OptimizerKind::RADAMW;
    }
    throw std::runtime_error(
        "ai_config.json: training.config.optimizer_kind must be \"adamw\" or \"radamw\", got \"" + value + "\"");
}

// Parameter-group precision policy.
enum class ParameterGroupPrecision : uint8_t {
    UNSPECIFIED,
    FP32,
    BF16_COMPUTE
};

inline const char* parameterGroupPrecisionToString(ParameterGroupPrecision precision) {
    switch (precision) {
        case ParameterGroupPrecision::UNSPECIFIED:  return "UNSPECIFIED";
        case ParameterGroupPrecision::FP32:         return "FP32";
        case ParameterGroupPrecision::BF16_COMPUTE: return "BF16_COMPUTE";
    }
    throw std::runtime_error("parameterGroupPrecisionToString: unknown ParameterGroupPrecision enum value");
}

inline const char* parameterGroupPrecisionToJsonString(ParameterGroupPrecision precision) {
    switch (precision) {
        case ParameterGroupPrecision::UNSPECIFIED:  return "unspecified";
        case ParameterGroupPrecision::FP32:         return "fp32";
        case ParameterGroupPrecision::BF16_COMPUTE: return "bf16_compute";
    }
    throw std::runtime_error("parameterGroupPrecisionToJsonString: unknown ParameterGroupPrecision enum value");
}

inline ParameterGroupPrecision parseParameterGroupPrecision(const std::string& value) {
    const std::string precision = normalizeHyperparameterEnumToken(value);

    if (precision == "fp32" || precision == "float32") {
        return ParameterGroupPrecision::FP32;
    }
    if (precision == "bf16_compute" || precision == "bf16" || precision == "bfloat16_compute") {
        return ParameterGroupPrecision::BF16_COMPUTE;
    }

    throw std::runtime_error(
        "parseParameterGroupPrecision: unknown precision value '" + value +
        "'. Valid values: fp32, bf16_compute");
}

inline void validateParameterGroupPrecision(ParameterGroupPrecision precision,
                                            const char* field,
                                            const char* caller) {
    if (precision == ParameterGroupPrecision::UNSPECIFIED) {
        throw std::runtime_error(
            std::string(caller) + ": " + field +
            " is UNSPECIFIED (valid: fp32, bf16_compute)");
    }
}

enum class HardcodedPattern {
    DISABLED,
    RANDOM_CENTERED,
    ORTHOGONAL_W277,
    ALIGNED_W277,
    CONSTANT_UNIFORM,
    ZERO_MEAN_SINE
};

inline const char* hardcodedPatternToString(HardcodedPattern pattern) {
    switch (pattern) {
        case HardcodedPattern::DISABLED: return "DISABLED";
        case HardcodedPattern::RANDOM_CENTERED: return "RANDOM_CENTERED";
        case HardcodedPattern::ORTHOGONAL_W277: return "ORTHOGONAL_W277";
        case HardcodedPattern::ALIGNED_W277: return "ALIGNED_W277";
        case HardcodedPattern::CONSTANT_UNIFORM: return "CONSTANT_UNIFORM";
        case HardcodedPattern::ZERO_MEAN_SINE: return "ZERO_MEAN_SINE";
    }
    throw std::runtime_error("hardcodedPatternToString: unknown HardcodedPattern enum value");
}

inline const char* hardcodedPatternToJsonString(HardcodedPattern pattern) {
    switch (pattern) {
        case HardcodedPattern::DISABLED: return "disabled";
        case HardcodedPattern::RANDOM_CENTERED: return "random_centered";
        case HardcodedPattern::ORTHOGONAL_W277: return "orthogonal_w277";
        case HardcodedPattern::ALIGNED_W277: return "aligned_w277";
        case HardcodedPattern::CONSTANT_UNIFORM: return "constant_uniform";
        case HardcodedPattern::ZERO_MEAN_SINE: return "zero_mean_sine";
    }
    throw std::runtime_error("hardcodedPatternToJsonString: unknown HardcodedPattern enum value");
}

inline HardcodedPattern parseHardcodedHiddenPattern(const std::string& pattern) {
    const std::string normalized = normalizeHyperparameterEnumToken(pattern);
    if (normalized == "disabled") return HardcodedPattern::DISABLED;
    if (normalized == "random_centered") return HardcodedPattern::RANDOM_CENTERED;
    if (normalized == "orthogonal_w277") return HardcodedPattern::ORTHOGONAL_W277;
    if (normalized == "aligned_w277") return HardcodedPattern::ALIGNED_W277;
    if (normalized == "constant_uniform") return HardcodedPattern::CONSTANT_UNIFORM;
    if (normalized == "zero_mean_sine") return HardcodedPattern::ZERO_MEAN_SINE;

    throw std::runtime_error(
        "ai_config.json: training.config.hardcoded_hidden_states_pattern has unknown value '" +
        pattern + "'");
}

} // namespace GRIM::HyperParameters

namespace nlohmann {

template <>
struct adl_serializer<GRIM::HyperParameters::TrainingStage> {
    static void to_json(json& j, const GRIM::HyperParameters::TrainingStage& value) {
        j = GRIM::HyperParameters::trainingStageToJsonString(value);
    }

    static void from_json(const json& j, GRIM::HyperParameters::TrainingStage& value) {
        value = GRIM::HyperParameters::parseTrainingStage(j.get<std::string>());
    }
};

template <>
struct adl_serializer<GRIM::HyperParameters::PositionalEncodingType> {
    static void to_json(json& j, const GRIM::HyperParameters::PositionalEncodingType& value) {
        j = GRIM::HyperParameters::positionalEncodingTypeToJsonString(value);
    }

    static void from_json(const json& j, GRIM::HyperParameters::PositionalEncodingType& value) {
        value = GRIM::HyperParameters::parsePositionalEncodingType(j.get<std::string>());
    }
};

template <>
struct adl_serializer<GRIM::HyperParameters::SamplingStrategy> {
    static void to_json(json& j, const GRIM::HyperParameters::SamplingStrategy& value) {
        j = GRIM::HyperParameters::samplingStrategyToJsonString(value);
    }

    static void from_json(const json& j, GRIM::HyperParameters::SamplingStrategy& value) {
        value = GRIM::HyperParameters::parseGenerationSamplingStrategy(j.get<std::string>());
    }
};

template <>
struct adl_serializer<GRIM::HyperParameters::OptimizerKind> {
    static void to_json(json& j, const GRIM::HyperParameters::OptimizerKind& value) {
        j = GRIM::HyperParameters::optimizerKindToString(value);
    }

    static void from_json(const json& j, GRIM::HyperParameters::OptimizerKind& value) {
        value = GRIM::HyperParameters::parseOptimizerKind(j.get<std::string>());
    }
};

template <>
struct adl_serializer<GRIM::HyperParameters::ParameterGroupPrecision> {
    static void to_json(json& j, const GRIM::HyperParameters::ParameterGroupPrecision& value) {
        j = GRIM::HyperParameters::parameterGroupPrecisionToJsonString(value);
    }

    static void from_json(const json& j, GRIM::HyperParameters::ParameterGroupPrecision& value) {
        value = GRIM::HyperParameters::parseParameterGroupPrecision(j.get<std::string>());
    }
};

template <>
struct adl_serializer<GRIM::HyperParameters::HardcodedPattern> {
    static void to_json(json& j, const GRIM::HyperParameters::HardcodedPattern& value) {
        j = GRIM::HyperParameters::hardcodedPatternToJsonString(value);
    }

    static void from_json(const json& j, GRIM::HyperParameters::HardcodedPattern& value) {
        value = GRIM::HyperParameters::parseHardcodedHiddenPattern(j.get<std::string>());
    }
};

} // namespace nlohmann

#endif // GRIM_SHARED_HYPERPARAMETER_ENUMS_HPP
