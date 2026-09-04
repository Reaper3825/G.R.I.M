#include "ParameterGroupRegistration.hpp"
#include "ParameterRegistry.hpp"

#include "ModelGpuState.hpp"

#include "../../../../GRIM/grim_language_model_cuda.hpp"
#include "../../../../Layers/Encoding/Encoding_GPU.hpp"
#include "../../../../Shared/Batching/BatchPayload.hpp"
#include "../../../../Shared/HyperParameters/HyperParameters_GPU.hpp"
#include "../../../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../../../../Shared/LogRecorder/LogRecorder.hpp"
#include "../../../../Shared/Optimizers/OptimizerState_GPU.hpp"
#include "../../../../Shared/UnigramByte/TokenLayout.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace GRIMText::Training::Startup::ModelRegistration {

#ifdef USE_CUDA

namespace {

constexpr auto kRegistrationModule = GRIM::Logging::ModuleId::Training;

using GRIM::ParameterGroup;
using GRIM::ParamStatsBucket;
using GRIM::ParamGroupType;
using GRIM::Tensor;
using GRIM::HyperParameters::ParameterGroupPrecision;
using GRIM::HyperParameters::OptimizerUpdateHP;

struct LoRAMatrixSpec {
    GRIM::LoRAMatrixClass matrix_class;
    const GRIM::HyperParameters::LoRAClassSettingsHP* settings;
    int a_rows;
    int a_cols;
    int b_rows;
    int b_cols;
    ParamGroupType group_type;
    const char* projection_name;
    const char* tensor_a_name;
    const char* tensor_b_name;
};

std::array<LoRAMatrixSpec, 5> makeLoRAMatrixSpecs(
    const GRIM::HyperParameters::EncoderLayerConstructionHP& encoder_hp,
    const GRIM::HyperParameters::LoRATrainingHP& lora_hp) {
    return {{
        {GRIM::LoRAMatrixClass::QKV, &lora_hp.qkv,
         static_cast<int>(lora_hp.qkv.rank), encoder_hp.d_model,
         encoder_hp.qkv_dim, static_cast<int>(lora_hp.qkv.rank),
         ParamGroupType::ATTENTION, "qkv", "lora_qkv_A", "lora_qkv_B"},
        {GRIM::LoRAMatrixClass::ATTENTION_OUTPUT, &lora_hp.o,
         static_cast<int>(lora_hp.o.rank), encoder_hp.d_model,
         encoder_hp.d_model, static_cast<int>(lora_hp.o.rank),
         ParamGroupType::ATTENTION, "wo", "lora_o_A", "lora_o_B"},
        {GRIM::LoRAMatrixClass::FFN_GATE, &lora_hp.gate,
         static_cast<int>(lora_hp.gate.rank), encoder_hp.d_ff,
         encoder_hp.d_model, static_cast<int>(lora_hp.gate.rank),
         ParamGroupType::FFN, "ffn_w_gate", "lora_gate_A", "lora_gate_B"},
        {GRIM::LoRAMatrixClass::FFN_UP, &lora_hp.w1,
         static_cast<int>(lora_hp.w1.rank), encoder_hp.d_ff,
         encoder_hp.d_model, static_cast<int>(lora_hp.w1.rank),
         ParamGroupType::FFN, "ffn_w1", "lora_w1_A", "lora_w1_B"},
        {GRIM::LoRAMatrixClass::FFN_DOWN, &lora_hp.w2,
         static_cast<int>(lora_hp.w2.rank), encoder_hp.d_model,
         encoder_hp.d_ff, static_cast<int>(lora_hp.w2.rank),
         ParamGroupType::FFN, "ffn_w2", "lora_w2_A", "lora_w2_B"},
    }};
}

std::uint64_t deriveLoRATargetSeed(std::uint64_t base_seed,
                                   int layer,
                                   GRIM::LoRAMatrixClass matrix_class) {
    if (layer < 0) {
        throw std::runtime_error("deriveLoRATargetSeed: layer must be non-negative");
    }
    std::uint64_t value = base_seed ^
        (static_cast<std::uint64_t>(layer) + 1ULL) * 0x9E3779B97F4A7C15ULL ^
        (static_cast<std::uint64_t>(ParameterRegistry::loraMatrixClassIndex(matrix_class)) + 1ULL) *
            0xD1B54A32D192ED03ULL;
    value ^= value >> 30;
    value *= 0xBF58476D1CE4E5B9ULL;
    value ^= value >> 27;
    value *= 0x94D049BB133111EBULL;
    value ^= value >> 31;
    return value;
}

void requireTensorShape(const Tensor& tensor,
                        int rows,
                        int cols,
                        const std::string& name,
                        const char* caller) {
    tensor.require(caller);
    if (!tensor.shape.is_2d_layout()) {
        throw std::runtime_error(std::string(caller) + ": " + name + " must be 2D");
    }
    const auto shape = tensor.shape.as_2d();
    if (shape.rows != rows || shape.cols != cols) {
        throw std::runtime_error(
            std::string(caller) + ": " + name + " shape mismatch; expected [" +
            std::to_string(rows) + "," + std::to_string(cols) + "] got [" +
            std::to_string(shape.rows) + "," + std::to_string(shape.cols) + "]");
    }
}

std::string tensorDebugSummary(const Tensor& tensor) {
    std::ostringstream oss;
    oss << "data=" << tensor.data
        << " grad=" << tensor.grad_data()
        << " has_grad=" << static_cast<int>(tensor.has_grad())
        << " numel=" << tensor.numel();
    return oss.str();
}

void emitInfo(const std::string& message) {
    GRIM::Logging::EmitModuleInfo(kRegistrationModule, message);
}

void validateAtomInsertionBoundaryParameterTensors(
    const GRIM::AtomInsertionBoundaryParameterTensors& parameters,
    const GRIM::HyperParameters::AtomInsertionBoundaryProjectionHP& hp,
    const char* caller) {
    if (!caller || caller[0] == '\0') {
        throw std::runtime_error(
            "validateAtomInsertionBoundaryParameterTensors: caller is empty");
    }
    if (!hp.enabled) {
        throw std::runtime_error(
            std::string(caller) + ": atom insertion is disabled");
    }
    if (hp.d_model <= 0) {
        throw std::runtime_error(
            std::string(caller) + ": d_model must be positive");
    }
    auto require_shape = [&](const Tensor& tensor,
                             int rows,
                             int cols,
                             const char* name) {
        tensor.require(caller);
        if (!tensor.shape.is_2d_layout()) {
            throw std::runtime_error(
                std::string(caller) + ": " + name + " must be 2D");
        }
        const auto shape = tensor.shape.as_2d();
        if (shape.rows != rows || shape.cols != cols) {
            throw std::runtime_error(
                std::string(caller) + ": " + name +
                " shape mismatch; expected [" + std::to_string(rows) + "," +
                std::to_string(cols) + "] got [" +
                std::to_string(shape.rows) + "," +
                std::to_string(shape.cols) + "]");
        }
    };
    require_shape(
        parameters.left_projection_weight,
        hp.d_model,
        hp.d_model,
        "left_projection_weight");
    require_shape(
        parameters.right_projection_weight,
        hp.d_model,
        hp.d_model,
        "right_projection_weight");
    require_shape(
        parameters.projection_bias,
        1,
        hp.d_model,
        "projection_bias");
}

void validateOutputUnigramPrior(const OutputUnigramPriorView& prior,
                                int expected_vocab_size,
                                const char* caller) {
    if (!prior.log_bias) {
        throw std::runtime_error(std::string(caller) + ": output unigram prior log_bias is NULL");
    }
    if (expected_vocab_size <= 0) {
        throw std::runtime_error(std::string(caller) + ": expected_vocab_size must be positive, got " +
                                 std::to_string(expected_vocab_size));
    }
    if (prior.vocab_size != static_cast<std::uint32_t>(expected_vocab_size)) {
        throw std::runtime_error(std::string(caller) + ": output unigram prior vocab_size=" +
                                 std::to_string(prior.vocab_size) + " != expected_vocab_size=" +
                                 std::to_string(expected_vocab_size));
    }
    if (prior.size != static_cast<std::size_t>(expected_vocab_size)) {
        throw std::runtime_error(std::string(caller) + ": output unigram prior size=" +
                                 std::to_string(prior.size) + " != expected_vocab_size=" +
                                 std::to_string(expected_vocab_size));
    }
    if (prior.total_targets == 0) {
        throw std::runtime_error(std::string(caller) + ": output unigram prior total_targets is zero");
    }
}

void uploadOutputUnigramPriorToBias(Tensor& bias,
                                    const OutputUnigramPriorView& prior,
                                    int expected_vocab_size,
                                    cudaStream_t stream,
                                    const char* caller,
                                    const char* tensor_name) {
    validateOutputUnigramPrior(prior, expected_vocab_size, caller);
    if (!bias.data) {
        throw std::runtime_error(std::string(caller) + ": " + tensor_name + " bias tensor data is NULL");
    }
    if (bias.numel() != static_cast<std::size_t>(expected_vocab_size)) {
        throw std::runtime_error(std::string(caller) + ": " + tensor_name + " bias numel=" +
                                 std::to_string(bias.numel()) + " != expected_vocab_size=" +
                                 std::to_string(expected_vocab_size));
    }
    const cudaError_t copy_err = cudaMemcpyAsync(
        bias.data,
        prior.log_bias,
        static_cast<std::size_t>(expected_vocab_size) * sizeof(float),
        cudaMemcpyHostToDevice,
        stream);
    if (copy_err != cudaSuccess) {
        throw std::runtime_error(std::string(caller) + ": cudaMemcpyAsync(" + tensor_name +
                                 " output unigram prior H2D) failed: " + cudaGetErrorString(copy_err));
    }
}

size_t paramGroupTypeIndex(ParamGroupType type) {
    switch (type) {
        case ParamGroupType::EMBEDDING:       return 0;
        case ParamGroupType::LM_HEAD:         return 1;
        case ParamGroupType::ATTENTION:       return 2;
        case ParamGroupType::FFN:             return 3;
        case ParamGroupType::RMSNORM:         return 4;
        case ParamGroupType::ARG_SELECTOR:    return 5;
        case ParamGroupType::COUNT: break;
    }
    throw std::runtime_error("[buildParameterGroups] invalid ParamGroupType::COUNT in registered group summary");
}

const char* paramGroupTypeSummaryName(ParamGroupType type) {
    switch (type) {
        case ParamGroupType::EMBEDDING:       return "embedding";
        case ParamGroupType::LM_HEAD:         return "lm_head";
        case ParamGroupType::ATTENTION:       return "attention";
        case ParamGroupType::FFN:             return "ffn";
        case ParamGroupType::RMSNORM:         return "rmsnorm";
        case ParamGroupType::ARG_SELECTOR:    return "arg_selector";
        case ParamGroupType::COUNT: break;
    }
    throw std::runtime_error("[buildParameterGroups] invalid ParamGroupType::COUNT in registered group summary");
}

size_t parameterPrecisionIndex(ParameterGroupPrecision precision) {
    switch (precision) {
        case ParameterGroupPrecision::FP32:         return 0;
        case ParameterGroupPrecision::BF16_COMPUTE: return 1;
        case ParameterGroupPrecision::UNSPECIFIED: break;
    }
    throw std::runtime_error("[buildParameterGroups] invalid ParameterGroupPrecision in registered group summary");
}

const char* parameterPrecisionSummaryName(ParameterGroupPrecision precision) {
    switch (precision) {
        case ParameterGroupPrecision::FP32:         return "FP32";
        case ParameterGroupPrecision::BF16_COMPUTE: return "BF16_COMPUTE";
        case ParameterGroupPrecision::UNSPECIFIED: break;
    }
    throw std::runtime_error("[buildParameterGroups] invalid ParameterGroupPrecision in registered group summary");
}

void validateRegisteredPrecisionSupport(const std::string& name,
                                        ParamGroupType type,
                                        ParameterGroupPrecision precision) {
    switch (precision) {
        case ParameterGroupPrecision::FP32:
            return;
        case ParameterGroupPrecision::BF16_COMPUTE:
            throw std::runtime_error("[buildParameterGroups] " + name +
                                     " requests BF16_COMPUTE for parameter group type " +
                                     paramGroupTypeSummaryName(type) +
                                     ", but TensorContract implicit BF16 compute consumers are not wired for optimizer-visible parameter tensors in this build; set training.config.parameter_precision_" +
                                     paramGroupTypeSummaryName(type) + " to fp32");
        case ParameterGroupPrecision::UNSPECIFIED:
            break;
    }
    throw std::runtime_error("[buildParameterGroups] " + name +
                             " has UNSPECIFIED or unknown ParameterGroupPrecision");
}

template <typename LayerT>
LayerT& requireLayer(LayerT* layer, const char* layer_name, const char* caller) {
    if (!layer) {
        throw std::runtime_error(std::string(caller) + ": " + layer_name +
                                 " is NULL - startup initialization order is broken");
    }
    return *layer;
}

void throwUntrainableTensor(const std::string& name, const Tensor& tensor, int layer) {
    throw std::runtime_error("[buildParameterGroups] " + name +
                             " is not a trainable parameter group tensor: " +
                             tensorDebugSummary(tensor) + " layer=" + std::to_string(layer));
}

class Registrar {
public:
    Registrar(std::vector<ParameterGroup>& groups,
              const GRIM::Config::AiConfigSnapshot& config)
    : groups_(groups), config_(config), optimizer_hp_(GRIM::HyperParameters::optimizerUpdateHP(config)) {}

    void addTensor(const std::string& name,
                   Tensor& tensor,
                   ParamGroupType type,
                   ParamStatsBucket stats_bucket,
                   int layer = -1,
                   float wd_mult = 1.0f,
                   float lr_mult = 1.0f) {
        if (name.empty()) {
            throw std::runtime_error(
                "[buildParameterGroups] parameter group name must not be empty");
        }
        if (!registered_names_.insert(name).second) {
            throw std::runtime_error(
                "[buildParameterGroups] duplicate parameter group name: " + name);
        }
        if (!tensor.data || !tensor.has_grad() || tensor.numel() == 0) {
            throwUntrainableTensor(name, tensor, layer);
        }
        if (stats_bucket == ParamStatsBucket::COUNT) {
            throw std::runtime_error("[buildParameterGroups] " + name +
                                     " has invalid ParamStatsBucket::COUNT");
        }
        const void* data_ptr = static_cast<const void*>(tensor.data);
        for (const void* registered_ptr : registered_data_) {
            if (registered_ptr == data_ptr) {
                throw std::runtime_error("[buildParameterGroups] duplicate tensor.data registration for " + name +
                                         " would double-step the same memory: " +
                                         tensorDebugSummary(tensor) + " layer=" + std::to_string(layer));
            }
        }
        registered_data_.push_back(data_ptr);

        const ParameterGroupPrecision precision = precisionForType(type);
        validateRegisteredPrecisionSupport(name, type, precision);

        const TensorContract::PrecisionType tensor_precision =
            TensorContract::precision_from_parameter_group_precision(precision);
        if (tensor.precision() != TensorContract::PrecisionType::FP32 &&
            tensor.precision() != tensor_precision) {
            throw std::runtime_error("[buildParameterGroups] " + name +
                                     " tensor precision metadata already equals " +
                                     TensorContract::precision_name(tensor.precision()) +
                                     " but registration requires " +
                                     TensorContract::precision_name(tensor_precision));
        }
        tensor.set_compute_precision(tensor_precision, "ParameterGroupRegistration::Registrar::addTensor");

        ParameterGroup group{};
        group.name = name;
        group.tensor = &tensor;
        group.m_tensor = nullptr;
        group.v_tensor = nullptr;
        group.type = type;
        group.parameter_precision = precision;
        group.stats_bucket = stats_bucket;
        group.layer_index = layer;
        group.weight_decay_multiplier = wd_mult;
        group.lr_multiplier = lr_mult;
        // Registration is the durable ownership boundary for verifier history.
        // Seed the observation point from the shared gradient tensor so a
        // rebuilt registry never mistakes pre-registration writes for delivery
        // by the first active backward it verifies.
        group.gradient_verification.last_observed_delivery_count =
            tensor.gradient_delivery_count();
        if (optimizer_hp_.use_depth_aware_upsilon && layer >= 0) {
            group.upsilon = GRIM::HyperParameters::UPSILON_BASE *
                std::sqrt(static_cast<float>(GRIM::HyperParameters::UPSILON_REFERENCE_LAYERS) /
                          static_cast<float>(layer + 1));
        }
        groups_.push_back(group);
    }

    void addConfigGatedTensor(const std::string& name,
                              Tensor& tensor,
                              ParamGroupType type,
                              ParamStatsBucket stats_bucket,
                              int layer,
                              bool enabled,
                              const char* disabled_reason) {
        if (enabled) {
            addTensor(name, tensor, type, stats_bucket, layer);
            return;
        }

        if (tensor.data || tensor.has_grad()) {
            throw std::runtime_error("[buildParameterGroups] " + name +
                                     " exists while disabled (" + disabled_reason + "): " +
                                     tensorDebugSummary(tensor));
        }
    }

    void addLoRATensor(const std::string& name,
                       Tensor& tensor,
                       ParamGroupType type,
                       ParamStatsBucket stats_bucket,
                       int layer,
                       ParameterGroupPrecision precision) {
        addTensorWithPrecision(name, tensor, type, stats_bucket, layer,
                               0.0f, 1.0f, precision);
    }

private:
    void addTensorWithPrecision(const std::string& name,
                                Tensor& tensor,
                                ParamGroupType type,
                                ParamStatsBucket stats_bucket,
                                int layer,
                                float wd_mult,
                                float lr_mult,
                                ParameterGroupPrecision precision) {
        if (name.empty()) {
            throw std::runtime_error(
                "[buildParameterGroups] parameter group name must not be empty");
        }
        if (!registered_names_.insert(name).second) {
            throw std::runtime_error(
                "[buildParameterGroups] duplicate parameter group name: " + name);
        }
        if (!tensor.data || !tensor.has_grad() || tensor.numel() == 0) {
            throwUntrainableTensor(name, tensor, layer);
        }
        if (stats_bucket == ParamStatsBucket::COUNT) {
            throw std::runtime_error("[buildParameterGroups] " + name +
                                     " has invalid ParamStatsBucket::COUNT");
        }
        const void* data_ptr = static_cast<const void*>(tensor.data);
        for (const void* registered_ptr : registered_data_) {
            if (registered_ptr == data_ptr) {
                throw std::runtime_error("[buildParameterGroups] duplicate tensor.data registration for " + name +
                                         " would double-step the same memory: " +
                                         tensorDebugSummary(tensor) + " layer=" + std::to_string(layer));
            }
        }
        registered_data_.push_back(data_ptr);
        validateRegisteredPrecisionSupport(name, type, precision);

        const TensorContract::PrecisionType tensor_precision =
            TensorContract::precision_from_parameter_group_precision(precision);
        if (tensor.precision() != TensorContract::PrecisionType::FP32 &&
            tensor.precision() != tensor_precision) {
            throw std::runtime_error("[buildParameterGroups] " + name +
                                     " tensor precision metadata already equals " +
                                     TensorContract::precision_name(tensor.precision()) +
                                     " but registration requires " +
                                     TensorContract::precision_name(tensor_precision));
        }
        tensor.set_compute_precision(tensor_precision, "ParameterGroupRegistration::Registrar::addTensorWithPrecision");

        ParameterGroup group{};
        group.name = name;
        group.tensor = &tensor;
        group.m_tensor = nullptr;
        group.v_tensor = nullptr;
        group.type = type;
        group.parameter_precision = precision;
        group.stats_bucket = stats_bucket;
        group.layer_index = layer;
        group.weight_decay_multiplier = wd_mult;
        group.lr_multiplier = lr_mult;
        group.gradient_verification.last_observed_delivery_count =
            tensor.gradient_delivery_count();
        if (optimizer_hp_.use_depth_aware_upsilon && layer >= 0) {
            group.upsilon = GRIM::HyperParameters::UPSILON_BASE *
                std::sqrt(static_cast<float>(GRIM::HyperParameters::UPSILON_REFERENCE_LAYERS) /
                          static_cast<float>(layer + 1));
        }
        groups_.push_back(group);
    }

    ParameterGroupPrecision precisionForType(ParamGroupType type) const {
        switch (type) {
            case ParamGroupType::EMBEDDING:       return GRIM::HyperParameters::snapshotTrainingConfigField<ParameterGroupPrecision>(config_, "parameter_precision_embedding");
            case ParamGroupType::LM_HEAD:         return GRIM::HyperParameters::snapshotTrainingConfigField<ParameterGroupPrecision>(config_, "parameter_precision_lm_head");
            case ParamGroupType::ATTENTION:       return GRIM::HyperParameters::snapshotTrainingConfigField<ParameterGroupPrecision>(config_, "parameter_precision_attention");
            case ParamGroupType::FFN:             return GRIM::HyperParameters::snapshotTrainingConfigField<ParameterGroupPrecision>(config_, "parameter_precision_ffn");
            case ParamGroupType::RMSNORM:         return GRIM::HyperParameters::snapshotTrainingConfigField<ParameterGroupPrecision>(config_, "parameter_precision_rmsnorm");
            case ParamGroupType::ARG_SELECTOR:    return GRIM::HyperParameters::snapshotTrainingConfigField<ParameterGroupPrecision>(config_, "parameter_precision_arg_selector");
            case ParamGroupType::COUNT: break;
        }
        throw std::runtime_error("[buildParameterGroups] invalid ParamGroupType::COUNT for parameter precision lookup");
    }

    std::vector<ParameterGroup>& groups_;
    const GRIM::Config::AiConfigSnapshot& config_;
    const OptimizerUpdateHP optimizer_hp_;
    std::unordered_set<std::string> registered_names_;
    std::vector<const void*> registered_data_;
};

void validateEmbeddingLmHeadAliasing(const Tensor& embedding_weights,
                                     const Tensor& lm_head_weights,
                                     const GRIM::Config::AiConfigSnapshot& config) {
    if (!embedding_weights.data || !lm_head_weights.data) {
        throw std::runtime_error("[buildParameterGroups] cannot validate embedding/LM-head aliasing with NULL data: embedding=" +
                                 tensorDebugSummary(embedding_weights) + " lm_head=" + tensorDebugSummary(lm_head_weights));
    }

    const bool tied = GRIM::HyperParameters::snapshotTrainingConfigField<bool>(config, "tie_embeddings");
    const bool same_data = embedding_weights.data == lm_head_weights.data;
    if (tied && !same_data) {
        throw std::runtime_error("[buildParameterGroups] tie_embeddings=true but embedding and LM-head data pointers differ: embedding=" +
                                 tensorDebugSummary(embedding_weights) + " lm_head=" + tensorDebugSummary(lm_head_weights));
    }
    if (!tied && same_data) {
        throw std::runtime_error("[buildParameterGroups] tie_embeddings=false but embedding and LM-head data pointers are identical: embedding=" +
                                 tensorDebugSummary(embedding_weights) + " lm_head=" + tensorDebugSummary(lm_head_weights));
    }
    const auto embedding_precision = GRIM::HyperParameters::snapshotTrainingConfigField<ParameterGroupPrecision>(config, "parameter_precision_embedding");
    const auto lm_head_precision = GRIM::HyperParameters::snapshotTrainingConfigField<ParameterGroupPrecision>(config, "parameter_precision_lm_head");
    if (tied && embedding_precision != lm_head_precision) {
        throw std::runtime_error("[buildParameterGroups] tie_embeddings=true requires parameter_precision_embedding and parameter_precision_lm_head to match; embedding=" +
                                 std::string(parameterPrecisionSummaryName(embedding_precision)) +
                                 " lm_head=" +
                                 std::string(parameterPrecisionSummaryName(lm_head_precision)));
    }
}

void registerTopLevelParameters(Startup::GpuModelState& gpu_model_state,
                                ParameterRegistry::StartupParameterRegistry& parameter_registry,
                                Registrar& registrar,
                                const GRIM::Config::AiConfigSnapshot& config) {
    (void)gpu_model_state;
    auto& embedding_parameters = parameter_registry.requireEmbeddingParameters("registerTopLevelParameters");
    auto& lm_head_parameters = parameter_registry.requireLmHeadParameters("registerTopLevelParameters");
    validateEmbeddingLmHeadAliasing(embedding_parameters.token_weights, lm_head_parameters.weights, config);

    const bool tie_embeddings = GRIM::HyperParameters::snapshotTrainingConfigField<bool>(config, "tie_embeddings");
    const bool lm_head_bias_enabled = GRIM::HyperParameters::snapshotTrainingConfigField<bool>(config, "lm_head_bias_enabled");
    const bool freeze_learned_rms_gammas = GRIM::HyperParameters::snapshotTrainingConfigField<bool>(config, "freeze_learned_rms_gammas");

    if (!tie_embeddings) {
        ParameterRegistry::registerEmbeddingParameters(embedding_parameters, registrar);

        registrar.addTensor("lm_head_weight",
                            lm_head_parameters.weights,
                            ParamGroupType::LM_HEAD,
                            ParamStatsBucket::LM_HEAD);
    } else {
        registrar.addTensor("embedding_lm_head_tied",
                            lm_head_parameters.weights,
                            ParamGroupType::LM_HEAD,
                            ParamStatsBucket::EMBEDDING);
    }

    registrar.addConfigGatedTensor("lm_head_bias",
                                   lm_head_parameters.bias,
                                   ParamGroupType::LM_HEAD,
                                   ParamStatsBucket::LM_HEAD,
                                   -1,
                                   lm_head_bias_enabled,
                                   "config.lm_head_bias_enabled=false");

    // Head-side residual SwiGLU adapter (config.lm_head_mlp_enabled).
    const bool lm_head_mlp_enabled = GRIM::HyperParameters::snapshotTrainingConfigField<bool>(config, "lm_head_mlp_enabled");
    registrar.addConfigGatedTensor("lm_head_mlp_w_gate",
                                   lm_head_parameters.mlp_W_gate,
                                   ParamGroupType::LM_HEAD,
                                   ParamStatsBucket::LM_HEAD,
                                   -1,
                                   lm_head_mlp_enabled,
                                   "config.lm_head_mlp_enabled=false");
    registrar.addConfigGatedTensor("lm_head_mlp_w_up",
                                   lm_head_parameters.mlp_W_up,
                                   ParamGroupType::LM_HEAD,
                                   ParamStatsBucket::LM_HEAD,
                                   -1,
                                   lm_head_mlp_enabled,
                                   "config.lm_head_mlp_enabled=false");
    registrar.addConfigGatedTensor("lm_head_mlp_w_down",
                                   lm_head_parameters.mlp_W_down,
                                   ParamGroupType::LM_HEAD,
                                   ParamStatsBucket::LM_HEAD,
                                   -1,
                                   lm_head_mlp_enabled,
                                   "config.lm_head_mlp_enabled=false");

    const Tensor& final_gamma = lm_head_parameters.final_rms_gamma;
    if (freeze_learned_rms_gammas) {
        if (final_gamma.has_grad()) {
            throw std::runtime_error("[buildParameterGroups] final_rms_gamma is frozen by config but still has a grad buffer: " +
                                     tensorDebugSummary(final_gamma));
        }
    } else {
        registrar.addTensor("final_rms_gamma",
                            lm_head_parameters.final_rms_gamma,
                            ParamGroupType::RMSNORM,
                            ParamStatsBucket::LM_HEAD);
    }
}

void registerEncoderParameters(Startup::GpuModelState& gpu_model_state,
                               ParameterRegistry::StartupParameterRegistry& parameter_registry,
                               Registrar& registrar,
                               const GRIM::Config::AiConfigSnapshot& config) {
    auto* gpu_encoder = gpu_model_state.gpu_encoder.get();
    if (!gpu_encoder) {
        throw std::runtime_error(
            "[buildParameterGroups] gpu_model_state.gpu_encoder is NULL - "
            "Startup::assembleGpuModel(config, training_state, gpu_model_state, parameter_registry, weight_init_seed) must complete before parameter registration");
    }
    const int num_layers = GRIM::HyperParameters::snapshotTrainingConfigField<int>(config, "num_layers");
    const auto encoder_hp = GRIM::HyperParameters::encoderLayerConstructionHP(config);
    const bool freeze_learned_rms_gammas = GRIM::HyperParameters::snapshotTrainingConfigField<bool>(config, "freeze_learned_rms_gammas");
    const bool use_layer_scale = GRIM::HyperParameters::snapshotTrainingConfigField<bool>(config, "use_layer_scale");
    if (static_cast<int>(parameter_registry.encodingLayerParameterTensors().size()) != num_layers) {
        throw std::runtime_error("[buildParameterGroups] encoding_layer_parameter_tensors size must equal config.num_layers. size=" +
                                 std::to_string(parameter_registry.encodingLayerParameterTensors().size()) +
                                 " num_layers=" + std::to_string(num_layers));
    }
    if (static_cast<int>(parameter_registry.feedForwardParameterTensors().size()) != num_layers) {
        throw std::runtime_error("[buildParameterGroups] feed_forward_parameter_tensors size must equal config.num_layers. size=" +
                                 std::to_string(parameter_registry.feedForwardParameterTensors().size()) +
                                 " num_layers=" + std::to_string(num_layers));
    }

    for (int layer = 0; layer < num_layers; ++layer) {
        if (!gpu_encoder->getLayer(layer)) {
            throw std::runtime_error("[buildParameterGroups] Encoder layer " + std::to_string(layer) +
                                     " is NULL - initGPU() did not build the configured topology");
        }

        const std::string prefix = "layer" + std::to_string(layer);
        auto& encoding_parameters = parameter_registry.requireEncodingLayerParameters(layer, "registerEncoderParameters");
        ParameterRegistry::registerEncodingLayerParameters(
            encoding_parameters,
            layer,
            encoder_hp.attention_qkv_bias_enabled,
            encoder_hp.attention_output_bias_enabled,
            freeze_learned_rms_gammas,
            use_layer_scale,
            encoder_hp.attention_residual_gate_enabled,
            registrar);

        auto& ffn_parameters = parameter_registry.requireFeedForwardParameters(layer, "registerEncoderParameters");
        ParameterRegistry::registerFeedForwardParameters(
            ffn_parameters,
            layer,
            encoder_hp.ffn_output_bias_enabled,
            registrar);

        if (freeze_learned_rms_gammas) {
            if (encoding_parameters.rms1_gamma.has_grad()) {
                throw std::runtime_error("[buildParameterGroups] " + prefix +
                                         "_rms1_gamma is frozen by config but still has a grad buffer: " +
                                         tensorDebugSummary(encoding_parameters.rms1_gamma));
            }
            if (encoding_parameters.rms2_gamma.has_grad()) {
                throw std::runtime_error("[buildParameterGroups] " + prefix +
                                         "_rms2_gamma is frozen by config but still has a grad buffer: " +
                                         tensorDebugSummary(encoding_parameters.rms2_gamma));
            }
        }
    }
}

void registerAtomInsertionBoundaryParameters(
    ParameterRegistry::StartupParameterRegistry& parameter_registry,
    Registrar& registrar,
    const GRIM::Config::AiConfigSnapshot& config) {
    auto* parameters = parameter_registry.getAtomInsertionBoundaryParameters();
    const auto atom_hp =
        GRIM::HyperParameters::atomInsertionBoundaryProjectionHP(config);
    if (!atom_hp.enabled) {
        if (parameters) {
            throw std::runtime_error(
                "[buildParameterGroups] atom insertion boundary parameter owner exists "
                "while config.atom_insertion_enabled=false");
        }
        return;
    }

    auto& owner = requireLayer(
        parameters,
        "AtomInsertionBoundaryParameterTensors",
        "registerAtomInsertionBoundaryParameters");
    ParameterRegistry::registerAtomInsertionBoundaryParameters(owner, registrar);
}

void registerSelectorParameters(ParameterRegistry::StartupParameterRegistry& parameter_registry,
                                 Registrar& registrar,
                                 const GRIM::Config::AiConfigSnapshot& config) {
    auto* selector_parameters = parameter_registry.getSelectorParameters();
    const bool selector_enabled =
        GRIM::HyperParameters::snapshotTrainingConfigField<bool>(config, "selector_enabled");
    if (!selector_enabled) {
        if (selector_parameters) {
            throw std::runtime_error("[buildParameterGroups] Selector parameter owner exists while config.selector_enabled=false");
        }
        return;
    }

    auto& selector_tensor_owner = requireLayer(
        selector_parameters,
        "SelectorParameterTensors",
        "registerSelectorParameters");
    ParameterRegistry::registerSelectorParameters(selector_tensor_owner, registrar);
}

void registerLocalAtomRetrievalParameters(
    ParameterRegistry::StartupParameterRegistry& parameter_registry,
    Registrar& registrar,
    const GRIM::Config::AiConfigSnapshot& config) {
    auto* parameters = parameter_registry.getLocalAtomRetrievalParameters();
    const auto model_hp = GRIM::HyperParameters::modelHP(config);
    if (!model_hp.local_atom_retrieval_enabled) {
        if (parameters) {
            throw std::runtime_error(
                "[buildParameterGroups] LocalAtomRetrieval parameter owner "
                "exists while config.local_atom_retrieval_enabled=false");
        }
        return;
    }
    auto& owner = parameter_registry.requireLocalAtomRetrievalParameters(
        "registerLocalAtomRetrievalParameters");
    ParameterRegistry::registerLocalAtomRetrievalParameters(owner, registrar);
}

void validateBaseParametersFrozen(
    const ParameterRegistry::StartupParameterRegistry& parameter_registry) {
    const auto require_frozen = [](const Tensor& tensor, const std::string& name) {
        if (tensor.requires_grad || tensor.has_grad()) {
            throw std::runtime_error(
                "[buildParameterGroups] LoRA model base tensor is not frozen: " +
                name + " " + tensorDebugSummary(tensor));
        }
    };

    if (const auto* embedding = parameter_registry.getEmbeddingParameters()) {
        require_frozen(embedding->token_weights, "embedding");
    }
    if (const auto* lm_head = parameter_registry.getLmHeadParameters()) {
        require_frozen(lm_head->weights, "lm_head_weight");
        require_frozen(lm_head->bias, "lm_head_bias");
        require_frozen(lm_head->final_rms_gamma, "final_rms_gamma");
        require_frozen(lm_head->mlp_W_gate, "lm_head_mlp_w_gate");
        require_frozen(lm_head->mlp_W_up, "lm_head_mlp_w_up");
        require_frozen(lm_head->mlp_W_down, "lm_head_mlp_w_down");
    }
    if (const auto* atom = parameter_registry.getAtomInsertionBoundaryParameters()) {
        require_frozen(atom->left_projection_weight, "atom_insertion_left_projection_weight");
        require_frozen(atom->right_projection_weight, "atom_insertion_right_projection_weight");
        require_frozen(atom->projection_bias, "atom_insertion_projection_bias");
    }
    for (std::size_t layer = 0;
         layer < parameter_registry.encodingLayerParameterTensors().size();
         ++layer) {
        const auto& tensors = parameter_registry.encodingLayerParameterTensors()[layer];
        const std::string prefix = "layer" + std::to_string(layer) + "_";
        for (const auto& spec : ParameterRegistry::kEncodingLayerTensorParameters) {
            require_frozen(tensors.*(spec.tensor_member), prefix + spec.name);
        }
        for (const auto& spec : ParameterRegistry::kAttentionResidualGateTensorParameters) {
            require_frozen(tensors.attention_residual_gate.*(spec.tensor_member),
                           prefix + spec.name);
        }
    }
    for (std::size_t layer = 0;
         layer < parameter_registry.feedForwardParameterTensors().size();
         ++layer) {
        const auto& tensors = parameter_registry.feedForwardParameterTensors()[layer];
        const std::string prefix = "layer" + std::to_string(layer) + "_";
        for (const auto& spec : ParameterRegistry::kFeedForwardTensorParameters) {
            require_frozen(tensors.*(spec.tensor_member), prefix + spec.name);
        }
    }
    if (const auto* selector = parameter_registry.getSelectorParameters()) {
        require_frozen(selector->W_q, "selector_W_q");
    }
    if (const auto* retrieval = parameter_registry.getLocalAtomRetrievalParameters()) {
        require_frozen(retrieval->type_no_reference_key,
                       "local_atom_retrieval_type_no_reference_key");
    }
}

void registerLoRAParameters(
    ParameterRegistry::StartupParameterRegistry& parameter_registry,
    Registrar& registrar,
    const GRIM::Config::AiConfigSnapshot& config) {
    const auto encoder_hp = GRIM::HyperParameters::encoderLayerConstructionHP(config);
    const auto lora_hp = GRIM::HyperParameters::loraTrainingHP(config);
    const auto specs = makeLoRAMatrixSpecs(encoder_hp, lora_hp);
    if (static_cast<int>(parameter_registry.loraLayerParameterPairs().size()) !=
        encoder_hp.num_layers) {
        throw std::runtime_error(
            "[buildParameterGroups] lora_layer_parameter_pairs size must equal num_layers");
    }

    for (int layer = 0; layer < encoder_hp.num_layers; ++layer) {
        for (const auto& spec : specs) {
            auto* pair = parameter_registry.getLoRAParameterPair(layer, spec.matrix_class);
            if (!spec.settings->enabled) {
                if (pair) {
                    throw std::runtime_error(
                        "[buildParameterGroups] disabled LoRA class owns tensors at layer " +
                        std::to_string(layer) + " projection=" + spec.projection_name);
                }
                continue;
            }
            if (!pair) {
                throw std::runtime_error(
                    "[buildParameterGroups] enabled LoRA class is missing tensors at layer " +
                    std::to_string(layer) + " projection=" + spec.projection_name);
            }
            if (pair->matrix_class != spec.matrix_class ||
                pair->rank != spec.settings->rank ||
                pair->alpha != spec.settings->alpha ||
                pair->precision != spec.settings->precision) {
                throw std::runtime_error(
                    "[buildParameterGroups] LoRA pair facts do not match authored class settings at layer " +
                    std::to_string(layer) + " projection=" + spec.projection_name);
            }
            const float expected_scale = pair->alpha / static_cast<float>(pair->rank);
            if (!std::isfinite(pair->scale) || pair->scale <= 0.0f ||
                pair->scale != expected_scale) {
                throw std::runtime_error(
                    "[buildParameterGroups] LoRA pair scale is invalid at layer " +
                    std::to_string(layer) + " projection=" + spec.projection_name);
            }
            requireTensorShape(pair->A, spec.a_rows, spec.a_cols,
                               std::string(spec.projection_name) + ".lora_A",
                               "registerLoRAParameters");
            requireTensorShape(pair->B, spec.b_rows, spec.b_cols,
                               std::string(spec.projection_name) + ".lora_B",
                               "registerLoRAParameters");
            const std::string prefix = "layer" + std::to_string(layer) + "_" +
                spec.projection_name;
            registrar.addLoRATensor(prefix + ".lora_A", pair->A, spec.group_type,
                                    ParamStatsBucket::ENCODER, layer, pair->precision);
            registrar.addLoRATensor(prefix + ".lora_B", pair->B, spec.group_type,
                                    ParamStatsBucket::ENCODER, layer, pair->precision);
        }
    }
}

void clearOptimizerBindings(std::vector<ParameterGroup>& groups) {
    for (auto& group : groups) {
        group.m_tensor = nullptr;
        group.v_tensor = nullptr;
    }
}

void validateOptimizerStateAllocation(const std::vector<ParameterGroup>& groups,
                                      const GRIM::OptimizerState& optimizer_state) {
    if (optimizer_state.m_states.size() != groups.size()) {
        throw std::runtime_error("[bindOptimizerState] optimizer m_states size mismatch: states=" +
                                 std::to_string(optimizer_state.m_states.size()) +
                                 " groups=" + std::to_string(groups.size()));
    }
    if (optimizer_state.v_states.size() != groups.size()) {
        throw std::runtime_error("[bindOptimizerState] optimizer v_states size mismatch: states=" +
                                 std::to_string(optimizer_state.v_states.size()) +
                                 " groups=" + std::to_string(groups.size()));
    }

    for (size_t i = 0; i < groups.size(); ++i) {
        const size_t group_size = groups[i].size();
        const size_t m_size = optimizer_state.m_states[i].numel();
        const size_t v_size = optimizer_state.v_states[i].numel();
        if (m_size != group_size) {
            throw std::runtime_error("[bindOptimizerState] optimizer m_state size mismatch for group " +
                                     std::to_string(i) + " (" + groups[i].name + "): state=" +
                                     std::to_string(m_size) + " group=" + std::to_string(group_size));
        }
        if (v_size != group_size) {
            throw std::runtime_error("[bindOptimizerState] optimizer v_state size mismatch for group " +
                                     std::to_string(i) + " (" + groups[i].name + "): state=" +
                                     std::to_string(v_size) + " group=" + std::to_string(group_size));
        }
    }
}

void validateRegisteredTensorPrecisionMetadata(const std::vector<ParameterGroup>& groups) {
    for (const auto& group : groups) {
        if (!group.tensor) {
            throw std::runtime_error("[buildParameterGroups] group " + group.name +
                                     " has NULL tensor while validating precision metadata");
        }
        validateRegisteredPrecisionSupport(group.name, group.type, group.parameter_precision);
        const TensorContract::PrecisionType expected =
            TensorContract::precision_from_parameter_group_precision(group.parameter_precision);
        const TensorContract::PrecisionType actual = group.tensor->precision();
        if (actual != expected) {
            throw std::runtime_error("[buildParameterGroups] group " + group.name +
                                     " parameter_precision=" +
                                     parameterPrecisionSummaryName(group.parameter_precision) +
                                     " but TensorContract tensor metadata is " +
                                     TensorContract::precision_name(actual));
        }
    }
}

void emitGroupSummary(const std::vector<ParameterGroup>& groups) {
    constexpr size_t kParamGroupTypeCount = static_cast<size_t>(ParamGroupType::COUNT);
    constexpr size_t kPrecisionCount = 2;
    static_assert(kParamGroupTypeCount == 6,
                  "Registered group precision summary must list every ParamGroupType");

    const std::array<ParamGroupType, kParamGroupTypeCount> group_types = {
        ParamGroupType::EMBEDDING,
        ParamGroupType::LM_HEAD,
        ParamGroupType::ATTENTION,
        ParamGroupType::FFN,
        ParamGroupType::RMSNORM,
        ParamGroupType::ARG_SELECTOR
    };
    const std::array<ParameterGroupPrecision, kPrecisionCount> precisions = {
        ParameterGroupPrecision::FP32,
        ParameterGroupPrecision::BF16_COMPUTE
    };
    std::array<std::array<int, kPrecisionCount>, kParamGroupTypeCount> precision_counts{};

    int emb_count = 0;
    int attn_count = 0;
    int ffn_count = 0;
    int rms_count = 0;
    int other_count = 0;

    for (const auto& group : groups) {
        switch (group.type) {
            case ParamGroupType::EMBEDDING: ++emb_count; break;
            case ParamGroupType::LM_HEAD: ++other_count; break;
            case ParamGroupType::ATTENTION: ++attn_count; break;
            case ParamGroupType::FFN: ++ffn_count; break;
            case ParamGroupType::RMSNORM: ++rms_count; break;
            case ParamGroupType::ARG_SELECTOR: ++other_count; break;
            case ParamGroupType::COUNT:
                throw std::runtime_error("[buildParameterGroups] group " + group.name +
                                         " has invalid ParamGroupType::COUNT");
        }

        if (group.parameter_precision == ParameterGroupPrecision::UNSPECIFIED) {
            throw std::runtime_error("[buildParameterGroups] group " + group.name +
                                     " has UNSPECIFIED parameter_precision");
        }

        ++precision_counts[paramGroupTypeIndex(group.type)][parameterPrecisionIndex(group.parameter_precision)];
    }

    emitInfo("[buildParameterGroups] Parameter group summary: total=" + std::to_string(groups.size()) +
             " emb=" + std::to_string(emb_count) +
             " attn=" + std::to_string(attn_count) +
             " ffn=" + std::to_string(ffn_count) +
             " rms=" + std::to_string(rms_count) +
             " other=" + std::to_string(other_count));

    std::ostringstream precision_summary;
    precision_summary << "[buildParameterGroups] Registered group precision summary:";
    for (const ParamGroupType type : group_types) {
        const size_t type_index = paramGroupTypeIndex(type);
        int type_total = 0;
        for (const ParameterGroupPrecision precision : precisions) {
            type_total += precision_counts[type_index][parameterPrecisionIndex(precision)];
        }
        if (type_total == 0) {
            continue;
        }

        precision_summary << ' ' << paramGroupTypeSummaryName(type) << "{";
        for (size_t precision_index = 0; precision_index < precisions.size(); ++precision_index) {
            if (precision_index != 0) {
                precision_summary << ',';
            }
            const ParameterGroupPrecision precision = precisions[precision_index];
            precision_summary << parameterPrecisionSummaryName(precision) << '='
                              << precision_counts[type_index][precision_index];
        }
        precision_summary << '}';
    }
    emitInfo(precision_summary.str());
}

void validateParameterRegistrationConfig(const GRIM::Config::AiConfigSnapshot& config) {
    const int num_layers = GRIM::HyperParameters::snapshotTrainingConfigField<int>(config, "num_layers");
    const int vocab_size = GRIM::HyperParameters::snapshotTrainingConfigField<int>(config, "vocab_size");
    if (num_layers <= 0) {
        throw std::runtime_error("buildParameterGroups: num_layers must be > 0, got " +
                                 std::to_string(num_layers));
    }
    if (vocab_size <= 0) {
        throw std::runtime_error("buildParameterGroups: vocab_size must be > 0, got " +
                                 std::to_string(vocab_size));
    }
    GRIM::HyperParameters::validateParameterGroupPrecision(GRIM::HyperParameters::snapshotTrainingConfigField<ParameterGroupPrecision>(config, "parameter_precision_embedding"), "parameter_precision_embedding", "buildParameterGroups");
    GRIM::HyperParameters::validateParameterGroupPrecision(GRIM::HyperParameters::snapshotTrainingConfigField<ParameterGroupPrecision>(config, "parameter_precision_lm_head"), "parameter_precision_lm_head", "buildParameterGroups");
    GRIM::HyperParameters::validateParameterGroupPrecision(GRIM::HyperParameters::snapshotTrainingConfigField<ParameterGroupPrecision>(config, "parameter_precision_attention"), "parameter_precision_attention", "buildParameterGroups");
    GRIM::HyperParameters::validateParameterGroupPrecision(GRIM::HyperParameters::snapshotTrainingConfigField<ParameterGroupPrecision>(config, "parameter_precision_ffn"), "parameter_precision_ffn", "buildParameterGroups");
    GRIM::HyperParameters::validateParameterGroupPrecision(GRIM::HyperParameters::snapshotTrainingConfigField<ParameterGroupPrecision>(config, "parameter_precision_rmsnorm"), "parameter_precision_rmsnorm", "buildParameterGroups");
    GRIM::HyperParameters::validateParameterGroupPrecision(GRIM::HyperParameters::snapshotTrainingConfigField<ParameterGroupPrecision>(config, "parameter_precision_number_encoder"), "parameter_precision_number_encoder", "buildParameterGroups");
    GRIM::HyperParameters::validateParameterGroupPrecision(GRIM::HyperParameters::snapshotTrainingConfigField<ParameterGroupPrecision>(config, "parameter_precision_arg_selector"), "parameter_precision_arg_selector", "buildParameterGroups");
}

} // namespace

void initializeFeedForwardParameterTensors(
    std::vector<GRIM::FeedForwardParameterTensors>& feed_forward_parameter_tensors,
    const GRIM::HyperParameters::EncoderLayerConstructionHP& encoder_hp,
    std::uint64_t weight_init_seed,
    cudaStream_t init_stream) {
    if (!init_stream) {
        throw std::runtime_error("initializeFeedForwardParameterTensors: init_stream is NULL");
    }
    if (encoder_hp.num_layers <= 0) {
        throw std::runtime_error("initializeFeedForwardParameterTensors: encoder_hp.num_layers must be > 0, got " +
                                 std::to_string(encoder_hp.num_layers));
    }
    if (encoder_hp.d_model <= 0) {
        throw std::runtime_error("initializeFeedForwardParameterTensors: encoder_hp.d_model must be > 0, got " +
                                 std::to_string(encoder_hp.d_model));
    }
    if (encoder_hp.d_ff <= 0) {
        throw std::runtime_error("initializeFeedForwardParameterTensors: encoder_hp.d_ff must be > 0, got " +
                                 std::to_string(encoder_hp.d_ff));
    }
    if (!feed_forward_parameter_tensors.empty()) {
        throw std::runtime_error("initializeFeedForwardParameterTensors: registry FFN tensor vector is already initialized; refusing to allocate twice. size=" +
                                 std::to_string(feed_forward_parameter_tensors.size()));
    }

    const auto ffn_hp = GRIM::HyperParameters::feedForwardLayerConstructionHP(encoder_hp);
    const float residual_projection_init_gain = ffn_hp.residual_projection_init_gain;
    if (!std::isfinite(residual_projection_init_gain) || residual_projection_init_gain <= 0.0f) {
        throw std::runtime_error("initializeFeedForwardParameterTensors: residual_projection_init_gain must be positive finite");
    }
    if (!std::isfinite(ffn_hp.dropout_rate) || ffn_hp.dropout_rate < 0.0f || ffn_hp.dropout_rate >= 1.0f) {
        throw std::runtime_error("initializeFeedForwardParameterTensors: dropout_rate must be finite and in [0,1), got " +
                                 std::to_string(ffn_hp.dropout_rate));
    }

    feed_forward_parameter_tensors.resize(static_cast<std::size_t>(encoder_hp.num_layers));
    for (int layer = 0; layer < encoder_hp.num_layers; ++layer) {
        auto& tensors = feed_forward_parameter_tensors[static_cast<std::size_t>(layer)];
        const std::uint64_t ffn_seed = weight_init_seed + 4 + static_cast<std::uint64_t>(layer) * 10ULL;

        tensors.W_gate = GRIM::Tensor::zeros({ffn_hp.d_model, ffn_hp.d_ff}, init_stream, "ffn_w_gate");
        tensors.W_gate.requires_grad_();
        tensors.W_gate.alloc_grad();
        GRIM::Tensor::xavier_uniform_(tensors.W_gate, ffn_seed, init_stream);

        tensors.W1 = GRIM::Tensor::zeros({ffn_hp.d_model, ffn_hp.d_ff}, init_stream, "ffn_w1");
        tensors.W1.requires_grad_();
        tensors.W1.alloc_grad();
        GRIM::Tensor::xavier_uniform_(tensors.W1, ffn_seed + 1, init_stream);

        tensors.W2 = GRIM::Tensor::zeros({ffn_hp.d_ff, ffn_hp.d_model}, init_stream, "ffn_w2");
        tensors.W2.requires_grad_();
        tensors.W2.alloc_grad();
        GRIM::Tensor::xavier_uniform_with_gain_(tensors.W2, ffn_seed + 2, residual_projection_init_gain, init_stream);

        if (ffn_hp.output_bias_enabled) {
            tensors.b2 = GRIM::Tensor::zeros({1, ffn_hp.d_model}, init_stream, "ffn_b2");
            tensors.b2.requires_grad_();
            tensors.b2.alloc_grad();
        }
    }

    const cudaError_t sync_err = cudaStreamSynchronize(init_stream);
    if (sync_err != cudaSuccess) {
        throw std::runtime_error(std::string("initializeFeedForwardParameterTensors: cudaStreamSynchronize failed: ") +
                                 cudaGetErrorString(sync_err));
    }

    emitInfo("[initializeFeedForwardParameterTensors] Initialized registry-owned FFN tensors for " +
             std::to_string(encoder_hp.num_layers) + " layers");
}

void initializeEncodingLayerParameterTensors(
    std::vector<GRIM::EncodingLayerParameterTensors>& encoding_layer_parameter_tensors,
    const GRIM::HyperParameters::EncoderLayerConstructionHP& encoder_hp,
    std::uint64_t weight_init_seed,
    cudaStream_t init_stream) {
    if (!init_stream) {
        throw std::runtime_error("initializeEncodingLayerParameterTensors: init_stream is NULL");
    }
    if (encoder_hp.num_layers <= 0) {
        throw std::runtime_error("initializeEncodingLayerParameterTensors: encoder_hp.num_layers must be > 0, got " +
                                 std::to_string(encoder_hp.num_layers));
    }
    if (encoder_hp.d_model <= 0) {
        throw std::runtime_error("initializeEncodingLayerParameterTensors: encoder_hp.d_model must be > 0, got " +
                                 std::to_string(encoder_hp.d_model));
    }
    if (encoder_hp.qkv_dim <= 0) {
        throw std::runtime_error("initializeEncodingLayerParameterTensors: encoder_hp.qkv_dim must be > 0, got " +
                                 std::to_string(encoder_hp.qkv_dim));
    }
    if (!encoding_layer_parameter_tensors.empty()) {
        throw std::runtime_error("initializeEncodingLayerParameterTensors: registry encoder tensor vector is already initialized; refusing to allocate twice. size=" +
                                 std::to_string(encoding_layer_parameter_tensors.size()));
    }

    const float residual_projection_init_gain = encoder_hp.residual_projection_init_gain;
    if (!std::isfinite(residual_projection_init_gain) || residual_projection_init_gain <= 0.0f) {
        throw std::runtime_error("initializeEncodingLayerParameterTensors: residual_projection_init_gain must be positive finite");
    }
    if (!std::isfinite(encoder_hp.layer_scale_init)) {
        throw std::runtime_error("initializeEncodingLayerParameterTensors: layer_scale_init must be finite");
    }

    encoding_layer_parameter_tensors.resize(static_cast<std::size_t>(encoder_hp.num_layers));
    std::vector<float> ones(static_cast<std::size_t>(encoder_hp.d_model), 1.0f);
    std::vector<float> layer_scale_init(static_cast<std::size_t>(encoder_hp.d_model), encoder_hp.layer_scale_init);

    for (int layer = 0; layer < encoder_hp.num_layers; ++layer) {
        auto& tensors = encoding_layer_parameter_tensors[static_cast<std::size_t>(layer)];
        const std::uint64_t layer_seed = weight_init_seed + 2 + static_cast<std::uint64_t>(layer) * 10ULL;

        tensors.rms1_gamma = GRIM::Tensor::zeros({encoder_hp.d_model}, init_stream, "enc_rms1_gamma");
        if (!encoder_hp.freeze_learned_rms_gammas) {
            tensors.rms1_gamma.requires_grad_();
            tensors.rms1_gamma.alloc_grad();
        }
        cudaError_t copy_err = cudaMemcpyAsync(
            tensors.rms1_gamma.data,
            ones.data(),
            static_cast<std::size_t>(encoder_hp.d_model) * sizeof(float),
            cudaMemcpyHostToDevice,
            init_stream);
        if (copy_err != cudaSuccess) {
            throw std::runtime_error(std::string("initializeEncodingLayerParameterTensors: cudaMemcpyAsync failed for rms1_gamma: ") +
                                     cudaGetErrorString(copy_err));
        }

        tensors.rms2_gamma = GRIM::Tensor::zeros({encoder_hp.d_model}, init_stream, "enc_rms2_gamma");
        if (!encoder_hp.freeze_learned_rms_gammas) {
            tensors.rms2_gamma.requires_grad_();
            tensors.rms2_gamma.alloc_grad();
        }
        copy_err = cudaMemcpyAsync(
            tensors.rms2_gamma.data,
            ones.data(),
            static_cast<std::size_t>(encoder_hp.d_model) * sizeof(float),
            cudaMemcpyHostToDevice,
            init_stream);
        if (copy_err != cudaSuccess) {
            throw std::runtime_error(std::string("initializeEncodingLayerParameterTensors: cudaMemcpyAsync failed for rms2_gamma: ") +
                                     cudaGetErrorString(copy_err));
        }

        tensors.W_qkv = GRIM::Tensor::zeros({encoder_hp.qkv_dim, encoder_hp.d_model}, init_stream, "enc_W_qkv");
        tensors.W_qkv.requires_grad_();
        tensors.W_qkv.alloc_grad();
        GRIM::Tensor::xavier_uniform_(tensors.W_qkv, layer_seed + 0, init_stream);

        tensors.W_o = GRIM::Tensor::zeros({encoder_hp.d_model, encoder_hp.d_model}, init_stream, "enc_W_o");
        tensors.W_o.requires_grad_();
        tensors.W_o.alloc_grad();
        GRIM::Tensor::xavier_uniform_with_gain_(tensors.W_o, layer_seed + 1, residual_projection_init_gain, init_stream);

        if (encoder_hp.attention_qkv_bias_enabled) {
            tensors.b_qkv = GRIM::Tensor::zeros({encoder_hp.qkv_dim}, init_stream, "enc_b_qkv");
            tensors.b_qkv.requires_grad_();
            tensors.b_qkv.alloc_grad();
        }

        if (encoder_hp.attention_output_bias_enabled) {
            tensors.b_o = GRIM::Tensor::zeros({encoder_hp.d_model}, init_stream, "enc_b_o");
            tensors.b_o.requires_grad_();
            tensors.b_o.alloc_grad();
        }

        if (encoder_hp.attention_residual_gate_enabled) {
            // The forward gate will use 2 * sigmoid(logit), so zero logits
            // initialize the residual multiplier to exactly 1.0.
            tensors.attention_residual_gate.W_gate = GRIM::Tensor::zeros(
                {encoder_hp.d_model, 1}, init_stream, "enc_attention_residual_gate_W");
            tensors.attention_residual_gate.W_gate.requires_grad_();
            tensors.attention_residual_gate.W_gate.alloc_grad();

            tensors.attention_residual_gate.b_gate = GRIM::Tensor::zeros(
                {1}, init_stream, "enc_attention_residual_gate_b");
            tensors.attention_residual_gate.b_gate.requires_grad_();
            tensors.attention_residual_gate.b_gate.alloc_grad();
        }

        if (encoder_hp.use_layer_scale) {
            tensors.layer_scale1 = GRIM::Tensor::zeros({1, encoder_hp.d_model}, init_stream, "enc_layer_scale1");
            tensors.layer_scale1.requires_grad_();
            tensors.layer_scale1.alloc_grad();
            copy_err = cudaMemcpyAsync(
                tensors.layer_scale1.data,
                layer_scale_init.data(),
                static_cast<std::size_t>(encoder_hp.d_model) * sizeof(float),
                cudaMemcpyHostToDevice,
                init_stream);
            if (copy_err != cudaSuccess) {
                throw std::runtime_error(std::string("initializeEncodingLayerParameterTensors: cudaMemcpyAsync failed for layer_scale1: ") +
                                         cudaGetErrorString(copy_err));
            }

            tensors.layer_scale2 = GRIM::Tensor::zeros({1, encoder_hp.d_model}, init_stream, "enc_layer_scale2");
            tensors.layer_scale2.requires_grad_();
            tensors.layer_scale2.alloc_grad();
            copy_err = cudaMemcpyAsync(
                tensors.layer_scale2.data,
                layer_scale_init.data(),
                static_cast<std::size_t>(encoder_hp.d_model) * sizeof(float),
                cudaMemcpyHostToDevice,
                init_stream);
            if (copy_err != cudaSuccess) {
                throw std::runtime_error(std::string("initializeEncodingLayerParameterTensors: cudaMemcpyAsync failed for layer_scale2: ") +
                                         cudaGetErrorString(copy_err));
            }
        }
    }

    const cudaError_t sync_err = cudaStreamSynchronize(init_stream);
    if (sync_err != cudaSuccess) {
        throw std::runtime_error(std::string("initializeEncodingLayerParameterTensors: cudaStreamSynchronize failed: ") +
                                 cudaGetErrorString(sync_err));
    }

    emitInfo("[initializeEncodingLayerParameterTensors] Initialized registry-owned encoder tensors for " +
             std::to_string(encoder_hp.num_layers) + " layers");
}

void initializeEmbeddingParameterTensors(
    ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const GRIM::HyperParameters::EmbeddingLayerConstructionHP& embedding_hp,
    std::uint64_t weight_init_seed,
    cudaStream_t init_stream,
    bool requires_grad) {
    if (!init_stream) {
        throw std::runtime_error("initializeEmbeddingParameterTensors: init_stream is NULL");
    }
    if (embedding_hp.vocab_size <= 0) {
        throw std::runtime_error("initializeEmbeddingParameterTensors: vocab_size must be positive, got " +
                                 std::to_string(embedding_hp.vocab_size));
    }
    if (embedding_hp.d_model <= 0) {
        throw std::runtime_error("initializeEmbeddingParameterTensors: d_model must be positive, got " +
                                 std::to_string(embedding_hp.d_model));
    }
    if (parameter_registry.getEmbeddingParameters()) {
        throw std::runtime_error("initializeEmbeddingParameterTensors: registry embedding tensor owner is already initialized");
    }

    parameter_registry.embedding_parameters = std::make_unique<GRIM::EmbeddingParameterTensors>();
    auto& embedding_parameters = *parameter_registry.embedding_parameters;
    embedding_parameters.token_weights = Tensor::zeros(
        {embedding_hp.vocab_size, embedding_hp.d_model},
        init_stream,
        "embedding.token_weights");
    if (requires_grad) {
        embedding_parameters.token_weights.requires_grad_();
        embedding_parameters.token_weights.alloc_grad();
    }
    Tensor::xavier_uniform_(embedding_parameters.token_weights, weight_init_seed, init_stream);

    emitInfo("[initializeEmbeddingParameterTensors] Initialized registry-owned embedding tensor [" +
             std::to_string(embedding_hp.vocab_size) + ", " +
             std::to_string(embedding_hp.d_model) + "]");
}

void initializeLmHeadParameterTensors(
    ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const GRIM::HyperParameters::LMHeadLayerConstructionHP& lm_head_hp,
    std::uint64_t weight_init_seed,
    cudaStream_t init_stream,
    GRIM::Tensor* tied_embedding_weights,
    const OutputUnigramPriorView* output_unigram_prior) {
    if (!init_stream) {
        throw std::runtime_error("initializeLmHeadParameterTensors: init_stream is NULL");
    }
    if (lm_head_hp.d_model <= 0) {
        throw std::runtime_error("initializeLmHeadParameterTensors: d_model must be positive, got " +
                                 std::to_string(lm_head_hp.d_model));
    }
    if (lm_head_hp.vocab_size <= 0) {
        throw std::runtime_error("initializeLmHeadParameterTensors: vocab_size must be positive, got " +
                                 std::to_string(lm_head_hp.vocab_size));
    }
    if (parameter_registry.getLmHeadParameters()) {
        throw std::runtime_error("initializeLmHeadParameterTensors: registry LM-head tensor owner is already initialized");
    }
    if (lm_head_hp.tie_embeddings != (tied_embedding_weights != nullptr)) {
        throw std::runtime_error(
            "initializeLmHeadParameterTensors: tie_embeddings grouping/runtime ownership mismatch. tie_embeddings=" +
            std::string(lm_head_hp.tie_embeddings ? "true" : "false") +
            " tied_embedding_weights=" +
            std::string(tied_embedding_weights ? "non-null" : "NULL"));
    }

    if (lm_head_hp.mlp_enabled && lm_head_hp.mlp_d_ff <= 0) {
        throw std::runtime_error("initializeLmHeadParameterTensors: lm_head_mlp_enabled=true requires lm_head_mlp_d_ff > 0, got " +
                                 std::to_string(lm_head_hp.mlp_d_ff));
    }

    parameter_registry.lm_head_parameters = std::make_unique<GRIM::LMHeadParameterTensors>();
    auto& parameter_tensors = *parameter_registry.lm_head_parameters;
    parameter_tensors.weights = Tensor();
    parameter_tensors.bias = Tensor();
    parameter_tensors.final_rms_gamma = Tensor();
    parameter_tensors.mlp_W_gate = Tensor();
    parameter_tensors.mlp_W_up = Tensor();
    parameter_tensors.mlp_W_down = Tensor();
    parameter_tensors.owns_weights = !tied_embedding_weights;

    if (tied_embedding_weights) {
        if (!tied_embedding_weights->data) {
            throw std::runtime_error(
                "initializeLmHeadParameterTensors: tied embedding weights are NULL — embedding MUST be initialized before LM head");
        }
        if (!tied_embedding_weights->shape.is_2d_layout()) {
            throw std::runtime_error(
                "initializeLmHeadParameterTensors: tied embedding weights must be 2D [vocab_size,d_model]");
        }
        const auto& embedding_shape = tied_embedding_weights->shape.as_2d();
        if (embedding_shape.rows != lm_head_hp.vocab_size || embedding_shape.cols != lm_head_hp.d_model) {
            throw std::runtime_error(
                "initializeLmHeadParameterTensors: tied embedding shape mismatch. Expected [" +
                std::to_string(lm_head_hp.vocab_size) + "," + std::to_string(lm_head_hp.d_model) +
                "], got [" + std::to_string(embedding_shape.rows) + "," +
                std::to_string(embedding_shape.cols) + "]");
        }

        parameter_tensors.weights = Tensor::from_ptr(
            tied_embedding_weights->data,
            tied_embedding_weights->shape,
            false,
            true,
            "lm_head.weights_tied");
        parameter_tensors.weights.share_grad(*tied_embedding_weights);
        parameter_tensors.weights.owns_data = false;
        parameter_tensors.weights.requires_grad = true;
    } else {
        parameter_tensors.weights = Tensor::zeros(
            {lm_head_hp.vocab_size, lm_head_hp.d_model},
            init_stream,
            "lm_head.weights");
        parameter_tensors.weights.requires_grad_();
        parameter_tensors.weights.alloc_grad();
        Tensor::xavier_uniform_(parameter_tensors.weights, weight_init_seed, init_stream);
    }

    if (lm_head_hp.bias_enabled) {
        parameter_tensors.bias = Tensor::zeros({lm_head_hp.vocab_size}, init_stream, "lm_head.bias");
        parameter_tensors.bias.requires_grad_();
        parameter_tensors.bias.alloc_grad();
        if (output_unigram_prior) {
            if (!lm_head_hp.unigram_bias) {
                throw std::runtime_error(
                    "initializeLmHeadParameterTensors: output unigram prior was provided while lm_head_unigram_bias=false");
            }
            uploadOutputUnigramPriorToBias(
                parameter_tensors.bias,
                *output_unigram_prior,
                lm_head_hp.vocab_size,
                init_stream,
                "initializeLmHeadParameterTensors",
                "lm_head");
            const cudaError_t sync_err = cudaStreamSynchronize(init_stream);
            if (sync_err != cudaSuccess) {
                throw std::runtime_error(std::string("initializeLmHeadParameterTensors: cudaStreamSynchronize failed after output unigram prior upload: ") +
                                         cudaGetErrorString(sync_err));
            }
        }
    } else if (output_unigram_prior) {
        throw std::runtime_error(
            "initializeLmHeadParameterTensors: output unigram prior requires lm_head_bias_enabled=true");
    }

    parameter_tensors.final_rms_gamma = Tensor::zeros({lm_head_hp.d_model}, init_stream, "final_rms_gamma");
    if (!lm_head_hp.freeze_learned_rms_gammas) {
        parameter_tensors.final_rms_gamma.requires_grad_();
        parameter_tensors.final_rms_gamma.alloc_grad();
    }

    std::vector<float> ones(static_cast<std::size_t>(lm_head_hp.d_model), 1.0f);
    cudaMemcpyAsync(parameter_tensors.final_rms_gamma.data,
                    ones.data(),
                    static_cast<std::size_t>(lm_head_hp.d_model) * sizeof(float),
                    cudaMemcpyHostToDevice,
                    init_stream);

    // Head-side residual SwiGLU adapter (capacity expansion, config-gated):
    //   u = z + mlp_alpha * (SiLU(z @ W_gate) ⊙ (z @ W_up)) @ W_down
    // W_down is deliberately ZERO-initialized so the head is exactly the
    // existing linear head at step 0 (adapter branch contributes nothing);
    // W_down's own gradient is nonzero from the first backward, so the branch
    // opens gradually instead of perturbing logit scale at init.
    if (lm_head_hp.mlp_enabled) {
        const std::uint64_t mlp_seed = weight_init_seed + 100;

        parameter_tensors.mlp_W_gate = Tensor::zeros(
            {lm_head_hp.d_model, lm_head_hp.mlp_d_ff}, init_stream, "lm_head.mlp_W_gate");
        parameter_tensors.mlp_W_gate.requires_grad_();
        parameter_tensors.mlp_W_gate.alloc_grad();
        Tensor::xavier_uniform_(parameter_tensors.mlp_W_gate, mlp_seed, init_stream);

        parameter_tensors.mlp_W_up = Tensor::zeros(
            {lm_head_hp.d_model, lm_head_hp.mlp_d_ff}, init_stream, "lm_head.mlp_W_up");
        parameter_tensors.mlp_W_up.requires_grad_();
        parameter_tensors.mlp_W_up.alloc_grad();
        Tensor::xavier_uniform_(parameter_tensors.mlp_W_up, mlp_seed + 1, init_stream);

        parameter_tensors.mlp_W_down = Tensor::zeros(
            {lm_head_hp.mlp_d_ff, lm_head_hp.d_model}, init_stream, "lm_head.mlp_W_down");
        parameter_tensors.mlp_W_down.requires_grad_();
        parameter_tensors.mlp_W_down.alloc_grad();
    }

    emitInfo("[initializeLmHeadParameterTensors] Initialized registry-owned LM-head tensors" +
             std::string(lm_head_hp.mlp_enabled
                 ? " (+ residual SwiGLU adapter [" + std::to_string(lm_head_hp.d_model) + "x" +
                   std::to_string(lm_head_hp.mlp_d_ff) + "], alpha=" + std::to_string(lm_head_hp.mlp_alpha) + ")"
                 : "") +
             std::string(output_unigram_prior ? " with output unigram bias prior" : ""));
}

void initializeAtomInsertionBoundaryParameterTensors(
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const GRIM::HyperParameters::AtomInsertionBoundaryProjectionHP& atom_hp,
    std::uint64_t weight_init_seed,
    cudaStream_t init_stream) {
    if (!atom_hp.enabled) {
        if (parameter_registry.getAtomInsertionBoundaryParameters()) {
            throw std::runtime_error(
                "initializeAtomInsertionBoundaryParameterTensors: atom insertion "
                "is disabled but the registry owner already exists");
        }
        return;
    }
    if (!init_stream) {
        throw std::runtime_error(
            "initializeAtomInsertionBoundaryParameterTensors: init_stream is NULL");
    }
    if (atom_hp.d_model <= 0) {
        throw std::runtime_error(
            "initializeAtomInsertionBoundaryParameterTensors: d_model must be positive, got " +
            std::to_string(atom_hp.d_model));
    }
    if (parameter_registry.getAtomInsertionBoundaryParameters()) {
        throw std::runtime_error(
            "initializeAtomInsertionBoundaryParameterTensors: registry owner is already initialized");
    }

    auto parameters = std::make_unique<
        GRIM::AtomInsertionBoundaryParameterTensors>();
    auto make_xavier = [&](const char* name, std::uint64_t seed) -> GRIM::Tensor {
        GRIM::Tensor tensor = GRIM::Tensor::zeros(
            {atom_hp.d_model, atom_hp.d_model}, init_stream, name);
        tensor.requires_grad_();
        tensor.alloc_grad();
        GRIM::Tensor::xavier_uniform_(tensor, seed, init_stream);
        return tensor;
    };

    parameters->left_projection_weight = make_xavier(
        "atom_insertion.left_projection_weight", weight_init_seed);
    parameters->right_projection_weight = make_xavier(
        "atom_insertion.right_projection_weight", weight_init_seed + 1);
    parameters->projection_bias = GRIM::Tensor::zeros(
        {1, atom_hp.d_model},
        init_stream,
        "atom_insertion.projection_bias");
    parameters->projection_bias.requires_grad_();
    parameters->projection_bias.alloc_grad();

    validateAtomInsertionBoundaryParameterTensors(
        *parameters,
        atom_hp,
        "initializeAtomInsertionBoundaryParameterTensors");

    const cudaError_t sync_err = cudaStreamSynchronize(init_stream);
    if (sync_err != cudaSuccess) {
        throw std::runtime_error(
            std::string(
                "initializeAtomInsertionBoundaryParameterTensors: "
                "cudaStreamSynchronize failed: ") +
            cudaGetErrorString(sync_err));
    }

    parameter_registry.atom_insertion_boundary_parameters = std::move(parameters);
    emitInfo(
        "[initializeAtomInsertionBoundaryParameterTensors] Initialized registry-owned "
        "atom boundary tensors (d_model=" + std::to_string(atom_hp.d_model) + ")");
}

void initializeSelectorParameterTensors(
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    bool selector_enabled,
    int d_model,
    std::uint64_t weight_init_seed,
    cudaStream_t init_stream) {
    if (!selector_enabled) {
        if (parameter_registry.getSelectorParameters()) {
            throw std::runtime_error("initializeSelectorParameterTensors: selector disabled but registry owner already exists");
        }
        return;
    }
    if (!init_stream) {
        throw std::runtime_error("initializeSelectorParameterTensors: init_stream is NULL");
    }
    if (parameter_registry.getSelectorParameters()) {
        throw std::runtime_error("initializeSelectorParameterTensors: registry selector tensor owner is already initialized");
    }
    if (d_model <= 0) {
        throw std::runtime_error("initializeSelectorParameterTensors: d_model must be > 0, got " +
                                 std::to_string(d_model));
    }

    auto params = std::make_unique<GRIM::SelectorParameterTensors>();
    GRIM::Tensor w_q = GRIM::Tensor::zeros({d_model, d_model}, init_stream, "selector.W_q");
    w_q.requires_grad_();
    w_q.alloc_grad();
    GRIM::Tensor::xavier_uniform_(w_q, weight_init_seed, init_stream);
    params->W_q = std::move(w_q);

    const cudaError_t sync_err = cudaStreamSynchronize(init_stream);
    if (sync_err != cudaSuccess) {
        throw std::runtime_error(std::string("initializeSelectorParameterTensors: cudaStreamSynchronize failed: ") +
                                 cudaGetErrorString(sync_err));
    }

    parameter_registry.selector_parameters = std::move(params);
    emitInfo("[initializeSelectorParameterTensors] Initialized registry-owned selector W_q (d_model=" +
             std::to_string(d_model) + ")");
}

void initializeLocalAtomRetrievalParameterTensors(
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    int d_model,
    std::uint64_t weight_init_seed,
    cudaStream_t init_stream) {
    if (!init_stream) {
        throw std::runtime_error(
            "initializeLocalAtomRetrievalParameterTensors: init_stream is NULL");
    }
    if (parameter_registry.getLocalAtomRetrievalParameters()) {
        throw std::runtime_error(
            "initializeLocalAtomRetrievalParameterTensors: registry owner is already initialized");
    }
    if (d_model <= 0) {
        throw std::runtime_error(
            "initializeLocalAtomRetrievalParameterTensors: d_model must be > 0, got " +
            std::to_string(d_model));
    }

    auto params = std::make_unique<GRIM::LocalAtomRetrievalParameterTensors>();
    params->type_no_reference_key = GRIM::Tensor::zeros(
        {GRIM::Tokenizer::kAtomTypeCount, d_model},
        init_stream,
        "local_atom_retrieval.type_no_reference_key");
    params->type_no_reference_key.requires_grad_();
    params->type_no_reference_key.alloc_grad();
    GRIM::Tensor::xavier_uniform_(
        params->type_no_reference_key,
        weight_init_seed,
        init_stream);

    const cudaError_t sync_err = cudaStreamSynchronize(init_stream);
    if (sync_err != cudaSuccess) {
        throw std::runtime_error(
            std::string(
                "initializeLocalAtomRetrievalParameterTensors: "
                "cudaStreamSynchronize failed: ") +
            cudaGetErrorString(sync_err));
    }

    parameter_registry.local_atom_retrieval_parameters = std::move(params);
    emitInfo(
        "[initializeLocalAtomRetrievalParameterTensors] Initialized registry-owned "
        "NO_REFERENCE keys (atom_types=" +
        std::to_string(GRIM::Tokenizer::kAtomTypeCount) +
        ", d_model=" + std::to_string(d_model) + ")");
}

void initializeLoRAParameterTensors(
    ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const GRIM::HyperParameters::EncoderLayerConstructionHP& encoder_hp,
    const GRIM::HyperParameters::LoRATrainingHP& lora_hp,
    std::uint64_t weight_init_seed,
    cudaStream_t init_stream) {
    if (!init_stream) {
        throw std::runtime_error("initializeLoRAParameterTensors: init_stream is NULL");
    }
    if (encoder_hp.num_layers <= 0 || encoder_hp.d_model <= 0 ||
        encoder_hp.qkv_dim <= 0 || encoder_hp.d_ff <= 0) {
        throw std::runtime_error(
            "initializeLoRAParameterTensors: encoder dimensions must be positive");
    }
    if (!parameter_registry.loraLayerParameterPairs().empty()) {
        throw std::runtime_error(
            "initializeLoRAParameterTensors: registry LoRA owner is already initialized");
    }
    if (!std::isfinite(lora_hp.learning_rate_lora) ||
        lora_hp.learning_rate_lora <= 0.0f) {
        throw std::runtime_error(
            "initializeLoRAParameterTensors: learning_rate_lora must be positive and finite");
    }

    const auto specs = makeLoRAMatrixSpecs(encoder_hp, lora_hp);
    bool any_enabled = false;
    for (const auto& spec : specs) {
        if (spec.settings->precision != ParameterGroupPrecision::FP32) {
            throw std::runtime_error(
                std::string("initializeLoRAParameterTensors: ") + spec.projection_name +
                " precision must be FP32 in v1");
        }
        if (!spec.settings->enabled) {
            continue;
        }
        any_enabled = true;
        if (spec.settings->rank == 0 ||
            spec.settings->rank > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
            static_cast<int>(spec.settings->rank) > std::min(spec.a_cols, spec.b_rows)) {
            throw std::runtime_error(
                std::string("initializeLoRAParameterTensors: invalid rank for ") +
                spec.projection_name);
        }
        if (!std::isfinite(spec.settings->alpha) || spec.settings->alpha <= 0.0f) {
            throw std::runtime_error(
                std::string("initializeLoRAParameterTensors: invalid alpha for ") +
                spec.projection_name);
        }
    }
    if (!any_enabled) {
        throw std::runtime_error(
            "initializeLoRAParameterTensors: no LoRA matrix class is enabled");
    }

    parameter_registry.loraLayerParameterPairs().resize(
        static_cast<std::size_t>(encoder_hp.num_layers));
    for (int layer = 0; layer < encoder_hp.num_layers; ++layer) {
        auto& layer_pairs = parameter_registry.loraLayerParameterPairs()[
            static_cast<std::size_t>(layer)];
        for (const auto& spec : specs) {
            if (!spec.settings->enabled) {
                continue;
            }
            auto pair = std::make_unique<GRIM::LoRAParameterPair>();
            pair->rank = spec.settings->rank;
            pair->alpha = spec.settings->alpha;
            pair->scale = pair->alpha / static_cast<float>(pair->rank);
            pair->matrix_class = spec.matrix_class;
            pair->precision = spec.settings->precision;
            if (!std::isfinite(pair->scale) || pair->scale <= 0.0f) {
                throw std::runtime_error(
                    std::string("initializeLoRAParameterTensors: invalid scale for ") +
                    spec.projection_name);
            }

            pair->A = Tensor::zeros(
                {spec.a_rows, spec.a_cols}, init_stream, spec.tensor_a_name);
            pair->A.requires_grad_();
            pair->A.alloc_grad();
            Tensor::xavier_uniform_(
                pair->A,
                deriveLoRATargetSeed(weight_init_seed, layer, spec.matrix_class),
                init_stream);

            pair->B = Tensor::zeros(
                {spec.b_rows, spec.b_cols}, init_stream, spec.tensor_b_name);
            pair->B.requires_grad_();
            pair->B.alloc_grad();

            layer_pairs.pairs[ParameterRegistry::loraMatrixClassIndex(spec.matrix_class)] =
                std::move(pair);
        }
    }

    const cudaError_t sync_err = cudaStreamSynchronize(init_stream);
    if (sync_err != cudaSuccess) {
        throw std::runtime_error(
            std::string("initializeLoRAParameterTensors: cudaStreamSynchronize failed: ") +
            cudaGetErrorString(sync_err));
    }
    emitInfo("[initializeLoRAParameterTensors] Initialized registry-owned LoRA tensors for " +
             std::to_string(encoder_hp.num_layers) + " layers");
}

void buildParameterGroups(const GRIM::Config::AiConfigSnapshot& config,
                          Startup::GpuModelState& gpu_model_state,
                          ParameterRegistry::StartupParameterRegistry& parameter_registry) {
    validateParameterRegistrationConfig(config);

    auto& registry_groups = parameter_registry.parameterGroups();
    std::vector<ParameterGroup> rebuilt_groups;
    rebuilt_groups.reserve(registry_groups.size());

    Registrar registrar(rebuilt_groups, config);
    const bool lora_model =
        GRIM::HyperParameters::snapshotTrainingConfigField<bool>(config, "lora_model");
    if (lora_model) {
        validateBaseParametersFrozen(parameter_registry);
        registerLoRAParameters(parameter_registry, registrar, config);
    } else {
        if (!parameter_registry.loraLayerParameterPairs().empty()) {
            throw std::runtime_error(
                "[buildParameterGroups] LoRA tensor owner exists while lora_model=false");
        }
        registerTopLevelParameters(gpu_model_state, parameter_registry, registrar, config);
        registerAtomInsertionBoundaryParameters(parameter_registry, registrar, config);
        registerEncoderParameters(gpu_model_state, parameter_registry, registrar, config);
        registerLocalAtomRetrievalParameters(parameter_registry, registrar, config);
    }
    validateRegisteredTensorPrecisionMetadata(rebuilt_groups);
    clearOptimizerBindings(rebuilt_groups);

    // Transaction boundary: parameter_registry.parameterGroups() is replaced only after the
    // complete configured inventory has been discovered and validated. A thrown
    // registration check must never leave StartupParameterRegistry with a half-built group
    // vector that downstream optimizer/checkpoint code could observe.
    registry_groups.swap(rebuilt_groups);

    emitInfo("[buildParameterGroups] Built " + std::to_string(registry_groups.size()) + " parameter groups");
    emitGroupSummary(registry_groups);
}

void bindOptimizerState(ParameterRegistry::StartupParameterRegistry& parameter_registry,
                        GRIM::OptimizerState& optimizer_state,
                        cudaStream_t stream) {
    auto& groups = parameter_registry.requireParameterGroups("bindOptimizerState");
    if (stream == nullptr) {
        throw std::runtime_error("[bindOptimizerState] stream is NULL - caller MUST provide valid CUDA stream");
    }

    std::vector<size_t> sizes;
    sizes.reserve(groups.size());
    for (const auto& group : groups) {
        sizes.push_back(group.size());
    }

    optimizer_state.allocate(sizes, stream);
    validateOptimizerStateAllocation(groups, optimizer_state);

    emitInfo("[bindOptimizerState] Allocated optimizer state tensors for " +
             std::to_string(groups.size()) + " parameter groups");
    for (size_t i = 0; i < groups.size(); ++i) {
        groups[i].m_tensor = &optimizer_state.m_states[i];
        groups[i].v_tensor = &optimizer_state.v_states[i];
    }
    emitInfo("[bindOptimizerState] Bound optimizer state tensors to parameter groups");
}

#endif // USE_CUDA

} // namespace GRIMText::Training::Startup::ModelRegistration

