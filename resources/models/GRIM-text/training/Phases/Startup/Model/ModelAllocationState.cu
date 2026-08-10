#include "ModelAllocationState.hpp"

#include "ModelGpuAssembly.hpp"
#include "ParameterGroupRegistration.hpp"

#include "../../Phase1_Startup.hpp"

#include "../../../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../../../../Shared/LogRecorder/LogRecorder.hpp"

#include <stdexcept>
#include <string>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace GRIMText::Training {

namespace Internal {

void initializeModel(
    const GRIM::Config::AiConfigSnapshot& config_snapshot,
    std::uint32_t actual_vocab_size,
    std::uint64_t weight_init_seed,
    GRIM::TrainingState& training_state,
    GRIM::GenerationState& generation_state,
    GRIM::PBM::PBMStateOwner& pbm_owner,
    Startup::GpuModelState& gpu_model_state,
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const GRIMText::Training::Startup::ModelRegistration::OutputUnigramPriorView* output_unigram_prior,
    TrainingLogger& logger)
{
    logger.log("Initializing model with weight_init_seed=" + std::to_string(weight_init_seed) + "...");
    logger.log("Model inputs prepared: actual_vocab_size=" +
               std::to_string(actual_vocab_size));

    if (GRIM::HyperParameters::snapshotTrainingConfigField<GRIM::HyperParameters::HardcodedPattern>(config_snapshot, "hardcoded_hidden_pattern") != GRIM::HyperParameters::HardcodedPattern::DISABLED) {
        logger.log("⚠️  Hardcoded hidden-state diagnostic mode is active; exact config values are listed by ConfigDump.");
        logger.log("⚠️  Encoder output will be REPLACED with synthetic patterns - this is a DIAGNOSTIC MODE ONLY!");
    }

    {
        logger.log("Initializing CUDA device context...");
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("FATAL: cudaSetDevice(0) failed: ") + cudaGetErrorString(err));
        }

        err = cudaFree(0);
        if (err != cudaSuccess && err != cudaErrorInvalidValue) {
            throw std::runtime_error(std::string("FATAL: CUDA context creation failed: ") + cudaGetErrorString(err));
        }

        int device;
        err = cudaGetDevice(&device);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("FATAL: cudaGetDevice() failed: ") + cudaGetErrorString(err));
        }

        cudaDeviceProp props;
        err = cudaGetDeviceProperties(&props, device);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("FATAL: cudaGetDeviceProperties() failed: ") + cudaGetErrorString(err));
        }

        logger.log("✓ CUDA device initialized: " + std::string(props.name));
    }

    {
        GRIM::StreamControllerConfig stream_config;
        stream_config.verbose = true;

        if (!training_state.stream_ctrl.initialize(stream_config)) {
            throw std::runtime_error("FATAL: Failed to initialize StreamController");
        }
        logger.log("✓ StreamController initialized");
    }

    logger.log("Initializing cuBLAS handle...");
    GRIMText::Training::Startup::initializeCuBLASHandle(training_state);
    logger.log("✓ cuBLAS handle initialized with Tensor Core acceleration");

    logger.log("Initializing RoPE (required before encoder construction)...");
    GRIMText::Training::Startup::initializePBM(
        config_snapshot,
        training_state,
        pbm_owner);
    logger.log("✓ RoPE initialized");

    logger.log("Assembling GPU model layers with weight_init_seed=" + std::to_string(weight_init_seed) + "...");
    GRIMText::Training::Startup::assembleGpuModel(
        config_snapshot,
        training_state,
        pbm_owner,
        gpu_model_state,
        parameter_registry,
        weight_init_seed,
        output_unigram_prior);
    logger.log("✓ GPU model layers fully assembled");

    logger.log("Registering trainable parameter groups...");
    GRIMText::Training::Startup::ModelRegistration::buildParameterGroups(config_snapshot, gpu_model_state, parameter_registry);
    logger.log("✓ Trainable parameter groups registered and verified");

    const auto execution_mode = GRIM::HyperParameters::snapshotExecutionMode(config_snapshot);
    if (execution_mode == GRIM::HyperParameters::ModelExecutionMode::TRAINING) {
        logger.log("Initializing TrainingState runtime workspaces...");
        GRIMText::Training::Startup::initializeTrainingRuntime(
            training_state,
            pbm_owner);
        logger.log("✓ TrainingState fully initialized");
    } else if (execution_mode == GRIM::HyperParameters::ModelExecutionMode::INFERENCE) {
        logger.log("Initializing inference runtime workspaces...");
        GRIMText::Training::Startup::initializeInferenceRuntime(
            config_snapshot,
            training_state,
            generation_state,
            pbm_owner,
            gpu_model_state,
            parameter_registry);
        logger.log("✓ Inference runtime fully initialized");
    } else {
        throw std::runtime_error("initializeModel: unsupported execution_mode");
    }

    logger.log("✓ Model initialized");
}

} // namespace Internal

void ModelAllocated(
    TrainingContext& ctx,
    std::uint32_t actual_vocab_size,
    std::uint64_t weight_init_seed) {
    using GRIM::Logging::EmitModuleError;
    using GRIM::Logging::EmitModuleInfo;
    using GRIM::Logging::ModuleId;

    const int vocab_size = GRIM::HyperParameters::snapshotTrainingConfigField<int>(ctx.config, "vocab_size");
    if (actual_vocab_size == 0) {
        throw std::runtime_error(
            "FATAL: actual_vocab_size is zero before model initialization");
    }
    if (vocab_size != static_cast<int>(actual_vocab_size)) {
        throw std::runtime_error(
            "FATAL: runtime model vocab_size drifted from startup fact: actual_vocab_size=" +
            std::to_string(actual_vocab_size) +
            " config.vocab_size=" + std::to_string(vocab_size));
    }

    if (ctx.data.vocab_size != 0 && ctx.data.vocab_size != actual_vocab_size) {
        throw std::runtime_error(
            "FATAL: actual_vocab_size drifted from SequenceData.vocab_size (startup=" +
            std::to_string(actual_vocab_size) +
            " sequence_data=" + std::to_string(ctx.data.vocab_size) + ")");
    }
    if (weight_init_seed == 0) {
        throw std::runtime_error("FATAL: weight_init_seed is zero before model initialization");
    }

    GRIMText::Training::Startup::ModelRegistration::OutputUnigramPriorView output_unigram_prior{};
    const auto execution_mode = GRIM::HyperParameters::snapshotExecutionMode(ctx.config);
    const bool unigram_bias_enabled = GRIM::HyperParameters::snapshotTrainingConfigField<bool>(
        ctx.config,
        "lm_head_unigram_bias");
    if (ctx.model_parameter_source_plan == ModelParameterSourcePlan::UNRESOLVED) {
        throw std::runtime_error(
            "ModelAllocated: parameter source is unresolved; call CheckpointPlanReady(ctx) before model allocation");
    }
    const bool fresh_parameter_initialization =
        ctx.model_parameter_source_plan == ModelParameterSourcePlan::FRESH_INITIALIZATION;
    const bool pass_output_unigram_prior =
        execution_mode == GRIM::HyperParameters::ModelExecutionMode::TRAINING &&
        unigram_bias_enabled && fresh_parameter_initialization;
    if (!fresh_parameter_initialization && !ctx.data.output_unigram_prior.log_bias.empty()) {
        throw std::runtime_error(
            "ModelAllocated: checkpoint restore plan carries a fresh-only output unigram prior");
    }
    EmitModuleInfo(
        ModuleId::Training,
        fresh_parameter_initialization
            ? "[MODEL_INIT] Model allocation uses fresh parameter initialization"
            : "[MODEL_INIT] Model allocation creates checkpoint destination buffers; corpus-derived parameter initialization is disabled",
        0);
    if (pass_output_unigram_prior) {
        const auto& prior = ctx.data.output_unigram_prior;
        if (prior.log_bias.empty()) {
            throw std::runtime_error(
                "ModelAllocated: lm_head_unigram_bias=true but LoadTrainingData did not author output_unigram_prior.log_bias");
        }
        output_unigram_prior.log_bias = prior.log_bias.data();
        output_unigram_prior.size = prior.log_bias.size();
        output_unigram_prior.vocab_size = prior.vocab_size;
        output_unigram_prior.seen_tokens = prior.seen_tokens;
        output_unigram_prior.total_targets = prior.total_targets;
    }

    try {
        if (!ctx.training_state) {
            throw std::runtime_error("FATAL: TrainingState owner is NULL before model initialization");
        }
        if (!ctx.generation_state) {
            throw std::runtime_error("FATAL: GenerationState owner is NULL before model initialization");
        }
        Internal::initializeModel(
            ctx.config,
            actual_vocab_size,
            weight_init_seed,
            *ctx.training_state,
            *ctx.generation_state,
            ctx.pbm_owner,
            ctx.gpu_model,
            ctx.parameter_registry,
            pass_output_unigram_prior ? &output_unigram_prior : nullptr,
            *ctx.logging.logger);
    } catch (const std::exception& e) {
        EmitModuleError(ModuleId::Training, std::string("FATAL: Model initialization failed: ") + e.what(), 0);
        throw;
    }

    if (pass_output_unigram_prior) {
        ctx.data.output_unigram_prior = {};
    }
}

} // namespace GRIMText::Training

