#include "Phase1_Startup.hpp"

#include "Startup/Logging.hpp"
#include "Startup/Capacity/MemorySnapshot.hpp"
#include "Startup/Capacity/CapacityStem.hpp"
#include "Startup/Model/ModelAllocationState.hpp"
#include "Startup/CheckpointLoad.hpp"
#include "Startup/Resume/ResumeState.hpp"
#include "Startup/Telemetry/TelemetryInitInputs.hpp"
#include "Startup/Epoch/EpochPlan.hpp"
#include "Startup/Batching/PlannedBatches.hpp"
#include "Startup/Validation/StartupValidation.hpp"
#include "Startup/Validation/Phase2Handoff.hpp"
#include "../Subprocess/tokenizer_subprocess.hpp"
#include "../../Shared/LogRecorder/LogRecorder.hpp"

#include <sstream>
#include <stdexcept>
#include <utility>

namespace GRIMText::Training {

Phase1Result executePhase1(GRIM::Config::AiConfigSnapshot config) {
    const auto execution_mode = GRIM::HyperParameters::snapshotExecutionMode(config);
    if (execution_mode != GRIM::HyperParameters::ModelExecutionMode::TRAINING &&
        execution_mode != GRIM::HyperParameters::ModelExecutionMode::INFERENCE) {
        throw std::runtime_error(
            "Phase1 requires a TRAINING or INFERENCE execution_mode config root");
    }

    TrainingContext ctx;
    ctx.config = std::move(config);

    LoggingReady(ctx);
    const MemorySnapshot startup_memory_snapshot = captureMemorySnapshotOrThrow();
    HyperparametersReady(ctx);

    if (GRIM::HyperParameters::snapshotExecutionMode(ctx.config) == GRIM::HyperParameters::ModelExecutionMode::INFERENCE) {
        GRIM::Logging::EmitModuleInfo(
            GRIM::Logging::ModuleId::Training,
            "[Phase1] Loading inference tokenizer artifact bundle...",
            0);
        auto inference_tokenizer = LoadInferenceTokenizer(ctx.config, *ctx.logging.logger);
        const std::uint32_t inference_vocab_size = static_cast<std::uint32_t>(inference_tokenizer->vocabSize());
        (void)GRIM::Tokenizer::tokenLayoutFromActualVocabOrThrow(
            inference_vocab_size,
            "executePhase1[inference]");
        syncRuntimeVocabSizeFromActualOrThrow(
            ctx.config,
            inference_vocab_size,
            "executePhase1[inference]");
        GRIM::Logging::EmitModuleInfo(
            GRIM::Logging::ModuleId::Training,
            "[Phase1] ✓ Inference tokenizer ready | vocab_size=" + std::to_string(inference_vocab_size),
            0);
        RngReady(ctx);
        if (inference_vocab_size == 0) {
            throw std::runtime_error("executePhase1[inference]: tokenizer vocab_size must be ready before layer assembly");
        }
        if (ctx.rng.init_seed == 0) {
            throw std::runtime_error("executePhase1[inference]: RNGContext.init_seed must be ready before layer assembly");
        }
        GRIM::Logging::EmitModuleInfo(
            GRIM::Logging::ModuleId::Training,
            "[Phase1] Preparing layer assembly inputs before model allocation...",
            0);
        ctx.layer_assembly = Startup::buildLayerAssembly(
            ctx.config,
            inference_vocab_size,
            ctx.rng.init_seed);
        {
            const auto& layer_assembly = ctx.layer_assembly.requireReady("executePhase1[inference]");
            GRIM::Logging::EmitModuleInfo(
                GRIM::Logging::ModuleId::Training,
                "[Phase1] ✓ Layer assembly ready | vocab=" + std::to_string(layer_assembly.inputs.actual_vocab_size) +
                    " | init_seed=" + std::to_string(layer_assembly.inputs.weight_init_seed),
                0);
        }
        ModelAllocated(ctx);
        CheckpointLoaded(ctx);
        ctx.start_time = std::chrono::steady_clock::now();
        Phase1Result result{Phase1Outcome::ready_for_inference, std::move(ctx)};
        result.inference_tokenizer = std::move(inference_tokenizer);
        return result;
    }

    using GRIM::Logging::EmitModuleError;
    using GRIM::Logging::EmitModuleInfo;
    using GRIM::Logging::ModuleId;
    using GRIMText::Subprocess::subprocess_outcome;

    EmitModuleInfo(ModuleId::Training,
        "[Phase1] Running tokenizer subprocess after hyperparameter validation...", 0);

    const auto tok_result = GRIMText::Subprocess::run_tokenizer_subprocess(ctx);

    Phase1Outcome tokenizer_outcome = Phase1Outcome::ready_for_training;
    switch (tok_result.outcome) {
        case subprocess_outcome::ok_proceed: {
            std::ostringstream oss;
            oss << "[Subprocess:" << tok_result.subprocess_name
                << "] ok_proceed | vocab=" << tok_result.vocab_path
                << " | data=" << tok_result.training_data_path
                << " | vocab_size=" << tok_result.vocab_size;
            EmitModuleInfo(ModuleId::Training, oss.str(), 0);
            tokenizer_outcome = Phase1Outcome::ready_for_training;
            break;
        }
        case subprocess_outcome::ok_one_off: {
            std::ostringstream oss;
            oss << "[Subprocess:" << tok_result.subprocess_name
                << "] ok_one_off | vocab=" << tok_result.vocab_path
                << " | data=" << tok_result.training_data_path
                << " | vocab_size=" << tok_result.vocab_size
                << " | ai_config.json subprocess.tokenizer.only_mode=true; "
                   "skipping remaining startup and Phases 2-3";
            EmitModuleInfo(ModuleId::Training, oss.str(), 0);
            tokenizer_outcome = Phase1Outcome::tokenizer_only_complete;
            break;
        }
        case subprocess_outcome::error: {
            std::ostringstream oss;
            oss << "[Subprocess:" << tok_result.subprocess_name
                << "] error: " << tok_result.error_message;
            EmitModuleError(ModuleId::Training, oss.str(), 0);
            throw std::runtime_error(oss.str());
        }
    }

    if (tokenizer_outcome == Phase1Outcome::tokenizer_only_complete) {
        Phase1Result result;
        result.outcome = Phase1Outcome::tokenizer_only_complete;
        return result;
    }

    LoadTrainingData(ctx, startup_memory_snapshot);
    RngReady(ctx);
    if (ctx.data.vocab_size == 0) {
        throw std::runtime_error("executePhase1[training]: SequenceData.vocab_size must be ready before layer assembly");
    }
    if (ctx.rng.init_seed == 0) {
        throw std::runtime_error("executePhase1[training]: RNGContext.init_seed must be ready before layer assembly");
    }
    GRIM::Logging::EmitModuleInfo(
        GRIM::Logging::ModuleId::Training,
        "[Phase1] Preparing layer assembly inputs before model allocation...",
        0);
    ctx.layer_assembly = Startup::buildLayerAssembly(
        ctx.config,
        ctx.data.vocab_size,
        ctx.rng.init_seed);
    {
        const auto& layer_assembly = ctx.layer_assembly.requireReady("executePhase1[training]");
        GRIM::Logging::EmitModuleInfo(
            GRIM::Logging::ModuleId::Training,
            "[Phase1] ✓ Layer assembly ready | vocab=" + std::to_string(layer_assembly.inputs.actual_vocab_size) +
                " | init_seed=" + std::to_string(layer_assembly.inputs.weight_init_seed),
            0);
    }
    ModelAllocated(ctx);
    CheckpointLoaded(ctx);
    ResumeStateReady(ctx);
    TelemetryReady(ctx);
    PlannedBatchesReady(ctx);
    EpochPlanReady(ctx);
    StartupValidated(ctx);
    Phase2HandoffReady(ctx);

    return Phase1Result{Phase1Outcome::ready_for_training, std::move(ctx)};
}

} // namespace GRIMText::Training
