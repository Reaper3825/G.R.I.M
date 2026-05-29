#include "Phase1_Startup.hpp"

#include "Startup/Logging.hpp"
#include "Startup/Capacity/MemorySnapshot.hpp"
#include "Startup/Capacity/CapacityStem.hpp"
#include "Startup/Data/TrainingData.hpp"
#include "Startup/Model/ModelAllocationState.hpp"
#include "Startup/CheckpointLoad.hpp"
#include "Startup/Resume/ResumeState.hpp"
#include "Startup/Telemetry/TelemetryInitInputs.hpp"
#include "Startup/Epoch/EpochPlan.hpp"
#include "Startup/Payload/PayloadBuildInputs.hpp"
#include "Startup/Batching/PlannedBatches.hpp"
#include "Startup/Validation/StartupValidation.hpp"
#include "Startup/Validation/Phase2Handoff.hpp"
#include "../Subprocess/tokenizer_subprocess.hpp"
#include "../../Shared/LogRecorder/LogRecorder.hpp"

#include <sstream>
#include <stdexcept>
#include <utility>

namespace GRIMText::Training {

namespace {

Phase1Outcome runTokenizerSubprocessAfterHyperparameters(const TrainingContext& ctx) {
    using GRIM::Logging::EmitModuleError;
    using GRIM::Logging::EmitModuleInfo;
    using GRIM::Logging::ModuleId;
    using GRIMText::Subprocess::subprocess_outcome;

    GRIMText::Subprocess::tokenizer_subprocess_request tok_req;
    tok_req.hp = GRIM::HyperParameters::tokenizerSubprocessHP(ctx.config);
    // Rebuild policy belongs to TokenizerHP inside train_tokenizer; Phase1
    // only supplies the config path so there is no second rebuild payload.

    EmitModuleInfo(ModuleId::Training,
        "[Phase1] Running tokenizer subprocess after hyperparameter validation...", 0);

    const auto tok_result =
        GRIMText::Subprocess::run_tokenizer_subprocess(tok_req);

    switch (tok_result.outcome) {
        case subprocess_outcome::ok_proceed: {
            std::ostringstream oss;
            oss << "[Subprocess:" << tok_result.subprocess_name
                << "] ok_proceed | vocab=" << tok_result.vocab_path
                << " | data=" << tok_result.training_data_path
                << " | vocab_size=" << tok_result.vocab_size;
            EmitModuleInfo(ModuleId::Training, oss.str(), 0);
            return Phase1Outcome::ready_for_training;
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
            return Phase1Outcome::tokenizer_only_complete;
        }
        case subprocess_outcome::error: {
            std::ostringstream oss;
            oss << "[Subprocess:" << tok_result.subprocess_name
                << "] error: " << tok_result.error_message;
            EmitModuleError(ModuleId::Training, oss.str(), 0);
            throw std::runtime_error(oss.str());
        }
    }

    throw std::runtime_error("Phase1 tokenizer subprocess returned an unknown outcome");
}

} // anonymous namespace

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
    MemorySnapshotReady(ctx);
    HyperparametersReady(ctx);

    if (GRIM::HyperParameters::snapshotExecutionMode(ctx.config) == GRIM::HyperParameters::ModelExecutionMode::INFERENCE) {
        LoadInferenceTokenizer(ctx);
        ModelAllocated(ctx);
        CheckpointLoaded(ctx);
        PayloadBuildInputsReady(ctx);
        ctx.start_time = std::chrono::steady_clock::now();
        return Phase1Result{Phase1Outcome::ready_for_inference, std::move(ctx)};
    }

    const Phase1Outcome tokenizer_outcome = runTokenizerSubprocessAfterHyperparameters(ctx);
    if (tokenizer_outcome == Phase1Outcome::tokenizer_only_complete) {
        Phase1Result result;
        result.outcome = Phase1Outcome::tokenizer_only_complete;
        return result;
    }

    LoadTrainingData(ctx);
    ModelAllocated(ctx);
    CheckpointLoaded(ctx);
    ResumeStateReady(ctx);
    TelemetryReady(ctx);
    PayloadBuildInputsReady(ctx);
    PlannedBatchesReady(ctx);
    EpochPlanReady(ctx);
    StartupValidated(ctx);
    Phase2HandoffReady(ctx);

    return Phase1Result{Phase1Outcome::ready_for_training, std::move(ctx)};
}

} // namespace GRIMText::Training
