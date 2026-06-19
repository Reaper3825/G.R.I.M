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
#include "../../Shared/TrainingState/TrainingState_GPU.hpp"  // GRIM::TrainingState (stream access)

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace GRIMText::Training {

namespace {

// Initialize the output-head biases to the empirical unigram log-marginal,
// bias[v] = log p(v), estimated from the training targets. This houses the
// unigram prior in dedicated parameters so neither attention nor the FFN is
// rewarded for injecting it as a shared residual common-mode direction (the
// driver of the rho/representation-collapse buildup).
//
// Applies to BOTH the main LM head AND every MTP auxiliary head: the MTP losses
// backpropagate into the SAME shared trunk, so a zero-init MTP bias would
// re-create the identical collapse incentive (scaled by mtp_alpha). Each MTP
// head k predicts the token at shift k+1, whose marginal equals the overall
// unigram p(v) up to edge truncation, so the same log p(v) vector is reused for
// every head.
//
// The bias tensors are allocated during model assembly (LM head when
// lm_head_unigram_bias=true; MTP heads always carry a bias); here we only fill
// values, accessed via the ParameterRegistry API rather than a TrainingContext
// handle. Fresh-init only — on resume the trained biases are kept.
void populateUnigramOutputBiases(TrainingContext& ctx) {
    const bool enabled = GRIM::HyperParameters::snapshotTrainingConfigField<bool>(
        ctx.config, "lm_head_unigram_bias");
    if (!enabled) {
        return;
    }

    // On resume the trained bias is restored from the checkpoint (serialization
    // is presence-driven), so it must not be overwritten with the prior.
    if (!ctx.loaded_checkpoint_path.empty()) {
        GRIM::Logging::EmitModuleInfo(
            GRIM::Logging::ModuleId::Training,
            "[Phase1] lm_head_unigram_bias: resumed from checkpoint — keeping restored bias",
            0);
        return;
    }

    auto& lm_head = ctx.parameter_registry.requireLmHeadParameters("populateUnigramLmHeadBias");
    if (!lm_head.bias.data) {
        throw std::runtime_error(
            "populateUnigramLmHeadBias: lm_head_unigram_bias=true but lm_head.bias is NULL — "
            "initializeLmHeadParameterTensors must allocate the bias when the flag is set");
    }

    const int vocab_size = static_cast<int>(ctx.data.vocab_size);
    if (vocab_size <= 0) {
        throw std::runtime_error(
            "populateUnigramLmHeadBias: invalid vocab_size=" + std::to_string(vocab_size));
    }

    // Empirical unigram marginal over training targets (what the LM head
    // predicts), with Laplace add-one smoothing so unseen tokens get a finite
    // floor instead of log(0) = -inf.
    std::vector<double> counts(static_cast<std::size_t>(vocab_size), 0.0);
    double total = 0.0;
    for (const auto& seq : ctx.data.train_seqs) {
        for (int tgt : seq.targets) {
            if (tgt >= 0 && tgt < vocab_size) {
                counts[static_cast<std::size_t>(tgt)] += 1.0;
                total += 1.0;
            }
        }
    }
    if (total <= 0.0) {
        throw std::runtime_error(
            "populateUnigramLmHeadBias: no valid training targets to estimate p(v)");
    }

    constexpr double kSmoothing = 1.0;  // Laplace add-one
    const double denom = total + kSmoothing * static_cast<double>(vocab_size);
    std::vector<float> h_bias(static_cast<std::size_t>(vocab_size));
    int seen_tokens = 0;
    for (int v = 0; v < vocab_size; ++v) {
        const double count_v = counts[static_cast<std::size_t>(v)];
        const double p_v = (count_v + kSmoothing) / denom;
        h_bias[static_cast<std::size_t>(v)] = static_cast<float>(std::log(p_v));
        if (count_v > 0.0) {
            ++seen_tokens;
        }
    }

    int mtp_heads_written = 0;
#ifdef USE_CUDA
    cudaStream_t stream =
        ctx.requireTrainingState("populateUnigramOutputBiases").stream_ctrl.getPrimaryStream();
    const std::size_t bytes = static_cast<std::size_t>(vocab_size) * sizeof(float);

    // Upload the same log p(v) vector to a device bias tensor [vocab_size].
    auto upload_bias = [&](const GRIM::Tensor& bias, const char* who) {
        cudaError_t copy_err = cudaMemcpyAsync(
            bias.data, h_bias.data(), bytes, cudaMemcpyHostToDevice, stream);
        if (copy_err != cudaSuccess) {
            throw std::runtime_error(
                std::string("populateUnigramOutputBiases: cudaMemcpyAsync(") + who +
                " bias H2D) failed: " + cudaGetErrorString(copy_err));
        }
    };

    upload_bias(lm_head.bias, "lm_head");

    // Mirror the prior into every MTP auxiliary head; they share the trunk and
    // would otherwise re-introduce the collapse incentive from a zero-init bias.
    for (auto& mtp_head : ctx.parameter_registry.mtpHeadParameterTensors()) {
        if (!mtp_head.bias.data) {
            continue;
        }
        if (mtp_head.bias.numel() != static_cast<std::size_t>(vocab_size)) {
            throw std::runtime_error(
                "populateUnigramOutputBiases: MTP head bias size " +
                std::to_string(mtp_head.bias.numel()) + " != vocab_size " +
                std::to_string(vocab_size));
        }
        upload_bias(mtp_head.bias, "mtp_head");
        ++mtp_heads_written;
    }

    cudaError_t sync_err = cudaStreamSynchronize(stream);
    if (sync_err != cudaSuccess) {
        throw std::runtime_error(
            std::string("populateUnigramOutputBiases: cudaStreamSynchronize failed: ") +
            cudaGetErrorString(sync_err));
    }
#endif

    GRIM::Logging::EmitModuleInfo(
        GRIM::Logging::ModuleId::Training,
        "[Phase1] lm_head_unigram_bias: initialized bias = log p(v) | vocab=" +
            std::to_string(vocab_size) + " seen_tokens=" + std::to_string(seen_tokens) +
            " total_targets=" + std::to_string(static_cast<long long>(total)) +
            " mtp_heads=" + std::to_string(mtp_heads_written),
        0);
}

} // namespace

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
    populateUnigramOutputBiases(ctx);
    ResumeStateReady(ctx);
    TelemetryReady(ctx);
    PlannedBatchesReady(ctx);
    EpochPlanReady(ctx);
    StartupValidated(ctx);
    Phase2HandoffReady(ctx);

    return Phase1Result{Phase1Outcome::ready_for_training, std::move(ctx)};
}

} // namespace GRIMText::Training
