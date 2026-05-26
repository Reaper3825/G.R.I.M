//======================================================//
//  Phase2_TrainingLoop.cu
//  Core training computation: epoch iteration, batch
//  processing, forward/backward, optimizer steps,
//  validation measurement.
//======================================================//
#include "Phase2_TrainingLoop.hpp"
#include "Phase3_Cleanup.hpp"
#include "../Diagnostics/Diagnostics.hpp"
#include "../../Shared/LogRecorder/LogRecorder.hpp"
#include "../../Shared/Gradients/GradStatsCollector.hpp"
#include "../../Shared/Gradients/GradientCC_GPU.hpp"       // GradClip::clipGradientNorms (registry-level clipping)
#include "../../Shared/Dynamic_LR/LRSchedule.hpp"          // GRIM::LR::LRSchedule (exposed LR curve)
#include "../../Shared/TrainingState/TrainingState_GPU.hpp"
#include "../../Shared/LogRecorder/BatchLogTape.hpp"
#include "../../Shared/Telemetry/TelemetryUpdate.hpp"
#include "../../Shared/UnigramByte/AtomTable.hpp"
#include "../../Shared/Batching/BatchPayload.hpp"
#include "../../Shared/Forward/ModelForwardRuntimePayload.hpp"
#include "../../Shared/Forward/ModelForward_GPU.hpp"
#include "../../Shared/Execution/ExecutionPayloadValidation.hpp"
#include "../Autograd/AutogradTraining.hpp"
#include "../../Shared/Optimizers/OptimizerUpdate_GPU.hpp"  // launchOptimizerUpdate
#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"  // single entry point; transitively pulls in control/ai_config_paths.hpp (resolveGrimRoot, etc.)
#include "../../Shared/HyperParameters/HyperparameterGroupings.hpp"  // grouped HP reads without duplicating authored config
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <utility>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

// Module logging aliases
using GRIM::Logging::ModuleId;
using GRIM::Logging::EmitModuleInfo;
using GRIM::Logging::EmitModuleError;

namespace GRIMText::Training {

TrainingLoopState::~TrainingLoopState() = default;

//======================================================//
//  Internal Helpers
//======================================================//

namespace Internal {

std::string formatScalar(float value, int precision) {
    std::ostringstream oss;
    if (std::isfinite(value)) {
         if (value == 0.0f) value = 0.0f; // kill -0.0
        oss << std::fixed << std::setprecision(precision) << value;
    } else if (std::isnan(value)) {
        oss << "nan";
    } else {
        oss << (value > 0 ? "inf" : "-inf");
    }
    return oss.str();
}

std::string formatMetric(std::string_view name, float value, int precision) {
    return std::string(name) + "=" + formatScalar(value, precision);
}

// Query LR schedule at a given step, honoring stability overrides.
float getScheduledLearningRate(
    const GRIM::LR::LRSchedule& schedule,
    int step,
    float base_lr,
    bool stability_overrides_enabled) {
    if (stability_overrides_enabled) {
        return base_lr;
    }
    return schedule.lr(step);
}

} // namespace Internal

//======================================================//
//  Phase2 plan validation
//
//  All BatchPayloads are authored once in Phase1
//  (Startup/Batching/PlannedBatches.cu); Phase2 only INDEXES into
//  ctx.train_payloads / ctx.val_payloads via ctx.epoch_batch_order. The
//  per-batch payload builder, the per-epoch BatchSchedule construction, and
//  the train_views shuffle that used to live here have all moved to
//  PlannedBatchesReady.
//======================================================//
namespace {

struct TapeSkipScope {
    GRIM::Logging::BatchLogTape* tape;
    bool prev;

    explicit TapeSkipScope(bool skip)
        : tape(GRIM::Logging::getGlobalTape()),
          prev(tape ? tape->skipThisPass() : false) {
        if (tape) {
            tape->setSkipThisPass(skip);
        }
    }

    ~TapeSkipScope() {
        if (tape) {
            tape->setSkipThisPass(prev);
        }
    }

    TapeSkipScope(const TapeSkipScope&) = delete;
    TapeSkipScope& operator=(const TapeSkipScope&) = delete;
};

float accumulationNormalizationScaleForOptimizerWindow(int accum_steps) {
    if (accum_steps <= 0) {
        throw std::runtime_error(
            "accumulationNormalizationScaleForOptimizerWindow: accum_steps must be > 0, got " +
            std::to_string(accum_steps));
    }
    return 1.0f / static_cast<float>(accum_steps);
}

void scaleRegisteredParameterGradientsForOptimizerWindow(
    std::vector<GRIM::ParameterGroup>& groups,
    float accumulation_scale,
    cudaStream_t stream)
{
    if (groups.empty()) {
        throw std::runtime_error(
            "scaleRegisteredParameterGradientsForOptimizerWindow: parameter groups are empty");
    }
    if (!stream) {
        throw std::runtime_error(
            "scaleRegisteredParameterGradientsForOptimizerWindow: stream is NULL");
    }
    if (!std::isfinite(accumulation_scale) || accumulation_scale <= 0.0f) {
        throw std::runtime_error(
            "scaleRegisteredParameterGradientsForOptimizerWindow: accumulation_scale must be finite and > 0, got " +
            std::to_string(accumulation_scale));
    }
    if (accumulation_scale == 1.0f) {
        return;
    }

    for (auto& group : groups) {
        if (!group.grads() || group.size() == 0) {
            continue;
        }
        if (group.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error(
                "scaleRegisteredParameterGradientsForOptimizerWindow: group '" + group.name +
                "' exceeds launchScaleGradients int element limit with size=" + std::to_string(group.size()));
        }
        launchScaleGradients(
            group.grads(),
            static_cast<int>(group.size()),
            accumulation_scale,
            stream);
    }
}

int validatedAccumulationSteps(const TrainingContext& ctx) {
    const auto schedule_hp =
        ::GRIM::HyperParameters::trainingScheduleHP(ctx.config);
    const int accum_steps = schedule_hp.gradient_accumulation_steps;
    if (accum_steps <= 0) {
        throw std::runtime_error("FATAL: gradient_accumulation_steps must be > 0 in Phase2 (got " +
                                 std::to_string(accum_steps) + ")");
    }
    return accum_steps;
}

void validateAccumulationPositionBeforeBackward(
    const OptimizerContext& optimizer,
    int accum_steps,
    int batch_idx,
    int global_step)
{
    if (accum_steps <= 0) {
        throw std::runtime_error(
            "validateAccumulationPositionBeforeBackward: accum_steps must be > 0, got " +
            std::to_string(accum_steps));
    }
    const int accumulation_slot = optimizer.accumulationSlot();
    try {
        if (accumulation_slot < 0 || accumulation_slot >= accum_steps) {
            throw std::runtime_error(
                "FATAL: accumulation slot cursor out of range before autograd pass");
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "\n[Phase2] FATAL: Training step attempted with accumulation_slot=%d outside accum_steps=%d\n",
                accumulation_slot, accum_steps);
        fprintf(stderr, "[Phase2] batch=%d global_step=%d\n", batch_idx + 1, global_step);
        fprintf(stderr, "[Phase2] %s\n", e.what());
        std::abort();
    }
}

bool shouldAccumulateGradients(const OptimizerContext& optimizer) {
    return optimizer.accumulationSlot() > 0;
}

bool advanceAccumulationOrThrow(
    OptimizerContext& optimizer,
    int accum_steps)
{
    if (accum_steps <= 0) {
        throw std::runtime_error(
            "advanceAccumulationOrThrow: accum_steps must be > 0, got " +
            std::to_string(accum_steps));
    }

    const int accumulation_slot = optimizer.accumulationSlot();
    if (accumulation_slot < 0 || accumulation_slot >= accum_steps) {
        throw std::runtime_error(
            "advanceAccumulationOrThrow: accumulation slot cursor out of range before advance (slot=" +
            std::to_string(accumulation_slot) + " accum_steps=" + std::to_string(accum_steps) + ")");
    }

    const int next_slot = accumulation_slot + 1;
    optimizer.setAccumulationSlot(next_slot);
    return next_slot >= accum_steps;
}

void completeOptimizerWindowBookkeepingOrThrow(
    OptimizerContext& optimizer,
    int accum_steps)
{
    if (accum_steps <= 0) {
        throw std::runtime_error(
            "completeOptimizerWindowBookkeepingOrThrow: accum_steps must be > 0, got " +
            std::to_string(accum_steps));
    }

    const int accumulation_slot = optimizer.accumulationSlot();
    if (accumulation_slot != accum_steps) {
        throw std::runtime_error(
            "FATAL: optimizer step requested before accumulation window completed (completed=" +
            std::to_string(accumulation_slot) + " required=" + std::to_string(accum_steps) + ")");
    }

    optimizer.setAccumulationSlot(0);
    optimizer.optimizer_step.step++;
}

std::unique_ptr<GRIM::Loss::LossSignalBus> makeValidationHighLossSignalBus(
    const ::GRIM::HyperParameters::LanguageModelConfig& hp)
{
    const auto auto_stop_hp = ::GRIM::HyperParameters::autoStopHP(hp);
    GRIM::Loss::LossSignalConfig sig_cfg{};
    sig_cfg.validation_high_threshold = auto_stop_hp.high_loss_threshold;
    sig_cfg.validation_high_patience  = auto_stop_hp.high_loss_patience;
    return std::make_unique<GRIM::Loss::LossSignalBus>(sig_cfg);
}

float scheduledLearningRateForOptimizerStep(
    const TrainingContext& ctx,
    int optimizer_step)
{
    if (!ctx.lr_schedule) {
        throw std::runtime_error("lr_schedule is not initialized at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    const auto lr_inputs = ::GRIM::HyperParameters::learningRateScheduleInputs(ctx.config);
    const auto stability_hp =
        ::GRIM::HyperParameters::stabilityOverrideHP(ctx.config);
    return Internal::getScheduledLearningRate(
        *ctx.lr_schedule,
        optimizer_step,
        lr_inputs.learning_rate,
        stability_hp.enabled);
}

float mtpAlphaEffectiveForBatch(
    const ::GRIM::HyperParameters::LanguageModelConfig& cfg,
    int global_step)
{
    if (!cfg.mtp_enabled || cfg.mtp_k <= 0) {
        return 0.0f;
    }
    if (global_step < 0) {
        throw std::runtime_error("mtpAlphaEffectiveForBatch: global_step must be >= 0, got " +
                                 std::to_string(global_step));
    }
    if (cfg.mtp_alpha_warmup_steps <= 0) {
        throw std::runtime_error("mtpAlphaEffectiveForBatch: mtp_alpha_warmup_steps must be > 0, got " +
                                 std::to_string(cfg.mtp_alpha_warmup_steps));
    }
    const float progress = std::min(
        1.0f,
        static_cast<float>(global_step) / static_cast<float>(cfg.mtp_alpha_warmup_steps));
    const float alpha = cfg.mtp_alpha * progress;
    if (!std::isfinite(alpha) || alpha < 0.0f) {
        throw std::runtime_error("mtpAlphaEffectiveForBatch: derived alpha must be finite and >= 0, got " +
                                 std::to_string(alpha));
    }
    return alpha;
}

void runOptimizerWindowFromEpoch(
    TrainingContext& ctx,
    TrainingLoopState& state,
    const GRIM::Batching::BatchPayload& payload,
    BatchResult& result,
    int batch_idx,
    int accum_steps,
    int optimizer_step)
{
    const bool sync_diag = GRIM::Diagnostics::shouldSyncDiagnostics(ctx, batch_idx);

    bool has_clip_metrics = false;
    GRIM::GradClip::ClipResult clip_metrics{};

    // Issue #135: gradient clipping is DEFERRED to post-accumulation. Clipping
    // ONCE on the averaged gradients matches PyTorch; the old per-slot clipping
    // crushed text gradients M×.
    const auto clipping_hp = ::GRIM::HyperParameters::gradientClippingHP(
        ctx.config);
    const auto optimizer_update_hp = ::GRIM::HyperParameters::optimizerUpdateHP(
        ctx.config);
    const float effective_per_token_limit = clipping_hp.effective_per_token_limit;
    const bool clipping_enabled = clipping_hp.enabled;

    GRIM::Diagnostics::WeightSample pre_sample{};
    if (sync_diag) {
        pre_sample = GRIM::Diagnostics::sampleWeightStats(
            ctx.model->getLmHeadLayer(), ctx.model->getTrainingState(), true);
    }

    auto& clip_ts = ctx.model->getTrainingState();
    cudaStream_t clip_stream = clip_ts.stream_ctrl.getPrimaryStream();
    auto& clip_groups = ctx.model->parameterGroups();
    const float accumulation_scale =
        accumulationNormalizationScaleForOptimizerWindow(accum_steps);

    // Accumulation normalization happens exactly once at the optimizer window
    // boundary, after the full microbatch window has completed and before any
    // clipping/update logic consumes the registered parameter gradients.
    scaleRegisteredParameterGradientsForOptimizerWindow(
        clip_groups,
        accumulation_scale,
        clip_stream);

    // Global clipping on post-accumulation normalized gradients.
    // Norm measurement, global aggregation, and clipping all happen inside
    // GradientCC against the registered ParameterGroup tensors.
    if (clipping_enabled) {
        GRIM::GradClip::ClipConfig clip_cfg;
        clip_cfg.max_rms = effective_per_token_limit;

        const auto clip = GRIM::GradClip::clipGradientNorms(
            clip_groups.data(), clip_groups.size(),
            clip_ts.grad_norm_scratch, clip_cfg, clip_stream);

        result.grad_rms = clip.global_rms_post;
        result.grad_rms_valid = true;
        result.gradient_clipped = clip.any_clipped();
        clip_metrics = clip;
        has_clip_metrics = true;

        GRIM::Diagnostics::runGradientNormClipDiagnostic(ctx, state, payload, clip, batch_idx, clip_stream);
    }

    // ════════════════════════════════════════════════════════════════════
    // RUNTIME tie_embeddings pointer verification (every optimizer step)
    // (extracted to Diagnostics/TieVerifyDiagnostic.cu)
    // ════════════════════════════════════════════════════════════════════
    GRIM::Diagnostics::runTieVerifyDiagnostic(ctx, batch_idx);

    // Optimizer Window: runEpoch owns the accumulation-complete boundary;
    // Shared/Optimizers owns configured optimizer dispatch. The window only
    // provides filled-window grads, LR, step, stream, and grouped HP.
    GRIM::launchOptimizerUpdate(ctx.model->parameterGroups(),
                                optimizer_update_hp,
                                result.learning_rate,
                                optimizer_step,
                                ctx.model->getTrainingState().stream_ctrl.getPrimaryStream());

    // Rule 20: post-optimizer weight NaN spot check. Stream ownership stays
    // in Phase2; the guard only inspects optimizer parameter groups.
    {
        auto& post_step_state = ctx.model->getTrainingState();
        cudaError_t post_step_sync = cudaStreamSynchronize(
            post_step_state.stream_ctrl.getPrimaryStream());
        if (post_step_sync != cudaSuccess) {
            throw std::runtime_error(
                std::string("[FATAL] Failed to synchronize stream before post-optimizer finite check: ") +
                cudaGetErrorString(post_step_sync));
        }

        const auto& post_step_groups = ctx.model->parameterGroups();
        GRIM::Diagnostics::checkPostOptimizerWeightsFinite(
            post_step_groups.data(),
            post_step_groups.size(),
            optimizer_step,
            result.learning_rate,
            batch_idx);
    }

    GRIM::Diagnostics::runOptimizerMomentDiagnostic(
        ctx, batch_idx, accum_steps, sync_diag);

    // Post-optimizer LM-head sample, GradTrace POST log, [UpdateMag],
    // and optimizer-boundary adaptive update trace.
    GRIM::Diagnostics::runPostOptimizerWeightTrace(
        ctx, result, optimizer_update_hp, pre_sample, sync_diag);

    completeOptimizerWindowBookkeepingOrThrow(ctx.optimizer, accum_steps);

    // Rule 20: an async CUDA error here means an optimizer-window kernel
    // launch faulted earlier. Crash with the exact error.
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("[CUDA] async error after optimizer step: ") +
                cudaGetErrorString(err));
        }
    }

    // Update telemetry lattice from the clipping-owned grad measurement only.
    // Phase2 does not launch diagnostic grad-norm measurement or manufacture
    // placeholder gradient values for telemetry.
    if (has_clip_metrics) {
        GRIM::Telemetry::TelemetryBatchInput tel_input;
        tel_input.loss              = result.loss;
        tel_input.preclip_grad_rms  = clip_metrics.global_rms_pre;
        tel_input.learning_rate     = result.learning_rate;
        tel_input.total_tokens      = payload.actual_tokens;
        tel_input.enc_rms_pre       = clip_metrics.encoder_rms_pre;
        tel_input.optimizer_step    = optimizer_step;
        tel_input.should_step       = true;
        tel_input.total_loss_value  = result.loss;
        tel_input.aux_loss          = result.aux_loss;
        tel_input.max_seq_len       = payload.max_seq_len;
        tel_input.batch_idx         = batch_idx;
        tel_input.global_step       = ctx.global_step;
        tel_input.actual_vocab_size = ctx.data_info.actual_vocab_size;
        tel_input.d_model           = ctx.model_config.d_model;

        GRIM::Telemetry::updateTelemetryObservations(ctx, tel_input, clip_metrics.metrics, &payload);
    }
}

void validateTrainingForwardInputs(
    GRIM::LanguageModel& model,
    const GRIM::Batching::BatchPayload& payload,
    const char* caller)
{
    payload.validate(caller);

    const auto& cfg = model.getConfig();
    GRIM::Execution::validateExecutionPayload(
        payload,
        caller,
        cfg.execution_block_num_slots,
        cfg.execution_block_num_ops,
        cfg.execution_block_num_steps);

    if (!payload.teacher_steps.empty() && !cfg.execution_block_enabled) {
        std::cerr << "[Phase2] WARN: batch has teacher_steps while execution_block_enabled=false; "
                  << "training with plain cross-entropy over text tokens" << std::endl;
    }

    if (cfg.execution_block_enabled) {
        if (!model.getExecutionBlockLayer()) {
            throw std::runtime_error(
                std::string(caller) + ": execution_block_enabled but ExecutionBlock layer is null");
        }
        GRIM::ScratchBlockLayer* scratch_block = model.getScratchBlockLayer();
        if (!scratch_block || !scratch_block->isEnabled()) {
            throw std::runtime_error(
                std::string(caller) + ": execution_block_enabled requires ScratchBlock enabled");
        }
    }
}

void configureAutogradLossInputs(
    GRIM::Autograd::AutogradContext& autograd_ctx,
    GRIM::LanguageModel& model,
    GRIM::TrainingState& training_state,
    const GRIM::Batching::BatchPayload& payload,
    const GRIM::HyperParameters::LossConfigHP& loss_config,
    bool skip_equation_logging)
{
    if (loss_config.class_balanced_enabled) {
        if (!training_state.class_weights_tensor.data) {
            throw std::runtime_error(
                "configureAutogradLossInputs: class_balanced_enabled=true but class_weights_tensor is NULL");
        }
        if (training_state.class_weights_vocab_size != payload.vocab_size) {
            throw std::runtime_error(
                "configureAutogradLossInputs: class_weights_vocab_size=" +
                std::to_string(training_state.class_weights_vocab_size) +
                " != payload.vocab_size=" + std::to_string(payload.vocab_size));
        }
        autograd_ctx.d_class_weights = training_state.class_weights_tensor.data;
    } else {
        autograd_ctx.d_class_weights = nullptr;
    }

    autograd_ctx.skip_equation_logging = skip_equation_logging;
    autograd_ctx.model = &model;
}

GRIM::Forward::ModelForwardRuntimePayload buildTrainingForwardRuntimePayload(
    GRIM::TrainingState& training_state)
{
    GRIM::Forward::ModelForwardRuntimePayload runtime_payload{};
    runtime_payload.autograd_intermediates = &training_state.autograd_intermediates;
    runtime_payload.execution_trace_by_row = &training_state.execution_trace_by_row;
    runtime_payload.trace_state_by_row = &training_state.trace_state_by_row;
    runtime_payload.read_gate_accum_tensor = &training_state.read_gate_accum_tensor;
    return runtime_payload;
}

GRIM::Forward::ModelForwardRequest buildTrainingForwardRequest(
    const GRIM::Autograd::AutogradContext& autograd_ctx)
{
    GRIM::Forward::ModelForwardRequest request{};
    request.config = autograd_ctx.config;
    request.gpu_encoder = autograd_ctx.gpu_encoder;
    request.cublas_handle = autograd_ctx.cublas_handle;
    request.stream = autograd_ctx.stream;
    request.embedding_layer = autograd_ctx.embedding_layer;
    request.lm_head = autograd_ctx.lm_head;
    request.scratch_block = autograd_ctx.scratch_block;
    request.reasoning_head = autograd_ctx.reasoning_head;
    request.execution_block = autograd_ctx.execution_block;
    request.payload = autograd_ctx.payload;
    request.bindings = autograd_ctx.device_bindings;
    request.batch_idx = autograd_ctx.batch_idx;
    request.graph = GRIM::Forward::ModelForwardGraphPolicy{
        /*connect_parameter_graph=*/true,
        /*retain_backward_graph=*/true,
        /*enable_dropout=*/true};
    return request;
}

void snapshotReadGateMean(
    GRIM::TrainingState& training_state,
    cudaStream_t stream)
{
    if (!training_state.read_gate_accum_tensor.data) {
        throw std::runtime_error(
            "snapshotReadGateMean: TrainingState.read_gate_accum_tensor is NULL");
    }

    float h_accum[2] = {0.0f, 0.0f};
    CUDA_CHECK(cudaMemcpyAsync(
        h_accum,
        training_state.read_gate_accum_tensor.data,
        2 * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    training_state.h_read_gate_mean = (h_accum[1] > 0.0f)
        ? (h_accum[0] / h_accum[1])
        : 0.0f;
}

} // namespace

//======================================================//
//  Batch Processing Implementation
//======================================================//

BatchResult processBatch(
    TrainingContext& ctx,
    TrainingLoopState& state,
    const GRIM::Batching::BatchPayload& payload,
    int batch_idx,
    int epoch_idx,
    const BatchAutogradPlan& plan) {
    
    BatchResult result;
    result.batch_idx = batch_idx;

    // Begin tape recording for this batch (clears prior entries, sets step/batch)
    if (ctx.logging.tape) {
        ctx.logging.tape->beginBatch(static_cast<int>(ctx.global_step), batch_idx);
    }

    // Issue #142b: drain deferred CUDA errors from prior ops (sample gen,
    // previous batch backward); without this they manifest deep inside
    // encoderForward as an SEH exception.
    {
        cudaError_t pre_err = cudaGetLastError();
        if (pre_err != cudaSuccess) {
            std::string err_msg = "[processBatch] CUDA error BEFORE batch " +
                std::to_string(batch_idx + 1) + ": " +
                std::string(cudaGetErrorString(pre_err)) +
                " (code=" + std::to_string(static_cast<int>(pre_err)) + ")";
            ctx.logging.logger->log(err_msg);
            fprintf(stderr, "%s\n", err_msg.c_str());
            cudaDeviceSynchronize();
            cudaGetLastError();
        }
    }

    if (payload.batch_size == 0) {
        // Rule 20: scheduler MUST NOT emit empty batches.
        throw std::runtime_error(
            "[processBatch] empty payload at batch_idx=" + std::to_string(batch_idx) +
            " — scheduler produced batch_size=0; fix the upstream filter");
    }

    // beginBatch() must run EVERY BatchPayload pass to clear previous entries;
    // otherwise accumulation slots 1+ inherit stale entries.
    GRIM::GradStats::beginBatch();

    PHASE2_DEBUG_STDERR("[DEBUG-PROCESS] After BATCH_INFO log, checking shouldLogAtomStats...\n");
    GRIM::Diagnostics::runAtomStatsDiagnostic(ctx, payload, batch_idx);

    PHASE2_DEBUG_STDERR("[DEBUG-PROCESS] After atom stats, entering boundary diagnostic...\n");
    GRIM::Diagnostics::runBoundaryDiagnostic(ctx, payload, batch_idx);

    PHASE2_DEBUG_STDERR("[DEBUG-PROCESS] After boundary diagnostic, entering forward pass...\n");
    static int forward_call_count = 0;
    ++forward_call_count;

    PHASE2_DEBUG_STDERR("[DEBUG-PROCESS] forward_call_count=%d, building target distribution...\n", forward_call_count);
    GRIM::Diagnostics::runTargetDistributionLog(ctx, payload, batch_idx);

    PHASE2_DEBUG_STDERR("[DEBUG-PROCESS] About to run explicit shared forward + autograd loss/backward...\n");
    // First-batch CUDA check: surface any error before explicit forward/loss/bwd.
    if (batch_idx == 0) {
        cudaError_t e = cudaDeviceSynchronize();
        cudaError_t last = (e != cudaSuccess) ? e : cudaGetLastError();
        if (last != cudaSuccess) {
            ctx.logging.logger->log("[CUDA] first_batch BEFORE explicit training forward: " + std::string(cudaGetErrorString(last)));
            cudaGetLastError();
        } else {
            ctx.logging.logger->log("[CUDA] first_batch BEFORE explicit training forward: ok");
        }
    }
    // Rule 20 ownership taxonomy: AutogradStepScope is the SINGLE owner of
    // AutogradIntermediates::clear() for this batch. Do NOT add an explicit
    // clear() anywhere inside this scope.
    GRIM::Autograd::AutogradStepScope autograd_step_scope(ctx.model->getTrainingState());
    // Sync slice: upload the prebuilt host BatchPayload once and reuse the
    // returned BatchDeviceBindings across shared forward, loss, and backward —
    // payload itself is host-only/immutable and never carries device pointers.
    const auto train_bindings = ctx.model->uploadBatchToDevice(payload);
    const auto loss_config = GRIM::HyperParameters::lossConfigHP(ctx.config);
    auto& model = *ctx.model;
    auto& training_state = model.getTrainingState();
    const auto& model_config = model.getConfig();

    validateTrainingForwardInputs(model, payload, "processBatch");

    training_state.autograd_batch_idx = plan.batch_idx;
    TapeSkipScope tape_skip_scope(plan.should_accumulate);

    if (!training_state.read_gate_accum_tensor.data) {
        throw std::runtime_error(
            "processBatch: TrainingState.read_gate_accum_tensor is NULL - "
            "Phase1 startup must allocate the read-gate workspace before Phase2 runs");
    }

    cudaStream_t stream = training_state.stream_ctrl.getPrimaryStream();
    CUDA_CHECK(cudaMemsetAsync(training_state.read_gate_accum_tensor.data, 0, 2 * sizeof(float), stream));

    GRIM::Autograd::AutogradContext autograd_ctx = GRIM::Autograd::initAutogradContext(
        &model_config,
        &training_state,
        &model.getGpuEncoder(),
        model.getEmbeddingLayer(),
        model.getLmHeadLayer(),
        model.getScratchBlockLayer(),
        model.getReasoningHeadLayer(),
        model.getExecutionBlockLayer(),
        training_state.cublas_handle.get(),
        stream,
        payload,
        train_bindings,
        plan.batch_idx);

    configureAutogradLossInputs(
        autograd_ctx,
        model,
        training_state,
        payload,
        loss_config,
        plan.should_accumulate);

    GRIM::Forward::ModelForwardRuntimePayload runtime_payload =
        buildTrainingForwardRuntimePayload(training_state);
    GRIM::Forward::ModelForwardRequest forward_request =
        buildTrainingForwardRequest(autograd_ctx);

    GRIM::Forward::executeModelForward(forward_request, runtime_payload);
    snapshotReadGateMean(training_state, stream);

    auto loss_result = GRIM::Autograd::computeAutogradLoss(
        autograd_ctx,
        loss_config,
        mtpAlphaEffectiveForBatch(ctx.model_config, ctx.global_step));
    if (!loss_result.success) {
        throw std::runtime_error(
            "[computeAutogradLoss] FAILED batch=" + std::to_string(batch_idx + 1) +
            ": " + loss_result.error_message);
    }
    if (!std::isfinite(loss_result.loss_value)) {
        throw std::runtime_error("Non-finite loss: " + std::to_string(loss_result.loss_value));
    }

    auto backward_result = GRIM::Autograd::executeAutogradBackward(
        autograd_ctx,
        plan.should_accumulate);
    if (!backward_result.success) {
        throw std::runtime_error(
            "[executeAutogradBackward] FAILED batch=" + std::to_string(batch_idx + 1) +
            ": " + backward_result.error_message);
    }

    training_state.sequence_weight_count = 0;

    result.loss = loss_result.loss_value;
    result.aux_loss = loss_result.aux_loss;
    result.mtp_diagnostics = std::move(loss_result.mtp_diagnostics);
    PHASE2_DEBUG_STDERR("[DEBUG-PROCESS] explicit forward + autograd loss/backward returned, loss=%f success=%d\n", 
                        result.loss, static_cast<int>(loss_result.success));

    // First-batch CUDA check: a fault here means explicit forward/loss/bwd produced it.
    // Rule 20: crash with the exact error — don't defer to teardown.
    if (batch_idx == 0) {
        cudaError_t e = cudaDeviceSynchronize();
        cudaError_t last = (e != cudaSuccess) ? e : cudaGetLastError();
        if (last != cudaSuccess) {
            throw std::runtime_error(
                std::string("[CUDA] first_batch AFTER explicit training forward/loss/backward: ") +
                cudaGetErrorString(last) +
                " (fault is in forward, loss, or backward)");
        }
        ctx.logging.logger->log("[CUDA] first_batch AFTER explicit training forward/loss/backward: ok");
    }

    auto& active_intermediates = ctx.model->getTrainingState().autograd_intermediates;
    if (!active_intermediates.logits_tensor.data) {
        throw std::runtime_error(
            "processBatch: live logits tensor is NULL after successful explicit shared forward — "
            "diagnostics must run before AutogradStepScope teardown");
    }
    GRIM::Tensor* live_lm_head_input = nullptr;
    if (active_intermediates.centered_encoder_output.data) {
        live_lm_head_input = &active_intermediates.centered_encoder_output;
    } else if (active_intermediates.encoder_output_tensor.data) {
        live_lm_head_input = &active_intermediates.encoder_output_tensor;
    }
    if (!live_lm_head_input || !live_lm_head_input->data) {
        throw std::runtime_error(
            "processBatch: live LM-head input tensor is NULL after successful explicit shared forward");
    }

    // Log model predictions (what it predicts vs targets) - uses ForwardPass module for filtering
    // (extracted to Diagnostics/PredictionDistributionDiagnostic.cu)
    GRIM::Diagnostics::runPredictionDistributionAndLogitTrace(
        ctx,
        payload,
        active_intermediates.logits_tensor,
        result.loss,
        batch_idx);
    // NOTE: Loss variance computation removed (was causing 5-second GPU sync bottleneck).
    // Variance is now tracked on GPU by TelemetryLattice (σ_tilde, v_σ fields).
    // Use computeTelemetryFeedback() to access grad_norm variance for adaptive decisions.

    // ========================================================================
    // BATCH_LOSS equation log + LossStats summary line
    // (extracted to Diagnostics/LossStatsDiagnostic.cu)
    // ========================================================================
    GRIM::Diagnostics::runLossStatsDiagnostic(ctx, payload, result, batch_idx);

    // ========================================================================
    // TRAINING SIGNAL: Logit Statistics (argmax distribution, confidence)
    // (extracted to Diagnostics/LogitScaleDiagnostic.cu)
    // ========================================================================
    GRIM::Diagnostics::runLogitScaleDiagnostic(
        ctx,
        payload,
        active_intermediates.logits_tensor,
        *live_lm_head_input,
        batch_idx);
    
    if (!std::isfinite(result.loss)) {
        throw std::runtime_error("Non-finite batch loss: " + std::to_string(result.loss));
    }

    // ========================================================================
    // Adaptive loss baseline tracking + invalid-token validation
    // (extracted to Diagnostics/LossBaselineDiagnostic.cu)
    // Mutates: state.initial_loss, state.min_observed_loss, state.warmup_batches.
    // Throws on data corruption (Rule 20).
    // ========================================================================
    GRIM::Diagnostics::runLossBaselineAndTokenValidation(
        ctx, state, payload, result.loss, batch_idx);

    // ═══════════════════════════════════════════════════════════════════════════
    // POST-BACKWARD: explicit shared forward + autograd backward has produced/accumulated grads.
    // Diagnostics below read from TrainingState before runEpoch may open the
    // optimizer window.
    // ═══════════════════════════════════════════════════════════════════════════

    // Issue #142: special-token weight & gradient verification
    GRIM::Diagnostics::runSpecialTokenDiagnostic(ctx, payload, batch_idx);

    const bool sync_diag = GRIM::Diagnostics::shouldSyncDiagnostics(ctx, batch_idx);

    if (sync_diag) {
        auto& training_state = ctx.model->getTrainingState();
        const auto flush_result = GRIM::GradStats::flushAndLog(
            training_state.stream_ctrl.getPrimaryStream(),
            ctx.global_step,
            ModuleId::BackwardPass);
        if (flush_result.has_explosion || flush_result.has_nan || flush_result.has_inf) {
            std::ostringstream oss;
            oss << "[GradStats] FATAL: gradient stats flagged "
                << (flush_result.has_explosion ? "explosion " : "")
                << (flush_result.has_nan ? "NaN " : "")
                << (flush_result.has_inf ? "Inf " : "")
                << "at batch=" << (batch_idx + 1)
                << " step=" << ctx.global_step;
            EmitModuleError(ModuleId::BackwardPass, oss.str(), ctx.global_step);
            throw std::runtime_error(oss.str());
        }
        
        // CRITICAL: sync stream BEFORE endOptimizerStep() zeros gradient buffers.
        // GradStats::flushAndLog() launches async D2H copies that read from grad
        // buffers; without this sync those copies see zeroed memory and produce
        // corrupted stats (e.g. negative max_abs, inverted min/max).
        cudaError_t sync_err = cudaStreamSynchronize(training_state.stream_ctrl.getPrimaryStream());
        if (sync_err != cudaSuccess) {
            std::ostringstream oss;
            oss << "[GradStats] FATAL: Failed to synchronize stream before optimizer step: "
                << cudaGetErrorString(sync_err);
            EmitModuleError(ModuleId::BackwardPass, oss.str(), ctx.global_step);
            throw std::runtime_error(oss.str());
        }
    }

    // Rule 20: an async CUDA error here means a microbatch/autograd kernel
    // launch faulted earlier. Crash with the exact error before returning to
    // the epoch-owned optimizer boundary.
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("[CUDA] async error after processBatch: ") +
                cudaGetErrorString(err));
        }
    }
    
    result.sequences_processed = payload.batch_size;
    result.tokens_processed = payload.actual_tokens;

    // First-batch CUDA check (runs even if telemetry disabled): last point
    // before returning to runEpoch's accumulation/optimizer boundary.
    if (batch_idx == 0) {
        cudaError_t e = cudaDeviceSynchronize();
        cudaError_t last = (e != cudaSuccess) ? e : cudaGetLastError();
        if (last != cudaSuccess) {
            ctx.logging.logger->log("[CUDA] first_batch END processBatch: " + std::string(cudaGetErrorString(last)));
            cudaGetLastError();
        } else {
            ctx.logging.logger->log("[CUDA] first_batch END processBatch: ok");
        }
    }

    // Rule 20 single-owner clear: AutogradStepScope at processBatch entry owns
    // the clear; the tape flush below does not touch autograd intermediates.
    if (ctx.logging.tape) {
        ctx.logging.tape->flush();
    }

    return result;
}

//======================================================//
//  Epoch Implementation
//======================================================//

EpochResult runEpoch(
    TrainingContext& ctx,
    TrainingLoopState& state,
    int epoch_idx,
    int num_epochs,
    int accum_steps) {
    
    EpochResult result;
    result.epoch = epoch_idx;
    
    if (num_epochs <= 0) {
        throw std::runtime_error("FATAL: Phase2 received non-positive epoch count");
    }
    if (static_cast<int>(ctx.epoch_batch_order.size()) != num_epochs) {
        throw std::runtime_error(
            "FATAL: Phase2 epoch count does not match Phase1 epoch_batch_order size (epochs=" +
            std::to_string(num_epochs) + " order_size=" +
            std::to_string(ctx.epoch_batch_order.size()) + ")");
    }
    
    ctx.logging.logger->log("Epoch " + std::to_string(epoch_idx + 1) + "/" + std::to_string(num_epochs));
    
    PHASE2_DEBUG_STDERR("[DEBUG-EPOCH] Indexing Phase1 schedule...\n");

    // Phase1 owns all batch packing (Startup/Batching/PlannedBatches.cu).
    // Phase2 NEVER shuffles ctx.data.train_views, never rebuilds
    // ctx.data.train_seq_lengths, and never calls buildEpochBatches /
    // buildBatchPayload. Phase1 authors ctx.epoch_batch_order as the exact
    // executable batch-index order for this epoch.
    if (ctx.train_payloads.empty()) {
        throw std::runtime_error(
            "FATAL: ctx.train_payloads is empty at runEpoch — "
            "PlannedBatchesReady must run during Phase1");
    }
    if (epoch_idx < 0 || epoch_idx >= static_cast<int>(ctx.epoch_batch_order.size())) {
        throw std::runtime_error(
            "FATAL: epoch_idx " + std::to_string(epoch_idx) +
            " out of range for ctx.epoch_batch_order (size=" +
            std::to_string(ctx.epoch_batch_order.size()) + ")");
    }
    const auto& batch_order = ctx.epoch_batch_order[epoch_idx];
    if (batch_order.empty()) {
        throw std::runtime_error(
            "FATAL: epoch_batch_order[" + std::to_string(epoch_idx) + "] is empty");
    }
    for (int active_idx : batch_order) {
        if (active_idx < 0 || active_idx >= static_cast<int>(ctx.train_payloads.size())) {
            throw std::runtime_error(
                "FATAL: epoch_batch_order[" + std::to_string(epoch_idx) +
                "] contains out-of-range train payload index " +
                std::to_string(active_idx));
        }
    }

    const int total_batches = static_cast<int>(batch_order.size());
    const auto epoch_start = std::chrono::steady_clock::now();
    float epoch_loss = 0.0f;
    int epoch_sequences_processed = 0;

    // Process batches: ctx.epoch_batch_order[epoch_idx] dictates which
    // Phase1-authored payload is active each step. The hard invariant from
    // the plan is:
    //     active_batch = ctx.train_payloads[ctx.epoch_batch_order[epoch_idx][batch_idx]]
    for (int batch_idx = 0; batch_idx < total_batches; ++batch_idx) {
        const GRIM::Batching::BatchPayload& payload =
            ctx.train_payloads[ctx.epoch_batch_order[epoch_idx][batch_idx]];

        validateAccumulationPositionBeforeBackward(
            ctx.optimizer, accum_steps, batch_idx, ctx.global_step);

        // runEpoch owns optimizer timing. processBatch receives an immutable
        // autograd plan and must not read or mutate OptimizerContext.
        BatchAutogradPlan autograd_plan;
        autograd_plan.should_accumulate = shouldAccumulateGradients(ctx.optimizer);
        autograd_plan.batch_idx = static_cast<uint64_t>(batch_idx);

        const int optimizer_step = static_cast<int>(ctx.optimizer.optimizer_step.step);

        BatchResult batch_result = processBatch(
            ctx, state, payload, batch_idx, epoch_idx, autograd_plan);

        // LR: index by optimizer step (NOT global_step). global_step is per
        // BatchPayload pass; using it advances warmup/decay accum_steps times
        // too fast.
        batch_result.learning_rate = scheduledLearningRateForOptimizerStep(
            ctx, optimizer_step);

        const bool should_step = advanceAccumulationOrThrow(ctx.optimizer, accum_steps);
        if (should_step) {
            runOptimizerWindowFromEpoch(
                ctx,
                state,
                payload,
                batch_result,
                batch_idx,
                accum_steps,
                optimizer_step);
        }

        // Flush device logs on the diagnostic sync interval after the optional
        // optimizer window has had a chance to emit device logs.
        if (GRIM::Diagnostics::shouldSyncDiagnostics(ctx, batch_idx)) {
            GRIM::Logging::FlushDeviceLogs();
        }

        // Rule 20: surface any first-batch CUDA error here so the real fault
        // shows up rather than a teardown cudaFree failure.
        if (batch_idx == 0) {
            cudaError_t sync_err = cudaDeviceSynchronize();
            cudaError_t last_err = (sync_err != cudaSuccess) ? sync_err : cudaGetLastError();
            if (last_err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("[CUDA] ERROR after first batch: ") + cudaGetErrorString(last_err) +
                    " (sync=" + (sync_err != cudaSuccess ? "failed" : "ok") + ")");
            }
        }

        epoch_loss += batch_result.loss;
        epoch_sequences_processed += payload.batch_size;
        result.batches_processed++;
        result.best_batch_loss = std::min(result.best_batch_loss, batch_result.loss);
        result.worst_batch_loss = std::max(result.worst_batch_loss, batch_result.loss);

        ctx.global_step++;
        
        GRIM::Telemetry::logIntervalTelemetry(ctx, state, batch_result);
        
        writeTrainingProgressStatus(
            ctx, epoch_idx, num_epochs, batch_idx, total_batches,
            batch_result.loss, epoch_loss, result.batches_processed);

    }
    
    if (result.batches_processed <= 0) {
        throw std::runtime_error("FATAL: epoch completed with 0 processed batches");
    }
    
    // No second validation/eval loop. The epoch metric is derived from the
    // training batches already executed by the explicit shared-forward +
    // autograd loss/backward path; every op and feature is verified in that
    // single path.
    result.validation.loss = epoch_loss / result.batches_processed;
    result.validation.sequences_processed = epoch_sequences_processed;
    result.validation.perplexity =
        (std::isfinite(result.validation.loss) && result.validation.loss < 50.0f)
            ? std::exp(result.validation.loss)
            : std::numeric_limits<float>::infinity();
    finalizeEpochOutcome(ctx, state, result, epoch_idx, epoch_loss, epoch_start);
    
    return result;
}

//======================================================//
//  Phase2 Main Entry Point
//======================================================//

bool executePhase2(TrainingContext& ctx) {
    // Initialize loop state
    TrainingLoopState state;
    const auto schedule_hp =
        ::GRIM::HyperParameters::trainingScheduleHP(ctx.config);

    // Construct the epoch high-loss policy detector. Train-loss spike/EWMA
    // tracking is owned by TelemetryLattice.
    state.loss_signals = makeValidationHighLossSignalBus(ctx.config);
    
    PHASE2_DEBUG_STDERR("[DEBUG] About to initialize training log...");
 
    // NOTE: per-field hyperparameter logging (warmup_fraction, learning_rate,
    // cosine_decay_*, soft_restart_*, auto_stop_*, embedding_freeze_*) is the
    // exclusive responsibility of dumpAllHyperparameters() — invoked from
    // Phase1_Startup. Do NOT re-log individual hp fields here; that creates a
    // second source of truth that drifts every time a field is added.
    EmitModuleInfo(ModuleId::Training, "Starting training...", ctx.global_step);

    const int accum_steps = validatedAccumulationSteps(ctx);
    const int num_epochs = schedule_hp.epochs;
    if (num_epochs <= 0) {
        throw std::runtime_error("FATAL: epochs must be > 0 in Phase2");
    }
    if (static_cast<int>(ctx.epoch_batch_order.size()) != num_epochs) {
        throw std::runtime_error(
            "FATAL: Phase2 epoch count does not match Phase1 epoch_batch_order size (epochs=" +
            std::to_string(num_epochs) + " order_size=" +
            std::to_string(ctx.epoch_batch_order.size()) + ")");
    }
    
    try {
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            EpochResult epoch_result = runEpoch(ctx, state, epoch, num_epochs, accum_steps);
            ctx.epochs_completed = epoch + 1;
            
            if (epoch_result.auto_stop_triggered) {
                EmitModuleInfo(ModuleId::Training,
                    std::string("Auto-stop engaged after epoch ") + std::to_string(epoch + 1) +
                    " (" + epoch_result.auto_stop_reason + "). Skipping remaining epochs.", ctx.global_step);
                break;
            }
            if (epoch + 1 < num_epochs) {
                EmitModuleInfo(ModuleId::Training,
                    "[Epoch boundary] Epoch " + std::to_string(epoch + 1) + "/" + std::to_string(num_epochs) +
                    " complete. Transitioning to epoch " + std::to_string(epoch + 2) + ".", ctx.global_step);
            } else {
                EmitModuleInfo(ModuleId::Training,
                    "[Epoch boundary] All " + std::to_string(num_epochs) + " epochs complete.", ctx.global_step);
            }
        }
    } catch (const std::exception& e) {
        EmitModuleError(ModuleId::Training, 
            std::string("TRAINING ERROR: ") + e.what(), ctx.global_step);
        
        writeTrainingErrorStatus(ctx, num_epochs, std::string(e.what()));
        
        throw;
    }
    return true;
}

} // namespace GRIMText::Training
