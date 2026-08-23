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
#include "../../Shared/Batching/BatchDeviceUpload.hpp"
#include "../../Shared/Forward/ModelForwardRuntimePayload.hpp"
#include "../../Shared/Forward/ModelForward_GPU.hpp"
#include "../../Shared/CudaAllocUtils.hpp"
#include "../../Shared/VerboseLogging.hpp"
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

void logMeanPoolHistogram(
    TrainingContext& ctx,
    const GRIM::Forward::ModelForwardOutputs& forward_outputs,
    int batch_idx,
    std::string_view phase,
    cudaStream_t stream)
{
    constexpr bool kEnableMeanPoolHistogramLogging = false;
    if (!kEnableMeanPoolHistogramLogging) {
        return;
    }

    constexpr int kBinCount = 32;
    constexpr float kMinValue = -1.0f;
    constexpr float kMaxValue = 1.0f;

    const GRIM::Tensor& mean_pool = forward_outputs.mean_pool;
    mean_pool.require("logMeanPoolHistogram mean_pool");

    const GRIM::Diagnostics::Histogram histogram =
        GRIM::Diagnostics::computeHistogram(
            mean_pool.data,
            mean_pool.numel(),
            kBinCount,
            kMinValue,
            kMaxValue,
            stream);
    GRIM::Diagnostics::logHistogram(
        ctx,
        histogram,
        "model.mean_pool",
        phase,
        batch_idx);
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

struct ProcessBatchStepStateClearScope {
    GRIM::Forward::ModelForwardOutputs& forward_outputs;
    GRIM::Autograd::AutogradLossState& loss_state;

    ~ProcessBatchStepStateClearScope() {
        forward_outputs.clear();
        loss_state.clear();
    }

    ProcessBatchStepStateClearScope(const ProcessBatchStepStateClearScope&) = delete;
    ProcessBatchStepStateClearScope& operator=(const ProcessBatchStepStateClearScope&) = delete;
};

// Device-wide usage plus stream-ordered (cudaMallocAsync/cudaFreeAsync) default
// memory-pool accounting. All queries here are non-synchronizing driver calls.
struct GpuMemoryBreakdown {
    std::uint64_t device_used = 0;
    std::uint64_t device_total = 0;
    std::uint64_t pool_reserved_current = 0;  // bytes the async pool holds from the OS
    std::uint64_t pool_reserved_high = 0;
    std::uint64_t pool_used_current = 0;       // bytes currently handed out by the pool
    std::uint64_t pool_used_high = 0;
    bool pool_ok = false;
};

bool queryGpuMemoryBreakdown(GpuMemoryBreakdown& out) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        (void)cudaGetLastError();  // don't let it masquerade as a later kernel fault
        return false;
    }
    out.device_total = static_cast<std::uint64_t>(total_bytes);
    out.device_used = static_cast<std::uint64_t>(total_bytes - free_bytes);

    int device = 0;
    cudaMemPool_t pool = nullptr;
    if (cudaGetDevice(&device) == cudaSuccess &&
        cudaDeviceGetDefaultMemPool(&pool, device) == cudaSuccess && pool != nullptr) {
        unsigned long long value = 0;
        auto readAttr = [&](cudaMemPoolAttr attr, std::uint64_t& dst) {
            value = 0;
            if (cudaMemPoolGetAttribute(pool, attr, &value) == cudaSuccess) {
                dst = static_cast<std::uint64_t>(value);
            }
        };
        readAttr(cudaMemPoolAttrReservedMemCurrent, out.pool_reserved_current);
        readAttr(cudaMemPoolAttrReservedMemHigh, out.pool_reserved_high);
        readAttr(cudaMemPoolAttrUsedMemCurrent, out.pool_used_current);
        readAttr(cudaMemPoolAttrUsedMemHigh, out.pool_used_high);
        out.pool_ok = true;
    } else {
        (void)cudaGetLastError();
    }
    return true;
}

int gpuMemoryLogInterval() {
    if (const char* raw = std::getenv("GRIM_GPU_MEM_INTERVAL")) {
        const int parsed = std::atoi(raw);
        if (parsed > 0) {
            return parsed;
        }
    }
    return 50;  // default: emit a breakdown line every 50 batches (plus the first)
}

bool shouldEmitGpuMemoryDiagnostics(const TrainingContext& ctx, int batch_idx) {
    if constexpr (!GRIM::VerboseLogging::ENABLE_GPU_MEMORY_DIAGNOSTICS) {
        return false;
    } else {
        if (!ctx.logging.logger) {
            return false;
        }
        const int interval = gpuMemoryLogInterval();
        return (batch_idx == 0) || (((batch_idx + 1) % interval) == 0);
    }
}

// Update the run-level peak GPU-memory high-water mark AND, at a modest interval,
// emit an informative [GPU_MEM] breakdown that splits device-used into async-pool
// reserved vs. non-pool bytes. This is the measurement that turns the previously
// "unaccounted" VRAM gap (VRAM_BREAKDOWN.md) into a tracked number: pool retention
// / fragmentation (async_pool_retained) vs. real working set (non_pool_used).
//
// cudaMemGetInfo and the pool-attribute queries are non-synchronizing driver calls
// (~microseconds), so sampling twice per batch is negligible next to forward+backward.
// Observability only — a failed query must never take down a training step.
void updatePeakGpuMemory(TrainingContext& ctx, int batch_idx, const char* phase) {
    GpuMemoryBreakdown mem;
    if (!queryGpuMemoryBreakdown(mem)) {
        return;
    }
    if (mem.device_used > ctx.peak_gpu_used_bytes) {
        ctx.peak_gpu_used_bytes = mem.device_used;
    }
    ctx.gpu_total_bytes = mem.device_total;

    if (!shouldEmitGpuMemoryDiagnostics(ctx, batch_idx)) {
        return;
    }

    constexpr double kMiB = 1024.0 * 1024.0;
    const double total_for_pct = static_cast<double>(mem.device_total > 0 ? mem.device_total : 1);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1)
        << "[GPU_MEM] batch=" << (batch_idx + 1)
        << " phase=" << (phase ? phase : "unknown")
        << " device_used=" << (mem.device_used / kMiB) << "MiB"
        << " device_total=" << (mem.device_total / kMiB) << "MiB"
        << " device_used_pct=" << (100.0 * static_cast<double>(mem.device_used) / total_for_pct)
        << " peak_used=" << (ctx.peak_gpu_used_bytes / kMiB) << "MiB";
    if (mem.pool_ok) {
        const std::uint64_t non_pool = (mem.device_used > mem.pool_reserved_current)
            ? (mem.device_used - mem.pool_reserved_current)
            : 0;
        const std::uint64_t pool_retained = (mem.pool_reserved_current > mem.pool_used_current)
            ? (mem.pool_reserved_current - mem.pool_used_current)
            : 0;
        oss << " async_pool_reserved=" << (mem.pool_reserved_current / kMiB) << "MiB"
            << " async_pool_reserved_high=" << (mem.pool_reserved_high / kMiB) << "MiB"
            << " async_pool_in_use=" << (mem.pool_used_current / kMiB) << "MiB"
            << " async_pool_retained=" << (pool_retained / kMiB) << "MiB"
            << " non_pool_used=" << (non_pool / kMiB) << "MiB";
    } else {
        oss << " async_pool=unavailable";
    }
    ctx.logging.logger->log(oss.str());
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
    const ::GRIM::Config::AiConfigSnapshot& hp)
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
    const std::int64_t schedule_step_wide =
        static_cast<std::int64_t>(ctx.lr_schedule_step_at_start) +
        static_cast<std::int64_t>(optimizer_step) -
        static_cast<std::int64_t>(ctx.lr_schedule_optimizer_step_at_start);
    if (schedule_step_wide < 0 ||
        schedule_step_wide > std::numeric_limits<int>::max()) {
        throw std::runtime_error(
            "LR schedule step is outside the supported integer range (optimizer_step=" +
            std::to_string(optimizer_step) + " optimizer_step_at_start=" +
            std::to_string(ctx.lr_schedule_optimizer_step_at_start) +
            " schedule_step_at_start=" +
            std::to_string(ctx.lr_schedule_step_at_start) + ")");
    }
    const int schedule_step = static_cast<int>(schedule_step_wide);
    return Internal::getScheduledLearningRate(
        *ctx.lr_schedule,
        schedule_step,
        lr_inputs.learning_rate,
        stability_hp.enabled);
}

void runOptimizerWindowFromEpoch(
    TrainingContext& ctx,
    TrainingLoopState& state,
    const GRIM::Batching::BatchPayload& payload,
    BatchResult& result,
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    GRIM::TrainingState& training_state,
    std::vector<GRIM::ParameterGroup>& parameter_groups,
    int batch_idx,
    int accum_steps,
    int optimizer_step)
{
    const bool sync_diag = GRIM::Diagnostics::shouldSyncDiagnostics(ctx, batch_idx);
    const GRIM::Tensor& lm_head_weights =
        parameter_registry.requireLmHeadParameters("runOptimizerWindowFromEpoch").weights;

    bool has_clip_metrics = false;
    GRIM::GradClip::ClipResult clip_metrics{};

    // Issue #135: gradient clipping is DEFERRED to post-accumulation. Clipping
    // ONCE on the averaged gradients matches PyTorch; the old per-slot clipping
    // crushed text gradients M×.
    const auto clipping_hp = ::GRIM::HyperParameters::gradientClippingHP(ctx.config);
    const auto schedule_hp = ::GRIM::HyperParameters::trainingScheduleHP(ctx.config);
    const auto optimizer_update_hp = ::GRIM::HyperParameters::optimizerUpdateHP(ctx.config);

    GRIM::Diagnostics::WeightSample pre_sample{};
    if (sync_diag) {
        pre_sample = GRIM::Diagnostics::sampleWeightStats(
            lm_head_weights, training_state, true);
    }

    cudaStream_t clip_stream = training_state.stream_ctrl.getPrimaryStream();

    const auto clip = GRIM::GradClip::clipGradientNorms(
        parameter_groups.data(), parameter_groups.size(),
        ctx.optimizer.optimizer_state.grad_norm_scratch, clipping_hp, schedule_hp, clip_stream);

    result.grad_rms = clip.global_rms_post;
    result.grad_rms_valid = true;
    result.gradient_clipped = clip.any_clipped();
    clip_metrics = clip;
    has_clip_metrics = true;

    if (clipping_hp.enabled) {
        GRIM::Diagnostics::runGradientNormClipDiagnostic(ctx, state, payload, clip, batch_idx, clip_stream);
    }

    // ════════════════════════════════════════════════════════════════════
    // RUNTIME tie_embeddings pointer verification (every optimizer step)
    // (extracted to Diagnostics/TieVerifyDiagnostic.cu)
    // ════════════════════════════════════════════════════════════════════
    GRIM::Diagnostics::runTieVerifyDiagnostic(ctx, parameter_registry, batch_idx);

    // Optimizer Window: runEpoch owns the accumulation-complete boundary;
    // Shared/Optimizers owns configured optimizer dispatch. The window only
    // provides filled-window grads, LR, step, stream, and grouped HP.
    GRIM::launchOptimizerUpdate(parameter_groups,
                                optimizer_update_hp,
                                result.learning_rate,
                                optimizer_step,
                                training_state.stream_ctrl.getPrimaryStream());

    // Rule 20: post-optimizer weight NaN spot check. Stream ownership stays
    // in Phase2; the guard only inspects optimizer parameter groups.
    {
        cudaError_t post_step_sync = cudaStreamSynchronize(
            training_state.stream_ctrl.getPrimaryStream());
        if (post_step_sync != cudaSuccess) {
            throw std::runtime_error(
                std::string("[FATAL] Failed to synchronize stream before post-optimizer finite check: ") +
                cudaGetErrorString(post_step_sync));
        }

        GRIM::Diagnostics::checkPostOptimizerWeightsFinite(
            parameter_groups.data(),
            parameter_groups.size(),
            optimizer_step,
            result.learning_rate,
            batch_idx);
    }

    GRIM::Diagnostics::runOptimizerMomentDiagnostic(
        ctx, batch_idx, accum_steps, sync_diag);

    // Post-optimizer LM-head sample, GradTrace POST log, [UpdateMag],
    // and optimizer-boundary adaptive update trace.
    GRIM::Diagnostics::runPostOptimizerWeightTrace(
        ctx,
        result,
        parameter_registry,
        training_state,
        parameter_groups,
        optimizer_update_hp,
        pre_sample,
        sync_diag);

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
        tel_input.text_loss         = result.text_loss;
        tel_input.selector_loss     = result.selector_loss;
        tel_input.max_seq_len       = payload.max_seq_len;
        tel_input.batch_idx         = batch_idx;
        tel_input.global_step       = ctx.global_step;
        tel_input.actual_vocab_size = payload.vocab_size;
        tel_input.d_model           = GRIM::HyperParameters::snapshotTrainingConfigField<int>(ctx.config, "d_model");

        GRIM::Telemetry::updateTelemetryObservations(
            ctx,
            training_state,
            ctx.gpu_model,
            parameter_registry,
            tel_input,
            clip_metrics.metrics);
    }
}

GRIMText::Training::Startup::ForwardTopologyView validateTrainingForwardInputs(
    const GRIM::Config::AiConfigSnapshot& config,
    GRIMText::Training::Startup::GpuModelState& gpu_model,
    const GRIM::Batching::BatchPayload& payload,
    const char* caller)
{
    payload.validate(caller);

    return gpu_model.requireForwardTopology(config, caller);
}

void configureAutogradLossInputs(
    GRIM::Autograd::AutogradContext& autograd_ctx,
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
}

GRIM::Forward::ModelForwardRuntimePayload buildTrainingForwardRuntimePayload()
{
    return {};
}

GRIM::Forward::ModelForwardRequest buildTrainingForwardRequest(
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const GRIM::PBM::PBMState& pbm,
    const GRIM::Config::AiConfigSnapshot& config,
    const GRIMText::Training::Startup::ForwardTopologyView& forward_topology,
    const GRIM::TrainingState& training_state,
    cudaStream_t stream,
    const GRIM::Batching::BatchPayload& payload,
    const GRIM::Batching::BatchDeviceBindings& bindings,
    uint64_t batch_idx,
    bool emit_selector_logits)
{
    GRIM::Forward::ModelForwardRequest request{};
    request.config = &config;
    request.gpu_encoder = forward_topology.gpu_encoder;
    request.parameter_registry = &parameter_registry;
    request.pbm = &pbm;
    request.cublas_handle = training_state.cublas_handle.get();
    request.stream = stream;
    request.payload = &payload;
    request.bindings = &bindings;
    request.batch_idx = batch_idx;
    request.graph = GRIM::Forward::ModelForwardGraphPolicy{
        /*connect_parameter_graph=*/true,
        /*enable_dropout=*/true,
        /*emit_selector_logits=*/emit_selector_logits};
    return request;
}

} // namespace

//======================================================//
//  Batch Processing Implementation
//======================================================//

BatchResult processBatch(
    TrainingContext& ctx,
    TrainingLoopState& state,
    GRIM::Batching::BatchPayload& payload,
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

    // Peak-memory sample BEFORE this batch's forward graph is allocated. The
    // previous batch's forward_outputs/loss_state were already cleared at that
    // batch's processBatch scope exit, so device_used here is the persistent
    // static floor: params + grads + optimizer state + CUDA context +
    // cuBLAS/FlashAttention workspace + durable TrainingState buffers. Compared
    // against the post_forward / post_backward samples this attributes the
    // per-step retained forward graph vs. the static baseline.
    updatePeakGpuMemory(ctx, batch_idx, "pre_forward");

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
    auto& training_state = ctx.requireTrainingState("processBatch");
    GRIM::Autograd::AutogradLossState autograd_loss_state;
    cudaStream_t stream = training_state.stream_ctrl.getPrimaryStream();
    // Sync slice: upload the prebuilt host BatchPayload once and reuse the
    // returned BatchDeviceBindings across shared forward, loss, and backward —
    // payload itself is host-only/immutable and never carries device pointers.
    const auto train_bindings = GRIM::Batching::uploadBatchToDevice(
        ctx.config,
        payload,
        stream);
    const auto loss_config = GRIM::HyperParameters::lossConfigHP(ctx.config);
    const auto& model_config = ctx.config;

    const auto forward_topology = validateTrainingForwardInputs(
        ctx.config,
        ctx.gpu_model,
        payload,
        "processBatch");

    training_state.autograd_batch_idx = plan.batch_idx;
    TapeSkipScope tape_skip_scope(plan.should_accumulate);

    GRIM::Forward::ModelForwardRuntimePayload runtime_payload =
        buildTrainingForwardRuntimePayload();
    const int optimizer_step = static_cast<int>(ctx.optimizer.optimizer_step.step);
    GRIM::Forward::ModelForwardRequest forward_request =
        buildTrainingForwardRequest(
            ctx.parameter_registry,
            ctx.pbm_owner.state(),
            model_config,
            forward_topology,
            training_state,
            stream,
            payload,
            train_bindings,
            plan.batch_idx,
            /*emit_selector_logits=*/false);

    if constexpr (GRIM::VerboseLogging::ENABLE_GPU_MEMORY_DIAGNOSTICS &&
                  GRIM::VerboseLogging::ENABLE_GPU_ALLOCATION_LEDGER) {
        GRIM::CudaAlloc::Ledger::beginScope(
            ("ForwardAllocationSizes batch=" + std::to_string(batch_idx + 1)).c_str());
    }
    auto forward_outputs = GRIM::Forward::executeModelForward(forward_request, runtime_payload);
    Internal::logMeanPoolHistogram(
        ctx,
        forward_outputs,
        batch_idx,
        "training",
        stream);
    if constexpr (GRIM::VerboseLogging::ENABLE_GPU_MEMORY_DIAGNOSTICS &&
                  GRIM::VerboseLogging::ENABLE_GPU_ALLOCATION_LEDGER) {
        if (ctx.logging.logger) {
            ctx.logging.logger->log(GRIM::CudaAlloc::Ledger::endScopeSummary(80));
        } else {
            GRIM::CudaAlloc::Ledger::endScopeSummary(0);
        }
    }
    {
        std::ostringstream oss;
        oss << "EXPLICIT_TRAINING_FORWARD_COMPLETE batch=" << (batch_idx + 1)
            << " total_tokens=" << payload.total_tokens
            << " max_seq_len=" << payload.max_seq_len
            << " optimizer_step=" << optimizer_step
            << " accumulate=" << (plan.should_accumulate ? "true" : "false");
        EmitModuleInfo(ModuleId::ForwardPass, oss.str(), ctx.global_step);
    }
    // Peak-memory sample: with all forward activations live alongside the
    // persistent params / grad buffers / optimizer state, this brackets the high
    // end of the step (backward then frees activations as it fills grads).
    updatePeakGpuMemory(ctx, batch_idx, "post_forward");
    // Per-tensor forward-output size breakdown: lists every live retained
    // ModelForwardOutputs tensor with element/byte size plus the total. Attributes
    // the post_forward memory jump to individual forward products.
    if (shouldEmitGpuMemoryDiagnostics(ctx, batch_idx)) {
        ctx.logging.logger->log(
            forward_outputs.describeRetainedSizes("batch=" + std::to_string(batch_idx + 1)));
    }
    // Rule 20 ownership taxonomy: processBatch owns the single batch-boundary
    // clear path for the active forward/loss step-state. Do NOT add a second
    // explicit clear() site inside this function.
    ProcessBatchStepStateClearScope step_state_clear_scope{
        forward_outputs,
        autograd_loss_state};

    GRIM::Autograd::AutogradContext autograd_ctx = GRIM::Autograd::initAutogradContext(
        &model_config,
        &training_state,
        forward_outputs,
        autograd_loss_state,
        forward_topology.gpu_encoder,
        forward_topology.execution_block_enabled,
        ctx.parameter_registry,
        training_state.cublas_handle.get(),
        stream,
        payload,
        train_bindings,
        plan.batch_idx,
        ctx.global_step);

    configureAutogradLossInputs(
        autograd_ctx,
        training_state,
        payload,
        loss_config,
        plan.should_accumulate);

    auto loss_result = GRIM::Autograd::computeAutogradLoss(
        autograd_ctx,
        payload,
        loss_config);
    if (!loss_result.success) {
        throw std::runtime_error(
            "[computeAutogradLoss] FAILED batch=" + std::to_string(batch_idx + 1) +
            ": " + loss_result.error_message);
    }
    if (!std::isfinite(loss_result.loss_value)) {
        throw std::runtime_error("Non-finite loss: " + std::to_string(loss_result.loss_value));
    }

    if constexpr (GRIM::VerboseLogging::ENABLE_RHO_BUILDUP_LOGS) {
        const GRIM::Diagnostics::RhoDiagnosticOptions rho_pre_backward_options{
            GRIM::Diagnostics::RhoDiagnosticPhase::PostForwardPreBackward,
            false,
            true};
        GRIM::Diagnostics::computeRhoDiagnostic(
            ctx,
            payload,
            forward_outputs,
            batch_idx,
            rho_pre_backward_options);
    }

    auto backward_result = GRIM::Autograd::executeAutogradBackward(
        autograd_ctx,
        plan.should_accumulate);
    if (!backward_result.success) {
        throw std::runtime_error(
            "[executeAutogradBackward] FAILED batch=" + std::to_string(batch_idx + 1) +
            ": " + backward_result.error_message);
    }
    {
        std::ostringstream oss;
        oss << "EXPLICIT_TRAINING_BACKWARD_COMPLETE batch=" << (batch_idx + 1)
            << " total_tokens=" << payload.total_tokens
            << " max_seq_len=" << payload.max_seq_len
            << " accumulate=" << (plan.should_accumulate ? "true" : "false");
        EmitModuleInfo(ModuleId::BackwardPass, oss.str(), ctx.global_step);
    }
    // Peak-memory sample: captures any backward-only transient (reduction
    // scratch, etc.) the post-forward sample missed. The high-water mark keeps
    // whichever bracket is larger.
    updatePeakGpuMemory(ctx, batch_idx, "post_backward");

    result.loss = loss_result.loss_value;
    result.text_loss = loss_result.text_loss;
    result.selector_loss = loss_result.selector_loss;
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

    if (!forward_outputs.logits_tensor.data) {
        throw std::runtime_error(
            "processBatch: live logits tensor is NULL after successful explicit shared forward — "
            "diagnostics must run before processBatch step-state teardown");
    }
    // Log model predictions (what it predicts vs targets) - uses ForwardPass module for filtering
    // (extracted to Diagnostics/PredictionDistributionDiagnostic.cu)
    GRIM::Diagnostics::runPredictionDistributionAndLogitTrace(
        ctx,
        payload,
        forward_outputs.logits_tensor,
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
    if constexpr (GRIM::VerboseLogging::ENABLE_LOGIT_SCALE_DIAGNOSTICS) {
        const GRIM::Tensor* live_lm_head_input = forward_outputs.liveLmHeadInputOrNull();
        if (!live_lm_head_input || !live_lm_head_input->data) {
            throw std::runtime_error(
                "processBatch: live LM-head input tensor is NULL after successful explicit shared forward");
        }
        GRIM::Diagnostics::runLogitScaleDiagnostic(
            ctx,
            ctx.parameter_registry,
            payload,
            forward_outputs.logits_tensor,
            *live_lm_head_input,
            batch_idx);
    }
    if constexpr (GRIM::VerboseLogging::ENABLE_RHO_BUILDUP_LOGS ||
                  GRIM::VerboseLogging::ENABLE_RHO_BUILDUP_TELEMETRY) {
        const GRIM::Diagnostics::RhoDiagnosticOptions rho_post_backward_options{
            GRIM::Diagnostics::RhoDiagnosticPhase::PostBackward,
            GRIM::VerboseLogging::ENABLE_RHO_BUILDUP_TELEMETRY,
            GRIM::VerboseLogging::ENABLE_RHO_BUILDUP_LOGS};
        GRIM::Diagnostics::computeRhoDiagnostic(
            ctx,
            payload,
            forward_outputs,
            batch_idx,
            rho_post_backward_options);
    }
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
    GRIM::Diagnostics::runSpecialTokenDiagnostic(
        ctx,
        ctx.parameter_registry,
        payload,
        batch_idx);

    const bool sync_diag = GRIM::Diagnostics::shouldSyncDiagnostics(ctx, batch_idx);

    if (sync_diag) {
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

    // Rule 20 single-owner clear: the processBatch-local step-state guard owns
    // the clear; the tape flush below does not touch autograd intermediates.
    if (ctx.logging.tape) {
        ctx.logging.tape->flush();
    }

    return result;
}

//======================================================//
//  End-of-Epoch Validation
//======================================================//
//
//  Forward-only evaluation over the Phase1-authored ctx.val_payloads
//  (val_sequence_count in the startup dump). This mirrors the processBatch
//  forward + loss path with three deliberate differences:
//
//    1. Read-only ModelForwardGraphPolicy — dropout DISABLED, no autograd edges
//       connected to durable parameters, no backward graph retained (identical
//       to the inference forward in Phase2_InferenceLoop.cu).
//    2. NO executeAutogradBackward(): computeAutogradLoss() only reads the loss
//       tensor value via a D2H copy; only backward writes parameter gradients.
//       So validation can never corrupt accumulated training gradients or the
//       model weights, and runs no optimizer step.
//    3. No training diagnostics / tape entries / equation-CSV rows.
//
//  Phase2 NEVER builds val batches here — PlannedBatchesReady authored
//  ctx.val_payloads at startup (Rule 20: fixed batch membership, no per-step
//  batch creation in the hot loop). result.validation feeds best-checkpoint
//  selection and auto-stop in finalizeEpochOutcome, so it must be the TRUE
//  held-out metric rather than the training loss.

ValidationResult runValidation(TrainingContext& ctx) {
    ValidationResult result;

    const int total_val_batches = static_cast<int>(ctx.val_payloads.size());
    if (total_val_batches == 0) {
        // No validation data configured. +inf so best-val tracking ignores it
        // (saveBestCheckpoint compares with `<`; auto-stop skips non-finite).
        result.loss = std::numeric_limits<float>::infinity();
        result.perplexity = std::numeric_limits<float>::infinity();
        result.sequences_processed = 0;
        ctx.logging.logger->log(
            "[Val] No validation payloads (val_sequence_count=0) — skipping end-of-epoch validation");
        return result;
    }

    ctx.logging.logger->log("[Val] Running end-of-epoch validation over " +
                            std::to_string(total_val_batches) +
                            " Phase1-authored val payloads");

    // PRE-VALIDATION SAFETY: drain any deferred CUDA error so it does not
    // surface as an SEH exception deep inside the eval forward.
    {
        cudaError_t sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            ctx.logging.logger->log("[Val] WARNING: cudaDeviceSynchronize before validation returned: " +
                std::string(cudaGetErrorString(sync_err)));
        }
        cudaError_t deferred_err = cudaGetLastError();
        if (deferred_err != cudaSuccess) {
            ctx.logging.logger->log("[Val] WARNING: cleared deferred CUDA error before validation: " +
                std::string(cudaGetErrorString(deferred_err)));
        }
    }

    auto& training_state = ctx.requireTrainingState("runValidation");
    cudaStream_t stream = training_state.stream_ctrl.getPrimaryStream();
    const auto loss_config = GRIM::HyperParameters::lossConfigHP(ctx.config);
    const auto& model_config = ctx.config;

    // Match the training loss composition so the validation number is comparable.

    // Validation must not record training-tape entries or equation-CSV rows.
    TapeSkipScope tape_skip_scope(/*skip=*/true);

    // Accumulate in double precision: 8k+ sequences across many batches.
    double val_loss_weighted = 0.0;
    int val_sequences_processed = 0;
    const auto val_start_time = std::chrono::steady_clock::now();

    for (int val_idx = 0; val_idx < total_val_batches; ++val_idx) {
        GRIM::Batching::BatchPayload& val_payload = ctx.val_payloads[val_idx];

        if (val_idx % 50 == 0) {
            ctx.logging.logger->log("[Val] batch " + std::to_string(val_idx + 1) + "/" +
                std::to_string(total_val_batches) +
                " (seqs=" + std::to_string(val_sequences_processed) + ")");
        }

        if (val_payload.batch_size == 0) {
            // Rule 20: PlannedBatches validates payloads at startup, so an empty
            // payload here means a startup invariant regressed.
            throw std::runtime_error(
                "[Val] empty payload at batch " + std::to_string(val_idx + 1) + "/" +
                std::to_string(total_val_batches) +
                " — Phase1 PlannedBatches authored an empty val payload");
        }

        // Sync slice: upload the host-only payload to device once per val batch,
        // then reuse the bindings across the eval forward + loss.
        const auto val_bindings = GRIM::Batching::uploadBatchToDevice(
            ctx.config, val_payload, stream);

        const auto forward_topology = validateTrainingForwardInputs(
            ctx.config, ctx.gpu_model, val_payload, "runValidation");

        GRIM::Forward::ModelForwardRuntimePayload runtime_payload =
            buildTrainingForwardRuntimePayload();

        GRIM::Forward::ModelForwardRequest forward_request{};
        forward_request.config = &model_config;
        forward_request.gpu_encoder = forward_topology.gpu_encoder;
        forward_request.parameter_registry = &ctx.parameter_registry;
        forward_request.pbm = &ctx.pbm_owner.state();
        forward_request.cublas_handle = training_state.cublas_handle.get();
        forward_request.stream = stream;
        forward_request.payload = &val_payload;
        forward_request.bindings = &val_bindings;
        forward_request.batch_idx = static_cast<uint64_t>(val_idx);
        // Eval policy: read-only forward (no autograd edges, no retained backward
        // graph) with dropout DISABLED — identical to the inference forward.
        // The text-CE and selector terms read only explicitly emitted outputs,
        // valid under this read-only policy.
        forward_request.graph = GRIM::Forward::ModelForwardGraphPolicy{
            /*connect_parameter_graph=*/false,
            /*enable_dropout=*/false,
            /*emit_selector_logits=*/false};

        auto forward_outputs = GRIM::Forward::executeModelForward(forward_request, runtime_payload);
        Internal::logMeanPoolHistogram(
            ctx,
            forward_outputs,
            val_idx,
            "validation",
            stream);

        GRIM::Autograd::AutogradLossState autograd_loss_state;
        // Rule 20 single-owner clear for this val step's forward/loss state.
        ProcessBatchStepStateClearScope step_state_clear_scope{
            forward_outputs, autograd_loss_state};

        GRIM::Autograd::AutogradContext autograd_ctx = GRIM::Autograd::initAutogradContext(
            &model_config,
            &training_state,
            forward_outputs,
            autograd_loss_state,
            forward_topology.gpu_encoder,
            forward_topology.execution_block_enabled,
            ctx.parameter_registry,
            training_state.cublas_handle.get(),
            stream,
            val_payload,
            val_bindings,
            static_cast<uint64_t>(val_idx),
            ctx.global_step);

        configureAutogradLossInputs(
            autograd_ctx, training_state, val_payload, loss_config,
            /*skip_equation_logging=*/true);

        const auto loss_result = GRIM::Autograd::computeAutogradLoss(
            autograd_ctx, val_payload, loss_config);
        if (!loss_result.success) {
            throw std::runtime_error(
                "[Val] computeAutogradLoss FAILED at batch " + std::to_string(val_idx + 1) +
                ": " + loss_result.error_message);
        }
        // Deliberately NO executeAutogradBackward() and NO optimizer step:
        // gradients and weights are left untouched by validation.

        if (!std::isfinite(loss_result.loss_value)) {
            throw std::runtime_error(
                "[Val] non-finite loss at batch " + std::to_string(val_idx + 1) +
                " (loss=" + std::to_string(loss_result.loss_value) +
                ") — fix the model/data, do not skip");
        }

        cudaError_t batch_err = cudaGetLastError();
        if (batch_err != cudaSuccess) {
            throw std::runtime_error(
                "[Val] CUDA error after batch " + std::to_string(val_idx + 1) +
                ": " + std::string(cudaGetErrorString(batch_err)));
        }

        val_loss_weighted += static_cast<double>(loss_result.loss_value) *
                             static_cast<double>(val_payload.batch_size);
        val_sequences_processed += val_payload.batch_size;
    }

    const auto val_duration = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - val_start_time);

    result.sequences_processed = val_sequences_processed;
    result.loss = (val_sequences_processed > 0)
        ? static_cast<float>(val_loss_weighted / static_cast<double>(val_sequences_processed))
        : std::numeric_limits<float>::infinity();
    result.perplexity = (std::isfinite(result.loss) && result.loss < 50.0f)
        ? std::exp(result.loss)
        : std::numeric_limits<float>::infinity();

    ctx.logging.logger->log("[Val] " + Internal::formatMetric("loss", result.loss) + " " +
                            Internal::formatMetric("ppl", result.perplexity, 3) +
                            " seqs=" + std::to_string(val_sequences_processed) +
                            " time=" + std::to_string(val_duration.count()) + "s");
    EmitModuleInfo(ModuleId::Validation,
        "Validation complete: loss=" + Internal::formatScalar(result.loss) +
        " ppl=" + Internal::formatScalar(result.perplexity, 3) +
        " seqs=" + std::to_string(val_sequences_processed), ctx.global_step);

    return result;
}

//======================================================//
//  Epoch Implementation
//======================================================//

EpochResult runEpoch(
    TrainingContext& ctx,
    TrainingLoopState& state,
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
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
    auto& training_state = ctx.requireTrainingState("runEpoch");
    auto& parameter_groups = parameter_registry.requireParameterGroups("runEpoch");
    float epoch_loss = 0.0f;

    // Process batches: ctx.epoch_batch_order[epoch_idx] dictates which
    // Phase1-authored payload is active each step. The hard invariant from
    // the plan is:
    //     active_batch = ctx.train_payloads[ctx.epoch_batch_order[epoch_idx][batch_idx]]
    for (int batch_idx = 0; batch_idx < total_batches; ++batch_idx) {
        GRIM::Batching::BatchPayload& payload =
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
                parameter_registry,
                training_state,
                parameter_groups,
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
    
    // End-of-epoch validation over the held-out ctx.val_payloads: forward-only,
    // dropout disabled, no backward/optimizer (see runValidation). result.validation
    // drives best-checkpoint selection and auto-stop in finalizeEpochOutcome, so it
    // MUST be the true held-out metric — not the training loss. The training-loss
    // average is still surfaced separately as result.avg_loss.
    result.validation = runValidation(ctx);
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
    auto& parameter_registry = ctx.parameter_registry;
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
            EpochResult epoch_result = runEpoch(ctx, state, parameter_registry, epoch, num_epochs, accum_steps);
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
