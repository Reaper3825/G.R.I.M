//======================================================//
//  Phase2_TrainingLoop.cu
//  Core training computation and logic
//======================================================//
//
//  IMPLEMENTATION
//  ==============
//  This file implements all training computation:
//  1. Epoch iteration with dynamic batching
//  2. Batch construction with content weighting
//  3. Forward/backward passes with gradient accumulation
//  4. Gradient clipping (token-normalized + adaptive)
//  5. Optimizer steps with dynamic LR
//  6. Validation and checkpointing
//  7. Auto-stop and soft restart logic
//
//  Author: Austin Wadkins
//  Date: December 2025
//  Version: 1.0.0
//======================================================//

#include "Phase2_TrainingLoop.hpp"
#include "../OptimizerCheckpoint.hpp"
#include "../Diagnostics/DiagnosticInference.hpp"
#include "../Diagnostics/RhoDiagnostic.hpp"

#include "../../Shared/LogRecorder/LogRecorder.hpp"
#include "../../Shared/CudaAllocUtils.hpp"
#include "../../Shared/Gradients/GradStatsCollector.hpp"

using GRIM::CudaAlloc::cudaMallocOrThrow;
#include "../../Shared/Gradients/GradientCC_GPU.hpp"       // GradClip::clipGradientNorms (registry-level clipping)
#include "../../Shared/Dynamic_LR/LRSchedule.hpp"          // GRIM::LR::LRSchedule (exposed LR curve)
#include "../../Shared/GradNorm/GradNormGPU.hpp"           // GradNorm::measureGradientNorms, GradMetrics
#include "../../Shared/TrainingState/TrainingState_GPU.hpp"
#include "../../Shared/LogRecorder/BatchLogTape.hpp"
#include "../../Shared/Telemetry/TelemetryUpdate.hpp"
#include "../Diagnostics/TrainingDiagnostics.hpp"
#include "../../Shared/UnigramByte/Unigram.hpp"
#include "../../Shared/UnigramByte/AtomTable.hpp"
#include "../../Shared/Batching/BatchPayload.hpp"
#include "../Autograd/AutogradTraining.hpp"  // autogradTrainingStep: unified forward+loss+backward
#include "../../Shared/Optimizers/AdamW/AdamW_Kernal_GPU.hpp"  // launchAdamWStep, resetAdamWMoments, scaleAdamWMoments
#include "../../Shared/Optimizers/RAdam/RAdam_Kernal_GPU.hpp"  // launchRAdamStep — selectable via training.config.optimizer.kind
#include "../../../../../control/ai_config_paths.hpp"  // For resolveGrimRoot()

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <filesystem>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

// Module logging aliases
using GRIM::Logging::ModuleId;
using GRIM::Logging::EmitModuleInfo;
using GRIM::Logging::EmitModuleWarning;
using GRIM::Logging::EmitModuleError;

namespace GRIMText::Training {
namespace fs = std::filesystem;

TrainingLoopState::~TrainingLoopState() = default;

//======================================================//
//  String Utilities
//======================================================//

namespace {

int readEnvInt(const char* name, int fallback) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) {
        return fallback;
    }
    char* end = nullptr;
    long value = std::strtol(raw, &end, 10);
    if (end == raw) {
        return fallback;
    }
    if (value < 0) {
        return fallback;
    }
    return static_cast<int>(value);
}

std::string readEnvString(const char* name, const std::string& fallback) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) {
        return fallback;
    }
    return std::string(raw);
}

bool isPhase2DebugEnabled() {
    static const bool enabled = readEnvInt("GRIM_PHASE2_DEBUG", 0) > 0;
    return enabled;
}

#define PHASE2_DEBUG_STDERR(...)             \
    do {                                     \
        if (isPhase2DebugEnabled()) {        \
            fprintf(stderr, __VA_ARGS__);    \
        }                                    \
    } while (0)

#define PHASE2_DEBUG_FLUSH_STDERR()          \
    do {                                     \
        if (isPhase2DebugEnabled()) {        \
            fflush(stderr);                  \
        }                                    \
    } while (0)

bool shouldSyncDiagnostics(const TrainingContext& ctx, std::size_t batch_idx) {
    // Skip expensive D2H syncs when equation logging disabled (avoids GPU pipeline drain)
    if (!ctx.logging.tape || !ctx.logging.tape->accepts(GRIM::Logging::LogLevel::Debug)) {
        return false;
    }
    const int default_interval = ctx.config.hyperparameters.log_interval;
    const int interval = readEnvInt("GRIM_SYNC_INTERVAL", default_interval);
    if (interval <= 0) {
        return false;
    }
    return ((batch_idx + 1) % static_cast<std::size_t>(interval)) == 0;
}

bool shouldLogLogitTrace(const TrainingContext& ctx, std::size_t batch_idx) {
    const auto& hp = ctx.config.hyperparameters;
    if (!hp.logit_update_trace_enabled) {
        return false;
    }
    const int interval = readEnvInt("GRIM_LOGIT_TRACE_INTERVAL", hp.logit_update_trace_interval);
    if (interval <= 0) {
        return false;
    }
    return ((batch_idx + 1) % static_cast<std::size_t>(interval)) == 0;
}

struct AtomStats {
    int total_atoms = 0;
    int total_tokens = 0;
    int min_atoms = 0;
    int max_atoms = 0;
    double avg_atoms = 0.0;
};

AtomStats computeAtomStats(const std::vector<std::vector<int>>& batch_inputs,
                           const GRIM::Tokenizer::UniByte& tokenizer,
                           std::vector<int>* per_seq_atoms,
                           std::vector<int>* per_seq_lengths) {
    AtomStats stats{};
    if (batch_inputs.empty()) {
        return stats;
    }

    stats.min_atoms = std::numeric_limits<int>::max();
    for (const auto& seq : batch_inputs) {
        int atom_count = 0;
        for (int tid : seq) {
            if (tokenizer.isAtomToken(tid)) {
                ++atom_count;
            }
        }
        if (per_seq_atoms) {
            per_seq_atoms->push_back(atom_count);
        }
        if (per_seq_lengths) {
            per_seq_lengths->push_back(static_cast<int>(seq.size()));
        }
        stats.total_atoms += atom_count;
        stats.total_tokens += static_cast<int>(seq.size());
        stats.min_atoms = std::min(stats.min_atoms, atom_count);
        stats.max_atoms = std::max(stats.max_atoms, atom_count);
    }

    stats.avg_atoms = static_cast<double>(stats.total_atoms) /
                      static_cast<double>(batch_inputs.size());
    return stats;
}

bool shouldLogAtomStats(const TrainingContext& ctx, int batch_idx) {
    const int interval = ctx.config.hyperparameters.atom_stats_interval;
    if (interval <= 0) {
        return false;
    }
    return ((batch_idx + 1) % interval) == 0;
}

constexpr int kMomentSamplePerGroup = 4;

struct MomentSample {
    bool valid = false;
    float m_rms = 0.0f;
    float v_rms = 0.0f;
    std::size_t groups = 0;
    std::size_t samples = 0;
};

MomentSample sampleOptimizerMomentStats(const GRIM::TrainingState& ts, bool sync_for_host = false) {
    MomentSample sample{};
    if (!sync_for_host) {
        return sample;
    }
    const std::size_t group_count = std::min(ts.optimizer_m_states.size(),
                                             ts.optimizer_v_states.size());
    if (group_count == 0) {
        return sample;
    }

    std::vector<float> m_host(group_count * kMomentSamplePerGroup, 0.0f);
    std::vector<float> v_host(group_count * kMomentSamplePerGroup, 0.0f);
    std::vector<std::size_t> counts(group_count, 0);

    cudaStream_t stream = ts.stream_ctrl.getPrimaryStream();
    bool has_copy = false;
    for (std::size_t i = 0; i < group_count; ++i) {
        const float* m_ptr = ts.optimizer_m_states[i].data;
        const float* v_ptr = ts.optimizer_v_states[i].data;
        const std::size_t size = ts.optimizer_m_states[i].numel();
        const std::size_t count = std::min<std::size_t>(kMomentSamplePerGroup, size);
        if (!m_ptr || !v_ptr || count == 0) {
            continue;
        }
        counts[i] = count;
        has_copy = true;
        cudaMemcpyAsync(m_host.data() + i * kMomentSamplePerGroup, m_ptr,
                        count * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(v_host.data() + i * kMomentSamplePerGroup, v_ptr,
                        count * sizeof(float), cudaMemcpyDeviceToHost, stream);
    }

    if (!has_copy) {
        return sample;
    }

    cudaStreamSynchronize(stream);

    double m_sum_sq = 0.0;
    double v_sum_sq = 0.0;
    std::size_t total_samples = 0;
    for (std::size_t i = 0; i < group_count; ++i) {
        const std::size_t count = counts[i];
        for (std::size_t j = 0; j < count; ++j) {
            const float m_val = m_host[i * kMomentSamplePerGroup + j];
            const float v_val = v_host[i * kMomentSamplePerGroup + j];
            m_sum_sq += static_cast<double>(m_val) * m_val;
            v_sum_sq += static_cast<double>(v_val) * v_val;
        }
        total_samples += count;
    }

    if (total_samples == 0) {
        return sample;
    }

    sample.m_rms = static_cast<float>(std::sqrt(m_sum_sq / total_samples));
    sample.v_rms = static_cast<float>(std::sqrt(v_sum_sq / total_samples));
    sample.groups = group_count;
    sample.samples = total_samples;
    sample.valid = true;
    return sample;
}


} // anonymous namespace

// GuessCacheScope, GuessCacheBatchBuffers implementations moved to
// Layers/GRIMTS/GuessCacheTraining.cu (namespace GRIMTS::Training)

//======================================================//
//  GPU-Native Telemetry Control
//======================================================//
// All decision logic runs on GPU via TelemetryControl::evaluate()
// See: TelemetryControl_GPU.{cu,hpp} for kernel implementation
// Only 48-byte ControlDecision struct synced to CPU

//======================================================//
//  Internal Helper Implementations
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

std::string formatGradientComponents(const GRIM::GradNorm::GradMetrics& gm, bool tied) {
    using GM = GRIM::GradNorm::GradMetrics;
    std::ostringstream comp_msg;
    comp_msg << "[GradTrace] COMPONENTS(rms):";
    
    // When tie_embeddings=true: tied buffer registered as LM_HEAD type (embedding_sum_sq=0)
    // When tie_embeddings=false: separate EMBEDDING and LM_HEAD groups
    // Use precision=6 so small per-parameter RMS values (e.g. attn ~0.00004) don't
    // display as 0.0000 with the default precision=4 (Issue #150)
    constexpr int kComponentPrecision = 10;

    if (tied) {
        comp_msg << " emb_lm_tied=" << formatScalar(GM::rms(gm.lm_head_sum_sq, gm.lm_head_count), kComponentPrecision);
    } else {
        comp_msg << " emb=" << formatScalar(GM::rms(gm.embedding_sum_sq, gm.embedding_count), kComponentPrecision)
                 << " lm=" << formatScalar(GM::rms(gm.lm_head_sum_sq, gm.lm_head_count), kComponentPrecision);
    }
    
    comp_msg << " attn=" << formatScalar(GM::rms(gm.attention_sum_sq, gm.attention_count), kComponentPrecision)
             << " ffn=" << formatScalar(GM::rms(gm.ffn_sum_sq, gm.ffn_count), kComponentPrecision)
             << " rmsnorm=" << formatScalar(GM::rms(gm.rmsnorm_sum_sq, gm.rmsnorm_count), kComponentPrecision);

    comp_msg << " tied=" << (tied ? "yes" : "no");
    
    // Include ScratchBlock if enabled
    if (gm.scratchblock_sum_sq > 0.0f) {
        comp_msg << " sb=" << formatScalar(GM::rms(gm.scratchblock_sum_sq, gm.scratchblock_count), kComponentPrecision);
    }
    
    return comp_msg.str();
}

/// Query the LR schedule at a given step, respecting stability overrides.
/// Delegates to GRIM::LR::LRSchedule for the actual curve computation.
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

GRIM::Batching::BatchSchedule buildEpochBatches(
    const TrainingContext& ctx,
    const GRIM::DynaSeq::Catalog& catalog,
    int batch_size,
    int global_step,
    int epoch,
    TrainingLogger& logger) {
    
    PHASE2_DEBUG_STDERR("[DEBUG-BUILD] ENTER buildEpochBatches batch_size=%d\n", batch_size);
    
    const bool warmup_phase = (global_step < kWarmupTokenSteps);
    const bool curriculum_active = (epoch < kCurriculumEpochs);
    
    PHASE2_DEBUG_STDERR("[DEBUG-BUILD] warmup_phase=%d curriculum_active=%d\n",
                        warmup_phase ? 1 : 0, curriculum_active ? 1 : 0);
    
    GRIM::Batching::BatchOptions opts;
    
    // Derive token budget directly from batch_size and max_seq_len
    const auto& model_cfg = ctx.model->getConfig();
    const uint32_t config_token_budget = static_cast<uint32_t>(batch_size) *
        static_cast<uint32_t>(model_cfg.max_seq_len);
    opts.max_tokens_per_batch = config_token_budget;
    
    opts.max_batch_size = static_cast<uint32_t>(batch_size);
    
    // Issue #90: GREEDY forced — SIMILARITY_GROUPED causes mode collapse promotes mode collapse (many small batches of similar sequences → unstable loss spikes)
    opts.strategy = GRIM::Batching::PackingStrategy::GREEDY;
    
    opts.similarity_threshold = 0.30f;
    opts.prefer_short_first = curriculum_active;
    opts.curriculum_progress = curriculum_active ? static_cast<float>(epoch + 1) / kCurriculumEpochs : 1.0f;
    opts.rng_seed = ctx.rng.data_seed + epoch;  // Vary by epoch for different shuffle each epoch
    // Use RANDOM ordering to avoid loss spikes at epoch end
    // LENGTH_ASCENDING causes easy→hard progression: loss plateaus then explodes
    // RANDOM interleaves difficulties for stable gradient flow
    opts.batch_ordering = GRIM::Batching::BatchOrdering::RANDOM;
    opts.interleave_overflow = true;
    
    PHASE2_DEBUG_STDERR("[DEBUG-BUILD] About to call buildBatches...\n");
    auto schedule = GRIM::Batching::buildBatches(catalog, opts);
    PHASE2_DEBUG_STDERR("[DEBUG-BUILD] buildBatches returned, batches=%zu\n", schedule.batches.size());
    PHASE2_DEBUG_FLUSH_STDERR();  // Force flush before potential crash

    PHASE2_DEBUG_STDERR("[DEBUG-BUILD] Checking schedule.batches.empty()...\n");
    PHASE2_DEBUG_FLUSH_STDERR();
    bool is_empty = schedule.batches.empty();
    PHASE2_DEBUG_STDERR("[DEBUG-BUILD] is_empty=%d\n", is_empty ? 1 : 0);
    PHASE2_DEBUG_FLUSH_STDERR();

    PHASE2_DEBUG_STDERR("[DEBUG-BUILD] Checking schedule stats...\n");
    PHASE2_DEBUG_FLUSH_STDERR();
    PHASE2_DEBUG_STDERR("[DEBUG-BUILD] min_batch=%u max_batch=%u avg_eff=%.2f\n",
                        schedule.min_batch_size_observed,
                        schedule.max_batch_size_observed,
                        schedule.avg_packing_efficiency);
    PHASE2_DEBUG_FLUSH_STDERR();

    PHASE2_DEBUG_STDERR("[DEBUG-BUILD] About to log batches created...\n");
    PHASE2_DEBUG_FLUSH_STDERR();
    logger.log("Created " + std::to_string(schedule.batches.size()) + " dynamic batches");
    PHASE2_DEBUG_STDERR("[DEBUG-BUILD] About to log token budget...\n");
    logger.log("[Batching] Token budget: " + std::to_string(opts.max_tokens_per_batch));
    PHASE2_DEBUG_STDERR("[DEBUG-BUILD] About to log strategy...\n");
    logger.log("[Batching] Strategy: GREEDY (forced - Issue #90)");
    PHASE2_DEBUG_STDERR("[DEBUG-BUILD] About to log batch size range...\n");
    logger.log("[Batching] Batch size range: " +
               std::to_string(schedule.min_batch_size_observed) + "-" +
               std::to_string(schedule.max_batch_size_observed));
    PHASE2_DEBUG_STDERR("[DEBUG-BUILD] About to log packing efficiency...\n");
    logger.log("[Batching] Packing efficiency: " + 
               std::to_string(static_cast<int>(schedule.avg_packing_efficiency * 100)) + "%");
    PHASE2_DEBUG_STDERR("[DEBUG-BUILD] All logger.log calls completed\n");
    
    return schedule;
}

bool maybeSaveCheckpoint(
    TrainingContext& ctx,
    float val_loss,
    int epoch) {
    
    if (val_loss >= ctx.best_val_loss) {
        return false;
    }
    
    ctx.best_val_loss = val_loss;
    ctx.logging.logger->log("✓ New best! Saving checkpoint...");
    
    std::string checkpoint_path = ctx.config.paths.checkpoint_dir + 
                                  "/checkpoint_epoch_" + std::to_string(epoch + 1) + ".bin";
    
    // NOTE: model->save() handles its own sync internally before reading weights
    // No need for device sync here - let save() manage sync granularity
    
    try {
        bool save_result = ctx.model->save(checkpoint_path);
        if (save_result) {
            ctx.logging.logger->log("  ✓ Checkpoint saved: " + checkpoint_path);
            if (fs::exists(checkpoint_path)) {
                auto file_size = fs::file_size(checkpoint_path);
                ctx.logging.logger->log("  File size: " + std::to_string(file_size / (1024*1024)) + " MB");
            }
            // Save optimizer sidecar (.opt) alongside model checkpoint
            try {
                std::string opt_path = optimizerSidecarPath(checkpoint_path);
                saveOptimizerState(ctx, opt_path);
            } catch (const std::exception& e) {
                ctx.logging.logger->log(std::string("  ⚠ Optimizer state save failed: ") + e.what());
            }
            return true;
        } else {
            ctx.logging.logger->log("  ✗ Save returned false");
        }
    } catch (const std::exception& e) {
        ctx.logging.logger->log(std::string("  ✗ Exception: ") + e.what());
    }
    
    return false;
}

} // namespace Internal

//======================================================//
//  Validation Implementation
//======================================================//

ValidationResult runValidation(TrainingContext& ctx) {
    ValidationResult result;
    
    ctx.logging.logger->log("Running validation...");

    // ═══════════════════════════════════════════════════════════════════════════
    // PRE-VALIDATION SAFETY: Sync GPU and check for deferred CUDA errors.
    // After 24+ hours of training, deferred errors can accumulate. If we don't
    // drain them here, they manifest as SEH exceptions inside the validation
    // loop — bypassing C++ catch blocks and silently killing the process.
    // ═══════════════════════════════════════════════════════════════════════════
    {
        cudaError_t sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            ctx.logging.logger->log("[Val] WARNING: cudaDeviceSynchronize before validation returned: " +
                std::string(cudaGetErrorString(sync_err)));
        }
        cudaError_t deferred_err = cudaGetLastError();
        if (deferred_err != cudaSuccess) {
            ctx.logging.logger->log("[Val] WARNING: Cleared deferred CUDA error before validation: " +
                std::string(cudaGetErrorString(deferred_err)));
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // PRE-VALIDATION CHECKPOINT: Save model state BEFORE validation.
    // Validation runs hundreds of forward passes that can crash via SEH.
    // Without this, a validation crash loses the entire epoch of training.
    // ═══════════════════════════════════════════════════════════════════════════
    {
        std::string safety_path = ctx.config.paths.checkpoint_dir + "/checkpoint_pre_validation.bin";
        ctx.logging.logger->log("[Val] Saving pre-validation safety checkpoint...");
        try {
            bool saved = ctx.model->save(safety_path);
            if (saved) {
                ctx.logging.logger->log("[Val] Safety checkpoint saved: " + safety_path);
            } else {
                ctx.logging.logger->log("[Val] WARNING: Safety checkpoint save returned false");
            }
        } catch (const std::exception& e) {
            ctx.logging.logger->log("[Val] WARNING: Safety checkpoint failed: " + std::string(e.what()));
            // Non-fatal — continue with validation anyway
        }
    }

    // Match GPU state to "start of training step" so validation has same memory as training.
    // Rule 20: pre-validation reset is one boundary; per-batch RAII handles the rest.
    {
        GRIM::Autograd::AutogradStepScope pre_val_scope(ctx.model->getTrainingState());
    }
    flushDeferredCleanup();
    cudaDeviceSynchronize();
    (void)cudaGetLastError();

    // Log available GPU memory before validation for diagnostics
    {
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        ctx.logging.logger->log("[Val] GPU memory: " +
            std::to_string(free_mem / (1024*1024)) + " MB free / " +
            std::to_string(total_mem / (1024*1024)) + " MB total");
    }

    // Use the same batch limits as training so validation uses the same memory path.
    // Training runs 4000+ batches with these limits; validation must not use different (e.g. halved) limits.
    const auto& model_cfg = ctx.model->getConfig();
    const int batch_size = std::max(1, ctx.config.hyperparameters.batch_size);
    const uint32_t config_token_budget = static_cast<uint32_t>(batch_size) *
        static_cast<uint32_t>(std::max(1, model_cfg.max_seq_len));

    ctx.logging.logger->log("[Val] Token budget: " + std::to_string(config_token_budget) +
        " (same as training: batch_size=" + std::to_string(batch_size) + " x max_seq_len=" + std::to_string(model_cfg.max_seq_len) + ")");

    GRIM::Batching::BatchOptions val_opts;
    val_opts.max_tokens_per_batch = config_token_budget;
    val_opts.max_batch_size = static_cast<uint32_t>(batch_size);
    val_opts.bucket_step = 256;
    
    auto val_schedule = GRIM::Batching::buildBatches(ctx.data.val_catalog, val_opts);
    const int total_val_batches = static_cast<int>(val_schedule.batches.size());
    ctx.logging.logger->log("Created " + std::to_string(total_val_batches) + " validation batches");
    
    float val_loss = 0.0f;
    int val_sequences_processed = 0;
    int val_batches_failed = 0;
    
    const auto& val_model_cfg = ctx.model->getConfig();
    const size_t val_max_cached_batch = static_cast<size_t>(std::max(1, val_model_cfg.max_cached_batch));
    const size_t val_max_cached_seq = static_cast<size_t>(std::max(1, std::min(val_model_cfg.max_seq_len, val_model_cfg.max_cached_seq_len)));
    
    const auto val_start_time = std::chrono::steady_clock::now();
    
    for (int val_idx = 0; val_idx < total_val_batches; ++val_idx) {
        const auto& val_batch = val_schedule.batches[val_idx];
        
        // Progress logging every 50 batches
        if (val_idx % 50 == 0) {
            ctx.logging.logger->log("[Val] Processing batch " + std::to_string(val_idx + 1) + 
                "/" + std::to_string(total_val_batches) +
                " (processed=" + std::to_string(val_sequences_processed) +
                " failed=" + std::to_string(val_batches_failed) + ")");
        }
        
        try {
            const auto val_token_layout = ctx.tokenizer.tokenLayout();
            auto val_payload = GRIM::Batching::buildBatchPayload(
                val_batch, ctx.data.val_views, ctx.config.actual_vocab_size,
                val_token_layout,
                val_max_cached_batch, val_max_cached_seq,
                val_model_cfg.execution_block_num_slots,
                val_model_cfg.execution_block_num_ops,
                val_model_cfg.execution_block_num_steps,
                0);  // mtp_k=0 for validation — no MTP shifting needed
            if (val_payload.batch_size == 0) continue;
            
            // Rule 20 single-owner clear: AutogradStepScope covers this validation
            // batch and unwinds on both normal exit and exception (catch handler
            // below MUST NOT call clear() again).
            float batch_val_loss = 0.0f;
            {
                GRIM::Autograd::AutogradStepScope val_step_scope(ctx.model->getTrainingState());
                batch_val_loss = ctx.model->computeLossBatch(val_payload, /*is_training=*/false);
            }
            flushDeferredCleanup();

            // Check for deferred CUDA errors after each batch
            cudaError_t batch_err = cudaGetLastError();
            if (batch_err != cudaSuccess) {
                ctx.logging.logger->log("[Val] CUDA error after batch " + std::to_string(val_idx + 1) + 
                    ": " + std::string(cudaGetErrorString(batch_err)));
                val_batches_failed++;
                // Sync and clear to attempt recovery for remaining batches
                cudaDeviceSynchronize();
                cudaGetLastError();
                continue;
            }
            
            // Validate the loss value before accumulating
            if (!std::isfinite(batch_val_loss)) {
                ctx.logging.logger->log("[Val] WARNING: Non-finite loss at batch " + 
                    std::to_string(val_idx + 1) + " (loss=" + std::to_string(batch_val_loss) + "), skipping");
                val_batches_failed++;
                continue;
            }
            
            val_loss += batch_val_loss * val_payload.batch_size;
            val_sequences_processed += val_payload.batch_size;
            
        } catch (const std::exception& e) {
            ctx.logging.logger->log("[Val] Exception at batch " + std::to_string(val_idx + 1) + 
                "/" + std::to_string(total_val_batches) + ": " + std::string(e.what()));
            val_batches_failed++;
            
            // Rule 20 single-owner clear: AutogradStepScope already destructed when
            // the try-block scope unwound the exception. Do NOT add a clear() here.
            
            // Sync and clear CUDA state for recovery — also flushes deferred GPU frees
            cudaDeviceSynchronize();
            flushDeferredCleanup();
            cudaGetLastError();
            
            // If too many batches fail (>10%), abort validation with partial results
            if (val_batches_failed > total_val_batches / 10 && val_batches_failed >= 5) {
                ctx.logging.logger->log("[Val] ABORTING: Too many failed batches (" + 
                    std::to_string(val_batches_failed) + "/" + std::to_string(val_idx + 1) + 
                    "). Using partial results.");
                break;
            }
            continue;
        }
    }
    
    auto val_duration = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - val_start_time);
    
    if (val_sequences_processed > 0) {
        result.loss = val_loss / val_sequences_processed;
    } else {
        ctx.logging.logger->log("[Val] ERROR: No validation sequences processed successfully!");
        result.loss = std::numeric_limits<float>::infinity();
    }
    result.sequences_processed = val_sequences_processed;
    result.perplexity = (std::isfinite(result.loss) && result.loss < 50.0f)
        ? std::exp(result.loss)
        : std::numeric_limits<float>::infinity();
    result.is_best = (result.loss < ctx.best_val_loss);
    
    ctx.logging.logger->log("[Val] " + Internal::formatMetric("loss", result.loss) + " " +
                            Internal::formatMetric("ppl", result.perplexity, 3) +
                            " seqs=" + std::to_string(val_sequences_processed) +
                            " failed=" + std::to_string(val_batches_failed) +
                            " time=" + std::to_string(val_duration.count()) + "s");
    
    return result;
}

//======================================================//
//  Batch Processing Implementation
//======================================================//

BatchResult processBatch(
    TrainingContext& ctx,
    TrainingLoopState& state,
    const GRIM::Batching::BatchPayload& payload,
    int batch_idx,
    int epoch_idx) {
    
    BatchResult result;
    result.batch_idx = batch_idx;
    
    // Begin tape recording for this batch (clears prior entries, sets step/batch)
    if (ctx.logging.tape) {
        ctx.logging.tape->beginBatch(static_cast<int>(ctx.global_step), batch_idx);
    }
    
    // Issue #142b: Eagerly detect any deferred CUDA errors from prior operations
    // (e.g., sample generation, previous batch backward). Without this, the error
    // manifests deep inside encoderForward as an SEH exception → silent exit.
    {
        cudaError_t pre_err = cudaGetLastError();
        if (pre_err != cudaSuccess) {
            std::string err_msg = "[processBatch] CUDA error BEFORE batch " +
                std::to_string(batch_idx + 1) + ": " +
                std::string(cudaGetErrorString(pre_err)) +
                " (code=" + std::to_string(static_cast<int>(pre_err)) + ")";
            ctx.logging.logger->log(err_msg);
            fprintf(stderr, "%s\n", err_msg.c_str());
            // Attempt recovery: sync and clear
            cudaDeviceSynchronize();
            cudaGetLastError();
        }
    }
    
    const auto& hp = ctx.config.hyperparameters;
    const bool logit_trace_enabled = shouldLogLogitTrace(ctx, batch_idx);
    
    // ========================================================================
    // RULE 20: Step Counter Clarity
    // Three step counters exist:
    // 1. batch_number = batch_idx + 1 (increases every batch: 1,2,3,...,N)
    // 2. ctx.global_step = training token counter (increments with every batch)
    // 3. ctx.optimizer.optimizer_state.step = actual optimizer updates (every accum_steps)
    //
    // CONVENTION: Log ONLY the relevant counter:
    // - During FORWARD/BACKWARD: use batch_number (most relevant to user)
    // - During OPTIMIZER step: use optimizer_step (shows actual weight updates)
    // - Remove global_step from logs (creates confusion with near-duplicate batch_number)
    // ========================================================================

    // Payload is built in runEpoch from scheduler (BatchAssignment); single source of truth here
    if (payload.batch_size == 0) {
        result.skipped = true;
        result.skip_reason = "filtered";
        return result;
    }

    // First batch diagnostics - check weight initialization
    if (batch_idx == 0 && ctx.global_step == 0) {
        ctx.logging.logger->log("[GradTrace] FIRST_BATCH: Checking initial model state...");
        auto model_stats = ctx.model->getModelStats();
        ctx.logging.logger->log("[GradTrace] FIRST_BATCH: total_params=" + std::to_string(model_stats.total_params) +
                                " embedding_params=" + std::to_string(model_stats.embedding_params) +
                                " encoder_params=" + std::to_string(model_stats.encoder_params) +
                                " lm_head_params=" + std::to_string(model_stats.lm_head_params));
    }
    
    // Token stats already computed in payload — single source of truth
    const auto& token_stats = payload.token_stats;
    // Per-batch seq_len from BatchPayload (single source of truth), not config
    const int long_seq_threshold = payload.max_seq_len;
    const auto clip_selection = GRIM::TNC::computeClipSelection(
        hp.grad_clip_norm, token_stats, 1.0f, long_seq_threshold);
    
    // Start gradient accumulation window only when at micro_step 0
    // (i.e., first micro-batch after an optimizer step or at very start)
    // Gradient zeroing is handled by backward() method based on accumulate parameter.
    // When micro_step=0, backward() is called with accumulate=false → zeros gradients.
    // When micro_step>0, backward() is called with accumulate=true → accumulates.
    const int accum_steps = std::max(1, ctx.config.hyperparameters.gradient_accumulation_steps);
    
    // BUG FIX: beginBatch() must be called EVERY batch to clear previous entries,
    // not just at accumulation window start. Otherwise micro-batches 1+ have stale entries.
    GRIM::GradStats::beginBatch();
    
    // DIAGNOSTIC: Disabled for performance (was causing 4 device syncs per batch)
    // static int diag_batch_count = 0;
    // ++diag_batch_count;
    // if (diag_batch_count <= 3) {
    //     const auto& ts = ctx.model->getTrainingState();
    //     ctx.logging.logger->log("[GradDiag] AFTER_ZERO: " + sampleEmbeddingGrads(ts));
    // }
    
    // Log batch info for diagnostics
    {
        std::ostringstream batch_info;
        batch_info << "[GradTrace] BATCH_INFO batch=" << (batch_idx + 1)
                   << " seqs=[";
        for (size_t i = 0; i < payload.seq_ids.size(); ++i) {
            batch_info << payload.seq_ids[i];
            if (i + 1 < payload.seq_ids.size()) batch_info << ",";
        }
        batch_info << "] lens=[";
        for (int i = 0; i < payload.batch_size; ++i) {
            batch_info << payload.seq_lengths[i];
            if (i + 1 < payload.batch_size) batch_info << ",";
        }
        batch_info << "]";
        ctx.logging.logger->log(batch_info.str());
    }

    PHASE2_DEBUG_STDERR("[DEBUG-PROCESS] After BATCH_INFO log, checking shouldLogAtomStats...\n");
    if (shouldLogAtomStats(ctx, batch_idx)) {
        PHASE2_DEBUG_STDERR("[DEBUG-PROCESS] shouldLogAtomStats=true, creating vectors...\n");
        std::vector<int> per_seq_atoms;
        std::vector<int> per_seq_lengths;
        per_seq_atoms.reserve(payload.batch_size);
        per_seq_lengths.reserve(payload.batch_size);

        // Reconstruct per-sequence views from flat payload for atom detection
        std::vector<std::vector<int>> seq_views;
        seq_views.reserve(payload.batch_size);
        int offset = 0;
        for (int i = 0; i < payload.batch_size; ++i) {
            const int len = payload.seq_lengths[i];
            seq_views.emplace_back(payload.input_ids.begin() + offset,
                                   payload.input_ids.begin() + offset + len);
            offset += payload.max_seq_len; // stride is padded length
        }
        PHASE2_DEBUG_STDERR("[DEBUG-PROCESS] About to call computeAtomStats...\n");
        const AtomStats stats = computeAtomStats(seq_views, ctx.tokenizer,
                                                 &per_seq_atoms, &per_seq_lengths);
        PHASE2_DEBUG_STDERR("[DEBUG-PROCESS] computeAtomStats returned\n");
        const double atom_ratio = stats.total_tokens > 0
            ? static_cast<double>(stats.total_atoms) / static_cast<double>(stats.total_tokens)
            : 0.0;

        PHASE2_DEBUG_STDERR("[DEBUG-PROCESS] Building atom_msg...\n");
        std::ostringstream atom_msg;
        atom_msg << "[AtomStats] batch=" << (batch_idx + 1)
                 << " seqs=" << payload.batch_size
                 << " atoms=" << stats.total_atoms
                 << " tokens=" << stats.total_tokens
                 << " atom_ratio=" << std::fixed << std::setprecision(4) << atom_ratio
                 << " min=" << stats.min_atoms
                 << " max=" << stats.max_atoms
                 << " avg=" << std::fixed << std::setprecision(2) << stats.avg_atoms;
        PHASE2_DEBUG_STDERR("[DEBUG-PROCESS] About to log atom_msg...\n");
        ctx.logging.logger->log(atom_msg.str());
        PHASE2_DEBUG_STDERR("[DEBUG-PROCESS] atom_msg logged\n");

        const int max_seq_log = std::max(0, ctx.config.hyperparameters.atom_stats_max_seqs);
        if (max_seq_log > 0 && !per_seq_atoms.empty()) {
            const int to_log = std::min<int>(max_seq_log,
                                             static_cast<int>(per_seq_atoms.size()));
            std::ostringstream per_seq_msg;
            per_seq_msg << "[AtomStats] per_seq=";
            for (int i = 0; i < to_log; ++i) {
                const int seq_len = per_seq_lengths[i];
                const int atom_count = per_seq_atoms[i];
                const double ratio = seq_len > 0
                    ? static_cast<double>(atom_count) / static_cast<double>(seq_len)
                    : 0.0;
                per_seq_msg << i << ":" << atom_count << "/" << seq_len
                            << "(" << std::fixed << std::setprecision(3) << ratio << ")";
                if (i + 1 < to_log) {
                    per_seq_msg << " ";
                }
            }
            if (static_cast<int>(per_seq_atoms.size()) > to_log) {
                per_seq_msg << " ...";
            }
            ctx.logging.logger->log(per_seq_msg.str());
        }
    }
    
    PHASE2_DEBUG_STDERR("[DEBUG-PROCESS] After atom stats, entering boundary diagnostic...\n");
    // ========================================================================
    // DIAGNOSTIC: Boundary crossing check (simplified for FlashAttention v2)
    // NOTE: FlashAttention v2 does NOT use O(seq²) attention buffers.
    // The O(N) memory tiled attention means seq_len boundaries are NOT
    // inherently problematic for memory. This diagnostic only checks:
    // - Position embedding bounds (seq_len vs model.max_seq_len)
    // - Training cache capacity (total tokens vs max_cached_tokens)
    // - Token ID validity (vocab bounds)
    // ========================================================================
    {
        // Use payload geometry — single source of truth
        const size_t max_seq_len = static_cast<size_t>(payload.max_seq_len);
        const size_t total_tokens = static_cast<size_t>(payload.actual_tokens);

        
        const auto& model_cfg_bd = ctx.model->getConfig();
        static bool logged_max_seq = false;
        const bool is_boundary_max_seq = (max_seq_len >= static_cast<size_t>(model_cfg_bd.max_seq_len) && !logged_max_seq);

        
        if (is_boundary_max_seq) {
            std::ostringstream diag;
            diag << "\n[BOUNDARY_DIAGNOSTIC] ========================================\n";
            diag << "[BOUNDARY_DIAGNOSTIC] Batch " << (batch_idx + 1) << " CROSSING BOUNDARY\n";
            
            // Identify which boundary was crossed
            if (is_boundary_max_seq) diag << "[BOUNDARY_DIAGNOSTIC] *** REACHED model.max_seq_len=" << model_cfg_bd.max_seq_len << " ***\n";

            diag << "[BOUNDARY_DIAGNOSTIC] max_seq_len=" << max_seq_len 
                 << " total_tokens=" << total_tokens 
                 << " batch_size=" << payload.batch_size << "\n";
            
            diag << "[BOUNDARY_DIAGNOSTIC] MODEL CONFIG:\n";
            diag << "  d_model=" << model_cfg_bd.d_model << "\n";
            diag << "  max_seq_len=" << model_cfg_bd.max_seq_len << "\n";
            diag << "  num_heads=" << model_cfg_bd.num_heads << "\n";
            diag << "  num_layers=" << model_cfg_bd.num_layers << "\n";
            diag << "  vocab_size=" << model_cfg_bd.vocab_size << "\n";
            
            // Position embedding checks (this IS a valid concern)
            diag << "[BOUNDARY_DIAGNOSTIC] POSITION EMBEDDING CHECKS:\n";
            diag << "  Current max_seq_len in batch: " << max_seq_len << "\n";
            diag << "  Model max_seq_len: " << model_cfg_bd.max_seq_len << "\n";
            diag << "  Position index range needed: [0, " << (max_seq_len - 1) << "]\n";
            if (max_seq_len > static_cast<size_t>(model_cfg_bd.max_seq_len)) {
                diag << "  *** ERROR: Sequence exceeds model max_seq_len! Position embeddings will OOB! ***\n";
            }
            
            // Per-sequence breakdown using payload geometry
            diag << "[BOUNDARY_DIAGNOSTIC] PER-SEQUENCE BREAKDOWN:\n";
            for (int s = 0; s < payload.batch_size; ++s) {
                const int seq_len = payload.seq_lengths[s];
                diag << "  seq[" << s << "]: len=" << seq_len;
                
                // Check for position IDs that would overflow
                if (seq_len > model_cfg_bd.max_seq_len) {
                    diag << " *** OVERFLOW pos=" << seq_len 
                         << " > max=" << model_cfg_bd.max_seq_len << " ***";
                }
                
                // Sample first and last tokens from flat payload
                if (seq_len > 0) {
                    const int flat_start = s * payload.max_seq_len;
                    diag << " tokens[0]=" << payload.input_ids[flat_start];
                    if (seq_len > 1) {
                        diag << " tokens[" << (seq_len-1) << "]=" << payload.input_ids[flat_start + seq_len - 1];
                    }
                }
                diag << "\n";
            }
            
            // Training state checks - TRAINING cache info (not inference KV cache)
            const auto& ts = ctx.model->getTrainingState();
            diag << "[BOUNDARY_DIAGNOSTIC] TRAINING STATE:\n";
            diag << "  cached_batch_size=" << ts.cached_batch_size << "\n";
            diag << "  cached_seq_len=" << ts.cached_seq_len << "\n";
            diag << "  cached_valid_tokens=" << ts.cached_valid_tokens << "\n";
            
            // Training cache allocation check (the correct fields!)
            diag << "  max_cached_batch=" << ts.max_cached_batch << "\n";
            diag << "  max_cached_seq_len=" << ts.max_cached_seq_len << "\n";
            diag << "  max_cached_tokens=" << ts.max_cached_tokens << "\n";
            
            // Check if sequence fits in TRAINING cache — use payload.total_tokens (already batch*max_seq)
            diag << "  Required tokens for this batch: " << payload.total_tokens << "\n";
            if (static_cast<size_t>(payload.total_tokens) > ts.max_cached_tokens) {
                diag << "  *** WARNING: Batch exceeds training cache capacity! ***\n";
                diag << "  *** Need " << payload.total_tokens << " but have " << ts.max_cached_tokens << " ***\n";
            }
            if (max_seq_len > static_cast<size_t>(ts.max_cached_seq_len)) {
                diag << "  *** WARNING: Sequence exceeds max_cached_seq_len! ***\n";
                diag << "  *** max_seq_len=" << max_seq_len << " > max_cached=" << ts.max_cached_seq_len << " ***\n";
            }
            
            // NOTE: FlashAttention v2 uses O(N) tiled attention, NOT O(N²) buffers.
            // No attention buffer check needed - memory scales linearly with seq_len.
            diag << "[BOUNDARY_DIAGNOSTIC] ATTENTION: Using FlashAttention v2 (O(N) memory)\n";
            
            // Token ID sanity check — scan flat payload
            diag << "[BOUNDARY_DIAGNOSTIC] TOKEN ID SANITY:\n";
            int max_token_id = 0;
            int min_token_id = INT_MAX;
            for (int s = 0; s < payload.batch_size; ++s) {
                const int flat_start = s * payload.max_seq_len;
                const int len = payload.seq_lengths[s];
                for (int t = 0; t < len; ++t) {
                    const int tok = payload.input_ids[flat_start + t];
                    max_token_id = std::max(max_token_id, tok);
                    min_token_id = std::min(min_token_id, tok);
                }
            }
            diag << "  Token ID range: [" << min_token_id << ", " << max_token_id << "]\n";
            diag << "  Vocab size: " << model_cfg_bd.vocab_size << "\n";
            if (max_token_id >= static_cast<int>(model_cfg_bd.vocab_size)) {
                diag << "  *** ERROR: Token ID exceeds vocab size! ***\n";
            }
    
            if (is_boundary_max_seq) logged_max_seq = true;
        }
    }
    
    PHASE2_DEBUG_STDERR("[DEBUG-PROCESS] After boundary diagnostic, entering forward pass...\n");
    // Forward pass
    static int forward_call_count = 0;
    ++forward_call_count;
    
    PHASE2_DEBUG_STDERR("[DEBUG-PROCESS] forward_call_count=%d, building target distribution...\n", forward_call_count);
    // Log target and prediction distributions (uses ForwardPass module for filtering)
    {
        // Log target distribution using flat payload target_ids
        std::map<int, int> target_counts;
        int total_valid = 0;
        int total_tokens_td = 0;
        for (int s = 0; s < payload.batch_size; ++s) {
            const int flat_start = s * payload.max_seq_len;
            const int len = payload.seq_lengths[s];
            for (int t = 0; t < len; ++t) {
                const int tid = payload.target_ids[flat_start + t];
                total_tokens_td++;
                if (tid >= 0) {
                    target_counts[tid]++;
                    total_valid++;
                }
            }
        }
        
        // Find top-10 most common targets
        std::vector<std::pair<int, int>> sorted_targets(target_counts.begin(), target_counts.end());
        std::sort(sorted_targets.begin(), sorted_targets.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        std::ostringstream target_info;
        target_info << "BATCH_TARGET_DIST batch=" << (batch_idx + 1)
                    << " total_tokens=" << total_tokens_td 
                    << " valid=" << total_valid 
                    << " unique=" << target_counts.size()
                    << " top10=[";
        for (size_t i = 0; i < std::min(sorted_targets.size(), size_t(10)); ++i) {
            target_info << "tid=" << sorted_targets[i].first 
                        << ":" << sorted_targets[i].second;
            if (i + 1 < std::min(sorted_targets.size(), size_t(10))) target_info << ", ";
        }
        target_info << "]";
        EmitModuleInfo(ModuleId::ForwardPass, target_info.str(), ctx.global_step);
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // UNIFIED TRAINING STEP: forward → loss → backward via autogradTrainingStep
    // Replaces the old computeLossBatch() + backward() two-call pattern.
    // ═══════════════════════════════════════════════════════════════════════════
    
    // Rule 20: micro_step MUST be within bounds before starting the step
    if (ctx.optimizer.current_micro_step >= accum_steps) {
        fprintf(stderr, "\n[Phase2] FATAL: Training step attempted with micro_step=%d >= accum_steps=%d\n", 
                ctx.optimizer.current_micro_step, accum_steps);
        fprintf(stderr, "[Phase2] batch=%d global_step=%d\n", batch_idx + 1, ctx.global_step);
        fprintf(stderr, "[Phase2] This indicates current_micro_step was not reset after optimizer step.\n");
        std::abort();
    }
    
    // BUG FIX Issue #22: First micro-batch overwrites (accumulate=false), subsequent accumulate
    const bool should_accumulate = ctx.optimizer.current_micro_step > 0;
    
    PHASE2_DEBUG_STDERR("[DEBUG-PROCESS] About to call autogradTrainingStep...\n");
    // First-batch CUDA checkpoint: surface any error before forward/loss/backward
    if (batch_idx == 0) {
        cudaError_t e = cudaDeviceSynchronize();
        cudaError_t last = (e != cudaSuccess) ? e : cudaGetLastError();
        if (last != cudaSuccess) {
            ctx.logging.logger->log("[CUDA] first_batch BEFORE autogradTrainingStep: " + std::string(cudaGetErrorString(last)));
            cudaGetLastError();
        } else {
            ctx.logging.logger->log("[CUDA] first_batch BEFORE autogradTrainingStep: ok");
        }
    }
    // STABILITY FIX (Issue #27/Math Audit): Apply 1/M scaling at the source (backward pass) to match PyTorch.
    // This ensures that gradients averaged per micro-batch are further averaged across the accumulation window,
    // preventing the "Sum of Averages" discrepancy that causes gradient explosion by a factor of M.
    const float grad_scale = 1.0f / static_cast<float>(accum_steps);
    // FIX: Pass optimizer step, not global_step (batch counter).
    // ctx.step is used downstream by MTP alpha warmup ramp and dropout seeds.
    // With gradient_accumulation_steps > 1, global_step advances accum_steps
    // times faster than actual weight updates, completing warmup ramps too early.
    //
    // Rule 20 ownership taxonomy: AutogradStepScope is the SINGLE owner of
    // AutogradIntermediates::clear() for this batch. It covers autogradTrainingStep
    // + GuessCache update + post-step diagnostics + tape logging. Do NOT add an
    // explicit clear() anywhere inside this scope.
    GRIM::Autograd::AutogradStepScope autograd_step_scope(ctx.model->getTrainingState());
    auto loss_result = GRIM::Autograd::autogradTrainingStep(
        *ctx.model,
        ctx.model->getTrainingState(),
        payload,
        should_accumulate,
        grad_scale,
        static_cast<uint64_t>(ctx.optimizer.optimizer_state.step)
    );
    result.loss = loss_result.loss_value;
    PHASE2_DEBUG_STDERR("[DEBUG-PROCESS] autogradTrainingStep returned, loss=%f success=%d\n", 
                        result.loss, static_cast<int>(loss_result.success));
    
    // First-batch CUDA checkpoint: fault is in forward/loss/backward if you see error here
    if (batch_idx == 0) {
        cudaError_t e = cudaDeviceSynchronize();
        cudaError_t last = (e != cudaSuccess) ? e : cudaGetLastError();
        if (last != cudaSuccess) {
            ctx.logging.logger->log("[CUDA] first_batch AFTER autogradTrainingStep: " + std::string(cudaGetErrorString(last)) +
                " (fault is in forward, loss, or backward)");
            cudaGetLastError();
        } else {
            ctx.logging.logger->log("[CUDA] first_batch AFTER autogradTrainingStep: ok");
        }
    }
    
    // Handle training step failure (NaN/Inf loss or backward error)
    if (!loss_result.success) {
        ctx.logging.logger->log("[autogradTrainingStep] FAILED batch=" + std::to_string(batch_idx + 1) +
                                " error: " + loss_result.error_message);
        // zeroGrad removed: next autogradTrainingStep with accumulate=false zeros gradients
        ctx.optimizer.current_micro_step = 0;
        result.skipped = true;
        result.skip_reason = "autograd_step_failed";
        ctx.global_step++;
        return result;
    }
    
    // Log gradient accumulation status
    ctx.logging.logger->log("[GradAccum] batch=" + std::to_string(batch_idx + 1) +
                            " micro_step=" + std::to_string(ctx.optimizer.current_micro_step) +
                            "/" + std::to_string(accum_steps) +
                            " accumulate=" + (should_accumulate ? "true" : "false"));

    // Update guess cache with predictions from this batch
    if (std::isfinite(result.loss) && ctx.config.hyperparameters.guess_aux_enabled) {
        GRIMTS::Training::updateGuessCacheFromBatch(
            ctx.model->getTrainingState(),
            payload,
            result.loss,
            epoch_idx,
            ctx.global_step,
            state.guess_cache);
    }
    
    // Log model predictions (what it predicts vs targets) - uses ForwardPass module for filtering
    // GUARDED: Blocking cudaMemcpy drains GPU pipeline - only run on diagnostic sync interval
    if (shouldSyncDiagnostics(ctx, batch_idx)) {
        const auto& ts = ctx.model->getTrainingState();
        cudaStream_t stream = ts.stream_ctrl.getPrimaryStream();
        if (ts.cached_logits_tensor.data && ts.cached_batch_size > 0 && ts.cached_seq_len > 0) {
            const int total_tokens = ts.cached_batch_size * ts.cached_seq_len;
            const int vocab_size = ctx.config.actual_vocab_size;
            const int sample_positions = total_tokens;
            const size_t logit_bytes = static_cast<size_t>(sample_positions) * vocab_size * sizeof(float);
            std::vector<float> logit_sample(sample_positions * vocab_size);
            cudaMemcpyAsync(logit_sample.data(), ts.cached_logits_tensor.data, logit_bytes, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            // Count argmax predictions
            std::map<int, int> pred_counts;
            for (int pos = 0; pos < sample_positions; ++pos) {
                float max_logit = -std::numeric_limits<float>::infinity();
                int argmax = 0;
                for (int v = 0; v < vocab_size; ++v) {
                    float logit = logit_sample[pos * vocab_size + v];
                    if (logit > max_logit) {
                        max_logit = logit;
                        argmax = v;
                    }
                }
                pred_counts[argmax]++;
            }
            
            // Sort by frequency
            std::vector<std::pair<int, int>> sorted_preds(pred_counts.begin(), pred_counts.end());
            std::sort(sorted_preds.begin(), sorted_preds.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });
            
            std::ostringstream pred_info;
            pred_info << "BATCH_PRED_DIST batch=" << batch_idx
                      << " sampled_pos=" << sample_positions
                      << " unique_preds=" << pred_counts.size()
                      << " top10=[";
            for (size_t i = 0; i < std::min(sorted_preds.size(), size_t(10)); ++i) {
                pred_info << "tid=" << sorted_preds[i].first 
                          << ":" << sorted_preds[i].second;
                if (i + 1 < std::min(sorted_preds.size(), size_t(10))) pred_info << ", ";
            }
            pred_info << "]";
            EmitModuleInfo(ModuleId::ForwardPass, pred_info.str(), ctx.global_step);

            if (logit_trace_enabled && sample_positions > 0) {
                int debug_pos = -1;
                int debug_b = -1;
                int debug_t = -1;
                int target_token = -1;
                for (int pos = 0; pos < sample_positions; ++pos) {
                    const int b = pos / ts.cached_seq_len;
                    const int t = pos % ts.cached_seq_len;
                    if (b < payload.batch_size &&
                        t < payload.seq_lengths[b]) {
                        const int candidate = payload.target_ids[b * payload.max_seq_len + t];
                        if (candidate >= 0 && candidate < vocab_size) {
                            debug_pos = pos;
                            debug_b = b;
                            debug_t = t;
                            target_token = candidate;
                            break;
                        }
                    }
                }
                if (debug_pos < 0) {
                    debug_pos = 0;
                    debug_b = 0;
                    debug_t = 0;
                    if (payload.batch_size > 0 && payload.seq_lengths[0] > 0) {
                        target_token = payload.target_ids[0];
                    }
                }

                const float* logits = logit_sample.data() + debug_pos * vocab_size;
                float max_logit = -std::numeric_limits<float>::infinity();
                int argmax = 0;
                for (int v = 0; v < vocab_size; ++v) {
                    const float logit = logits[v];
                    if (logit > max_logit) {
                        max_logit = logit;
                        argmax = v;
                    }
                }

                double sum_exp = 0.0;
                for (int v = 0; v < vocab_size; ++v) {
                    sum_exp += std::exp(static_cast<double>(logits[v] - max_logit));
                }
                if (sum_exp <= 0.0) {
                    sum_exp = 1.0;
                }

                double p_t = -1.0;
                if (target_token >= 0 && target_token < vocab_size) {
                    p_t = std::exp(static_cast<double>(logits[target_token] - max_logit)) / sum_exp;
                }

                std::ostringstream trace_msg;
                trace_msg << std::fixed << std::setprecision(6);
                trace_msg << "[LogitTrace][PostLoss] source=cached_logits"
                          << " batch=" << (batch_idx + 1)
                          << " pos=" << debug_pos
                          << " b=" << debug_b
                          << " t=" << debug_t
                          << " target=" << target_token;
                if (p_t >= 0.0) {
                    trace_msg << " p_t=" << p_t;
                } else {
                    trace_msg << " p_t=N/A";
                }
                // sum_exp = Σ exp(logit[v] - max_logit), NOT Σ exp(logit[v])!
                // This is the SHIFTED partition function. Low values (e.g. 5.5) mean
                // the distribution is PEAKED (only ~5 tokens have significant mass).
                // logsumexp = log(sum_exp) + max_logit = log(Σ exp(logit[v]))
                const double logsumexp = std::log(sum_exp) + static_cast<double>(max_logit);
                trace_msg << " max_logit=" << max_logit
                          << " argmax=" << argmax
                          << " sum_exp_shifted=" << sum_exp
                          << " logsumexp=" << logsumexp
                          << " logit_range=" << Internal::formatScalar(max_logit - static_cast<float>(*std::min_element(logits, logits + vocab_size)), 4)
                          << " loss=" << Internal::formatScalar(result.loss, 4);
                ctx.logging.logger->log(trace_msg.str());
            }
        }
    }
    
    if (forward_call_count <= 3) {
        ctx.logging.logger->log("[GradTrace] POST-autogradTrainingStep call #" + std::to_string(forward_call_count) + 
                                " returned=" + std::to_string(result.loss));
        
    }
    
    // NOTE: Loss variance computation removed (was causing 5-second GPU sync bottleneck).
    // Variance is now tracked on GPU by TelemetryLattice (σ_tilde, v_σ fields).
    // Use computeTelemetryFeedback() to access grad_norm variance for adaptive decisions.
    
    ctx.logging.logger->log("[GradTrace] POST-FORWARD loss=" + Internal::formatScalar(result.loss, 4));

    // ========================================================================
    // EQUATION LOGGING: Per-batch loss computation trace (Issue #120 debug)
    // ========================================================================
    {
        const auto& ts_eq = ctx.model->getTrainingState();
        const int valid_tokens_eq = ts_eq.cached_valid_tokens;
        const float expected_random_loss = std::log(static_cast<float>(ctx.config.actual_vocab_size));
        
        std::ostringstream eq;
        eq << "[BATCH_LOSS] loss = -sum(log(p_target)) / valid_tokens\n";
        eq << "  valid_tokens=" << valid_tokens_eq << " vocab_size=" << ctx.config.actual_vocab_size << "\n";
        eq << "  EXPECTED loss (random) = ln(" << ctx.config.actual_vocab_size << ") = " << expected_random_loss << "\n";
        eq << "  ACTUAL loss = " << result.loss << "\n";
        EQ_LOG(ctx.logging.tape.get(), GRIM::Logging::LogGroup::Loss, GRIM::Logging::LogPhase::LOSS_COMPUTATION, -1, "BATCH_LOSS", eq.str().c_str());
    }

    {
        const auto& ts = ctx.model->getTrainingState();
        const int valid_tokens = ts.cached_valid_tokens;
        const int total_tokens = ts.cached_batch_size * ts.cached_seq_len;
        const int masked_tokens = std::max(total_tokens - valid_tokens, 0);
        const float loss_sum = (valid_tokens > 0)
            ? result.loss * static_cast<float>(valid_tokens)
            : 0.0f;
        std::ostringstream loss_stats;
        loss_stats << "[LossStats] batch=" << (batch_idx + 1)
                   << " loss_mean=" << Internal::formatScalar(result.loss, 4)
                   << " loss_sum=" << Internal::formatScalar(loss_sum, 4)
                   << " valid_tokens=" << valid_tokens
                   << " masked_tokens=" << masked_tokens
                   << " total_tokens=" << total_tokens;
        ctx.logging.logger->log(loss_stats.str());
    }
    
    // ========================================================================
    // TRAINING SIGNAL: Logit Statistics (argmax distribution, confidence)
    // ========================================================================
    {
        const auto& ts = ctx.model->getTrainingState();
        if (ts.cached_logits_tensor.data && ts.cached_batch_size > 0 && ts.cached_seq_len > 0) {
            const int total_tokens = ts.cached_batch_size * ts.cached_seq_len;
            const int vocab_size = ctx.config.actual_vocab_size;
            const int d_model = ctx.model->getConfig().d_model;
            
            // Use full batch for logit statistics
            const int sample_positions = total_tokens;
            const size_t logit_bytes = static_cast<size_t>(sample_positions) * vocab_size * sizeof(float);
            std::vector<float> logit_sample(sample_positions * vocab_size);
            cudaMemcpy(logit_sample.data(), ts.cached_logits_tensor.data, logit_bytes, cudaMemcpyDeviceToHost);
            
            // Compute argmax predictions and logit statistics
            std::map<int, int> argmax_counts;
            float logit_mean = 0.0f;
            float logit_max = -std::numeric_limits<float>::infinity();
            float logit_min = std::numeric_limits<float>::infinity();
            float max_logit_per_pos_sum = 0.0f;
            float margin_sum = 0.0f;  // Top-2 margin: logit[max] - logit[second]
            
            for (int pos = 0; pos < sample_positions; ++pos) {
                float pos_max = -std::numeric_limits<float>::infinity();
                float pos_second = -std::numeric_limits<float>::infinity();
                int pos_argmax = 0;
                
                for (int v = 0; v < vocab_size; ++v) {
                    float logit = logit_sample[pos * vocab_size + v];
                    logit_mean += logit;
                    logit_max = std::max(logit_max, logit);
                    logit_min = std::min(logit_min, logit);
                    
                    // Rule 20: Check for inf/nan IMMEDIATELY when detected (issue #142)
                    if (!std::isfinite(logit)) {
                        throw std::runtime_error(
                            "[LogitSignal] FATAL: Logit is inf/nan at batch " + 
                            std::to_string(batch_idx + 1) + 
                            ", pos=" + std::to_string(pos) + 
                            ", vocab=" + std::to_string(v) + 
                            " (value=" + std::to_string(logit) + 
                            "). Forward pass produced non-finite values. " +
                            "Check: (1) LM head matmul overflow, (2) hidden state explosion, (3) weight norm explosion."
                        );
                    }
                    
                    if (logit > pos_max) {
                        pos_second = pos_max;  // Previous max becomes second
                        pos_max = logit;
                        pos_argmax = v;
                    } else if (logit > pos_second) {
                        pos_second = logit;
                    }
                }
                argmax_counts[pos_argmax]++;
                max_logit_per_pos_sum += pos_max;
                margin_sum += (pos_max - pos_second);  // margin = max - second
            }
            
            logit_mean /= (sample_positions * vocab_size);
            float avg_max_logit = max_logit_per_pos_sum / sample_positions;
            float avg_margin = margin_sum / sample_positions;  // Average top-2 margin
            
            // Find top argmax predictions
            std::vector<std::pair<int, int>> sorted_argmax(argmax_counts.begin(), argmax_counts.end());
            std::sort(sorted_argmax.begin(), sorted_argmax.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });
            
            const int total_argmax = sample_positions;
            const int top1_count = sorted_argmax.empty() ? 0 : sorted_argmax[0].second;
            int top5_count = 0;
            for (size_t i = 0; i < std::min(sorted_argmax.size(), size_t(5)); ++i) {
                top5_count += sorted_argmax[i].second;
            }
            const float top1_frac = (total_argmax > 0) ? static_cast<float>(top1_count) / total_argmax : 0.0f;
            const float top5_frac = (total_argmax > 0) ? static_cast<float>(top5_count) / total_argmax : 0.0f;

            std::ostringstream logit_stats;
            logit_stats << "[LogitSignal] batch=" << (batch_idx + 1)
                        << " logit_mean=" << Internal::formatScalar(logit_mean, 4)
                        << " logit_max=" << Internal::formatScalar(logit_max, 4)
                        << " logit_min=" << Internal::formatScalar(logit_min, 4)
                        << " avg_max_logit=" << Internal::formatScalar(avg_max_logit, 4)
                        << " top2_margin=" << Internal::formatScalar(avg_margin, 4)
                        << " argmax_top1_frac=" << Internal::formatScalar(top1_frac, 4)
                        << " argmax_top5_frac=" << Internal::formatScalar(top5_frac, 4)
                        << " unique_argmax=" << argmax_counts.size()
                        << " top_argmax=[";
            for (size_t i = 0; i < std::min(sorted_argmax.size(), size_t(5)); ++i) {
                logit_stats << "tok" << sorted_argmax[i].first << ":" << sorted_argmax[i].second;
                if (i + 1 < std::min(sorted_argmax.size(), size_t(5))) logit_stats << ",";
            }
            logit_stats << "]";
            ctx.logging.logger->log(logit_stats.str());
            
            // ================================================================
            // LOGIT SCALE EQUATION DIAGNOSTIC (Rule 21)
            //
            // logit[v] = Σ_d hidden[t,d] × W[v,d] = h · W[v]^T
            // |logit[v]| ≤ ||h|| × ||W[v]||
            // logit_range = logit_max - logit_min
            // logit_std = sqrt(Var(logits))
            //
            // EXPECTED: logit_std ≈ sqrt(d_model) × h_rms × W_rms.
            // Track trends over time in stable metrics (logit_std, max_logit,
            // top2_margin, argmax concentration) rather than a fixed range cutoff.
            //
            // This diagnostic traces:
            //   1. Logit std/range across sampled positions×vocab
            //   2. Hidden state norms at LM head input (sampled)
            //   3. Weight norms for top-5 + random-10 vocab tokens
            //   4. Expected vs actual logit magnitude
            // ================================================================
            {
                // --- Logit statistics ---
                const float logit_range = logit_max - logit_min;
                // Compute logit variance (already have logit_mean from above)
                double logit_var_sum = 0.0;
                for (int pos = 0; pos < sample_positions; ++pos) {
                    for (int v = 0; v < vocab_size; ++v) {
                        const float diff = logit_sample[pos * vocab_size + v] - logit_mean;
                        logit_var_sum += static_cast<double>(diff) * diff;
                    }
                }
                const float logit_std = std::sqrt(static_cast<float>(logit_var_sum / (static_cast<double>(sample_positions) * vocab_size)));
                
                // Rule 20: FAIL LOUD on NaN/inf logits (issue #142)
                if (!std::isfinite(logit_max) || !std::isfinite(logit_min)) {
                    throw std::runtime_error(
                        "[LOGIT_SCALE_EQUATION] FATAL: Logits contain inf/nan at batch " + 
                        std::to_string(batch_idx + 1) + 
                        " (logit_max=" + std::to_string(logit_max) + 
                        ", logit_min=" + std::to_string(logit_min) + 
                        ", logit_mean=" + std::to_string(logit_mean) + 
                        "). Root cause: Numerical explosion in forward pass. " +
                        "Check: (1) hidden state norms, (2) weight norms, (3) gradient explosion in previous batch."
                    );
                }
                if (!std::isfinite(logit_std)) {
                    throw std::runtime_error(
                        "[LOGIT_SCALE_EQUATION] FATAL: logit_std is nan/inf at batch " + 
                        std::to_string(batch_idx + 1) + 
                        " (logit_std=" + std::to_string(logit_std) + 
                        ", logit_mean=" + std::to_string(logit_mean) + 
                        "). Variance computation produced NaN from inf logits."
                    );
                }
                
                // --- Per-position logit range ---
                float per_pos_range_sum = 0.0f;
                float per_pos_range_max = 0.0f;
                for (int pos = 0; pos < sample_positions; ++pos) {
                    float pos_max = -std::numeric_limits<float>::infinity();
                    float pos_min = std::numeric_limits<float>::infinity();
                    for (int v = 0; v < vocab_size; ++v) {
                        const float l = logit_sample[pos * vocab_size + v];
                        pos_max = std::max(pos_max, l);
                        pos_min = std::min(pos_min, l);
                    }
                    const float pos_range = pos_max - pos_min;
                    per_pos_range_sum += pos_range;
                    per_pos_range_max = std::max(per_pos_range_max, pos_range);
                }
                const float avg_per_pos_range = per_pos_range_sum / sample_positions;
                
                // --- Hidden state norms at LM head input ---
                // cached_encoder_output contains centered data (overwritten after LM head forward)
                const float* h_src = ts.cached_encoder_output.data;
                
                float h_rms_max = -std::numeric_limits<float>::infinity();
                float h_rms_min = std::numeric_limits<float>::infinity();
                float h_rms_mean = 0.0f;
                if (h_src) {
                    const size_t h_bytes = static_cast<size_t>(sample_positions) * d_model * sizeof(float);
                    std::vector<float> h_sample(sample_positions * d_model);
                    cudaMemcpy(h_sample.data(), h_src, h_bytes, cudaMemcpyDeviceToHost);
                    
                    for (int pos = 0; pos < sample_positions; ++pos) {
                        float sum_sq = 0.0f;
                        for (int d = 0; d < d_model; ++d) {
                            const float val = h_sample[pos * d_model + d];
                            sum_sq += val * val;
                        }
                        const float rms = std::sqrt(sum_sq / d_model);
                        h_rms_mean += rms;
                        h_rms_max = std::max(h_rms_max, rms);
                        h_rms_min = std::min(h_rms_min, rms);
                    }
                    h_rms_mean /= sample_positions;
                    
                    // Rule 20: Verify hidden state norms are finite (issue #142)
                    if (!std::isfinite(h_rms_mean) || !std::isfinite(h_rms_max)) {
                        throw std::runtime_error(
                            "[HIDDEN_STATE_NORMS] FATAL: Hidden state RMS contain inf/nan at batch " + 
                            std::to_string(batch_idx + 1) + 
                            " (h_rms_mean=" + std::to_string(h_rms_mean) + 
                            ", h_rms_max=" + std::to_string(h_rms_max) + 
                            "). Encoder output exploded. " +
                            "Check: (1) attention gradient explosion, (2) FFN activation overflow, (3) RMSNorm inverse explosion."
                        );
                    }
                }
                
                // --- Weight norm statistics (sample random + top tokens) ---
                // Issue #138 FIX: Compute E[||W||²] (mean of squared norms) for correct expected logit_std.
                // Old code used E[||W||] (mean of norms) which underestimates by Jensen's inequality
                // when ||W|| distribution is skewed.
                const float* lm_head_weights = ctx.model->getLmHeadLayer()->weights().data;
                const float* embedding_weights = ctx.model->getEmbeddingLayer()->tokenWeights().data;
                if (ctx.model->getConfig().tie_embeddings &&
                    lm_head_weights &&
                    embedding_weights &&
                    lm_head_weights != embedding_weights) {
                    throw std::runtime_error("Tied embeddings: lm_head_weights and embedding_weights must alias the same buffer.");
                }

                float w_rms_mean = 0.0f, w_rms_sq_mean = 0.0f, w_rms_max = 0.0f;
                int w_rms_max_tok = -1;
                const int w_sample_count = std::min(500, vocab_size);  // Sample 500 rows for better coverage
                if (lm_head_weights) {
                    std::vector<float> w_row(d_model);
                    
                    // Helper: compute RMS of a weight row on host
                    auto compute_row_rms = [&](int tok) -> float {
                        const size_t row_offset = static_cast<size_t>(tok) * d_model;
                        cudaMemcpy(w_row.data(),
                                   lm_head_weights + row_offset,
                                   d_model * sizeof(float), cudaMemcpyDeviceToHost);
                        float sum_sq = 0.0f;
                        for (int d = 0; d < d_model; ++d) {
                            sum_sq += w_row[d] * w_row[d];
                        }
                        return std::sqrt(sum_sq / d_model);
                    };
                    
                    // Sample evenly-spaced vocab tokens for mean/rms statistics
                    const int stride = std::max(1, vocab_size / w_sample_count);
                    int sampled = 0;
                    for (int tok = 0; tok < vocab_size && sampled < w_sample_count; tok += stride, ++sampled) {
                        const float rms = compute_row_rms(tok);
                        w_rms_mean += rms;
                        const float rms_sq = rms * rms;
                        w_rms_sq_mean += rms_sq;
                        if (rms > w_rms_max) {
                            w_rms_max = rms;
                            w_rms_max_tok = tok;
                        }
                    }
                    w_rms_mean /= sampled;
                    w_rms_sq_mean /= sampled;
                    
                    // FIX: Also check top-predicted tokens for ||W||_max.
                    // The strided sample (stride=100) can miss tokens with growing norms.
                    // Include top-argmax tokens to ensure ||W||_max is accurate.
                    for (size_t i = 0; i < std::min(sorted_argmax.size(), size_t(5)); ++i) {
                        const int tok = sorted_argmax[i].first;
                        // Skip if already in strided sample
                        if (tok % stride == 0 && tok / stride < w_sample_count) continue;
                        const float rms = compute_row_rms(tok);
                        if (rms > w_rms_max) {
                            w_rms_max = rms;
                            w_rms_max_tok = tok;
                        }
                    }
                }
                
                // --- Expected logit magnitude ---
                // Issue #138 FIX: Correct formula for dot product variance.
                // Var(h·W) = d × Var(h_i) × Var(W_j) = h_rms² × E[||W||²]
                // Old code used h_rms × E[||W||] which underestimates due to Jensen's inequality.
                // Correct: logit_std ≈ h_rms × sqrt(E[||W||²]) = h_rms × ||W||_rms
                const float w_rms_rms = std::sqrt(w_rms_sq_mean);  // sqrt(E[rms²])
                const float expected_logit_std = std::sqrt(static_cast<float>(d_model)) * h_rms_mean * w_rms_rms;
                const float logit_std_ratio = (expected_logit_std > 1e-10f) ? logit_std / expected_logit_std : 0.0f;

                // ============================================================
                // h↔W ALIGNMENT DIAGNOSTICS (Issue #149, telemetry streams 39-44)
                //
                // Detects the LM-head leak channel: grad_W[v] += (p_v - y_v) * h_t
                // accumulating across batches makes every W row drift along the
                // dominant h direction. This produces a coherent (h-aligned) bias
                // in W that inflates logit_std beyond the random-baseline formula
                // sqrt(d) * h_rms * W_rms (which assumes uncorrelated h, W).
                //
                // Random baseline (uncorrelated unit vectors in R^d):
                //   RMS(cos(h, W_v)) ≈ 1 / sqrt(d_model)        (≈ 0.0361 at d=768)
                // logit_std_ratio² ≈ 1 + d_model · cos_hW_rms²  (correlation correction)
                //
                // ALL ACCUMULATORS USE DOUBLE PRECISION (Rule 21: numerical precision).
                // Reads full-row data once; cost is ~500 D2H copies (already paid by
                // existing W sampling above, plus h_sample copy above). LOGIT_SCALE
                // cadence is already low-frequency so this is negligible overhead.
                // ============================================================
                float hw_cos_rms = 0.0f;
                float hw_cos_signed_mean = 0.0f;
                float hw_cos_abs_max = 0.0f;
                float hw_hbar_wbar_cos = 0.0f;
                float hw_h_dc_mean = 0.0f;
                float hw_h_dc_abs_max = 0.0f;
                if (h_src && lm_head_weights) {
                    // (1) Re-fetch h_sample (out-of-scope from earlier block).
                    const size_t h_bytes = static_cast<size_t>(sample_positions) * d_model * sizeof(float);
                    std::vector<float> h_sample(static_cast<size_t>(sample_positions) * d_model);
                    cudaMemcpy(h_sample.data(), h_src, h_bytes, cudaMemcpyDeviceToHost);

                    // (2) Per-position ||h_t||² and Σ_d h[t,d] (DC component) — double accum.
                    std::vector<double> h_norm_sq(sample_positions, 0.0);
                    std::vector<double> h_dc(sample_positions, 0.0);
                    for (int t = 0; t < sample_positions; ++t) {
                        const float* row = &h_sample[static_cast<size_t>(t) * d_model];
                        double sum_sq = 0.0;
                        double sum   = 0.0;
                        for (int d = 0; d < d_model; ++d) {
                            const double v = static_cast<double>(row[d]);
                            sum_sq += v * v;
                            sum   += v;
                        }
                        h_norm_sq[t] = sum_sq;
                        h_dc[t]      = sum / static_cast<double>(d_model);
                    }
                    // h_bar = mean_t h_t (per-dimension), accumulated in double.
                    std::vector<double> h_bar(d_model, 0.0);
                    for (int t = 0; t < sample_positions; ++t) {
                        const float* row = &h_sample[static_cast<size_t>(t) * d_model];
                        for (int d = 0; d < d_model; ++d) {
                            h_bar[d] += static_cast<double>(row[d]);
                        }
                    }
                    const double inv_T = 1.0 / static_cast<double>(sample_positions);
                    for (int d = 0; d < d_model; ++d) h_bar[d] *= inv_T;

                    // h DC summary stats.
                    double dc_sum = 0.0;
                    double dc_abs_max = 0.0;
                    for (int t = 0; t < sample_positions; ++t) {
                        dc_sum     += h_dc[t];
                        const double a = std::abs(h_dc[t]);
                        if (a > dc_abs_max) dc_abs_max = a;
                    }
                    hw_h_dc_mean    = static_cast<float>(dc_sum * inv_T);
                    hw_h_dc_abs_max = static_cast<float>(dc_abs_max);

                    // (3) Stream sampled W rows; compute cos(h_t, W_v) pairs with double accum.
                    //     Re-use the same strided sample as the W_rms loop above for consistency.
                    const int hw_stride = std::max(1, vocab_size / w_sample_count);
                    std::vector<float>  w_row_buf(d_model);
                    std::vector<double> w_bar(d_model, 0.0);

                    long double cos_sq_sum  = 0.0L;  // long double for RMS over up to 500*sample_positions terms
                    long double cos_signed  = 0.0L;
                    double      cos_abs_max_d = 0.0;
                    int64_t     pair_count  = 0;
                    int64_t     w_sampled   = 0;

                    for (int tok = 0; tok < vocab_size && w_sampled < w_sample_count;
                         tok += hw_stride, ++w_sampled)
                    {
                        const size_t row_offset = static_cast<size_t>(tok) * d_model;
                        cudaMemcpy(w_row_buf.data(),
                                   lm_head_weights + row_offset,
                                   d_model * sizeof(float),
                                   cudaMemcpyDeviceToHost);

                        // ||W_v||² in double.
                        double w_norm_sq = 0.0;
                        for (int d = 0; d < d_model; ++d) {
                            const double v = static_cast<double>(w_row_buf[d]);
                            w_norm_sq += v * v;
                            w_bar[d]  += v;
                        }
                        if (w_norm_sq <= 0.0 || !std::isfinite(w_norm_sq)) continue;
                        const double inv_w_norm = 1.0 / std::sqrt(w_norm_sq);

                        for (int t = 0; t < sample_positions; ++t) {
                            if (h_norm_sq[t] <= 0.0 || !std::isfinite(h_norm_sq[t])) continue;
                            const float* h_row = &h_sample[static_cast<size_t>(t) * d_model];
                            // dot(h_t, W_v) in double.
                            double dot = 0.0;
                            for (int d = 0; d < d_model; ++d) {
                                dot += static_cast<double>(h_row[d]) * static_cast<double>(w_row_buf[d]);
                            }
                            const double inv_h_norm = 1.0 / std::sqrt(h_norm_sq[t]);
                            const double cos_tv = dot * inv_h_norm * inv_w_norm;
                            cos_signed += static_cast<long double>(cos_tv);
                            cos_sq_sum += static_cast<long double>(cos_tv) * static_cast<long double>(cos_tv);
                            const double a = std::abs(cos_tv);
                            if (a > cos_abs_max_d) cos_abs_max_d = a;
                            ++pair_count;
                        }
                    }

                    if (pair_count > 0) {
                        const long double inv_n = 1.0L / static_cast<long double>(pair_count);
                        hw_cos_rms         = static_cast<float>(std::sqrt(static_cast<double>(cos_sq_sum * inv_n)));
                        hw_cos_signed_mean = static_cast<float>(static_cast<double>(cos_signed * inv_n));
                        hw_cos_abs_max     = static_cast<float>(cos_abs_max_d);
                    }

                    // (4) Rank-1 DC channel: cos(h_bar, W_bar).
                    if (w_sampled > 0) {
                        const double inv_W = 1.0 / static_cast<double>(w_sampled);
                        double hbar_dot_wbar = 0.0;
                        double hbar_norm_sq  = 0.0;
                        double wbar_norm_sq  = 0.0;
                        for (int d = 0; d < d_model; ++d) {
                            const double w = w_bar[d] * inv_W;
                            const double h = h_bar[d];
                            hbar_dot_wbar += h * w;
                            hbar_norm_sq  += h * h;
                            wbar_norm_sq  += w * w;
                        }
                        if (hbar_norm_sq > 0.0 && wbar_norm_sq > 0.0) {
                            hw_hbar_wbar_cos = static_cast<float>(
                                hbar_dot_wbar / std::sqrt(hbar_norm_sq * wbar_norm_sq));
                        }
                    }

                    // Rule 20: fail loud on NaN/Inf in alignment metrics.
                    if (!std::isfinite(hw_cos_rms) || !std::isfinite(hw_cos_signed_mean) ||
                        !std::isfinite(hw_cos_abs_max) || !std::isfinite(hw_hbar_wbar_cos) ||
                        !std::isfinite(hw_h_dc_mean) || !std::isfinite(hw_h_dc_abs_max))
                    {
                        throw std::runtime_error(
                            "[HW_ALIGNMENT] FATAL: h↔W alignment metrics contain NaN/Inf at batch " +
                            std::to_string(batch_idx + 1));
                    }
                }

                // Publish to telemetry lattice (streams 39-44). CSV logger picks these up.
                ctx.telemetry.last_obs[39] = hw_cos_rms;          // HW_COS_RMS
                ctx.telemetry.last_obs[40] = hw_cos_signed_mean;  // HW_COS_SIGNED_MEAN
                ctx.telemetry.last_obs[41] = hw_cos_abs_max;      // HW_COS_ABS_MAX
                ctx.telemetry.last_obs[42] = hw_hbar_wbar_cos;    // HW_HBAR_WBAR_COS
                ctx.telemetry.last_obs[43] = hw_h_dc_mean;        // HW_H_DC_MEAN
                ctx.telemetry.last_obs[44] = hw_h_dc_abs_max;     // HW_H_DC_ABS_MAX

                struct LogitTrendState {
                    bool initialized = false;
                    float ema_logit_std = 0.0f;
                    float ema_logit_max = 0.0f;
                    float ema_top2_margin = 0.0f;
                    float ema_top1_frac = 0.0f;
                };
                static LogitTrendState trend_state;
                const float trend_alpha = 0.10f;
                if (!trend_state.initialized) {
                    trend_state.initialized = true;
                    trend_state.ema_logit_std = logit_std;
                    trend_state.ema_logit_max = logit_max;
                    trend_state.ema_top2_margin = avg_margin;
                    trend_state.ema_top1_frac = top1_frac;
                } else {
                    trend_state.ema_logit_std = trend_alpha * logit_std + (1.0f - trend_alpha) * trend_state.ema_logit_std;
                    trend_state.ema_logit_max = trend_alpha * logit_max + (1.0f - trend_alpha) * trend_state.ema_logit_max;
                    trend_state.ema_top2_margin = trend_alpha * avg_margin + (1.0f - trend_alpha) * trend_state.ema_top2_margin;
                    trend_state.ema_top1_frac = trend_alpha * top1_frac + (1.0f - trend_alpha) * trend_state.ema_top1_frac;
                }
                
                std::ostringstream scale_eq;
                scale_eq << std::fixed << std::setprecision(6);
                scale_eq << "[LOGIT_SCALE_EQUATION] logit[v] = h · W[v]^T, logit_range = max - min\n";
                scale_eq << "  LOGIT STATS: std=" << logit_std << " range=" << logit_range
                         << " avg_per_pos_range=" << avg_per_pos_range
                         << " max_per_pos_range=" << per_pos_range_max << "\n";
                scale_eq << "  HIDDEN (LM input): h_rms_mean=" << h_rms_mean
                         << " h_rms_max=" << h_rms_max << " h_rms_min=" << h_rms_min << "\n";
                scale_eq << "  WEIGHTS (LM head): W_rms_mean=" << w_rms_mean
                         << " W_rms_rms=" << w_rms_rms
                         << " W_rms_max=" << w_rms_max << " (tok=" << w_rms_max_tok << ")"
                         << " d_model=" << d_model << "\n";
                scale_eq << "  EXPECTED logit_std = sqrt(d_model) × h_rms × W_rms_rms\n";
                scale_eq << "                      = sqrt(" << d_model << ") × " << h_rms_mean
                         << " × " << w_rms_rms << "\n";
                scale_eq << "                      = " << expected_logit_std << "\n";
                scale_eq << "  ACTUAL logit_std = " << logit_std
                         << " ratio(actual/expected)=" << logit_std_ratio << "\n";
                scale_eq << "  TREND (EMA α=" << trend_alpha << ")"
                         << " logit_std_delta=" << (logit_std - trend_state.ema_logit_std)
                         << " max_logit_delta=" << (logit_max - trend_state.ema_logit_max)
                         << " top2_margin_delta=" << (avg_margin - trend_state.ema_top2_margin)
                         << " top1_frac_delta=" << (top1_frac - trend_state.ema_top1_frac) << "\n";
                if (w_rms_max > 2.0f) {
                    scale_eq << "  [ANOMALY] WEIGHT_RMS_EXPLOSION: W_rms_max=" << w_rms_max
                             << " (tok=" << w_rms_max_tok << ") >> 2.0. Weight decay too weak or gradient bias.\n";
                }
                if (logit_std_ratio > 3.0f) {
                    scale_eq << "  [ANOMALY] LOGIT_STD_MISMATCH: actual/expected=" << logit_std_ratio
                             << " >> 3.0. Possible hidden-weight alignment or missing 1/sqrt(d) scaling.\n";
                }
                ctx.logging.logger->log(scale_eq.str());
            }
            
            // RHO_BUILDUP_EQUATION: Per-layer hidden state correlation
            // (moved to Diagnostics/RhoDiagnostic.cu)
            GRIM::Diagnostics::computeRhoDiagnostic(ctx, payload, batch_idx);
            
            // ================================================================
            // LM HEAD DIAGNOSTICS: Row norms ||W[v]|| for top predicted tokens
            // ================================================================
            const float* lm_head_weights_for_norms = ctx.model->getLmHeadLayer()->weights().data;
            if (lm_head_weights_for_norms) {
                // Copy LM head rows for top-5 predicted tokens
                std::ostringstream lm_stats;
                lm_stats << "[LMHeadNorm] batch=" << (batch_idx + 1) << " rms(W[v])=[";
                
                std::vector<float> row_buffer(d_model);
                for (size_t i = 0; i < std::min(sorted_argmax.size(), size_t(5)); ++i) {
                    int tok_id = sorted_argmax[i].first;
                    // Copy row [tok_id, :] from W [vocab_size, d_model]
                    const size_t row_offset = static_cast<size_t>(tok_id) * d_model;
                    cudaMemcpy(row_buffer.data(), 
                               lm_head_weights_for_norms + row_offset,
                               d_model * sizeof(float), cudaMemcpyDeviceToHost);
                    
                    // Compute RMS of row
                    float sum_sq = 0.0f;
                    for (int d = 0; d < d_model; ++d) {
                        sum_sq += row_buffer[d] * row_buffer[d];
                    }
                    float row_rms = std::sqrt(sum_sq / d_model);
                    lm_stats << "tok" << tok_id << ":" << Internal::formatScalar(row_rms, 10);
                    if (i + 1 < std::min(sorted_argmax.size(), size_t(5))) lm_stats << ",";
                }
                lm_stats << "]";
                ctx.logging.logger->log(lm_stats.str());
            }
        }
    }
    
    if (!std::isfinite(result.loss)) {
        throw std::runtime_error("Non-finite batch loss: " + std::to_string(result.loss));
    }
    
    // DIAGNOSTIC: If loss is suspiciously high (>50), log per-sequence breakdown
    if (result.loss > 50.0f) {
        std::ostringstream spike_diag;
        spike_diag << "[LossDiag] SPIKE DETECTED loss=" << result.loss << " batch=" << (batch_idx + 1) << "\n";
        spike_diag << "  Sequences: ";
        size_t max_seq_in_batch = 0;
        for (int i = 0; i < payload.batch_size; ++i) {
            spike_diag << payload.seq_ids[i] << "(len=" << payload.seq_lengths[i] << ")";
            max_seq_in_batch = std::max(max_seq_in_batch, static_cast<size_t>(payload.seq_lengths[i]));
            if (i + 1 < payload.batch_size) spike_diag << ", ";
        }
        const bool stability_seq_override = ctx.config.stability.enabled && ctx.config.stability.max_seq_len > 0;
        const int config_seq_len_limit = stability_seq_override
            ? ctx.config.stability.max_seq_len
            : ctx.config.hyperparameters.max_seq_len;
        const int effective_seq_len_limit = config_seq_len_limit > 0
            ? config_seq_len_limit
            : ctx.model->getConfig().max_seq_len;
        spike_diag << "\n  MAX_SEQ_LEN=" << max_seq_in_batch;
        spike_diag << " d_model=" << ctx.model->getConfig().d_model;
        spike_diag << " limit=" << effective_seq_len_limit;
        if (stability_seq_override) {
            spike_diag << " (stability_override)";
        }
        if (effective_seq_len_limit > 0 &&
            max_seq_in_batch >= static_cast<size_t>(effective_seq_len_limit)) {
            spike_diag << " *** BOUNDARY CROSSED (seq_len >= " << effective_seq_len_limit
                       << (stability_seq_override ? " = stability_override.max_seq_len" : " = max_seq_len")
                       << ") ***";
        }
        spike_diag << "\n  First 10 targets per seq: ";
        for (int s = 0; s < payload.batch_size; ++s) {
            spike_diag << "[";
            const int flat_start = s * payload.max_seq_len;
            const int len = payload.seq_lengths[s];
            for (int t = 0; t < std::min(10, len); ++t) {
                spike_diag << payload.target_ids[flat_start + t];
                if (t + 1 < std::min(10, len)) spike_diag << ",";
            }
            spike_diag << "] ";
        }
        spike_diag << "\n  Loss indicates model p_t ~ exp(-" << result.loss << ") -> EXTREME WRONG CONFIDENCE";
        spike_diag << "\n  HYPOTHESIS: Position embedding corrupted for pos >= " << effective_seq_len_limit;
        spike_diag << "\n  Check: Is attention collapsing? Are embeddings for these tokens corrupted?";
        ctx.logging.logger->log(spike_diag.str());
    }
    
    // Adaptive loss tracking - record baseline and minimum
    // NOTE: Loss alone is NOT sufficient to skip batches. Skipping hard examples
    // based on loss biases the model away from difficult regions of the data manifold,
    // causing apparent early improvement but later generalization failure.
    // Only skip on CONFIRMED corruption: invalid tokens, NaNs, or gradient spikes.
    if (state.initial_loss == 0.0f) {
        state.initial_loss = result.loss;
        state.min_observed_loss = result.loss;
        // Calculate expected random baseline for reference (ln(vocab_size))
        const float expected_random_baseline = std::log(static_cast<float>(ctx.config.actual_vocab_size));
        const bool likely_from_checkpoint = result.loss < (expected_random_baseline - 1.0f);
        std::string baseline_note = likely_from_checkpoint
            ? "(from checkpoint, expected random=" + Internal::formatScalar(expected_random_baseline) + ")"
            : "(random baseline for vocab=" + std::to_string(ctx.config.actual_vocab_size) + ")";
        ctx.logging.logger->log("[LossBaseline] Initial loss=" + Internal::formatScalar(state.initial_loss) +
                                " " + baseline_note);
    } else {
        state.warmup_batches++;
        
        // Track minimum observed loss
        if (result.loss < state.min_observed_loss) {
            state.min_observed_loss = result.loss;
        }
        
        // Check for CONFIRMED corruption: invalid token IDs
        // NaN/Inf loss is already handled by autogradTrainingStep() → early return above.
        // Loss-only skipping removes hard examples and destroys generalization
        bool has_invalid_tokens = false;
        
        // Scan for token IDs outside vocab range (actual corruption) — using flat payload
        for (int s = 0; s < payload.batch_size && !has_invalid_tokens; ++s) {
            const int flat_start = s * payload.max_seq_len;
            const int len = payload.seq_lengths[s];
            for (int t = 0; t < len; ++t) {
                if (payload.input_ids[flat_start + t] < 0 ||
                    payload.input_ids[flat_start + t] >= static_cast<int>(ctx.config.actual_vocab_size)) {
                    has_invalid_tokens = true;
                    break;
                }
            }
        }
        for (int s = 0; s < payload.batch_size && !has_invalid_tokens; ++s) {
            const int flat_start = s * payload.max_seq_len;
            const int len = payload.seq_lengths[s];
            for (int t = 0; t < len; ++t) {
                const int tid = payload.target_ids[flat_start + t];
                // targets can be -1 for masked positions, but not other negatives or OOB
                if (tid < -1 || tid >= static_cast<int>(ctx.config.actual_vocab_size)) {
                    has_invalid_tokens = true;
                    break;
                }
            }
        }
        
        // Rule 20: Data corruption = CRASH. Fix the data pipeline, don't silently skip.
        if (has_invalid_tokens) {
            throw std::runtime_error(
                "DATA CORRUPTION: batch " + std::to_string(batch_idx + 1) +
                " contains token IDs outside vocab range [0, " + std::to_string(ctx.config.actual_vocab_size) +
                ") — fix data pipeline at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
        }
        
        // Log high loss for monitoring, but DO NOT skip
        // Hard examples are valuable for learning - removing them biases the model
        float loss_threshold = (state.warmup_batches < 100) 
            ? state.initial_loss * 2.0f 
            : state.min_observed_loss * 1.5f + 2.0f;
        if (result.loss > loss_threshold) {
            ctx.logging.logger->log("[LossMonitor] HIGH_LOSS batch=" + std::to_string(batch_idx + 1) +
                                    " loss=" + Internal::formatScalar(result.loss) +
                                    " threshold=" + Internal::formatScalar(loss_threshold));
        }
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // POST-STEP: Backward already ran inside autogradTrainingStep().
    // Diagnostics below read from TrainingState (persists through backward).
    // ═══════════════════════════════════════════════════════════════════════════
    
    ctx.logging.logger->log("[GradTrace] POST-BACKWARD batch=" + std::to_string(batch_idx + 1) + 
                            " loss=" + Internal::formatScalar(result.loss) + 
                            " valid_tokens=" + std::to_string(payload.valid_tokens));
    
    // NOTE: Gradient component logging happens later after measureGradientNorms()
    // via formatGradientComponents(). Premature logging here would use undefined variables.

    // ========================================================================
    // DIAGNOSTIC: Issue #142 - Special Token Weight & Gradient Verification
    // (Rule 21 Equation-Based)
    //
    // After removing -inf logit masking (Issue #142), special tokens
    // (UNK=0, PAD=1, BOS=2, EOS=3) participate naturally in softmax.
    // Loss masking via target=-1 handles them correctly (zero loss, zero grad
    // at those TARGET positions). But their EMBEDDING WEIGHT ROWS still receive
    // gradients from:
    //   1. LM head backward: grad_W[v,d] = Σ_t hidden[t,d] * grad_logits[t,v]
    //      (summed across ALL positions, not just target-v positions)
    //   2. Embedding forward: grad_W[tok_id] += grad_encoder[t] * scale
    //      (only at positions where tok_id appears as INPUT)
    //
    // This diagnostic tracks whether special token weight rows are:
    //   - Receiving meaningful gradients (they SHOULD, from LM head backward)
    //   - Diverging in norm from content tokens (ANOMALY if so)
    //   - Drifting toward zero (could indicate implicit suppression)
    //
    // [SPECIAL_TOKEN_EQUATION] W_special health check:
    //   ||W[v]|| should be ~same magnitude as ||W[content]||_mean
    //   ||grad_W[v]|| should be non-zero (from LM head backward)
    // ========================================================================
    if (shouldSyncDiagnostics(ctx, batch_idx) && ctx.logging.tape && ctx.logging.tape->accepts(GRIM::Logging::LogLevel::Debug)) {
        const auto& cfg = ctx.model->getConfig();
        const float* weights_ptr = ctx.model->getLmHeadLayer()->weights().data;
        // Issue #150: When tied=no, LM head and embedding are DIFFERENT tensors.
        // Read gradients from the SAME layer as weights (LM head) so rms(W) and
        // rms(grad) refer to the same parameter. Previously read embedding grads,
        // which showed PAD scatter-add accumulation (~1754 positions) as 76x spike
        // vs BOS/EOS — misleading because that gradient doesn't affect LM head.
        const bool weights_tied = ctx.model->getEmbeddingLayer()->tokenWeights().data
                               == ctx.model->getLmHeadLayer()->weights().data;
        const float* grads_ptr = weights_tied
            ? ctx.model->getEmbeddingLayer()->tokenWeights().grad_data()   // tied: same tensor, either pointer works
            : ctx.model->getLmHeadLayer()->weights().grad_data();          // untied: use LM head's own gradients

        if (weights_ptr) {
                constexpr int SPECIAL_IDS[] = {
                    GRIM::Tokenizer::UNK_TOKEN_ID,   // 0
                    GRIM::Tokenizer::PAD_TOKEN_ID,    // 1
                    GRIM::Tokenizer::BOS_TOKEN_ID,    // 2
                    GRIM::Tokenizer::EOS_TOKEN_ID     // 3
                };
                constexpr const char* SPECIAL_NAMES[] = {"UNK", "PAD", "BOS", "EOS"};
                constexpr int NUM_SPECIALS = 4;

                std::vector<float> row_buf(cfg.d_model);
                std::ostringstream diag;
                diag << std::fixed << std::setprecision(8);
                diag << "[SPECIAL_TOKEN_EQUATION] batch=" << (batch_idx + 1)
                     << " W_special health: logit[v] = h · W[v]^T\n";

                // Also sample a few content token norms for comparison baseline
                double content_norm_sum = 0.0;
                int content_norm_count = 0;
                constexpr int CONTENT_SAMPLE_IDS[] = {512, 1000, 5000, 10000, 25000, 40000};
                for (int cid : CONTENT_SAMPLE_IDS) {
                    if (cid >= cfg.vocab_size) continue;
                    const size_t off = static_cast<size_t>(cid) * cfg.d_model;
                    cudaMemcpy(row_buf.data(), weights_ptr + off,
                               cfg.d_model * sizeof(float), cudaMemcpyDeviceToHost);
                    double sq = 0.0;
                    for (int d = 0; d < cfg.d_model; ++d) sq += static_cast<double>(row_buf[d]) * row_buf[d];
                    content_norm_sum += std::sqrt(sq / cfg.d_model);
                    content_norm_count++;
                }
                const double content_norm_mean = (content_norm_count > 0)
                    ? content_norm_sum / content_norm_count : 0.0;

                for (int s = 0; s < NUM_SPECIALS; ++s) {
                    const int tok_id = SPECIAL_IDS[s];
                    const size_t row_offset = static_cast<size_t>(tok_id) * cfg.d_model;

                    // Weight row
                    cudaMemcpy(row_buf.data(), weights_ptr + row_offset,
                               cfg.d_model * sizeof(float), cudaMemcpyDeviceToHost);
                    double w_sq = 0.0, w_sum = 0.0;
                    for (int d = 0; d < cfg.d_model; ++d) {
                        w_sq += static_cast<double>(row_buf[d]) * row_buf[d];
                        w_sum += row_buf[d];
                    }
                    const float w_rms = static_cast<float>(std::sqrt(w_sq / cfg.d_model));
                    const float w_mean = static_cast<float>(w_sum / cfg.d_model);

                    // Gradient row (may be null if not yet computed)
                    float g_rms = 0.0f, g_sum = 0.0f;
                    bool has_grad = false;
                    if (grads_ptr) {
                        cudaMemcpy(row_buf.data(), grads_ptr + row_offset,
                                   cfg.d_model * sizeof(float), cudaMemcpyDeviceToHost);
                        double g_sq = 0.0, gs = 0.0;
                        bool any_nonzero = false;
                        for (int d = 0; d < cfg.d_model; ++d) {
                            g_sq += static_cast<double>(row_buf[d]) * row_buf[d];
                            gs += row_buf[d];
                            if (row_buf[d] != 0.0f) any_nonzero = true;
                        }
                        g_rms = static_cast<float>(std::sqrt(g_sq / cfg.d_model));
                        g_sum = static_cast<float>(gs);
                        has_grad = any_nonzero;
                    }

                    // Count appearances as INPUT token in this batch
                    // (special tokens only appear as input if BOS is prepended, etc.)
                    // We don't have input_ids readily available here, so skip input count.

                    diag << "  " << SPECIAL_NAMES[s] << "(id=" << tok_id << "): "
                         << "rms(W)=" << w_rms
                         << " w_mean=" << w_mean;
                    if (grads_ptr) {
                        diag << " rms(grad)=" << g_rms
                             << " grad_sum=" << g_sum
                             << (has_grad ? "" : " [ZERO_GRAD]");
                    } else {
                        diag << " [NO_GRAD_BUFFER]";
                    }

                    // Anomaly: special token weight RMS diverging from content tokens
                    if (content_norm_mean > 0.0 && w_rms > 3.0f * content_norm_mean) {
                        diag << " [ANOMALY] rms(W)=" << w_rms
                             << " >> content_mean=" << Internal::formatScalar(static_cast<float>(content_norm_mean), 6);
                    }
                    if (w_rms < 1e-6f) {
                        diag << " [ANOMALY] NEAR_ZERO_WEIGHT";
                    }
                    if (!std::isfinite(w_rms) || !std::isfinite(g_rms)) {
                        throw std::runtime_error(
                            "[SPECIAL_TOKEN_EQUATION] Non-finite special token weight/grad: "
                            + std::string(SPECIAL_NAMES[s]) + " rms(W)=" + std::to_string(w_rms)
                            + " rms(grad)=" + std::to_string(g_rms)
                            + " at batch " + std::to_string(batch_idx + 1));
                    }
                    diag << "\n";
                }
                diag << "  content_baseline: rms(W)_mean=" << Internal::formatScalar(static_cast<float>(content_norm_mean), 6)
                     << " (sampled " << content_norm_count << " tokens)";

                ctx.logging.logger->log(diag.str());
                EQ_LOG(ctx.logging.tape.get(), GRIM::Logging::LogGroup::Embedding, GRIM::Logging::LogPhase::GRADIENT_CLIP, 0, "SPECIAL_TOKEN_EQUATION", diag.str().c_str());
            }
    }
    
    // NOTE: Window closes automatically via beginOptimizerStep() → endOptimizerStep()
    // State flow: ACCUMULATING → READY_FOR_STEP → IDLE
    
    // DISABLED for performance: Diagnostic causes device sync
    // if (diag_batch_count <= 3) {
    //     const auto& ts = ctx.model->getTrainingState();
    //     ctx.logging.logger->log("[GradDiag] AFTER_BACKWARD: " + sampleEmbeddingGrads(ts, ts.stream_ctrl.getPrimaryStream()));
    // }
    
    // FIX Issue #23: Sync gradient norms EVERY batch for accurate diagnostics.
    // Previously: only synced every 10th batch (batch_idx % 10 == 0), used stale cached values.
    // Evidence: Batches 611-620 all showed grad_norm=2.5877 despite actual norms varying 2.1-5.3.
    // Impact: Diagnostics showed wrong values, spike detection used stale data.
    // NOTE: Performance cost is ~1ms per batch (acceptable for correctness).
    // NOTE: After Issue #135, clipping happens LATER (in should_step block) and recomputes
    //       its own norm on scaled gradients. This norm is for diagnostics/spike detection only.

    ctx.logging.logger->log("[GradTrace] PRE-GRADNORM batch=" + std::to_string(batch_idx + 1) +
                            " cached_preclip=" + Internal::formatScalar(state.last_grad_rms));
    
    // === TIMING GUARD: Track expensive operations between POST-BACKWARD logs ===
    auto grad_ops_start = std::chrono::steady_clock::now();
    
    // Compute gradient norm on GPU via free functions (no wrapper overhead)
    auto norm_start = std::chrono::steady_clock::now();
    
    auto& training_state = ctx.model->getTrainingState();
    cudaStream_t stream = training_state.stream_ctrl.getPrimaryStream();
    const auto& groups = ctx.model->parameterGroups();

    // ════════════════════════════════════════════════════════════════════
    // DIAGNOSTIC: Sample gradient values BEFORE measurement to verify
    // backward results survive to this point.  Companion to [GRAD_DIAG]
    // POST-BACKWARD in AutogradTraining.cu.
    // ════════════════════════════════════════════════════════════════════
    {
        cudaStreamSynchronize(stream);
        float lm_sample = 0.0f;
        float* lm_grads = ctx.model->getLmHeadLayer()->weights().grad_data();
        if (lm_grads) {
            cudaMemcpy(&lm_sample, lm_grads, sizeof(float), cudaMemcpyDeviceToHost);
        }
        // Also verify the ParameterGroup sees the same pointer
        float pg_sample = 0.0f;
        for (size_t g = 0; g < groups.size(); ++g) {
            if (groups[g].type == GRIM::ParamGroupType::LM_HEAD) {
                float* pg_grads = groups[g].grads();
                if (pg_grads) {
                    cudaMemcpy(&pg_sample, pg_grads, sizeof(float), cudaMemcpyDeviceToHost);
                }
                fprintf(stderr,
                    "[GRAD_DIAG] PRE-MEASURE batch=%d micro=%d "
                    "lm_grad[0]=%.10e lm_ptr=%p pg_grad[0]=%.10e pg_ptr=%p match=%s\n",
                    batch_idx + 1, ctx.optimizer.current_micro_step,
                    lm_sample, static_cast<void*>(lm_grads),
                    pg_sample, static_cast<void*>(pg_grads),
                    (lm_grads == pg_grads) ? "YES" : "NO");
                break;
            }
        }
    }

    // Lazy-allocate scratch buffers on first use
    if (!training_state.grad_norm_scratch) {
        training_state.grad_norm_scratch = GRIM::GradNorm::allocateGradNormScratch(
            groups.size() * 2, stream);
        if (!training_state.grad_norm_scratch) {
            throw std::runtime_error("[FATAL] Failed to allocate grad_norm_scratch at batch " +
                                     std::to_string(batch_idx + 1));
        }
    }
    
    // Issue #138: Record CUDA event BEFORE launching grad norm kernels
    cudaEvent_t pre_norm_event = nullptr, post_norm_event = nullptr;
    cudaEventCreate(&pre_norm_event);
    cudaEventRecord(pre_norm_event, stream);
    
    auto norm_status = GRIM::GradNorm::measureGradientNormsLaunch(
        groups.data(), groups.size(), training_state.grad_norm_scratch, stream);
    if (norm_status != GRIM::GradNorm::GradNormStatus::SUCCESS) {
        throw std::runtime_error("[FATAL] measureGradientNormsLaunch failed: " +
                                 std::string(GRIM::GradNorm::statusToString(norm_status)) +
                                 " at batch " + std::to_string(batch_idx + 1));
    }
    cudaEventCreate(&post_norm_event);
    cudaEventRecord(post_norm_event, stream);
    cudaStreamSynchronize(stream);  // Wait for D2H before Finalize
    norm_status = GRIM::GradNorm::measureGradientNormsFinalize(
        groups.data(), groups.size(), training_state.grad_norm_scratch);
    if (norm_status != GRIM::GradNorm::GradNormStatus::SUCCESS) {
        throw std::runtime_error("[FATAL] measureGradientNormsFinalize failed: " +
                                 std::string(GRIM::GradNorm::statusToString(norm_status)) +
                                 " at batch " + std::to_string(batch_idx + 1));
    }

    const auto& gm = *training_state.grad_norm_scratch->h_metrics;
    // Compute per-component RMS matching the clipping strategy (Issue #139)
    const bool tied_for_norm = ctx.model->getConfig().tie_embeddings;
    const float emb_sum_sq_pre = tied_for_norm ? gm.lm_head_sum_sq : (gm.lm_head_sum_sq + gm.embedding_sum_sq);
    const int64_t emb_count_pre = tied_for_norm ? gm.lm_head_count : (gm.lm_head_count + gm.embedding_count);
    const float enc_sum_sq_pre = gm.attention_sum_sq + gm.ffn_sum_sq + gm.rmsnorm_sum_sq + gm.scratchblock_sum_sq + gm.reasoning_head_sum_sq + gm.execution_block_sum_sq;
    const int64_t enc_count_pre = gm.attention_count + gm.ffn_count + gm.rmsnorm_count + gm.scratchblock_count + gm.reasoning_head_count + gm.execution_block_count;
    const float emb_rms_pre = (emb_count_pre > 0) ? std::sqrt(emb_sum_sq_pre / static_cast<float>(emb_count_pre)) : 0.0f;
    const float enc_rms_pre = (enc_count_pre > 0) ? std::sqrt(enc_sum_sq_pre / static_cast<float>(enc_count_pre)) : 0.0f;
    // Separate sb_rms for POST-GRADNORM visibility (Issue #150)
    const float sb_rms_pre = (gm.scratchblock_count > 0) ? std::sqrt(gm.scratchblock_sum_sq / static_cast<float>(gm.scratchblock_count)) : 0.0f;
    // Combined RMS across all parameter groups
    const float total_sum_sq_pre = emb_sum_sq_pre + enc_sum_sq_pre;
    const int64_t total_count_pre = emb_count_pre + enc_count_pre;
    result.grad_rms = (total_count_pre > 0) ? std::sqrt(total_sum_sq_pre / static_cast<float>(total_count_pre)) : 0.0f;
    result.normalized_grad_rms = result.grad_rms;
    const float preclip_grad_rms = result.grad_rms;
    auto norm_elapsed_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - norm_start).count();
    
    // Issue #138: Decompose wall time into kernel time + backward drain time
    float gpu_kernel_ms = 0.0f;
    cudaEventElapsedTime(&gpu_kernel_ms, pre_norm_event, post_norm_event);
    cudaEventDestroy(pre_norm_event);
    cudaEventDestroy(post_norm_event);
    const float drain_ms = norm_elapsed_ms - gpu_kernel_ms;
    ctx.logging.logger->log("[GradTrace] POST-BACKWARD synced measureGradientNorms in " +
                                Internal::formatScalar(norm_elapsed_ms, 2) + "ms (kernel=" +
                                Internal::formatScalar(gpu_kernel_ms, 2) + "ms drain=" +
                                Internal::formatScalar(drain_ms, 2) + "ms)");

    // NaN/Inf check — RULE 20: Fail loud!
    if (gm.has_nan || gm.has_inf) {
        std::ostringstream nf_log;
        nf_log << "[GradTrace] NON-FINITE grads detected"
               << " nan=" << (gm.has_nan ? "true" : "false")
               << " inf=" << (gm.has_inf ? "true" : "false")
               << " first_nan_group=" << gm.first_nan_group
               << " first_nan_value=" << gm.first_nan_value
               << " first_inf_group=" << gm.first_inf_group
               << " first_inf_value=" << gm.first_inf_value
               << " groups_processed=" << gm.groups_processed;
        ctx.logging.logger->log(nf_log.str());
        
        throw std::runtime_error("[FATAL] NaN/Inf detected in gradients at batch " + 
                                std::to_string(batch_idx + 1) + 
                                " first_nan_group=" + std::to_string(gm.first_nan_group) +
                                " - investigate the backward pass!");
    }

    const bool tied = ctx.model->getConfig().tie_embeddings;
    std::string comp_log = Internal::formatGradientComponents(gm, tied);
    ctx.logging.logger->log(comp_log);

    ctx.logging.logger->log("[GradTrace] POST-GRADNORM preclip=" + Internal::formatScalar(preclip_grad_rms) +
                            " emb_rms=" + Internal::formatScalar(emb_rms_pre, 6) +
                            " enc_rms=" + Internal::formatScalar(enc_rms_pre, 6) +
                            " sb_rms=" + Internal::formatScalar(sb_rms_pre, 6));
    
    // ========================================================================
    // DIAGNOSTIC: [EMB_GRAD_EQUATION] Embedding gradient spike analysis (Issue #141)
    // Rule 21 equation-based logging for tied-weight gradient decomposition.
    // Runs every diag_interval batches (same cadence as other sync diagnostics).
    // Identifies which token rows concentrate gradient mass and whether
    // atomicAdd scatter density correlates with spike magnitude.
    // ========================================================================
    {
        static float prev_emb_rms_for_spike_diag = 0.0f;
        
        // Gate behind shouldSyncDiagnostics so full vocab gradient D2H only runs on diagnostic sync interval
        static int emb_grad_diag_interval = 10;
        const bool kEmbGradDiagEnabled = shouldSyncDiagnostics(ctx, batch_idx) &&
            ctx.logging.tape && ctx.logging.tape->accepts(GRIM::Logging::LogLevel::Debug) &&
            (batch_idx == 0 || (batch_idx + 1) % std::max(emb_grad_diag_interval, 1) == 0);
        
        // gm is already in scope from measureGradientNorms above
        const float curr_emb_rms = (gm.lm_head_count > 0) 
            ? std::sqrt(gm.lm_head_sum_sq / static_cast<float>(gm.lm_head_count)) : 0.0f;
        
        if (kEmbGradDiagEnabled) {
            const auto& ts = ctx.model->getTrainingState();
            const auto& cfg = ctx.model->getConfig();
            cudaStream_t stream = ts.stream_ctrl.getPrimaryStream();
            
            const int total_tokens_diag = ts.cached_batch_size * ts.cached_seq_len;
            const int* d_tok_ids = reinterpret_cast<const int*>(ts.cached_token_ids_tensor.data);
            
            if (d_tok_ids && total_tokens_diag > 0) {
                GRIM::Diagnostics::EmbGradEquationDiag emb_diag = GRIM::Diagnostics::computeEmbGradEquation(
                    ctx.model->getEmbeddingLayer(), d_tok_ids, total_tokens_diag,
                    cfg.d_model, cfg.vocab_size,
                    prev_emb_rms_for_spike_diag, curr_emb_rms,
                    stream);
                
                std::string emb_eq_str = GRIM::Diagnostics::formatEmbGradEquation(emb_diag, batch_idx);
                ctx.logging.logger->log(emb_eq_str);
                
                EQ_LOG(ctx.logging.tape.get(), GRIM::Logging::LogGroup::Embedding, GRIM::Logging::LogPhase::GRADIENT_CLIP, 0, "EMB_GRAD_EQUATION", emb_eq_str.c_str());
            }
        }
        
        prev_emb_rms_for_spike_diag = curr_emb_rms;
    }
    
    // === TIMING GUARD: Track operations between POST-BACKWARD and PRE-OPTIMIZER ===
    auto pre_optimizer_start = std::chrono::steady_clock::now();
    
    // Gradient spike handling removed (Rule 20: spikes indicate real bugs, not something to silently skip)
    
    // Telemetry control interventions removed (Rule 20: monitoring-only, crash on real bugs)
    
    // ========================================================================
    // Issue #135: Gradient clipping DEFERRED to after accumulation scaling
    //
    // OLD BUG: Clipping ran EVERY micro-batch, crushing text gradients 3x
    //   (once per micro-batch × 3 accum steps). With accum_steps=3, text
    //   gradients that were already tiny (~0.04) got clipped 3 times.
    //
    // FIX: Clipping now runs ONCE, inside should_step block, AFTER
    //   accum_scale(1/accum_steps). This means gradients are:
    //   1. Accumulated across all micro-batches (summed)
    //   2. Scaled by 1/accum_steps (averaged)
    //   3. Norm recomputed on the averaged gradients
    //   4. Clipped once against the limit
    //   5. Fed to optimizer
    // ========================================================================
    auto clipping_start = std::chrono::steady_clock::now();
    
    const float effective_per_token_limit = clip_selection.per_token_limit;
    const bool clipping_enabled = (hp.grad_clip_norm > 0.0f);
    
    // No clipping here — deferred to post-accumulation inside should_step
    auto clipping_elapsed_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - clipping_start).count();
    
    // Learning rate computation (accum_steps already computed above)
    // FIX: Use optimizer step count, NOT global_step (batch counter).
    // global_step increments every micro-batch; optimizer_state.step increments
    // only on actual weight updates. With gradient_accumulation_steps > 1,
    // using global_step makes warmup/decay advance accum_steps times too fast.
    auto lr_start = std::chrono::steady_clock::now();
    const int optimizer_step = static_cast<int>(ctx.optimizer.optimizer_state.step);
    if (!ctx.lr_schedule) {
        throw std::runtime_error("lr_schedule is not initialized at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    const float scheduled_lr = Internal::getScheduledLearningRate(
        *ctx.lr_schedule, optimizer_step, hp.learning_rate,
        ctx.config.stability.enabled);
    
    result.learning_rate = scheduled_lr;
    
    auto lr_elapsed_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - lr_start).count();
    
    // Optimizer step
    const bool sync_diag = shouldSyncDiagnostics(ctx, batch_idx);
    auto optimizer_step_start = std::chrono::steady_clock::now();
    float sample_elapsed_ms = 0.0f;
    GRIM::Diagnostics::WeightSample pre_sample{};
    std::string pre_weights = "lm_head_weights=skipped";
    if (sync_diag) {
        auto sample_start = std::chrono::steady_clock::now();
        pre_sample = GRIM::Diagnostics::sampleWeightStats(ctx.model->getLmHeadLayer(), ctx.model->getTrainingState(), true);
        if (pre_sample.valid) {
            pre_weights = GRIM::Diagnostics::formatWeightSample(pre_sample);
        }
        sample_elapsed_ms = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - sample_start).count();
    }
    
    // Log total time for pre-optimizer setup (spike handling, telemetry, clipping, LR)
    auto pre_optimizer_elapsed_ms = std::chrono::duration<float, std::milli>(
        optimizer_step_start - pre_optimizer_start).count();
    if (pre_optimizer_elapsed_ms > 1000.0f) {  // Log if > 1 second
        ctx.logging.logger->log("[PERF] Pre-optimizer setup took " + Internal::formatScalar(pre_optimizer_elapsed_ms, 2) + 
                                "ms (clipping=" + Internal::formatScalar(clipping_elapsed_ms, 2) + 
                                "ms, lr=" + Internal::formatScalar(lr_elapsed_ms, 2) + 
                                "ms, sample=" + Internal::formatScalar(sample_elapsed_ms, 2) + "ms)");
    }
    
    ctx.logging.logger->log("[GradTrace] PRE-OPTIMIZER batch=" + std::to_string(batch_idx + 1) +
                            " lr=" + Internal::formatScalar(result.learning_rate, 8) +
                            " grad_rms=" + Internal::formatScalar(result.grad_rms) +
                            " step=" + std::to_string(ctx.optimizer.optimizer_state.step) +
                            " " + pre_weights);
    
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
        
        // CRITICAL FIX: Synchronize stream BEFORE endOptimizerStep() zeros gradient buffers!
        // GradStats::flushAndLog() launches async D2H copies that read from gradient buffers.
        // If endOptimizerStep() zeros those buffers before the copies complete, GradStats
        // will read zeros instead of actual gradient values, causing corrupted stats.
        // This was the root cause of: max_abs=-1.53835e+13 (negative!), min/max inverted,
        // and unexplained zeros in fields that should contain computed values.
        cudaError_t sync_err = cudaStreamSynchronize(training_state.stream_ctrl.getPrimaryStream());
        if (sync_err != cudaSuccess) {
            std::ostringstream oss;
            oss << "[GradStats] FATAL: Failed to synchronize stream before optimizer step: "
                << cudaGetErrorString(sync_err);
            EmitModuleError(ModuleId::BackwardPass, oss.str(), ctx.global_step);
            throw std::runtime_error(oss.str());
        }
    }
    
    // Only run optimizer step when accumulation is complete
    // This enables gradient accumulation across multiple micro-batches
    // Increment micro_step BEFORE checking, since we already processed this batch's backward
    ctx.optimizer.current_micro_step++;
    const bool should_step = (ctx.optimizer.current_micro_step >= accum_steps);
    
    if (should_step) {
        const int micro_step_for_log = ctx.optimizer.current_micro_step;
        const int accum_steps_for_log = accum_steps;
        
        // ========================================================================
        // Issue #149: Zero PAD/UNK gradients before optimizer step
        //
        // PAD (id=1) and UNK (id=0) are never valid targets, yet they accumulate
        // non-zero gradients through two paths:
        //   1. LM head backward: label smoothing redistributes tiny gradient to ALL
        //      vocab rows, including PAD/UNK (via softmax Jacobian)
        //   2. Embedding backward: attention backward leaks gradient from valid
        //      positions to PAD input positions through K/V cross-attention,
        //      then scatter-adds to PAD's embedding row
        // Path 2 is dominant (~76x larger than BOS/EOS) because attention has
        // cross-position interactions even for masked-target positions.
        //
        // Zeroing these BEFORE norm computation ensures PAD/UNK don't inflate
        // gradient norms or waste optimizer capacity.
        // ========================================================================
        {
            auto& zero_ts = ctx.model->getTrainingState();
            cudaStream_t zero_stream = zero_ts.stream_ctrl.getPrimaryStream();
            const auto& zero_cfg = ctx.model->getConfig();
            const size_t row_bytes = static_cast<size_t>(zero_cfg.d_model) * sizeof(float);
            
            constexpr int NON_TRAINABLE_TOKENS[] = {
                GRIM::Tokenizer::UNK_TOKEN_ID,  // 0
                GRIM::Tokenizer::PAD_TOKEN_ID   // 1
            };
            
            // Zero embedding gradients for non-trainable tokens
            float* emb_grads = ctx.model->getEmbeddingLayer()->tokenWeights().grad_data();
            if (emb_grads) {
                for (int tok : NON_TRAINABLE_TOKENS) {
                    cudaMemsetAsync(
                        emb_grads + static_cast<size_t>(tok) * zero_cfg.d_model,
                        0, row_bytes, zero_stream);
                }
            }
            
            // Zero LM head gradients for non-trainable tokens
            float* lm_grads = ctx.model->getLmHeadLayer()->weights().grad_data();
            if (lm_grads) {
                for (int tok : NON_TRAINABLE_TOKENS) {
                    cudaMemsetAsync(
                        lm_grads + static_cast<size_t>(tok) * zero_cfg.d_model,
                        0, row_bytes, zero_stream);
                }
            }
        }
        
        // ========================================================================
        // Issue #135: POST-ACCUMULATION gradient clipping
        //
        // Clipping runs ONCE on the fully accumulated + scaled gradients.
        // Recompute grad norm since accum_scale changed magnitudes.
        // Per-component clipping (Issue #139):
        //   1. emb_clip  — LM_HEAD (+ EMBEDDING if untied)
        //   2. enc_clip  — ATTENTION + FFN + RMSNORM + SCRATCHBLOCK +
        //                  MTP + REASONING_HEAD + EXECUTION_BLOCK
        //
        // Clipping operates through the ParameterGroup tensor registry
        // via GradClip::clipGradientNorms() — norm measurement, bucket
        // aggregation, and gradient scaling all happen inside GradientCC
        // against the registered tensors.
        // ========================================================================
        if (clipping_enabled) {
            clipping_start = std::chrono::steady_clock::now();
            
            auto& clip_ts = ctx.model->getTrainingState();
            cudaStream_t clip_stream = clip_ts.stream_ctrl.getPrimaryStream();
            auto& clip_groups = ctx.model->parameterGroups();
            
            if (!clip_ts.grad_norm_scratch) {
                throw std::runtime_error("[FATAL] grad_norm_scratch is NULL at clipping stage - "
                                         "diagnostic norm should have allocated it");
            }
            
            GRIM::GradClip::ClipConfig clip_cfg;
            clip_cfg.max_rms = effective_per_token_limit;
            clip_cfg.tie_embeddings = ctx.model->getConfig().tie_embeddings;
            
            const auto clip = GRIM::GradClip::clipGradientNorms(
                clip_groups.data(), clip_groups.size(),
                clip_ts.grad_norm_scratch, clip_cfg, clip_stream);
            
            result.grad_rms = clip.total_rms_post;
            result.normalized_grad_rms = clip.total_rms_post;
            result.gradient_clipped = clip.any_clipped();
            
            ctx.logging.logger->log("[PostAccumClip] batch=" + std::to_string(batch_idx + 1) +
                                    " post_accum_rms=" + Internal::formatScalar(clip.total_rms_pre, 6) +
                                    " emb_rms=" + Internal::formatScalar(clip.emb_rms, 6) +
                                    " enc_rms=" + Internal::formatScalar(clip.enc_rms, 6) +
                                    " emb_clipped=" + (clip.emb_clipped ? "YES" : "NO") +
                                    " enc_clipped=" + (clip.enc_clipped ? "YES" : "NO") +
                                    " post_clip_total=" + Internal::formatScalar(clip.total_rms_post, 6));
            
            clipping_elapsed_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - clipping_start).count();
        }
        
        // ════════════════════════════════════════════════════════════════════
        // RUNTIME tie_embeddings pointer verification (every batch)
        // Startup logging only proves state at init. This proves state at
        // the moment the optimizer actually consumes the buffers.
        // ════════════════════════════════════════════════════════════════════
        {
            const float* emb_w = ctx.model->getEmbeddingLayer()->tokenWeights().data;
            const float* emb_g = ctx.model->getEmbeddingLayer()->tokenWeights().grad_data();
            const float* lm_w  = ctx.model->getLmHeadLayer()->weights().data;
            const float* lm_g  = ctx.model->getLmHeadLayer()->weights().grad_data();
            const bool cfg_tied = ctx.model->getConfig().tie_embeddings;
            const bool w_same = (emb_w == lm_w);
            const bool g_same = (emb_g == lm_g);

            // Count parameter groups referencing each buffer
            int emb_w_groups = 0, lm_w_groups = 0;
            for (const auto& pg : ctx.model->parameterGroups()) {
                if (pg.tensor && pg.tensor->data == emb_w) ++emb_w_groups;
                if (pg.tensor && pg.tensor->data == lm_w)  ++lm_w_groups;
            }

            // Log every 10 batches to avoid spam, but ALWAYS log if inconsistent
            const bool inconsistent = (cfg_tied != w_same) || (cfg_tied != g_same);
            if (inconsistent || (batch_idx % 10 == 0)) {
                std::ostringstream oss;
                oss << "[TIE_VERIFY] B=" << (batch_idx + 1)
                    << " step=" << ctx.optimizer.optimizer_state.step
                    << " cfg_tied=" << (cfg_tied ? "yes" : "no")
                    << " w_ptrs=" << (w_same ? "SAME" : "DIFF")
                    << " g_ptrs=" << (g_same ? "SAME" : "DIFF")
                    << " emb_w=" << (const void*)emb_w
                    << " lm_w=" << (const void*)lm_w
                    << " emb_g=" << (const void*)emb_g
                    << " lm_g=" << (const void*)lm_g
                    << " emb_w_groups=" << emb_w_groups
                    << " lm_w_groups=" << lm_w_groups;
                if (inconsistent) {
                    oss << " [ANOMALY] POINTER ALIASING MISMATCH — cfg says "
                        << (cfg_tied ? "tied" : "untied")
                        << " but weights " << (w_same ? "match" : "DIFFER")
                        << " and grads " << (g_same ? "match" : "DIFFER");
                }
                ctx.logging.logger->log(oss.str());
            }

            // Rule 20: crash on mismatch — this is an architectural bug
            if (cfg_tied && !w_same) {
                throw std::runtime_error("[TIE_VERIFY] FATAL: tie_embeddings=true but weight pointers differ at batch "
                    + std::to_string(batch_idx + 1) + " emb=" + std::to_string(reinterpret_cast<uintptr_t>(emb_w))
                    + " lm=" + std::to_string(reinterpret_cast<uintptr_t>(lm_w)));
            }
            if (cfg_tied && !g_same) {
                throw std::runtime_error("[TIE_VERIFY] FATAL: tie_embeddings=true but grad pointers differ at batch "
                    + std::to_string(batch_idx + 1) + " emb_g=" + std::to_string(reinterpret_cast<uintptr_t>(emb_g))
                    + " lm_g=" + std::to_string(reinterpret_cast<uintptr_t>(lm_g)));
            }
        }

        const int emb_freeze_step = ctx.config.hyperparameters.embedding_freeze_enabled
            ? ctx.config.hyperparameters.embedding_freeze_after_step : -1;

        if (emb_freeze_step > 0 && ctx.optimizer.optimizer_state.step == emb_freeze_step) {
            if (ctx.config.architecture.tie_embeddings) {
                ctx.logging.logger->log("[EmbeddingFreeze] WARNING: tie_embeddings=true — "
                    "embedding and LM head share weights. Set tie_embeddings=false to freeze "
                    "embedding independently. Freeze has no effect on tied weights.");
            } else {
                ctx.logging.logger->log("[EmbeddingFreeze] Embedding weights FROZEN at step "
                    + std::to_string(emb_freeze_step) + " — no further embedding updates");
            }
        }

        // Optimizer dispatch — single source of truth: ctx.config.hyperparameters
        // (loaded from training.config.optimizer in ai_config.json). Kernel hyperparams
        // are passed by signature so kernels read no globals. Rule 20: kind already
        // validated at config-load time ("adamw" | "radam" only).
        const auto& opt_hp = ctx.config.hyperparameters;
        if (opt_hp.optimizer_kind == "radam") {
            GRIM::launchRAdamStep(ctx.model->parameterGroups(),
                                  result.learning_rate,
                                  opt_hp.weight_decay,
                                  ctx.optimizer.optimizer_state.step,
                                  opt_hp.optimizer_beta1,
                                  opt_hp.optimizer_beta2,
                                  opt_hp.optimizer_epsilon,
                                  opt_hp.radam_compute_b2_halflife,
                                  ctx.model->getTrainingState().stream_ctrl.getPrimaryStream(),
                                  emb_freeze_step);
        } else {
            GRIM::launchAdamWStep(ctx.model->parameterGroups(),
                                  result.learning_rate,
                                  opt_hp.weight_decay,
                                  ctx.optimizer.optimizer_state.step,
                                  ctx.model->getTrainingState().stream_ctrl.getPrimaryStream(),
                                  emb_freeze_step);
        }
        
        // RULE 20: Post-optimizer weight NaN spot check
        // Catches the EXACT batch where optimizer corrupts weights (instead of crashing
        // on the NEXT batch's forward pass with an unhelpful "logits NaN" message).
        {
            cudaStreamSynchronize(ctx.model->getTrainingState().stream_ctrl.getPrimaryStream());
            const auto& groups = ctx.model->parameterGroups();
            for (size_t g = 0; g < groups.size(); ++g) {
                if (!groups[g].weights() || groups[g].size() == 0) continue;
                // Sample first element of each parameter group (fast: 1 float per group)
                float h_sample = 0.0f;
                cudaMemcpy(&h_sample, groups[g].weights(), sizeof(float), cudaMemcpyDeviceToHost);
                if (!std::isfinite(h_sample)) {
                    throw std::runtime_error("[FATAL] Post-optimizer NaN/Inf in parameter group '" +
                        groups[g].name + "' (group " + std::to_string(g) + ") at batch " +
                        std::to_string(batch_idx + 1) + " optimizer_step=" +
                        std::to_string(ctx.optimizer.optimizer_state.step) +
                        " lr=" + std::to_string(result.learning_rate) +
                        " — THIS batch's optimizer step corrupted weights. "
                        "Check gradient magnitude and clipping for this group.");
                }
            }
        }

        // Reset micro_step counter after optimizer step completes
        ctx.optimizer.current_micro_step = 0;

        // Periodic tape flush (every 10 optimizer steps) — sinks handle I/O
        // (tape.flush() already called at end of processBatch; this is a mid-batch safety flush)

        const auto moment_sample = sampleOptimizerMomentStats(ctx.model->getTrainingState(), sync_diag);
        if (moment_sample.valid) {
            ctx.logging.logger->log("[OptState] batch=" + std::to_string(batch_idx + 1) +
                                    " step=" + std::to_string(ctx.optimizer.optimizer_state.step) +
                                    " micro_step=" + std::to_string(micro_step_for_log) +
                                    "/" + std::to_string(accum_steps_for_log) +
                                    " m_rms=" + Internal::formatScalar(moment_sample.m_rms, 10) +
                                    " v_rms=" + Internal::formatScalar(moment_sample.v_rms, 10) +
                                    " groups=" + std::to_string(moment_sample.groups) +
                                    " samples=" + std::to_string(moment_sample.samples));
        }
        
        GRIM::Diagnostics::WeightSample post_sample{};
        std::string post_weights = "lm_head_weights=skipped";
        if (sync_diag) {
            post_sample = GRIM::Diagnostics::sampleWeightStats(ctx.model->getLmHeadLayer(), ctx.model->getTrainingState(), true);
            if (post_sample.valid) {
                post_weights = GRIM::Diagnostics::formatWeightSample(post_sample);
            }
        }
        ctx.logging.logger->log("[GradTrace] POST-OPTIMIZER batch=" + std::to_string(batch_idx + 1) + 
                                " lr=" + Internal::formatScalar(result.learning_rate, 8) +
                                " step=" + std::to_string(ctx.optimizer.optimizer_state.step) +
                                " t=" + std::to_string(ctx.optimizer.optimizer_state.step) +
                                " " + post_weights);
        
        if (pre_sample.valid && post_sample.valid) {
            const float update_rms = GRIM::Diagnostics::computeUpdateRms(pre_sample, post_sample);
            const std::string update_msg = "[UpdateMag] batch=" + std::to_string(batch_idx + 1) +
                                           " update_rms=" + Internal::formatScalar(update_rms, 8) +
                                           " param_rms=" + Internal::formatScalar(pre_sample.rms, 8);
            ctx.logging.logger->log(update_msg);
            EmitModuleInfo(ModuleId::Optimizer, update_msg, ctx.global_step);
        }
        
        // Per-component Adam update_rms diagnostic (Issue #150)
        // Answers: "Does Adam normalize the gradient gap across component types?"
        // Only on diagnostic-sync batches to avoid blocking the pipeline.
        if (sync_diag) {
            const auto update_trace = GRIM::Diagnostics::computePerComponentUpdateTrace(
                ctx.model->parameterGroups(),
                result.learning_rate,
                ctx.optimizer.optimizer_state.step + 1,  // 1-based iteration count (matches AdamW bias correction)
                ctx.model->getTrainingState().stream_ctrl.getPrimaryStream()
            );
            if (update_trace.valid) {
                const std::string trace_str = GRIM::Diagnostics::formatUpdateTrace(
                    update_trace, batch_idx + 1, ctx.model->getConfig().tie_embeddings);
                ctx.logging.logger->log(trace_str);
            }
        }
        
        ctx.optimizer.optimizer_state.step++;
    } else {
        ctx.logging.logger->log("[GradTrace] ACCUMULATING batch=" + std::to_string(batch_idx + 1) + 
                                " micro_step=" + std::to_string(ctx.optimizer.current_micro_step) +
                                " of " + std::to_string(accum_steps) +
                                " (skipping optimizer step)");
    }
    
    // Check for CUDA errors (non-blocking)
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            ctx.logging.logger->log("[CUDA ERROR] Async check: " + std::string(cudaGetErrorString(err)));
        }
    }
    
    result.sequences_processed = payload.batch_size;
    result.tokens_processed = clip_selection.stats.total_tokens;
    
    // Flush device logs on the diagnostic sync interval.
    if (sync_diag) {
        GRIM::Logging::FlushDeviceLogs();
    }
    
    // Record layer log for post-run analysis (matches old train_gpu.cu behavior)
    GRIM::Logging::RecordLayerLogHost(
        GRIM::LayerType::kEncoding,         // aggregate marker
        -1,                                 // no specific layer (-1 = global)
        static_cast<std::uint64_t>(ctx.global_step),
        result.grad_rms,                   // primary: gradient RMS
        result.normalized_grad_rms,        // secondary: per-token RMS
        "grad_rms",
        "post_backward");
    
    // Update telemetry lattice — all metric computation delegated to TelemetryUpdate.cu
    {
        GRIM::Telemetry::TelemetryBatchInput tel_input;
        tel_input.loss              = result.loss;
        tel_input.preclip_grad_rms  = preclip_grad_rms;
        tel_input.learning_rate     = result.learning_rate;
        tel_input.total_tokens      = payload.token_stats.total_tokens;
        tel_input.enc_rms_pre       = enc_rms_pre;
        tel_input.optimizer_step    = static_cast<int>(ctx.optimizer.optimizer_state.step);
        tel_input.should_step       = should_step;
        tel_input.total_loss_value  = loss_result.loss_value;
        tel_input.aux_loss          = loss_result.aux_loss;
        tel_input.max_seq_len       = payload.max_seq_len;
        tel_input.batch_idx         = batch_idx;
        tel_input.global_step       = ctx.global_step;
        tel_input.actual_vocab_size = ctx.config.actual_vocab_size;
        tel_input.d_model           = ctx.model->getConfig().d_model;

        GRIM::Telemetry::updateTelemetryObservations(ctx, tel_input, gm, &payload);
    }
    
    // First-batch CUDA checkpoint (runs even if telemetry disabled): last point before step++
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
    
    ctx.global_step++;
    state.last_grad_rms = result.grad_rms;

    // UpdateProbe consumer deleted (Rule 26): hasUpdateProbe() never returned
    // true because update_probe_ready_ was never set anywhere. Entire probe
    // subsystem removed; PostLoss cached_logits trace at line ~1362 remains.

    // Rule 20 single-owner clear: handled by AutogradStepScope at processBatch entry.
    // Tape flush below does not read autograd intermediates.
    
    // Flush tape: sort by phase, dispatch to all sinks
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
    int epoch_idx) {
    
    EpochResult result;
    result.epoch = epoch_idx;
    
    const auto& hp = ctx.config.hyperparameters;
    const int num_epochs = hp.epochs;
    
    ctx.logging.logger->log("Epoch " + std::to_string(epoch_idx + 1) + "/" + std::to_string(num_epochs));
    
    // Reset guess cache at epoch start
    if (ctx.config.hyperparameters.guess_aux_enabled) {
        GRIMTS::Training::resetGuessCacheForEpoch(
            ctx.model->getTrainingState(), state.guess_cache);
    }
    
    PHASE2_DEBUG_STDERR("[DEBUG-EPOCH] After ResetGuessCache, checking shuffle...\n");
    
    // Shuffle train catalog if enabled
    const bool shuffle_this_epoch = hp.shuffle_train_enabled &&
        (hp.shuffle_train_epochs == 0 || epoch_idx < hp.shuffle_train_epochs);
    PHASE2_DEBUG_STDERR("[DEBUG-EPOCH] shuffle_this_epoch=%d hp.shuffle_train_enabled=%d\n",
                        shuffle_this_epoch ? 1 : 0, hp.shuffle_train_enabled ? 1 : 0);
    
    if (shuffle_this_epoch) {
        PHASE2_DEBUG_STDERR("[DEBUG-EPOCH] Entering shuffle block\n");
        ctx.logging.logger->log("[Shuffle] Randomizing train catalog for epoch " + std::to_string(epoch_idx + 1));
        PHASE2_DEBUG_STDERR("[DEBUG-EPOCH] train_views.size()=%zu\n", ctx.data.train_views.size());
        
        std::vector<size_t> perm(ctx.data.train_views.size());
        for (size_t i = 0; i < perm.size(); ++i) perm[i] = i;
        std::shuffle(perm.begin(), perm.end(), ctx.rng.data_rng);
        
        std::vector<TrainingSequence*> shuffled;
        shuffled.reserve(ctx.data.train_views.size());
        GRIM::DynaSeq::Catalog shuffled_catalog;
        
        for (size_t new_idx = 0; new_idx < perm.size(); ++new_idx) {
            auto* seq = ctx.data.train_views[perm[new_idx]];
            shuffled.push_back(seq);
            const uint32_t len = static_cast<uint32_t>(seq->token_ids.size());
            shuffled_catalog.add(len, len, 0, 0, 0);
        }
        
        ctx.data.train_views.swap(shuffled);
        ctx.data.train_catalog = std::move(shuffled_catalog);
        PHASE2_DEBUG_STDERR("[DEBUG-EPOCH] Shuffle complete\n");
    }
    
    PHASE2_DEBUG_STDERR("[DEBUG-EPOCH] About to build epoch batches...\n");
    // Build batches for this epoch
    auto schedule = Internal::buildEpochBatches(
        ctx,
        ctx.data.train_catalog,
        hp.batch_size,
        ctx.global_step,
        epoch_idx,
        *ctx.logging.logger);
    
    const int total_batches = static_cast<int>(schedule.batches.size());
    if (ctx.estimated_total_steps == 0 && total_batches > 0) {
        // estimated_total_steps counts OPTIMIZER STEPS (not micro-batches).
        // LR schedule, warmup, and cosine decay all index by optimizer step.
        const int accum = std::max(1, hp.gradient_accumulation_steps);
        ctx.estimated_total_steps = (num_epochs * total_batches) / accum;
        
        // Derive warmup_steps from warmup_fraction now that total_steps is known.
        GRIM::Config::deriveWarmupSteps(ctx.config.hyperparameters, ctx.estimated_total_steps);
        
        // Construct the deterministic LR schedule now that total_steps is known.
        // Cosine decay spans all epochs; warm restarts are disabled.
        GRIM::LR::LRScheduleConfig lr_cfg;
        lr_cfg.base_lr = hp.learning_rate;
        lr_cfg.cosine_decay_min_lr = hp.cosine_decay_min_lr;
        lr_cfg.warmup_steps = ctx.config.hyperparameters.warmup_steps;
        lr_cfg.total_steps = ctx.estimated_total_steps;
        lr_cfg.steps_per_epoch = total_batches / accum;
        lr_cfg.cosine_decay_enabled = hp.cosine_decay_enabled;
        lr_cfg.warm_restarts = hp.cosine_warm_restarts;
        ctx.lr_schedule.emplace(lr_cfg);
        
        ctx.logging.logger->log("[LRSchedule] warmup_fraction=" + std::to_string(hp.warmup_fraction)
            + " -> warmup_steps=" + std::to_string(ctx.config.hyperparameters.warmup_steps)
            + " / total_steps=" + std::to_string(ctx.estimated_total_steps));
    }
    int total_batches_to_run = total_batches;
    const bool single_batch_overfit = hp.single_batch_overfit_enabled;
    if (single_batch_overfit) {
        if (total_batches == 0) {
            ctx.logging.logger->log("[SingleBatch] WARNING: No batches available to repeat.");
            total_batches_to_run = 0;
        } else {
            total_batches_to_run = std::max(1, hp.single_batch_overfit_max_steps);
            ctx.logging.logger->log("[SingleBatch] Enabled: repeating batch 1 for " +
                                    std::to_string(total_batches_to_run) + " steps.");
        }
    }
    const auto epoch_start = std::chrono::steady_clock::now();
    float epoch_loss = 0.0f;
    
    // Get accumulation steps from hyperparameters
    const int accum_steps = std::max(1, ctx.config.hyperparameters.gradient_accumulation_steps);
    
    // Process batches: scheduler (BatchAssignment) dictates order; we build BatchPayload and act on it
    const auto& model_cfg = ctx.model->getConfig();
    const auto token_layout = ctx.tokenizer.tokenLayout();
    const size_t max_cached_batch = static_cast<size_t>(std::max(1, model_cfg.max_cached_batch));
    const size_t max_cached_seq = static_cast<size_t>(std::max(1, std::min(model_cfg.max_seq_len, model_cfg.max_cached_seq_len)));

    for (int batch_idx = 0; batch_idx < total_batches_to_run; ++batch_idx) {
        const auto& assignment = schedule.batches[single_batch_overfit ? 0 : batch_idx];
        GRIM::Batching::BatchPayload payload = GRIM::Batching::buildBatchPayload(
            assignment, ctx.data.train_views, ctx.config.actual_vocab_size,
            token_layout, max_cached_batch, max_cached_seq,
            model_cfg.execution_block_num_slots,
            model_cfg.execution_block_num_ops,
            model_cfg.execution_block_num_steps,
            model_cfg.mtp_enabled ? model_cfg.mtp_k : 0);

        // Log progress periodically (from payload — single source of truth)
        if (batch_idx % 5 == 0) {
            std::ostringstream msg;
            msg << "[Batch " << (batch_idx + 1) << "/" << total_batches_to_run << "] "
                << "size=" << payload.seq_ids.size()
                << " len=" << payload.min_seq_len << "-" << payload.max_seq_len
                << " eff=" << static_cast<int>(payload.packing_efficiency * 100) << "%"
                << " accum_steps=" << accum_steps;
            ctx.logging.logger->log(msg.str());
        }

        BatchResult batch_result = processBatch(ctx, state, payload, batch_idx, epoch_idx);
        
        // After first batch: sync and surface any CUDA error so we see the real fault
        // (otherwise it only appears later as cudaFree failures during teardown)
        if (batch_idx == 0 && !batch_result.skipped) {
            cudaError_t sync_err = cudaDeviceSynchronize();
            cudaError_t last_err = (sync_err != cudaSuccess) ? sync_err : cudaGetLastError();
            if (last_err != cudaSuccess) {
                ctx.logging.logger->log("[CUDA] ERROR after first batch: " + std::string(cudaGetErrorString(last_err)) +
                    " (sync=" + (sync_err != cudaSuccess ? "failed" : "ok") + "). "
                    "Fix this to avoid cudaFree 'illegal memory access' during teardown.");
                cudaGetLastError(); // clear so subsequent code can run if desired
            }
        }
        
        if (batch_result.skipped) {
            result.batches_skipped++;
            ctx.logging.logger->log("[Batch " + std::to_string(batch_idx + 1) + 
                                    "] skipped (" + batch_result.skip_reason + ")");
            continue;
        }
        
        epoch_loss += batch_result.loss;
        result.batches_processed++;
        result.best_batch_loss = std::min(result.best_batch_loss, batch_result.loss);
        result.worst_batch_loss = std::max(result.worst_batch_loss, batch_result.loss);
        
        // Log at interval
        if (ctx.global_step % hp.log_interval == 0) {
            ctx.logging.logger->log("[Step " + std::to_string(ctx.global_step) + "] " +
                                    Internal::formatMetric("loss", batch_result.loss) + " " +
                                    Internal::formatMetric("lr", batch_result.learning_rate, 8));
            // MTP diagnostics: per-head loss, acc, loss_ratio, alpha_effective, L_total
            {
                auto& ts = ctx.model->getTrainingState();
                if (ts.mtp_diagnostics.valid && !ts.mtp_diagnostics.head_loss.empty()) {
                    const float L0 = ts.mtp_diagnostics.L0_main > 0.0f ? ts.mtp_diagnostics.L0_main : batch_result.loss;
                    std::ostringstream mtp_log;
                    for (size_t i = 0; i < ts.mtp_diagnostics.head_loss.size(); ++i) {
                        const float Lk = ts.mtp_diagnostics.head_loss[i];
                        const float acc = i < ts.mtp_diagnostics.head_acc.size() ? ts.mtp_diagnostics.head_acc[i] : 0.0f;
                        const float ratio = (L0 > 0.0f) ? (Lk / L0) : 0.0f;
                        mtp_log << "[MTP_EQUATION] head_k=" << (i + 1) << ": loss=" << Internal::formatScalar(Lk, 4)
                                << " acc=" << Internal::formatScalar(acc, 2) << "%"
                                << " loss_ratio=" << Internal::formatScalar(ratio, 4) << " ";
                    }
                    mtp_log << "alpha_effective=" << Internal::formatScalar(ts.mtp_diagnostics.alpha_effective, 4)
                            << " L_total=" << Internal::formatScalar(ts.mtp_diagnostics.L_total, 4);
                    ctx.logging.logger->log(mtp_log.str());
                    // MTP Monitor: Lk/L0 with healthy-range indication (configurable via log_ratio_monitor)
                    if (hp.mtp_log_ratio_monitor) {
                        static const float kHealthyLow[] = { 1.1f, 1.3f, 1.5f, 1.6f };
                        static const float kHealthyHigh[] = { 1.3f, 1.6f, 2.0f, 2.2f };
                        std::ostringstream mon;
                        mon << "[MTP_Monitor]";
                        for (size_t i = 0; i < ts.mtp_diagnostics.head_loss.size(); ++i) {
                            const float ratio = (L0 > 0.0f) ? (ts.mtp_diagnostics.head_loss[i] / L0) : 0.0f;
                            const int k = static_cast<int>(i) + 1;
                            const size_t idx = std::min(static_cast<size_t>(k - 1), static_cast<size_t>(4));
                            const float lo = kHealthyLow[idx];
                            const float hi = kHealthyHigh[idx];
                            const bool ok = (ratio >= lo && ratio <= hi);
                            mon << " k=" << k << ": Lk/L0=" << Internal::formatScalar(ratio, 3)
                                << " (healthy " << Internal::formatScalar(lo, 1) << "-" << Internal::formatScalar(hi, 1)
                                << (ok ? " OK" : " OUT_OF_RANGE") << ")";
                        }
                        ctx.logging.logger->log(mon.str());
                    }
                }
            }
            if (ctx.config.hyperparameters.guess_aux_enabled) {
                GRIMTS::Training::logGuessCacheTelemetry(state.guess_cache, ctx.global_step);
            }
        }

        logDiagnosticSample(ctx, state);
        
        // Status update
        float current_avg_loss = epoch_loss / result.batches_processed;
        float train_perplexity = (std::isfinite(current_avg_loss) && current_avg_loss < 50.0f)
            ? std::exp(current_avg_loss)
            : std::numeric_limits<float>::infinity();
        
        ctx.logging.status_writer->writeStatus(
            GRIMText::Control::TrainingState_Training,
            epoch_idx + 1, num_epochs,
            batch_idx + 1, total_batches_to_run,
            batch_result.loss, current_avg_loss,
            train_perplexity, 0.0f, 0.0f, 0.0f,
            "Training epoch " + std::to_string(epoch_idx + 1) + " batch " + std::to_string(batch_idx + 1));
        

    }
    
    result.avg_loss = epoch_loss / std::max(result.batches_processed, 1);
    result.duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - epoch_start);
    
    ctx.logging.logger->log("[Epoch " + std::to_string(epoch_idx + 1) + "] " +
                            Internal::formatMetric("avg_loss", result.avg_loss));
    
    // Log telemetry vectors (multi-scale summary)
    GRIM::Telemetry::logTelemetrySummary(ctx);
    
    // Run validation
    result.validation = runValidation(ctx);
    
    // Auto-stop checks
    if (hp.auto_stop_enabled && !ctx.auto_stop_triggered) {
        const float prev_best = ctx.best_val_loss;
        bool significant_improvement = true;
        if (std::isfinite(prev_best)) {
            significant_improvement = (prev_best - result.validation.loss) > hp.auto_stop_plateau_min_delta;
        }
        
        // Plateau detection
        if (hp.auto_stop_plateau_patience > 0) {
            if (significant_improvement) {
                state.plateau_epochs_without_improvement = 0;
            } else {
                state.plateau_epochs_without_improvement++;
                if (state.plateau_epochs_without_improvement >= hp.auto_stop_plateau_patience) {
                    ctx.auto_stop_triggered = true;
                    ctx.auto_stop_reason = "plateau";
                    ctx.auto_stop_epoch = epoch_idx + 1;
                    ctx.auto_stop_metric = result.validation.loss;
                    result.auto_stop_triggered = true;
                    result.auto_stop_reason = "plateau";
                }
            }
        }
        
        // High loss detection
        if (!ctx.auto_stop_triggered && hp.auto_stop_high_loss_patience > 0) {
            if (result.validation.loss >= hp.auto_stop_high_loss_threshold) {
                state.high_loss_epochs++;
            } else {
                state.high_loss_epochs = 0;
            }
            if (state.high_loss_epochs >= hp.auto_stop_high_loss_patience) {
                ctx.auto_stop_triggered = true;
                ctx.auto_stop_reason = "high_loss";
                ctx.auto_stop_epoch = epoch_idx + 1;
                ctx.auto_stop_metric = result.validation.loss;
                result.auto_stop_triggered = true;
                result.auto_stop_reason = "high_loss";
            }
        }
    }
    
    // Checkpoint
    if (result.validation.is_best) {
        Internal::maybeSaveCheckpoint(ctx, result.validation.loss, epoch_idx);
    }
    
    // Update status - GPU memory query at epoch end only (not per-batch)
    // cudaMemGetInfo is implicitly synchronizing on some drivers, but acceptable
    // at epoch granularity (once per ~100-1000 batches)
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    float gpu_used_mb = static_cast<float>((total_mem - free_mem)) / (1024.0f*1024.0f);
    float gpu_total_mb = static_cast<float>(total_mem) / (1024.0f*1024.0f);
    
    ctx.logging.status_writer->writeStatus(
        GRIMText::Control::TrainingState_Training,
        epoch_idx + 1, num_epochs,
        total_batches, total_batches,
        result.validation.loss, result.avg_loss,
        result.validation.perplexity, 0.0f,
        gpu_used_mb, gpu_total_mb,
        "Validation complete - epoch " + std::to_string(epoch_idx + 1));
    
    return result;
}

//======================================================//
//  Phase2 Main Entry Point
//======================================================//

bool executePhase2(TrainingContext& ctx) {
    const auto& hp = ctx.config.hyperparameters;
    
    // Initialize loop state
    TrainingLoopState state;
    
    // Initialize guess cache (Rule 22: pass TrainingState for buffer allocation)
    auto guess_cache_scope = GRIMTS::Training::initGuessCache(
        ctx.model->getTrainingState(),
        ctx.config.hyperparameters.guess_aux_enabled,
        ctx.config.cuda_exec.single_stream_mode,
        ctx.global_step,
        state.guess_cache);
    
    PHASE2_DEBUG_STDERR("[DEBUG] About to initialize training log...");

    // Log configuration
    EmitModuleInfo(ModuleId::Training, "Starting training...", ctx.global_step);
    EmitModuleInfo(ModuleId::Training, std::string("  Warmup fraction: ") + std::to_string(hp.warmup_fraction) + " (steps derived after batch count known)", ctx.global_step);
    EmitModuleInfo(ModuleId::Training, std::string("  Target learning rate: ") + std::to_string(hp.learning_rate), ctx.global_step);
    if (hp.cosine_decay_enabled) {
        EmitModuleInfo(ModuleId::Training,
            std::string("  Cosine decay: enabled, min_lr=") + std::to_string(hp.cosine_decay_min_lr), ctx.global_step);
    }
    EmitModuleInfo(ModuleId::Training, 
        std::string("  Soft restart: ") + (hp.soft_restart_enabled ? "enabled" : "disabled"), ctx.global_step);
    EmitModuleInfo(ModuleId::Training, 
        std::string("  Auto-stop: ") + (hp.auto_stop_enabled ? "enabled" : "disabled"), ctx.global_step);
    if (hp.embedding_freeze_enabled) {
        EmitModuleInfo(ModuleId::Training,
            std::string("  Embedding freeze: after step ") + std::to_string(hp.embedding_freeze_after_step)
            + (ctx.config.architecture.tie_embeddings ? " (WARNING: tie_embeddings=true, set to false for freeze to take effect)" : ""),
            ctx.global_step);
    }
    
    const int num_epochs = std::max(1, hp.epochs);
    EmitModuleInfo(ModuleId::Training,
        std::string("Total epochs to run: ") + std::to_string(num_epochs), ctx.global_step);
    
    // Reconstruct adam_cumulative_disp = Σlr(0..optimizer_step-1) for checkpoint resume correctness.
    // On fresh start this is a no-op. On resume, it reconstructs the exact cumulative
    // displacement by iterating over actual optimizer steps (not micro-batches).
    // FIX: Used to loop over ctx.global_step (batch counter), overcounting by
    // gradient_accumulation_steps. Now loops over optimizer_state.step.
    const int resumed_optimizer_step = static_cast<int>(ctx.optimizer.optimizer_state.step);
    if (resumed_optimizer_step > 0 && ctx.telemetry.adam_cumulative_disp == 0.0f) {
        float reconstructed = 0.0f;
        for (int t = 0; t < resumed_optimizer_step; ++t) {
            reconstructed += ctx.lr_schedule->lr(t);
        }
        ctx.telemetry.adam_cumulative_disp = reconstructed;
        EmitModuleInfo(ModuleId::Training,
            "[AdamCausation] Reconstructed cumulative_displacement=" + std::to_string(reconstructed) +
            " from " + std::to_string(resumed_optimizer_step) + " optimizer steps", ctx.global_step);
    }
    
    try {
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            EpochResult epoch_result = runEpoch(ctx, state, epoch);
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
        
        ctx.logging.status_writer->writeStatus(
            GRIMText::Control::TrainingState_Error,
            0, num_epochs, 0, 0,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            "Training error", std::string(e.what()));
        
        throw;
    }
    return true;
}

} // namespace GRIMText::Training
