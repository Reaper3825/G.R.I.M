#include "ForwardOps_Orchestrator.hpp"
#include "../../Shared/LogRecorder/LogRecorder.hpp"

#include <chrono>
#include <sstream>
#include <iomanip>

namespace GRIM {
namespace Forward {

namespace {
using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;
} // namespace

ForwardStatus executeForward(ForwardContext& ctx) {
    const bool log_info = GRIM::ForwardOps::ShouldLogForwardInfo();
    auto start_time = log_info ? Clock::now() : Clock::time_point{};

    FWD_INFO("[ForwardOrch] ====== FORWARD PASS START ======");
    FWD_INFO("[ForwardOrch] mode=" << modeToString(ctx.mode)
                                  << " batch=" << ctx.batch_size
                                  << " seq=" << ctx.seq_len);

    ForwardStatus validation = ctx.validate();
    if (validation != ForwardStatus::SUCCESS) {
        ctx.error_message = "Forward context validation failed";
        FWD_ERROR("[ForwardOrch] Context validation failed: " << statusToString(validation));
        return validation;
    }

    auto phase3_start = log_info ? Clock::now() : Clock::time_point{};
    ForwardStatus phase3_status = executePhase3_InputLayer(ctx);
    auto phase3_end = log_info ? Clock::now() : Clock::time_point{};
    Duration phase3_time = log_info ? (phase3_end - phase3_start) : Duration::zero();
    ctx.phase3_status = phase3_status;
    if (phase3_status != ForwardStatus::SUCCESS) {
        FWD_ERROR("[ForwardOrch] Phase 3 FAILED: " << statusToString(phase3_status));
        FWD_ERROR("[ForwardOrch] Error: " << ctx.error_message);
        return phase3_status;
    }

    auto phase2_start = log_info ? Clock::now() : Clock::time_point{};
    ForwardStatus phase2_status = executePhase2_Encoder(ctx);
    auto phase2_end = log_info ? Clock::now() : Clock::time_point{};
    Duration phase2_time = log_info ? (phase2_end - phase2_start) : Duration::zero();
    ctx.phase2_status = phase2_status;
    if (phase2_status != ForwardStatus::SUCCESS) {
        FWD_ERROR("[ForwardOrch] Phase 2 FAILED: " << statusToString(phase2_status));
        FWD_ERROR("[ForwardOrch] Error: " << ctx.error_message);
        if (ctx.error_layer >= 0) {
            FWD_ERROR("[ForwardOrch] Failed at layer: " << ctx.error_layer);
        }
        return phase2_status;
    }

    auto phase1_start = log_info ? Clock::now() : Clock::time_point{};
    ForwardStatus phase1_status = executePhase1_OutputLayer(ctx);
    auto phase1_end = log_info ? Clock::now() : Clock::time_point{};
    Duration phase1_time = log_info ? (phase1_end - phase1_start) : Duration::zero();
    ctx.phase1_status = phase1_status;
    if (phase1_status != ForwardStatus::SUCCESS) {
        FWD_ERROR("[ForwardOrch] Phase 1 FAILED: " << statusToString(phase1_status));
        FWD_ERROR("[ForwardOrch] Error: " << ctx.error_message);
        return phase1_status;
    }

    auto end_time = log_info ? Clock::now() : Clock::time_point{};
    Duration total_time = log_info ? (end_time - start_time) : Duration::zero();

    FWD_INFO("[ForwardOrch] ====== FORWARD PASS COMPLETE ======");
    FWD_INFO("[ForwardOrch] Total: " << std::fixed << std::setprecision(2) << total_time.count() << "ms");
    FWD_INFO("[ForwardOrch]  Phase 3 (Input):  " << std::setw(8) << phase3_time.count() << "ms");
    FWD_INFO("[ForwardOrch]  Phase 2 (Encoder):" << std::setw(8) << phase2_time.count() << "ms");
    FWD_INFO("[ForwardOrch]  Phase 1 (Output): " << std::setw(8) << phase1_time.count() << "ms");

    return ForwardStatus::SUCCESS;
}

ForwardContext initForwardContext(
    LanguageModel& model,
    ForwardMode mode,
    int batch_size,
    int seq_len,
    ForwardLogitsTarget logits_target,
    const int* host_tokens,
    bool tokens_on_device,
    int new_token,
    int query_pos,
    bool enable_scratch_block,
    bool enable_activation_quantization,
    bool enable_entropy_output) {
    ForwardContext ctx{};

    ctx.config = &model.getConfig();
    ctx.training_state = &model.getTrainingState();
    ctx.model = &model;

    ctx.gpu_encoder = &model.getGpuEncoder();
    ctx.embedding_runtime = &model.getGpuEmbedder();
    ctx.scratch_block = model.getScratchBlockLayer();
    // GUARD: Only assign atom/numeric buffers if ScratchBlock is enabled (buffers may be nullptr)
    ctx.token_numeric_values = ctx.training_state->cached_token_numeric_values;
    ctx.token_numeric_mask = ctx.training_state->cached_token_numeric_mask;
    // GRMT v4: wire text features for ScratchBlock (only if allocated)
    ctx.token_text_features = ctx.training_state->cached_token_text_features;
    ctx.token_text_mask = ctx.training_state->cached_token_text_mask;
    ctx.alibi = model.getAlibiPtr();

    ctx.cublas_handle = ctx.training_state->cublas_handle;
    ctx.stream = ctx.training_state->stream_ctrl.getPrimaryStream();

    ctx.mode = mode;
    ctx.logits_target = logits_target;
    ctx.host_tokens = host_tokens;
    ctx.tokens_on_device = tokens_on_device;
    ctx.batch_size = batch_size;
    ctx.seq_len = seq_len;
    ctx.total_tokens = batch_size * seq_len;
    ctx.new_token = new_token;
    ctx.query_pos = query_pos;

    ctx.enable_scratch_block = enable_scratch_block;
    ctx.enable_activation_quantization = enable_activation_quantization;
    ctx.enable_entropy_output = enable_entropy_output;

    ctx.layer_caches = ctx.training_state->forward_layer_caches;
    ctx.layer_cache_count = ctx.training_state->forward_layer_cache_count;

    if (mode == ForwardMode::DecodeIncremental) {
        ctx.encoder_output = ctx.training_state->single_token_hidden;
        ctx.logits_output = ctx.training_state->single_token_logits;
    } else {
        ctx.encoder_output = ctx.training_state->cached_encoder_outputs;
        ctx.logits_output = (logits_target == ForwardLogitsTarget::FullSequence)
                                ? ctx.training_state->cached_logits
                                : ctx.training_state->single_token_logits;
    }

    return ctx;
}

std::string getForwardErrorReport(const ForwardContext& ctx) {
    std::ostringstream oss;
    oss << "========== FORWARD PASS ERROR REPORT ==========\n";
    oss << "Mode: " << modeToString(ctx.mode) << "\n";
    oss << "Batch: " << ctx.batch_size << " seq: " << ctx.seq_len << "\n";
    oss << "Phase Status:\n";
    oss << "  Phase 3 (Input):  " << statusToString(ctx.phase3_status) << "\n";
    oss << "  Phase 2 (Encoder): " << statusToString(ctx.phase2_status) << "\n";
    oss << "  Phase 1 (Output): " << statusToString(ctx.phase1_status) << "\n";
    if (!ctx.error_message.empty()) {
        oss << "Error: " << ctx.error_message << "\n";
    }
    if (ctx.error_layer >= 0) {
        oss << "Layer: " << ctx.error_layer << "\n";
    }
    oss << "===============================================\n";
    return oss.str();
}

void logForwardSummary(const ForwardContext& ctx) {
    FWD_INFO("[ForwardOrch] ====== FORWARD PASS SUMMARY ======");
    FWD_INFO("[ForwardOrch] Mode: " << modeToString(ctx.mode));
    FWD_INFO("[ForwardOrch] Batch: " << ctx.batch_size << " seq=" << ctx.seq_len);
    FWD_INFO("[ForwardOrch] Phase 3: " << statusToString(ctx.phase3_status));
    FWD_INFO("[ForwardOrch] Phase 2: " << statusToString(ctx.phase2_status));
    FWD_INFO("[ForwardOrch] Phase 1: " << statusToString(ctx.phase1_status));
    if (!ctx.error_message.empty()) {
        FWD_WARN("[ForwardOrch] Error: " << ctx.error_message);
    }
    if (ctx.error_layer >= 0) {
        FWD_WARN("[ForwardOrch] Error layer: " << ctx.error_layer);
    }
    FWD_INFO("[ForwardOrch] =====================================");
}

} // namespace Forward
} // namespace GRIM
