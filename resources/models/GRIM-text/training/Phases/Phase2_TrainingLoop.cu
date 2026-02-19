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

#include "../../Shared/LogRecorder/LogRecorder.hpp"
#include "../../Shared/Gradients/GradStatsCollector.hpp"
#include "../../Shared/TrainingState/TrainingState_GPU.hpp"
#include "../../Shared/EquationLogging/EquationLogging.hpp"
#include "../Diagnostics/TrainingDiagnostics.hpp"
#include "../../Shared/UnigramByte/Unigram.hpp"
#include "../../Shared/UnigramByte/AtomTable.hpp"
#include "../../Shared/Batching/BatchPayload.hpp"
#include "../Autograd/AutogradTraining.hpp"  // autogradTrainingStep: unified forward+loss+backward
#include "../../../../../control/ai_config_paths.hpp"  // For resolveGrimRoot()

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <thread>
#include <future>
#include <atomic>
#include <filesystem>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
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

std::string trimSampleText(const std::string& text, std::size_t max_chars) {
    if (text.size() <= max_chars) {
        return text;
    }
    return text.substr(0, max_chars) + "...";
}

std::string decodeWithNumericSideChannel(const GRIM::Tokenizer::UniByte& tokenizer,
                                         const std::vector<int>& token_ids,
                                         const std::vector<float>& numeric_values,
                                         const std::vector<uint8_t>& numeric_mask) {
    std::string result;
    const size_t numeric_count = std::min(numeric_values.size(), numeric_mask.size());

    for (size_t i = 0; i < token_ids.size(); ++i) {
        const int tid = token_ids[i];

        if (tokenizer.isByteToken(tid)) {
            const uint8_t byte_val = tokenizer.byteEncoder().tokenToByte(tid);
            result.push_back(static_cast<char>(byte_val));
            continue;
        }

        if (tokenizer.isAtomToken(tid)) {
            const GRIM::Tokenizer::AtomType type = GRIM::Tokenizer::tokenIdToAtomType(tid);
            if (GRIM::Tokenizer::isNumericAtom(type) && i < numeric_count && numeric_mask[i]) {
                const float value = numeric_values[i];
                if (std::isfinite(value)) {
                    std::ostringstream oss;
                    if (std::fabs(value - std::round(value)) < 1e-6f) {
                        oss << static_cast<long long>(std::llround(value));
                    } else {
                        oss << std::setprecision(6) << value;
                    }
                    result += oss.str();
                    continue;
                }
            }
            // BUG FIX: Atom tokens without valid numeric values indicate corrupted training data.
            // Fail loud instead of silently showing placeholder text like <PATH>, <TIME>, etc.
            throw std::runtime_error(
                "Atom token " + std::to_string(tid) + " (" + tokenizer.tokenToString(tid) + 
                ") at position " + std::to_string(i) + " has no valid numeric/text value. " +
                "This indicates the model was trained on corrupted data containing literal atom placeholders. " +
                "Training data must be cleaned before retraining."
            );
        }

        result += tokenizer.tokenToString(tid);
    }

    return result;
}

bool shouldSyncDiagnostics(const TrainingContext& ctx, std::size_t batch_idx) {
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

void logInferenceSample(TrainingContext& ctx, TrainingLoopState& state) {
    const auto& hp = ctx.config.hyperparameters;
    const int interval = readEnvInt("GRIM_SAMPLE_INTERVAL", hp.log_interval);
    if (interval <= 0) {
        return;
    }

    const int optimizer_step = ctx.optimizer.optimizer_state.step;
    if (optimizer_step <= 0 || optimizer_step % interval != 0 || optimizer_step == state.last_sample_step) {
        return;
    }
    state.last_sample_step = optimizer_step;

    if (!ctx.model || !ctx.logging.logger) {
        return;
    }

    const std::string prompt = readEnvString("GRIM_SAMPLE_PROMPT", "What is a Planet?");
    const int max_new_tokens = readEnvInt("GRIM_SAMPLE_TOKENS", 80);
    const int max_chars = readEnvInt("GRIM_SAMPLE_MAX_CHARS", 300);
    if (max_new_tokens <= 0 || max_chars <= 0) {
        return;
    }

    auto prompt_result = ctx.tokenizer.encodeWithMetadata(prompt);
    std::vector<int> prompt_tokens = std::move(prompt_result.token_ids);
    std::vector<float> prompt_numeric_values = std::move(prompt_result.token_numeric_values);
    std::vector<uint8_t> prompt_numeric_mask = std::move(prompt_result.token_numeric_mask);
    if (prompt_tokens.empty()) {
        ctx.logging.logger->log("[Sample] prompt tokenization returned empty tokens");
        return;
    }

    const int max_seq_len = hp.max_seq_len;
    if (max_seq_len > 1 && static_cast<int>(prompt_tokens.size()) >= max_seq_len) {
        const size_t keep = static_cast<size_t>(max_seq_len - 1);
        const size_t drop = prompt_tokens.size() - keep;
        prompt_tokens.erase(prompt_tokens.begin(), prompt_tokens.begin() + drop);
        if (prompt_numeric_values.size() >= drop) {
            prompt_numeric_values.erase(prompt_numeric_values.begin(), prompt_numeric_values.begin() + drop);
        }
        if (prompt_numeric_mask.size() >= drop) {
            prompt_numeric_mask.erase(prompt_numeric_mask.begin(), prompt_numeric_mask.begin() + drop);
        }
    }

    GRIM::GenerationConfig cfg;
    // Use greedy for deterministic/reproducible samples (like PyTorch baseline)
    // With untrained model: greedy produces repetition, sampling produces gibberish
    cfg.strategy = GRIM::SamplingStrategy::GREEDY;
    cfg.do_sample = false;
    cfg.max_new_tokens = max_new_tokens;
    cfg.min_new_tokens = std::max(1, max_new_tokens / 4);  // Generate at least some tokens before EOS
    cfg.temperature = 1.0f;  // Ignored for greedy
    cfg.top_p = 0.9f;        // Ignored for greedy
    cfg.num_return_sequences = 1;
    cfg.eos_token_id = ctx.tokenizer.eosId();
    cfg.pad_token_id = ctx.tokenizer.padId();
    cfg.seed = static_cast<unsigned int>(optimizer_step);

    try {
        const auto start = std::chrono::steady_clock::now();
        std::vector<GRIM::GeneratedSequence> outputs = ctx.model->generate(
            prompt_tokens,
            prompt_numeric_values,
            prompt_numeric_mask,
            &cfg);
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();

        if (outputs.empty()) {
            ctx.logging.logger->log("[Sample] step=" + std::to_string(optimizer_step) + " empty output");
            return;
        }

        // DEBUG: Log raw token IDs to diagnose mode collapse
        const auto& gen_tokens = outputs.front().token_ids;
        const size_t prompt_len = prompt_tokens.size();
        std::ostringstream token_debug;
        token_debug << "[Sample] step=" << optimizer_step << " generated_tokens(first20): [";
        for (size_t ti = prompt_len; ti < std::min(prompt_len + 20, gen_tokens.size()); ++ti) {
            token_debug << gen_tokens[ti];
            if (ti < std::min(prompt_len + 19, gen_tokens.size() - 1)) token_debug << ", ";
        }
        token_debug << "] total_generated=" << (gen_tokens.size() - prompt_len);
        ctx.logging.logger->log(token_debug.str());
        
        // DEBUG: Decode individual token IDs to see what they map to
        if (gen_tokens.size() > prompt_len) {
            int first_gen_token = gen_tokens[prompt_len];
            std::string first_decoded = ctx.tokenizer.decode({first_gen_token});
            std::ostringstream tid_decode;
            tid_decode << "[TokenDecode] token_id=" << first_gen_token 
                       << " decodes_to=\"" << first_decoded << "\""
                       << " (len=" << first_decoded.size() << " bytes)";
            ctx.logging.logger->log(tid_decode.str());
        }

        std::string decoded = decodeWithNumericSideChannel(ctx.tokenizer,
                                                           outputs.front().token_ids,
                                                           outputs.front().token_numeric_values,
                                                           outputs.front().token_numeric_mask);
        decoded = trimSampleText(decoded, static_cast<std::size_t>(max_chars));
        ctx.logging.logger->log("[Sample] step=" + std::to_string(optimizer_step) +
                                " ms=" + std::to_string(elapsed_ms) +
                                " prompt=\"" + prompt + "\"");
        ctx.logging.logger->log("[Sample] " + decoded);
    } catch (const std::exception& e) {
        ctx.logging.logger->log(std::string("[Sample] generation failed: ") + e.what());
    }

    // Issue #142b: Check for deferred CUDA errors after generate().
    // generate() runs 80+ incremental forward passes (forwardInit + forwardStep).
    // CUDA kernel launches are async — errors may not surface until the NEXT sync.
    // Without this check, deferred errors corrupt batch N+1's forward pass,
    // triggering an SEH exception that bypasses C++ catch blocks → silent exit.
    {
        cudaError_t sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            std::string err_msg = "[Sample] CUDA ERROR after generate(): " +
                std::string(cudaGetErrorString(sync_err)) +
                " (code=" + std::to_string(static_cast<int>(sync_err)) + ")";
            ctx.logging.logger->log(err_msg);
            fprintf(stderr, "%s\n", err_msg.c_str());
            // Clear the error so training can attempt to continue
            cudaGetLastError();
        }
        // Also check for sticky errors from kernel launches
        cudaError_t peek_err = cudaGetLastError();
        if (peek_err != cudaSuccess) {
            std::string err_msg = "[Sample] CUDA sticky error after generate(): " +
                std::string(cudaGetErrorString(peek_err)) +
                " (code=" + std::to_string(static_cast<int>(peek_err)) + ")";
            ctx.logging.logger->log(err_msg);
            fprintf(stderr, "%s\n", err_msg.c_str());
        }
    }
}

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


// Detected at runtime — was token 277 under old layout
static int g_collapse_token_id = -1;  // Set dynamically from argmax of predictions

} // anonymous namespace

//======================================================//
//  GuessCache RAII Classes (Rule 22 Compliant)
//======================================================//

GuessCacheScope::GuessCacheScope(::GRIM::TrainingState& training_state,
                                 std::size_t capacity, 
                                 bool enable_async)
    : training_state_(training_state), active_(false), buffers_allocated_(false) {
    
    // RULE 22: Allocate buffers through TrainingState
    const bool enable_diversity = true;
    const std::size_t diversity_bloom_bits = 65536;
    const std::size_t pinned_buffer_size = enable_async ? 8192 : 0;
    
    training_state_.allocateGuessCacheBuffers(
            capacity, enable_diversity, diversity_bloom_bits, pinned_buffer_size);
    buffers_allocated_ = true;
    
    // Build GRIMTS config
    GRIMTS::CacheConfig config{};
    config.initial_capacity = capacity;
    config.min_capacity = 4096;
    config.max_capacity = 262144;
    config.grow_threshold = 0.85f;
    config.shrink_threshold = 0.25f;
    config.evict_window = 32;
    config.enable_diversity_tracking = enable_diversity;
    config.diversity_bloom_bits = static_cast<int>(diversity_bloom_bits);
    config.enable_async_transfers = enable_async;
    config.pinned_buffer_size = pinned_buffer_size;
    config.enable_histograms = false;
    
    // RULE 22: Get stream from centralized controller
    // getPrimaryStream() throws if not initialized (Rule 20)
    cudaStream_t primary_stream = training_state_.stream_ctrl.getPrimaryStream();
    
    // Convert TrainingState::GuessCacheBuffers to GRIMTS::GuessCacheBuffers
    GRIMTS::GuessCacheBuffers grimts_buffers{};
    grimts_buffers.records = training_state_.guess_cache_buffers.records;
    grimts_buffers.keys = training_state_.guess_cache_buffers.keys;
    grimts_buffers.size = training_state_.guess_cache_buffers.size;
    grimts_buffers.evict_cursor = training_state_.guess_cache_buffers.evict_cursor;
    grimts_buffers.diversity_bloom = training_state_.guess_cache_buffers.diversity_bloom;
    grimts_buffers.bloom_words = training_state_.guess_cache_buffers.bloom_words;
    grimts_buffers.calibration_offset = training_state_.guess_cache_buffers.calibration_offset;
    grimts_buffers.single_meta_buffer = training_state_.guess_cache_buffers.single_meta_buffer;
    grimts_buffers.single_reward_buffer = training_state_.guess_cache_buffers.single_reward_buffer;
    grimts_buffers.pinned_meta = training_state_.guess_cache_buffers.pinned_meta;
    grimts_buffers.pinned_rewards = training_state_.guess_cache_buffers.pinned_rewards;
    grimts_buffers.pinned_capacity = training_state_.guess_cache_buffers.pinned_capacity;
    grimts_buffers.capacity = training_state_.guess_cache_buffers.capacity;
    grimts_buffers.allocated = training_state_.guess_cache_buffers.allocated;
    
    // Wire up GRIMTS logging to training log system
    GRIMTS::Logging::RegisterLogCallback([](GRIMTS::Logging::LogLevel level, std::string_view message) {
        // Convert GRIMTS log level to our module logging
        switch (level) {
            case GRIMTS::Logging::LogLevel::Error:
                EmitModuleError(ModuleId::GuessCache, std::string(message), 0);
                break;
            case GRIMTS::Logging::LogLevel::Warning:
                EmitModuleWarning(ModuleId::GuessCache, std::string(message), 0);
                break;
            case GRIMTS::Logging::LogLevel::Info:
            case GRIMTS::Logging::LogLevel::Debug:
            default:
                EmitModuleInfo(ModuleId::GuessCache, std::string(message), 0);
                break;
        }
    });
    
    // Initialize GRIM-TS with pre-allocated buffers
    active_ = GRIMTS::InitializeGuessCache(config, grimts_buffers, primary_stream);
    if (active_) {
        GRIMTS::ResetGuessCache(primary_stream);
    } else {
        fprintf(stderr, "[ERROR] GuessCacheScope: GRIMTS::InitializeGuessCache failed!\n");
        training_state_.freeGuessCacheBuffers();
        buffers_allocated_ = false;
    }
}

GuessCacheScope::~GuessCacheScope() {
    if (active_) {
        GRIMTS::ShutdownGuessCache();
        active_ = false;
    }
    // Clear logging callbacks to avoid dangling references
    GRIMTS::Logging::ClearLogCallbacks();
    // RULE 22: TrainingState owns the buffers, we allocated them so we free them
    if (buffers_allocated_) {
        training_state_.freeGuessCacheBuffers();
        buffers_allocated_ = false;
    }
}

GuessCacheBatchBuffers::~GuessCacheBatchBuffers() {
    release();
}

void GuessCacheBatchBuffers::release() {
    if (device_metadata_) { cudaFree(device_metadata_); device_metadata_ = nullptr; }
    if (device_rewards_) { cudaFree(device_rewards_); device_rewards_ = nullptr; }
    if (device_stats_) { cudaFree(device_stats_); device_stats_ = nullptr; }
    capacity_ = 0;
}

cudaError_t GuessCacheBatchBuffers::ensure(std::size_t capacity) {
    if (capacity == 0) return cudaSuccess;
    if (capacity <= capacity_) return cudaSuccess;
    
    release();
    capacity_ = capacity;
    
    cudaError_t err = cudaMalloc(&device_metadata_, capacity * sizeof(GRIMTS::GuessMetadata));
    if (err != cudaSuccess) { release(); return err; }
    
    err = cudaMalloc(&device_rewards_, capacity * sizeof(float));
    if (err != cudaSuccess) { release(); return err; }
    
    err = cudaMalloc(&device_stats_, capacity * sizeof(GRIMTS::GuessRewardStats));
    if (err != cudaSuccess) { release(); return err; }
    
    return cudaSuccess;
}

MicroValidationScope::MicroValidationScope(int step) : step_(step) {
    GRIMTS::BeginMicroValidation(step_);
}

MicroValidationScope::~MicroValidationScope() {
    if (!completed_) {
        GRIMTS::CompleteMicroValidation({});
    }
}

void MicroValidationScope::complete(const GRIMTS::MicroValidationPulse& pulse) {
    GRIMTS::CompleteMicroValidation(pulse);
    completed_ = true;
}

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

std::string formatGradientComponents(GRIM::LanguageModel* model) {
    if (!model->hasGradientMetrics()) return "";
    
    const auto& gm = model->gradientMetrics();
    const bool tied = model->getConfig().tie_embeddings;
    
    std::ostringstream comp_msg;
    comp_msg << "[GradTrace] COMPUTED COMPONENTS: "
             << "total=" << formatScalar(gm.total_norm);
    
    // When tie_embeddings=true: tied buffer registered as LM_HEAD type (embedding_norm=0)
    // When tie_embeddings=false: separate EMBEDDING and LM_HEAD groups
    if (tied) {
        comp_msg << " emb_lm_tied=" << formatScalar(gm.lm_head_norm);
    } else {
        comp_msg << " emb=" << formatScalar(gm.embedding_norm)
                 << " lm=" << formatScalar(gm.lm_head_norm);
    }
    
    comp_msg << " attn=" << formatScalar(gm.attention_norm)
             << " ffn=" << formatScalar(gm.ffn_norm)
             << " rms=" << formatScalar(gm.rmsnorm_norm);

    comp_msg << " tied=" << (tied ? "yes" : "no");
    
    // Include ScratchBlock if enabled
    if (gm.scratchblock_norm > 0.0f) {
        comp_msg << " sb=" << formatScalar(gm.scratchblock_norm);
    }
    
    return comp_msg.str();
}

float getScheduledLearningRate(
    int step,
    float base_lr,
    int warmup_steps,
    bool stability_overrides_enabled) {
    
    if (stability_overrides_enabled) {
        return base_lr;
    }
    
    if (step < warmup_steps) {
        return base_lr * (static_cast<float>(step + 1) / warmup_steps);
    }
    
    // Constant LR after warmup
    return base_lr;
}

bool isBatchQuarantined(
    const std::vector<uint32_t>& seq_ids,
    const std::unordered_set<uint32_t>& quarantined_seqs) {
    
    for (uint32_t sid : seq_ids) {
        if (quarantined_seqs.count(sid)) return true;
    }
    return false;
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
    // Issue #90: Pass RNG seed for reproducible shuffling in GREEDY mode
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

void maybeRunMicroValidation(
    TrainingContext& ctx,
    TrainingLoopState& state,
    float latest_batch_loss,
    float current_lr) {
    
    const auto& hp = ctx.config.hyperparameters;
    if (!hp.micro_validation_enabled || ctx.data.val_seqs.empty()) return;
    if (hp.micro_validation_interval <= 0) return;
    if (ctx.global_step < hp.micro_validation_min_step) return;
    if ((ctx.global_step % hp.micro_validation_interval) != 0) return;
    
    // Ensure micro-validation batches are built
    if (state.micro_validation_batches.empty()) {
        const int micro_batch_size = std::max(1, std::min(hp.batch_size, 8));
        GRIM::Batching::BatchOptions mv_opts;
        const auto& mv_model_cfg = ctx.model->getConfig();
        mv_opts.max_tokens_per_batch = static_cast<uint32_t>(micro_batch_size) *
            static_cast<uint32_t>(mv_model_cfg.max_seq_len);
        mv_opts.max_batch_size = static_cast<uint32_t>(micro_batch_size);
        mv_opts.bucket_step = 128;
        mv_opts.prefer_short_first = hp.micro_validation_prefer_short;
        
        auto micro_schedule = GRIM::Batching::buildBatches(ctx.data.val_catalog, mv_opts);
        state.micro_validation_batches = std::move(micro_schedule.batches);
        state.micro_validation_cursor = 0;
    }
    
    if (state.micro_validation_batches.empty()) return;
    
    MicroValidationScope scope(ctx.global_step);
    const auto micro_start = std::chrono::steady_clock::now();
    float accumulated_loss = 0.0f;
    int sequences_processed = 0;
    int batches_executed = 0;
    
    const int batches_to_eval = std::min(hp.micro_validation_batch_limit,
                                         static_cast<int>(state.micro_validation_batches.size()));
    
    const auto& mv_model_cfg = ctx.model->getConfig();
    const size_t mv_max_cached_batch = static_cast<size_t>(std::max(1, mv_model_cfg.max_cached_batch));
    const size_t mv_max_cached_seq = static_cast<size_t>(std::max(1, std::min(mv_model_cfg.max_seq_len, mv_model_cfg.max_cached_seq_len)));
    
    for (int i = 0; i < batches_to_eval; ++i) {
        if (state.micro_validation_cursor >= state.micro_validation_batches.size()) {
            state.micro_validation_cursor = 0;
        }
        const auto& dyn_batch = state.micro_validation_batches[state.micro_validation_cursor++];
        
        const auto mv_token_layout = ctx.tokenizer.tokenLayout();
        auto mv_payload = GRIM::Batching::buildBatchPayload(
            dyn_batch, ctx.data.val_views, ctx.config.actual_vocab_size,
            mv_token_layout,
            mv_max_cached_batch, mv_max_cached_seq);
        if (mv_payload.batch_size == 0) continue;
        
        float micro_batch_loss = ctx.model->computeLossBatch(mv_payload);
        ctx.model->getTrainingState().autograd_intermediates.clear();
        // NOTE: No device sync here - computeLossBatch returns synchronously
        // Loss value is already on CPU after the call returns
        
        accumulated_loss += micro_batch_loss * static_cast<float>(mv_payload.batch_size);
        sequences_processed += mv_payload.batch_size;
        batches_executed++;
    }
    
    if (batches_executed == 0 || sequences_processed == 0) return;
    
    const float avg_val_loss = accumulated_loss / static_cast<float>(sequences_processed);
    const float val_perplexity = (std::isfinite(avg_val_loss) && avg_val_loss < 50.0f)
        ? std::exp(avg_val_loss)
        : std::numeric_limits<float>::infinity();
    const float duration_ms = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - micro_start).count();
    
    GRIMTS::MicroValidationPulse pulse;
    pulse.global_step = ctx.global_step;
    pulse.train_loss = latest_batch_loss;
    pulse.learning_rate = current_lr;
    pulse.val_loss = avg_val_loss;
    pulse.val_perplexity = val_perplexity;
    pulse.duration_ms = duration_ms;
    pulse.batches = batches_executed;
    pulse.sequences = sequences_processed;
    scope.complete(pulse);
    
    std::ostringstream msg;
    msg << "[ValMicro] step=" << ctx.global_step
        << " batches=" << batches_executed
        << " seq=" << sequences_processed
        << " loss=" << formatScalar(avg_val_loss)
        << " ppl=" << formatScalar(val_perplexity, 3)
        << " lr=" << formatScalar(current_lr, 6)
        << " dt=" << formatScalar(duration_ms, 2) << "ms";
    ctx.logging.logger->log(msg.str());
}

bool handleGradientSpike(
    TrainingContext& ctx,
    TrainingLoopState& state,
    const GRIM::Batching::BatchAssignment& batch,
    float preclip_grad_norm,
    float preclip_norm_grad,
    float batch_loss,
    const GRIM::TNC::ClipSelection& clip_selection,
    int batch_idx) {
    
    const float token_skip_threshold = clip_selection.per_token_limit * 50.0f;
    const float raw_skip_threshold = 500000.0f;
    
    if (preclip_norm_grad <= token_skip_threshold || preclip_grad_norm <= raw_skip_threshold) {
        return false;  // Not a spike
    }
    
    bool SkipGradGuard = true; // debug override
    if (ctx.config.stability.enabled || SkipGradGuard) {
        return false;  // Skip spike handling when stability overrides enabled
    }
    
    std::ostringstream skip_msg;
    skip_msg << "[GradGuard] skip optimizer step preclip_norm="
             << formatScalar(preclip_grad_norm)
             << " per_token=" << formatScalar(preclip_norm_grad, 4)
             << " batch=" << (batch_idx + 1)
             << " loss=" << formatScalar(batch_loss)
             << " tokens=" << clip_selection.stats.total_tokens;
    ctx.logging.logger->log(skip_msg.str());
    
    // Log gradient components if available
    std::string comp_log = Internal::formatGradientComponents(ctx.model.get());
    if (!comp_log.empty()) {
        ctx.logging.logger->log(comp_log);
    }
    
    // Quarantine sequences
    for (uint32_t sid : batch.seq_ids) {
        state.spike_counts[sid]++;
        state.quarantined_seqs.insert(sid);
    }
    
    std::ostringstream qmsg;
    qmsg << "[GradGuard] quarantined seqs=[";
    for (size_t si = 0; si < batch.seq_ids.size(); ++si) {
        qmsg << batch.seq_ids[si];
        if (si + 1 < batch.seq_ids.size()) qmsg << ",";
    }
    qmsg << "]";
    ctx.logging.logger->log(qmsg.str());
    
    // Log problematic sequences to bad_sequences.log with text content
    std::string bad_seq_path = ctx.config.paths.checkpoint_dir + "/bad_sequences.log";
    std::ofstream bad_seq_log(bad_seq_path, std::ios::app);
    if (bad_seq_log.is_open()) {
        bad_seq_log << "\n========================================\n";
        bad_seq_log << "[Batch " << (batch_idx + 1) << "] Gradient spike: " 
                   << preclip_grad_norm << " (raw) / " << preclip_norm_grad << " (per-token)\n";
        bad_seq_log << "Loss: " << batch_loss << " | Tokens: " << clip_selection.stats.total_tokens << "\n";
        bad_seq_log << "Sequence IDs: ";
        for (size_t si = 0; si < batch.seq_ids.size(); ++si) {
            bad_seq_log << batch.seq_ids[si];
            if (si + 1 < batch.seq_ids.size()) bad_seq_log << ", ";
        }
        bad_seq_log << "\n";
        
        // Decode and dump the actual text content
        for (size_t bi = 0; bi < batch.seq_ids.size(); ++bi) {
            uint32_t sid = batch.seq_ids[bi];
            bad_seq_log << "\n--- Sequence " << sid << " ---\n";
            
            // Get the sequence from training data seqs
            if (sid < ctx.data.train_seqs.size()) {
                const std::vector<int>& token_ids = ctx.data.train_seqs[sid].token_ids;
                std::string decoded_text = ctx.tokenizer.decode(token_ids);
                
                bad_seq_log << "Length: " << token_ids.size() << " tokens\n";
                bad_seq_log << "Text preview (first 500 chars):\n";
                bad_seq_log << decoded_text.substr(0, std::min<size_t>(500, decoded_text.length())) << "\n";
                if (decoded_text.length() > 500) {
                    bad_seq_log << "... (truncated, total " << decoded_text.length() << " chars)\n";
                }
            } else {
                bad_seq_log << "ERROR: Sequence ID out of bounds\n";
            }
        }
        bad_seq_log << "========================================\n";
        bad_seq_log.close(); 
    }
    
    return true;  // Spike detected
}

//======================================================//
//  Numerical Gradient Check (Finite Difference)
//======================================================//
// Verifies backward pass correctness by comparing analytical gradients
// against numerical gradients computed via finite differences:
//   numerical_grad ≈ (L(θ + ε) - L(θ - ε)) / (2ε)
//
// This is an expensive operation (2 extra forward passes) and should only
// be enabled for debugging. Control via GRIM_GRAD_CHECK_ENABLED env var.
//======================================================//

struct NumericalGradCheckResult {
    bool performed = false;
    bool passed = false;
    float analytical_grad = 0.0f;
    float numerical_grad = 0.0f;
    float relative_error = 0.0f;
    float loss_plus = 0.0f;
    float loss_minus = 0.0f;
    std::string param_name;
    int param_index = 0;
};

NumericalGradCheckResult performNumericalGradientCheck(
    TrainingContext& ctx,
    const GRIM::Batching::BatchPayload& payload,
    int batch_idx)
{
    NumericalGradCheckResult result;
    
    // Check if enabled via environment variable
    static const bool enabled = []() {
        const char* env = std::getenv("GRIM_GRAD_CHECK_ENABLED");
        return env && (std::string(env) == "1" || std::string(env) == "true");
    }();
    
    // Only run on first few batches to avoid performance impact
    static const int max_check_batches = readEnvInt("GRIM_GRAD_CHECK_MAX_BATCHES", 3);
    
    if (!enabled || batch_idx >= max_check_batches) {
        return result;
    }
    
    result.performed = true;
    
    auto& ts = ctx.model->getTrainingState();
    const auto& cfg = ctx.model->getConfig();
    cudaStream_t stream = ts.stream_ctrl.getPrimaryStream();
    
    // Choose a parameter to check - use embedding weight at a specific index
    // This is a good choice because it's directly involved in both forward and backward
    float* weight_ptr = ctx.model->getLmHeadLayer()->weights().data;  // [vocab_size, d_model]
    const float* grad_ptr = ctx.model->getLmHeadLayer()->weights().grad_data();
    
    if (!weight_ptr || !grad_ptr) {
        ctx.logging.logger->log("[NumGradCheck] ERROR: lm_head weights or grads not available");
        return result;
    }
    
    // Pick a random but deterministic index based on batch number
    // Use middle of vocabulary to avoid edge cases
    const int vocab_mid = cfg.vocab_size / 2;
    const int d_mid = cfg.d_model / 2;
    result.param_index = vocab_mid * cfg.d_model + d_mid;
    result.param_name = "lm_head_weights[" + std::to_string(vocab_mid) + "," + std::to_string(d_mid) + "]";
    
    // Read analytical gradient from backward pass
    cudaMemcpyAsync(&result.analytical_grad, grad_ptr + result.param_index, 
                    sizeof(float), cudaMemcpyDeviceToHost, stream);
    
    // Read original weight value
    float original_weight = 0.0f;
    cudaMemcpyAsync(&original_weight, weight_ptr + result.param_index, 
                    sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // Epsilon for finite difference - choose based on weight magnitude
    // Rule of thumb: ε ≈ sqrt(machine_epsilon) * max(1, |θ|)
    const float eps = 1e-4f * std::max(1.0f, std::abs(original_weight));
    
    // === Perturb +ε ===
    float perturbed_plus = original_weight + eps;
    cudaMemcpyAsync(weight_ptr + result.param_index, &perturbed_plus, 
                    sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    
    // Forward pass with perturbed weight (no gradient computation needed)
    // NOTE: We use the same batch data to ensure consistency
    result.loss_plus = ctx.model->computeLossBatch(payload);
    ts.autograd_intermediates.clear();
    
    // === Perturb -ε ===
    float perturbed_minus = original_weight - eps;
    cudaMemcpyAsync(weight_ptr + result.param_index, &perturbed_minus, 
                    sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    
    // Forward pass with perturbed weight
    result.loss_minus = ctx.model->computeLossBatch(payload);
    ts.autograd_intermediates.clear();
    
    // === Restore original weight ===
    cudaMemcpyAsync(weight_ptr + result.param_index, &original_weight, 
                    sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    
    // Compute numerical gradient: (L(θ+ε) - L(θ-ε)) / (2ε)
    result.numerical_grad = (result.loss_plus - result.loss_minus) / (2.0f * eps);
    
    // Compute relative error
    // Use formula that handles small gradients: |a - n| / max(|a|, |n|, 1e-8)
    float max_grad = std::max(std::abs(result.analytical_grad), std::abs(result.numerical_grad));
    max_grad = std::max(max_grad, 1e-8f);  // Avoid division by zero
    result.relative_error = std::abs(result.analytical_grad - result.numerical_grad) / max_grad;
    
    // Threshold for pass/fail (typically 1e-4 to 1e-2 depending on precision)
    const float error_threshold = 0.01f;  // 1% relative error
    result.passed = (result.relative_error < error_threshold);
    
    // Log results
    std::ostringstream log_msg;
    log_msg << std::fixed << std::setprecision(8);
    log_msg << "[NumGradCheck] batch=" << (batch_idx + 1) << " " << result.param_name << "\n";
    log_msg << "  analytical_grad = " << result.analytical_grad << "\n";
    log_msg << "  numerical_grad  = " << result.numerical_grad << "\n";
    log_msg << "  L(θ+ε) = " << result.loss_plus << ", L(θ-ε) = " << result.loss_minus << "\n";
    log_msg << "  ε = " << eps << "\n";
    log_msg << "  relative_error  = " << (result.relative_error * 100.0f) << "%\n";
    log_msg << "  " << (result.passed ? "✓ PASSED" : "✗ FAILED (threshold=" + std::to_string(error_threshold * 100) + "%)");
    
    ctx.logging.logger->log(log_msg.str());
    
    // Also print to stderr for visibility during debugging
    PHASE2_DEBUG_STDERR("\n%s\n", log_msg.str().c_str());
    
    return result;
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
    
    // Issue #85 FIX: Use model's actual buffer capacity, NOT kDefaultMaxTokensPerBatch (8192)!
    // Training allocates buffers based on batch_size * max_seq_len (e.g., 7 * 1024 = 7168).
    // Using 8192 for validation exceeds this → buffer overflow → crash!
    const auto& model_cfg = ctx.model->getConfig();
    const int safe_token_budget = static_cast<int>(model_cfg.max_cached_batch * model_cfg.max_cached_seq_len);

    ctx.logging.logger->log("[Val] Token budget: " + std::to_string(safe_token_budget) + 
        " (model limit: " + std::to_string(safe_token_budget) + ")");
    
    GRIM::Batching::BatchOptions val_opts;
    val_opts.max_tokens_per_batch = static_cast<uint32_t>(safe_token_budget);
    val_opts.max_batch_size = static_cast<uint32_t>(ctx.config.hyperparameters.batch_size);
    val_opts.bucket_step = 256;
    
    auto val_schedule = GRIM::Batching::buildBatches(ctx.data.val_catalog, val_opts);
    ctx.logging.logger->log("Created " + std::to_string(val_schedule.batches.size()) + " validation batches");
    
    float val_loss = 0.0f;
    int val_sequences_processed = 0;
    
    const auto& val_model_cfg = ctx.model->getConfig();
    const size_t val_max_cached_batch = static_cast<size_t>(std::max(1, val_model_cfg.max_cached_batch));
    const size_t val_max_cached_seq = static_cast<size_t>(std::max(1, std::min(val_model_cfg.max_seq_len, val_model_cfg.max_cached_seq_len)));
    
    for (const auto& val_batch : val_schedule.batches) {
        const auto val_token_layout = ctx.tokenizer.tokenLayout();
        auto val_payload = GRIM::Batching::buildBatchPayload(
            val_batch, ctx.data.val_views, ctx.config.actual_vocab_size,
            val_token_layout,
            val_max_cached_batch, val_max_cached_seq);
        if (val_payload.batch_size == 0) continue;
        
        float batch_val_loss = ctx.model->computeLossBatch(val_payload);
        ctx.model->getTrainingState().autograd_intermediates.clear();
        
        val_loss += batch_val_loss * val_payload.batch_size;
        val_sequences_processed += val_payload.batch_size;
    }
    
    result.loss = val_loss / val_sequences_processed;
    result.sequences_processed = val_sequences_processed;
    result.perplexity = (std::isfinite(result.loss) && result.loss < 50.0f)
        ? std::exp(result.loss)
        : std::numeric_limits<float>::infinity();
    result.is_best = (result.loss < ctx.best_val_loss);
    
    ctx.logging.logger->log("[Val] " + Internal::formatMetric("loss", result.loss) + " " +
                            Internal::formatMetric("ppl", result.perplexity, 3));
    
    return result;
}

//======================================================//
//  Batch Processing Implementation
//======================================================//

BatchResult processBatch(
    TrainingContext& ctx,
    TrainingLoopState& state,
    const GRIM::Batching::BatchAssignment& batch,
    int batch_idx,
    int total_batches,
    int epoch_idx) {
    
    BatchResult result;
    result.batch_idx = batch_idx;
    
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
    const std::uint64_t global_step_at_batch_start = ctx.global_step;  // Token counter (informational only)
    
    // Build unified batch payload — single source of truth for all metadata
    const auto& model_cfg = ctx.model->getConfig();
    const auto token_layout = ctx.tokenizer.tokenLayout();
    auto payload = GRIM::Batching::buildBatchPayload(
        batch, ctx.data.train_views, ctx.config.actual_vocab_size,
        token_layout,
        static_cast<size_t>(std::max(1, model_cfg.max_cached_batch)),
        static_cast<size_t>(std::max(1, std::min(model_cfg.max_seq_len, model_cfg.max_cached_seq_len))));
    
    if (payload.batch_size == 0) {
        result.skipped = true;
        result.skip_reason = "filtered";
        return result;
    }

    // Extract filtered_seq_ids for quarantine check
    std::vector<uint32_t> filtered_seq_ids(batch.seq_ids.begin(), batch.seq_ids.end());
    
    // Check quarantine
    if (Internal::isBatchQuarantined(filtered_seq_ids, state.quarantined_seqs)) {
        result.skipped = true;
        result.skip_reason = "quarantined";
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
    const int long_seq_threshold = ctx.config.max_seq_len;
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
        for (size_t i = 0; i < batch.seq_ids.size(); ++i) {
            batch_info << batch.seq_ids[i];
            if (i + 1 < batch.seq_ids.size()) batch_info << ",";
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
                per_seq_msg << filtered_seq_ids[i] << ":" << atom_count << "/" << seq_len
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
            
            // Dump ALL token IDs for the first batch only
            if (batch_idx == 0) {
                diag << "[BOUNDARY_DIAGNOSTIC] FULL TOKEN DUMP (batch 1):\n";
                for (int s = 0; s < payload.batch_size; ++s) {
                    const int flat_start = s * payload.max_seq_len;
                    const int len = payload.seq_lengths[s];
                    diag << "  seq[" << s << "] (" << len << " tokens):";
                    for (int t = 0; t < len; ++t) {
                        if (t % 32 == 0) diag << "\n    ";
                        diag << payload.input_ids[flat_start + t];
                        if (t + 1 < len) diag << ",";
                    }
                    diag << "\n";
                }
            }

            diag << "[BOUNDARY_DIAGNOSTIC] ========================================\n";
            ctx.logging.logger->log(diag.str());
            

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
    
    // Sync dynamic collapse token to training state for gradient attribution diagnostic
    ctx.model->getTrainingState().tracked_collapse_token = g_collapse_token_id;
    
    PHASE2_DEBUG_STDERR("[DEBUG-PROCESS] About to call autogradTrainingStep...\n");
    // STABILITY FIX (Issue #27/Math Audit): Apply 1/M scaling at the source (backward pass) to match PyTorch.
    // This ensures that gradients averaged per micro-batch are further averaged across the accumulation window,
    // preventing the "Sum of Averages" discrepancy that causes gradient explosion by a factor of M.
    const float grad_scale = 1.0f / static_cast<float>(accum_steps);
    auto loss_result = GRIM::Autograd::autogradTrainingStep(
        *ctx.model,
        ctx.model->getTrainingState(),
        payload,
        should_accumulate,
        grad_scale,
        ctx.global_step
    );
    result.loss = loss_result.loss_value;
    PHASE2_DEBUG_STDERR("[DEBUG-PROCESS] autogradTrainingStep returned, loss=%f success=%d\n", 
                        result.loss, static_cast<int>(loss_result.success));
    
    // Handle training step failure (NaN/Inf loss or backward error)
    if (!loss_result.success) {
        ctx.logging.logger->log("[autogradTrainingStep] FAILED batch=" + std::to_string(batch_idx + 1) +
                                " error: " + loss_result.error_message);
        ctx.model->zeroGrad();
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

    if (std::isfinite(result.loss) &&
        ctx.config.hyperparameters.guess_aux_enabled &&
        state.guess_cache_ready && !state.guess_cache_faulted) {
        if (!state.guess_cache_buffers) {
            state.guess_cache_buffers = std::make_unique<GuessCacheBatchBuffers>();
        }

        const std::size_t guess_count = static_cast<std::size_t>(payload.batch_size);
        if (guess_count > 0 && state.guess_cache_buffers) {
            auto& buffers = *state.guess_cache_buffers;
            auto& training_state = ctx.model->getTrainingState();
            if (!training_state.stream_ctrl.isInitialized()) {
                state.guess_cache_faulted = true;
                EmitModuleError(ModuleId::GuessCache,
                                "Guess cache update failed: stream controller not initialized",
                                ctx.global_step);
            } else if (!training_state.autograd_intermediates.hasLogits() ||
                       training_state.cached_batch_size != static_cast<int>(guess_count) ||
                       training_state.cached_seq_len <= 0) {
                EmitModuleWarning(ModuleId::GuessCache,
                                  "Guess cache skipped: cached logits not ready for prediction-based pass",
                                  ctx.global_step);
            } else {
                cudaStream_t stream = training_state.stream_ctrl.getPrimaryStream();
                const int batch_size = static_cast<int>(guess_count);
                const int vocab_size = ctx.config.actual_vocab_size;
                const int cached_seq_len = training_state.cached_seq_len;
                std::vector<int> guess_positions(batch_size, -1);
                for (int i = 0; i < batch_size; ++i) {
                    const int seq_len = payload.seq_lengths[i];
                    if (seq_len <= 0) {
                        continue;
                    }
                    const int flat_start = i * payload.max_seq_len;
                    int pos = seq_len - 1;
                    while (pos >= 0 && payload.target_ids[flat_start + pos] < 0) {
                        --pos;
                    }
                    if (pos >= 0 && pos < cached_seq_len) {
                        guess_positions[i] = pos;
                    }
                }

                std::vector<float> pred_logits;
                pred_logits.resize(static_cast<std::size_t>(batch_size) * vocab_size);
                cudaError_t err = cudaSuccess;
                for (int i = 0; i < batch_size; ++i) {
                    const int pos = guess_positions[i];
                    if (pos < 0) {
                        continue;
                    }
                    const std::size_t offset =
                        (static_cast<std::size_t>(i) * cached_seq_len + pos) * vocab_size;
                    err = cudaMemcpyAsync(pred_logits.data() + static_cast<std::size_t>(i) * vocab_size,
                                          training_state.autograd_intermediates.logits_tensor.data + offset,
                                          static_cast<std::size_t>(vocab_size) * sizeof(float),
                                          cudaMemcpyDeviceToHost, stream);
                    if (err != cudaSuccess) {
                        break;
                    }
                }
                if (err == cudaSuccess) {
                    err = cudaStreamSynchronize(stream);
                }
                if (err != cudaSuccess) {
                    state.guess_cache_faulted = true;
                    EmitModuleError(ModuleId::GuessCache,
                                    std::string("Guess cache logit sync failed: ") +
                                        cudaGetErrorString(err),
                                    ctx.global_step);
                } else {
                    std::vector<GRIMTS::GuessMetadata> pred_metadata;
                    std::vector<GRIMTS::GuessMetadata> reward_metadata;
                    std::vector<float> rewards;
                    pred_metadata.reserve(guess_count);
                    reward_metadata.reserve(guess_count);
                    rewards.reserve(guess_count);

                    const float reward = 1.0f / (1.0f + result.loss);
                    for (int i = 0; i < batch_size; ++i) {
                        const int pos = guess_positions[i];
                        if (pos < 0) {
                            continue;
                        }
                        const int flat_start_i = i * payload.max_seq_len;
                        const int target_token = payload.target_ids[flat_start_i + pos];
                        if (target_token < 0 || target_token >= vocab_size) {
                            continue;
                        }
                        const float* logits = pred_logits.data() + static_cast<std::size_t>(i) * vocab_size;
                        float top1 = -std::numeric_limits<float>::infinity();
                        float top2 = -std::numeric_limits<float>::infinity();
                        int pred_token = 0;
                        for (int v = 0; v < vocab_size; ++v) {
                            const float logit = logits[v];
                            if (logit > top1) {
                                top2 = top1;
                                top1 = logit;
                                pred_token = v;
                            } else if (logit > top2) {
                                top2 = logit;
                            }
                        }

                        const float margin = top1 - top2;
                        const float confidence = 1.0f / (1.0f + std::exp(-margin));
                        const float clamped_confidence = std::min(1.0f, std::max(0.0f, confidence));
                        const std::uint64_t prompt_hash = GRIMTS::HashSignature(
                            payload.input_ids.data() + flat_start_i,
                            payload.seq_lengths[i] * sizeof(int));

                        GRIMTS::GuessMetadata pred_meta{};
                        pred_meta.prompt_hash = prompt_hash;
                        pred_meta.guess_hash = GRIMTS::HashSignature(&pred_token, sizeof(int));
                        pred_meta.confidence = clamped_confidence;
                        pred_meta.sequence_length = static_cast<std::uint16_t>(
                            std::min<std::size_t>(payload.seq_lengths[i],
                                                  std::numeric_limits<std::uint16_t>::max()));
                        pred_meta.prompt_length = static_cast<std::uint16_t>(
                            std::min<std::size_t>(payload.seq_lengths[i],
                                                  std::numeric_limits<std::uint16_t>::max()));
                        pred_meta.epoch = static_cast<std::uint32_t>(epoch_idx + 1);
                        pred_metadata.push_back(pred_meta);

                        GRIMTS::GuessMetadata reward_meta = pred_meta;
                        reward_meta.guess_hash = GRIMTS::HashSignature(&target_token, sizeof(int));
                        reward_metadata.push_back(reward_meta);
                        rewards.push_back(reward);
                    }

                    if (!pred_metadata.empty()) {
                        err = buffers.ensure(pred_metadata.size());
                        if (err != cudaSuccess) {
                            state.guess_cache_faulted = true;
                            EmitModuleError(ModuleId::GuessCache,
                                            std::string("Guess cache buffer allocation failed: ") +
                                                cudaGetErrorString(err),
                                            ctx.global_step);
                        } else {
                            err = cudaMemcpyAsync(buffers.metadata(), pred_metadata.data(),
                                                  pred_metadata.size() * sizeof(GRIMTS::GuessMetadata),
                                                  cudaMemcpyHostToDevice, stream);
                            if (err == cudaSuccess) {
                                err = GRIMTS::CacheGuessBatchGPU(
                                    buffers.metadata(),
                                    pred_metadata.size(),
                                    stream);
                            }
                            if (err != cudaSuccess) {
                                state.guess_cache_faulted = true;
                                EmitModuleError(ModuleId::GuessCache,
                                                std::string("Guess cache insert failed: ") +
                                                    cudaGetErrorString(err),
                                                ctx.global_step);
                            } else {
                                err = cudaMemcpyAsync(buffers.metadata(), reward_metadata.data(),
                                                      reward_metadata.size() * sizeof(GRIMTS::GuessMetadata),
                                                      cudaMemcpyHostToDevice, stream);
                                if (err == cudaSuccess) {
                                    err = cudaMemcpyAsync(buffers.rewards(), rewards.data(),
                                                          rewards.size() * sizeof(float),
                                                          cudaMemcpyHostToDevice, stream);
                                }
                                if (err != cudaSuccess) {
                                    state.guess_cache_faulted = true;
                                    EmitModuleError(ModuleId::GuessCache,
                                                    std::string("Guess cache H2D copy failed: ") +
                                                        cudaGetErrorString(err),
                                                    ctx.global_step);
                                } else {
                                    err = GRIMTS::ApplyRewardBatchGPU(
                                        buffers.metadata(),
                                        buffers.rewards(),
                                        reward_metadata.size(),
                                        kGuessRewardMomentum,
                                        buffers.stats(),
                                        stream);
                                    if (err != cudaSuccess) {
                                        state.guess_cache_faulted = true;
                                        EmitModuleError(ModuleId::GuessCache,
                                                        std::string("Guess cache reward update failed: ") +
                                                            cudaGetErrorString(err),
                                                        ctx.global_step);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Log model predictions (what it predicts vs targets) - uses ForwardPass module for filtering
    {
        const auto& ts = ctx.model->getTrainingState();
        if (ts.autograd_intermediates.hasLogits() && ts.cached_batch_size > 0 && ts.cached_seq_len > 0) {
            const int total_tokens = ts.cached_batch_size * ts.cached_seq_len;
            const int vocab_size = ctx.config.actual_vocab_size;
            
            // Sample logits to find top predicted tokens (only sample first N positions to avoid slow sync)
            const int sample_positions = std::min(total_tokens, 100);
            const size_t logit_bytes = static_cast<size_t>(sample_positions) * vocab_size * sizeof(float);
            std::vector<float> logit_sample(sample_positions * vocab_size);
            cudaMemcpy(logit_sample.data(), ts.autograd_intermediates.logits_tensor.data, logit_bytes, cudaMemcpyDeviceToHost);
            
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
                constexpr int kDebugTokenId = 277;
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
                double p_277 = -1.0;
                if (vocab_size > kDebugTokenId) {
                    p_277 = std::exp(static_cast<double>(logits[kDebugTokenId] - max_logit)) / sum_exp;
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
                if (p_277 >= 0.0) {
                    trace_msg << " p_277=" << p_277;
                } else {
                    trace_msg << " p_277=N/A";
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
            
            // === COLLAPSE TOKEN MODE COLLAPSE DIAGNOSTIC ===
            // Track logit values for the dynamically-detected collapse token
            const int COLLAPSE_TOKEN_ID = g_collapse_token_id;
            if (COLLAPSE_TOKEN_ID >= 0 && vocab_size > COLLAPSE_TOKEN_ID) {
                // Sample first 10 positions to get logit stats for collapse token
                const int diag_positions = std::min(sample_positions, 10);
                float collapse_logit_sum = 0.0f;
                float collapse_logit_max = -std::numeric_limits<float>::infinity();
                float collapse_logit_min = std::numeric_limits<float>::infinity();
                int collapse_is_argmax_count = 0;
                
                // Also track the overall max logit for comparison
                float global_max_logit = -std::numeric_limits<float>::infinity();
                int global_argmax_token = -1;
                
                for (int pos = 0; pos < diag_positions; ++pos) {
                    float collapse_logit = logit_sample[pos * vocab_size + COLLAPSE_TOKEN_ID];
                    collapse_logit_sum += collapse_logit;
                    collapse_logit_max = std::max(collapse_logit_max, collapse_logit);
                    collapse_logit_min = std::min(collapse_logit_min, collapse_logit);
                    
                    // Find argmax for this position
                    float pos_max = -std::numeric_limits<float>::infinity();
                    int pos_argmax = 0;
                    for (int v = 0; v < vocab_size; ++v) {
                        float logit = logit_sample[pos * vocab_size + v];
                        if (logit > pos_max) {
                            pos_max = logit;
                            pos_argmax = v;
                        }
                    }
                    if (pos_argmax == COLLAPSE_TOKEN_ID) collapse_is_argmax_count++;
                    if (pos_max > global_max_logit) {
                        global_max_logit = pos_max;
                        global_argmax_token = pos_argmax;
                    }
                }
                
                float collapse_logit_mean = collapse_logit_sum / diag_positions;
                
                // Log collapse token diagnostic
                std::ostringstream collapse_diag;
                collapse_diag << std::fixed << std::setprecision(4);
                collapse_diag << "[CollapseTokenDiag] batch=" << (batch_idx + 1)
                           << " token=" << COLLAPSE_TOKEN_ID
                           << " logit: mean=" << collapse_logit_mean
                           << " min=" << collapse_logit_min
                           << " max=" << collapse_logit_max
                           << " is_argmax=" << collapse_is_argmax_count << "/" << diag_positions
                           << " global_max_logit=" << global_max_logit
                           << " global_argmax_token=" << global_argmax_token;
                ctx.logging.logger->log(collapse_diag.str());
            }
            
            // ================================================================
            // PERIODIC p_t / p_collapse DUMP
            // p_t = softmax probability assigned to the CORRECT target token
            // p_collapse = softmax probability assigned to the dynamically
            //   detected collapse token at positions where it is NOT the target
            // Fires every kPtPvDumpInterval batches for a configurable number
            // of positions so we can track whether the model is learning.
            // ================================================================
            constexpr int kPtPvDumpInterval   = 10;   // every N batches (set higher to reduce noise)
            constexpr int kPtPvDumpPositions  = 20;  // how many positions to dump
            const int dump_tracked_token = (g_collapse_token_id >= 0) ? g_collapse_token_id : -1;
            
            if ((batch_idx % kPtPvDumpInterval) == 0) {
                const int dump_n = kPtPvDumpPositions;
                
                // Issue #147/149: Collect positions with VALID targets first.
                // The scan must look through the FULL batch, not just sample_positions,
                // because sliding window overlaps can mask the first ~129 tokens.
                std::vector<int> valid_positions;
                valid_positions.reserve(dump_n);
                for (int pos = 0; pos < total_tokens && static_cast<int>(valid_positions.size()) < dump_n; ++pos) {
                    const int b = pos / ts.cached_seq_len;
                    const int t = pos % ts.cached_seq_len;
                    if (b < payload.batch_size && t < payload.seq_lengths[b]) {
                        const int tgt = payload.target_ids[b * payload.max_seq_len + t];
                        if (tgt >= 0) {
                            valid_positions.push_back(pos);
                        }
                    }
                }
                // Fallback: if no valid targets found, sample linearly (at least shows the data)
                if (valid_positions.empty()) {
                    for (int pos = 0; pos < std::min(total_tokens, dump_n); ++pos) valid_positions.push_back(pos);
                }
                
                // Accumulators for summary stats
                double sum_p_t = 0.0, sum_p_v = 0.0;
                int    count_p_t = 0, count_p_v = 0;
                float  max_p_t = 0.0f, min_p_t = 1.0f;
                float  max_p_v = 0.0f, min_p_v = 1.0f;
                
                std::ostringstream pt_pv;
                pt_pv << std::fixed << std::setprecision(8);
                pt_pv << "[PtPvDump] batch=" << (batch_idx + 1) << " positions=" << static_cast<int>(valid_positions.size()) << "\n";
                
                for (int vi = 0; vi < static_cast<int>(valid_positions.size()); ++vi) {
                    const int pos = valid_positions[vi];
                    
                    // Fetch logits for this specific position (Issue #149: handle positions outside sample_positions)
                    std::vector<float> row_buffer;
                    const float* logits_pos = nullptr;
                    if (pos < sample_positions) {
                        logits_pos = logit_sample.data() + static_cast<size_t>(pos) * vocab_size;
                    } else {
                        row_buffer.resize(vocab_size);
                        const float* device_ptr = static_cast<const float*>(ts.autograd_intermediates.logits_tensor.data) + static_cast<size_t>(pos) * vocab_size;
                        cudaMemcpy(row_buffer.data(), device_ptr, vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
                        logits_pos = row_buffer.data();
                    }
                    
                    // Stable softmax: subtract max
                    float lmax = -std::numeric_limits<float>::infinity();
                    for (int v = 0; v < vocab_size; ++v) {
                        if (logits_pos[v] > lmax) lmax = logits_pos[v];
                    }
                    double denom = 0.0;
                    for (int v = 0; v < vocab_size; ++v) {
                        denom += std::exp(static_cast<double>(logits_pos[v] - lmax));
                    }
                    if (denom <= 0.0) denom = 1.0;
                    
                    // Resolve target for this position from flat payload
                    const int b = pos / ts.cached_seq_len;
                    const int t = pos % ts.cached_seq_len;
                    int target = -1;
                    if (b < payload.batch_size &&
                        t < payload.seq_lengths[b]) {
                        target = payload.target_ids[b * payload.max_seq_len + t];
                    }
                    
                    // p_t: probability of correct target
                    float p_t_val = 0.0f;
                    if (target >= 0 && target < vocab_size) {
                        p_t_val = static_cast<float>(
                            std::exp(static_cast<double>(logits_pos[target] - lmax)) / denom);
                        sum_p_t += p_t_val;
                        max_p_t = std::max(max_p_t, p_t_val);
                        min_p_t = std::min(min_p_t, p_t_val);
                        count_p_t++;
                    }
                    
                    // p_collapse: probability of collapse token at this position
                    float p_v_val = 0.0f;
                    if (dump_tracked_token >= 0 && vocab_size > dump_tracked_token) {
                        p_v_val = static_cast<float>(
                            std::exp(static_cast<double>(logits_pos[dump_tracked_token] - lmax)) / denom);
                        if (target != dump_tracked_token) {
                            sum_p_v += p_v_val;
                            max_p_v = std::max(max_p_v, p_v_val);
                            min_p_v = std::min(min_p_v, p_v_val);
                            count_p_v++;
                        }
                    }
                    
                    // Find argmax for context
                    int am = 0; float am_val = logits_pos[0];
                    for (int v = 1; v < vocab_size; ++v) {
                        if (logits_pos[v] > am_val) { am_val = logits_pos[v]; am = v; }
                    }
                    
                    pt_pv << "  pos=" << pos
                           << " target=" << target
                           << " p_t=" << p_t_val
                           << " p_collapse=" << p_v_val
                           << " argmax=" << am
                           << (dump_tracked_token >= 0 && target == dump_tracked_token ? " [TGT=COLLAPSE]" : "")
                           << "\n";
                }
                
                // Summary line
                pt_pv << "  SUMMARY: "
                       << "avg_p_t=" << (count_p_t > 0 ? sum_p_t / count_p_t : 0.0)
                       << " [" << min_p_t << ".." << max_p_t << "] (n=" << count_p_t << ")"
                       << " | avg_p_277_at_other=" << (count_p_v > 0 ? sum_p_v / count_p_v : 0.0)
                       << " [" << min_p_v << ".." << max_p_v << "] (n=" << count_p_v << ")"
                       << " | uniform_baseline=" << (1.0 / vocab_size);
                ctx.logging.logger->log(pt_pv.str());
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
        EQ_LOG("BATCH_LOSS", eq.str(), batch_idx, -1, ctx.global_step, GRIM::EquationPhase::LOSS_COMPUTATION);
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
        if (ts.autograd_intermediates.hasLogits() && ts.cached_batch_size > 0 && ts.cached_seq_len > 0) {
            const int total_tokens = ts.cached_batch_size * ts.cached_seq_len;
            const int vocab_size = ctx.config.actual_vocab_size;
            const int d_model = ctx.model->getConfig().d_model;
            
            // Sample logits for statistics (limit to avoid expensive D2H copy)
            const int sample_positions = std::min(total_tokens, 50);
            const size_t logit_bytes = static_cast<size_t>(sample_positions) * vocab_size * sizeof(float);
            std::vector<float> logit_sample(sample_positions * vocab_size);
            cudaMemcpy(logit_sample.data(), ts.autograd_intermediates.logits_tensor.data, logit_bytes, cudaMemcpyDeviceToHost);
            
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
                const bool use_centered_src = ctx.model->getConfig().lm_head_center_hidden_states;
                const float* h_src = (use_centered_src && ts.centering_scratch_tensor.data)
                    ? ts.centering_scratch_tensor.data
                    : ts.cached_encoder_output.data;
                
                float h_norm_mean = 0.0f;
                float h_norm_max = -std::numeric_limits<float>::infinity();
                float h_norm_min = std::numeric_limits<float>::infinity();
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
                        const float norm = std::sqrt(sum_sq);
                        const float rms = std::sqrt(sum_sq / d_model);
                        h_norm_mean += norm;
                        h_norm_max = std::max(h_norm_max, norm);
                        h_norm_min = std::min(h_norm_min, norm);
                        h_rms_mean += rms;
                    }
                    h_norm_mean /= sample_positions;
                    h_rms_mean /= sample_positions;
                    
                    // Rule 20: Verify hidden state norms are finite (issue #142)
                    if (!std::isfinite(h_norm_mean) || !std::isfinite(h_norm_max) || !std::isfinite(h_rms_mean)) {
                        throw std::runtime_error(
                            "[HIDDEN_STATE_NORMS] FATAL: Hidden state norms contain inf/nan at batch " + 
                            std::to_string(batch_idx + 1) + 
                            " (h_norm_mean=" + std::to_string(h_norm_mean) + 
                            ", h_norm_max=" + std::to_string(h_norm_max) + 
                            ", h_rms_mean=" + std::to_string(h_rms_mean) + 
                            "). Encoder output exploded. " +
                            "Check: (1) attention gradient explosion, (2) FFN activation overflow, (3) RMSNorm inverse explosion."
                        );
                    }
                }
                
                // --- Weight norm statistics (sample random + top tokens) ---
                // Issue #138 FIX: Compute E[||W||²] (mean of squared norms) for correct expected logit_std.
                // Old code used E[||W||] (mean of norms) which underestimates by Jensen's inequality
                // when ||W|| distribution is skewed (e.g., tok277 has ||W||=1.67 vs mean=0.20).
                const float* lm_head_weights = ctx.model->getLmHeadLayer()->weights().data;
                const float* embedding_weights = ctx.model->getEmbeddingLayer()->tokenWeights().data;
                if (ctx.model->getConfig().tie_embeddings &&
                    lm_head_weights &&
                    embedding_weights &&
                    lm_head_weights != embedding_weights) {
                    throw std::runtime_error("Tied embeddings: lm_head_weights and embedding_weights must alias the same buffer.");
                }

                float w_norm_mean = 0.0f, w_norm_sq_mean = 0.0f, w_norm_max = 0.0f;
                int w_norm_max_tok = -1;
                const int w_sample_count = std::min(500, vocab_size);  // Sample 500 rows for better coverage
                if (lm_head_weights) {
                    std::vector<float> w_row(d_model);
                    
                    // Helper: compute L2 norm of a weight row on host
                    auto compute_row_norm = [&](int tok) -> float {
                        const size_t row_offset = static_cast<size_t>(tok) * d_model;
                        cudaMemcpy(w_row.data(),
                                   lm_head_weights + row_offset,
                                   d_model * sizeof(float), cudaMemcpyDeviceToHost);
                        float norm_sq = 0.0f;
                        for (int d = 0; d < d_model; ++d) {
                            norm_sq += w_row[d] * w_row[d];
                        }
                        return std::sqrt(norm_sq);
                    };
                    
                    // Sample evenly-spaced vocab tokens for mean/rms statistics
                    const int stride = std::max(1, vocab_size / w_sample_count);
                    int sampled = 0;
                    for (int tok = 0; tok < vocab_size && sampled < w_sample_count; tok += stride, ++sampled) {
                        const float norm = compute_row_norm(tok);
                        w_norm_mean += norm;
                        const float norm_sq = norm * norm;
                        w_norm_sq_mean += norm_sq;
                        if (norm > w_norm_max) {
                            w_norm_max = norm;
                            w_norm_max_tok = tok;
                        }
                    }
                    w_norm_mean /= sampled;
                    w_norm_sq_mean /= sampled;
                    
                    // FIX: Also check top-predicted tokens for ||W||_max.
                    // The strided sample (stride=100) systematically misses tokens like
                    // 262, 277, 488 which are the ones with growing norms causing mode
                    // collapse. The strided sample reported ||W||_max=0.197 while these
                    // tokens had ||W||=0.323 (64% above reported "max").
                    for (size_t i = 0; i < std::min(sorted_argmax.size(), size_t(5)); ++i) {
                        const int tok = sorted_argmax[i].first;
                        // Skip if already in strided sample
                        if (tok % stride == 0 && tok / stride < w_sample_count) continue;
                        const float norm = compute_row_norm(tok);
                        if (norm > w_norm_max) {
                            w_norm_max = norm;
                            w_norm_max_tok = tok;
                        }
                    }
                }
                
                // --- Expected logit magnitude ---
                // Issue #138 FIX: Correct formula for dot product variance.
                // Var(h·W) = d × Var(h_i) × Var(W_j) = h_rms² × E[||W||²]
                // Old code used h_rms × E[||W||] which underestimates due to Jensen's inequality.
                // Correct: logit_std ≈ h_rms × sqrt(E[||W||²]) = h_rms × ||W||_rms
                const float w_norm_rms = std::sqrt(w_norm_sq_mean);  // sqrt(E[||W||²])
                const float expected_logit_std = std::sqrt(static_cast<float>(d_model)) * h_rms_mean * w_norm_rms;
                const float logit_std_ratio = (expected_logit_std > 1e-10f) ? logit_std / expected_logit_std : 0.0f;

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
                scale_eq << "  HIDDEN (LM input): ||h||_mean=" << h_norm_mean
                         << " ||h||_max=" << h_norm_max << " ||h||_min=" << h_norm_min
                         << " h_rms_mean=" << h_rms_mean << "\n";
                scale_eq << "  WEIGHTS (LM head): ||W||_mean=" << w_norm_mean
                         << " ||W||_rms=" << w_norm_rms
                         << " ||W||_max=" << w_norm_max << " (tok=" << w_norm_max_tok << ")"
                         << " d_model=" << d_model << "\n";
                scale_eq << "  EXPECTED logit_std = sqrt(d_model) × h_rms × ||W||_rms\n";
                scale_eq << "                      = sqrt(" << d_model << ") × " << h_rms_mean
                         << " × " << w_norm_rms << "\n";
                scale_eq << "                      = " << expected_logit_std << "\n";
                scale_eq << "  ACTUAL logit_std = " << logit_std
                         << " ratio(actual/expected)=" << logit_std_ratio << "\n";
                scale_eq << "  TREND (EMA α=" << trend_alpha << ")"
                         << " logit_std_delta=" << (logit_std - trend_state.ema_logit_std)
                         << " max_logit_delta=" << (logit_max - trend_state.ema_logit_max)
                         << " top2_margin_delta=" << (avg_margin - trend_state.ema_top2_margin)
                         << " top1_frac_delta=" << (top1_frac - trend_state.ema_top1_frac) << "\n";
                if (w_norm_max > 2.0f) {
                    scale_eq << "  [ANOMALY] WEIGHT_NORM_EXPLOSION: ||W||_max=" << w_norm_max
                             << " (tok=" << w_norm_max_tok << ") >> 2.0. Weight decay too weak or gradient bias.\n";
                }
                if (logit_std_ratio > 3.0f) {
                    scale_eq << "  [ANOMALY] LOGIT_STD_MISMATCH: actual/expected=" << logit_std_ratio
                             << " >> 3.0. Possible hidden-weight alignment or missing 1/sqrt(d) scaling.\n";
                }
                ctx.logging.logger->log(scale_eq.str());
            }
            
            // ================================================================
            // HIDDEN STATE DIAGNOSTICS: Cosine similarity between positions
            // Issue #115 FIX: Use centered buffer when centering is enabled!
            // ts.cached_encoder_output = RAW encoder output BEFORE centering
            // ts.centering_scratch_tensor = CENTERED output (what LM head actually uses)
            // ================================================================
            const bool use_centering_for_diag = ctx.model->getConfig().lm_head_center_hidden_states;
            const float* hidden_source = (use_centering_for_diag && ts.centering_scratch_tensor.data)
                ? ts.centering_scratch_tensor.data  // CENTERED: actual LM head input
                : ts.cached_encoder_output.data;    // UNCENTERED: raw encoder output
            
            if (hidden_source && sample_positions >= 2) {
                const size_t hidden_bytes = static_cast<size_t>(sample_positions) * d_model * sizeof(float);
                std::vector<float> hidden_sample(sample_positions * d_model);
                cudaMemcpy(hidden_sample.data(), hidden_source, hidden_bytes, cudaMemcpyDeviceToHost);
                
                // Compute cosine similarity between position pairs (sample 5 pairs)
                auto compute_cosine = [&](int i, int j) -> float {
                    float dot = 0.0f, norm_i = 0.0f, norm_j = 0.0f;
                    for (int d = 0; d < d_model; ++d) {
                        float hi = hidden_sample[i * d_model + d];
                        float hj = hidden_sample[j * d_model + d];
                        dot += hi * hj;
                        norm_i += hi * hi;
                        norm_j += hj * hj;
                    }
                    return dot / (std::sqrt(norm_i * norm_j) + 1e-8f);
                };
                
                // Compute pairwise cosines: (0,1), (0,10), (0,25), (10,25), (25,49)
                std::ostringstream hidden_stats;
                hidden_stats << "[HiddenCosine] batch=" << (batch_idx + 1) << " cos(h_i,h_j)=[";
                const int pairs[][2] = {{0, 1}, {0, std::min(10, sample_positions-1)}, 
                                        {0, std::min(25, sample_positions-1)},
                                        {std::min(10, sample_positions-1), std::min(25, sample_positions-1)},
                                        {std::min(25, sample_positions-1), sample_positions-1}};
                for (int p = 0; p < 5; ++p) {
                    int i = pairs[p][0], j = pairs[p][1];
                    if (i < sample_positions && j < sample_positions && i != j) {
                        float cos_ij = compute_cosine(i, j);
                        hidden_stats << "(" << i << "," << j << "):" << Internal::formatScalar(cos_ij, 3);
                        if (p < 4) hidden_stats << ",";
                    }
                }
                hidden_stats << "]";
                
                // Compute average pairwise cosine (measure of hidden state isotropy)
                float cos_sum = 0.0f;
                int cos_count = 0;
                for (int i = 0; i < std::min(10, sample_positions); ++i) {
                    for (int j = i + 1; j < std::min(10, sample_positions); ++j) {
                        cos_sum += compute_cosine(i, j);
                        cos_count++;
                    }
                }
                float avg_cos = (cos_count > 0) ? cos_sum / cos_count : 0.0f;
                hidden_stats << " avg_cos=" << Internal::formatScalar(avg_cos, 8);
                ctx.logging.logger->log(hidden_stats.str());
            }
            
            // ================================================================
            // LM HEAD DIAGNOSTICS: Row norms ||W[v]|| for top predicted tokens
            // ================================================================
            const float* lm_head_weights_for_norms = ctx.model->getLmHeadLayer()->weights().data;
            if (lm_head_weights_for_norms) {
                // Copy LM head rows for top-5 predicted tokens
                std::ostringstream lm_stats;
                lm_stats << "[LMHeadNorm] batch=" << (batch_idx + 1) << " ||W[v]||=[";
                
                std::vector<float> row_buffer(d_model);
                for (size_t i = 0; i < std::min(sorted_argmax.size(), size_t(5)); ++i) {
                    int tok_id = sorted_argmax[i].first;
                    // Copy row [tok_id, :] from W [vocab_size, d_model]
                    const size_t row_offset = static_cast<size_t>(tok_id) * d_model;
                    cudaMemcpy(row_buffer.data(), 
                               lm_head_weights_for_norms + row_offset,
                               d_model * sizeof(float), cudaMemcpyDeviceToHost);
                    
                    // Compute L2 norm of row
                    float norm_sq = 0.0f;
                    for (int d = 0; d < d_model; ++d) {
                        norm_sq += row_buffer[d] * row_buffer[d];
                    }
                    float row_norm = std::sqrt(norm_sq);
                    lm_stats << "tok" << tok_id << ":" << Internal::formatScalar(row_norm, 10);
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
        
        // Only skip on confirmed invalid token corruption
        if (has_invalid_tokens) {
            ctx.logging.logger->log("[DataGuard] SKIPPING batch=" + std::to_string(batch_idx + 1) +
                                    " loss=" + Internal::formatScalar(result.loss) +
                                    " invalid_tokens=true");
            
            // Quarantine corrupted sequences
            for (uint32_t sid : batch.seq_ids) {
                state.spike_counts[sid]++;
                state.quarantined_seqs.insert(sid);
            }
            
            // Log to bad_sequences.log with corruption details
            std::string bad_seq_path = ctx.config.paths.checkpoint_dir + "/bad_sequences.log";
            std::ofstream bad_seq_log(bad_seq_path, std::ios::app);
            if (bad_seq_log.is_open()) {
                bad_seq_log << "\n========================================\n";
                bad_seq_log << "[Batch " << (batch_idx + 1) << "] DATA CORRUPTION DETECTED\n";
                bad_seq_log << "Loss: " << result.loss << " | Invalid tokens: " << (has_invalid_tokens ? "yes" : "no") << "\n";
                bad_seq_log << "Sequence IDs: ";
                for (size_t si = 0; si < batch.seq_ids.size(); ++si) {
                    bad_seq_log << batch.seq_ids[si];
                    if (si + 1 < batch.seq_ids.size()) bad_seq_log << ", ";
                }
                bad_seq_log << "\n";
                
                // CRITICAL: Decode sequences and check for data corruption
                for (size_t bi = 0; bi < batch.seq_ids.size(); ++bi) {
                    uint32_t sid = batch.seq_ids[bi];
                    bad_seq_log << "\n--- Sequence " << sid << " ---\n";
                    
                    if (sid < ctx.data.train_seqs.size()) {
                        const std::vector<int>& token_ids = ctx.data.train_seqs[sid].token_ids;
                        bad_seq_log << "Length: " << token_ids.size() << " tokens\n";
                        
                        // Check for invalid token IDs (root cause of loss=165)
                        int invalid_count = 0;
                        int max_invalid = -1;
                        for (int tid : token_ids) {
                            if (tid < 0 || tid >= static_cast<int>(ctx.config.actual_vocab_size)) {
                                invalid_count++;
                                if (tid > max_invalid) max_invalid = tid;
                            }
                        }
                        
                        if (invalid_count > 0) {
                            bad_seq_log << "**CORRUPTION DETECTED**: " << invalid_count 
                                       << " tokens out of vocab range [0, " << ctx.config.actual_vocab_size 
                                       << "). Max invalid token: " << max_invalid << "\n";
                        }
                        
                        // Decode text preview
                        std::string decoded_text = ctx.tokenizer.decode(token_ids);
                        bad_seq_log << "Text preview (first 500 chars):\n";
                        bad_seq_log << decoded_text.substr(0, std::min<size_t>(500, decoded_text.length())) << "\n";
                        if (decoded_text.length() > 500) {
                            bad_seq_log << "... (truncated, total " << decoded_text.length() << " chars)\n";
                        }
                    } else {
                        bad_seq_log << "ERROR: Sequence ID " << sid << " out of bounds (max: " 
                                   << ctx.data.train_seqs.size() << ")\n";
                    }
                }
                bad_seq_log << "========================================\n";
                bad_seq_log.close();
            }
            
            // IMPORTANT: Reset state before early return
            // Zero gradients to ensure corrupted data doesn't persist
            ctx.model->zeroGrad();
            ctx.optimizer.current_micro_step = 0;  // Reset accumulation window
            ctx.model->getTrainingState().autograd_intermediates.clear();
            
            result.skipped = true;
            result.skip_reason = "data_corruption";
            ctx.global_step++;
            return result;
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
    
    // NOTE: Gradient component logging happens later after computeGradNorm() at line ~2994
    // via formatGradientComponents(). Premature logging here would use undefined variables.
    
    // ========================================================================
    // DIAGNOSTIC: Dynamic collapse token gradient analysis - Issue #36.5
    // Track WHY p_collapse increases: if grad_sum > 0, weight row increases
    // For tied weights: logit[T] = x @ W[T].T, so larger W[T] → larger logit_T
    // ========================================================================
    // Use centralized EquationLogger to gate expensive diagnostics
    // Issue #134: Run EXPENSIVE diagnostics every N batches to avoid sync bottleneck
    // These copy GB of data D2H (logits are 7168*50377*4 = 1.4GB per call!)
    static int debug_collapse_diag_interval = 10;  // Debug knob: set to N for every N batches
    const int diag_interval = std::max(debug_collapse_diag_interval, 1);
    const bool kCollapseDiagEnabled = GRIM::getEquationLogger().isEnabled() &&
        (batch_idx == 0 || (batch_idx + 1) % diag_interval == 0);
    
    if (kCollapseDiagEnabled) {
        const auto& ts = ctx.model->getTrainingState();
        const auto& cfg = ctx.model->getConfig();
        cudaStream_t stream = ts.stream_ctrl.getPrimaryStream();
        
        // ====================================================================
        // Issue #137: Use GPU's cached_targets_tensor directly.
        //
        // GPU buffers (grad_logits, hidden states, logits) are laid out as
        // [batch_size × max_seq_len] where shorter sequences are padded.
        // The GPU targets buffer (cached_targets_tensor) is ALREADY in this
        // exact layout — just D2H copy it instead of reconstructing from
        // payload.target_ids (which is already flat-padded).
        // ====================================================================
        const int diag_total_tokens = ts.cached_batch_size * ts.cached_seq_len;
        std::vector<int> flat_targets(diag_total_tokens);
        if (ts.cached_targets_tensor.data && diag_total_tokens > 0) {
            cudaMemcpy(flat_targets.data(),
                       reinterpret_cast<const int*>(ts.cached_targets_tensor.data),
                       diag_total_tokens * sizeof(int),
                       cudaMemcpyDeviceToHost);
        }
        
        // ====================================================================
        // DYNAMIC ARGMAX DETECTION: Find which token the model is collapsing to.
        // Sample a subset of positions from cached logits, find the most frequent
        // argmax token. This replaces the old hardcoded token 277.
        // ====================================================================
        if (ts.autograd_intermediates.hasLogits() && diag_total_tokens > 0) {
            // Sample up to 128 non-pad positions to find the dominant argmax
            constexpr int kArgmaxSampleSize = 128;
            const int stride = std::max(1, diag_total_tokens / kArgmaxSampleSize);
            
            // Copy logits for sampled positions to count argmax tokens
            // Instead of copying ALL logits (1.4GB), copy one row at a time for sampled positions
            std::vector<float> logit_row(cfg.vocab_size);
            std::unordered_map<int, int> argmax_counts;
            int sampled = 0;
            
            for (int t = 0; t < diag_total_tokens && sampled < kArgmaxSampleSize; t += stride) {
                if (flat_targets[t] < 0 || flat_targets[t] == GRIM::Tokenizer::PAD_TOKEN_ID) continue;
                
                cudaMemcpy(logit_row.data(),
                           ts.autograd_intermediates.logits_tensor.data + static_cast<size_t>(t) * cfg.vocab_size,
                           cfg.vocab_size * sizeof(float),
                           cudaMemcpyDeviceToHost);
                
                // Find argmax for this position
                int best_tok = 0;
                float best_val = logit_row[0];
                for (int v = 1; v < cfg.vocab_size; ++v) {
                    if (logit_row[v] > best_val) {
                        best_val = logit_row[v];
                        best_tok = v;
                    }
                }
                argmax_counts[best_tok]++;
                sampled++;
            }
            
            // Find the most frequent argmax token
            int most_common_token = -1;
            int most_common_count = 0;
            for (const auto& [tok, cnt] : argmax_counts) {
                if (cnt > most_common_count) {
                    most_common_count = cnt;
                    most_common_token = tok;
                }
            }
            
            if (most_common_token >= 0) {
                const float collapse_fraction = static_cast<float>(most_common_count) / std::max(1, sampled);
                
                // Log the detection
                ctx.logging.logger->log(
                    "[COLLAPSE_DETECT] batch=" + std::to_string(batch_idx + 1) +
                    " dominant_argmax_token=" + std::to_string(most_common_token) +
                    " count=" + std::to_string(most_common_count) + "/" + std::to_string(sampled) +
                    " (" + std::to_string(static_cast<int>(collapse_fraction * 100)) + "%)");
                
                g_collapse_token_id = most_common_token;
            }
        }
        
        // Only run detailed diagnostics if we have a tracked token
        if (g_collapse_token_id >= 0) {
            GRIM::Diagnostics::Token277Diagnostic tok277 = GRIM::Diagnostics::computeToken277Diagnostic(
                ctx.model->getLmHeadLayer(), ctx.model->getEmbeddingLayer(), flat_targets, cfg.d_model, g_collapse_token_id, stream);
            ctx.logging.logger->log(GRIM::Diagnostics::formatToken277Diagnostic(tok277, batch_idx));
            
            // Structured equation logging to CSV
            EQ_LOG("TOKEN_COLLAPSE_WEIGHT_GRADIENT", GRIM::Diagnostics::formatToken277Diagnostic(tok277, batch_idx),
                   static_cast<int>(batch_idx), 0, 0, GRIM::EquationPhase::GRADIENT_CLIP);
            
            // Issue #37: Hidden state analysis - understand WHY grad_W[T].sum is positive
            // Issue #115: Pass use_centering to read correct buffer (centered vs raw)
            GRIM::Diagnostics::HiddenState277Analysis hidden277 = GRIM::Diagnostics::computeHiddenState277Analysis(
                ts, flat_targets, cfg.d_model, cfg.vocab_size, g_collapse_token_id,
                cfg.lm_head_center_hidden_states,  // Issue #115: Use centered buffer when active
                stream);
            ctx.logging.logger->log(GRIM::Diagnostics::formatHiddenState277Analysis(hidden277, batch_idx));
            
            // Structured equation logging to CSV
            EQ_LOG("HIDDEN_STATE_EQUATION", GRIM::Diagnostics::formatHiddenState277Analysis(hidden277, batch_idx),
                   static_cast<int>(batch_idx), 0, 0, GRIM::EquationPhase::GRADIENT_CLIP);
            
            // ========================================================================
            // DIAGNOSTIC: Issue #114 - Feedback Loop Detection (Rule 21 Equation-Based)
            // Tracks the self-reinforcing collapse: logit[T] = ||h|| × ||W[T]|| × cos(h, W[T])
            // Detects: hidden norm explosion, cosine collapse, weight paradox, alignment explosion
            // ========================================================================
            // Issue #115: Pass use_centering to read correct buffer (centered vs raw)
            GRIM::Diagnostics::FeedbackLoopDiagnostic feedback_diag = GRIM::Diagnostics::computeFeedbackLoopDiagnostic(
                ctx.model->getLmHeadLayer(), ts, flat_targets, cfg.d_model, cfg.vocab_size, batch_idx, g_collapse_token_id,
                cfg.lm_head_center_hidden_states,  // Issue #115: Use centered buffer when active
                stream);
            ctx.logging.logger->log(GRIM::Diagnostics::formatFeedbackLoopDiagnostic(feedback_diag, batch_idx));
            
            // Structured equation logging to CSV
            EQ_LOG("FEEDBACK_LOOP_EQUATION", GRIM::Diagnostics::formatFeedbackLoopDiagnostic(feedback_diag, batch_idx),
                   static_cast<int>(batch_idx), 0, 0, GRIM::EquationPhase::GRADIENT_CLIP);

            // ========================================================================
            // DIAGNOSTIC: Issue #134 - PC1 Causality Test (Rule 21 Equation-Based)
            // Estimates top principal component of hidden states via power iteration,
            // projects it out, recomputes logits, measures tracked token deflation and
            // entropy improvement to determine if PC1 is the mode collapse mechanism.
            // NOTE: This is EXPENSIVE (copies full weight matrix + N×V logit recompute
            // at sampled positions). Runs at same frequency as other expensive diagnostics.
            // ========================================================================
            GRIM::Diagnostics::PC1CausalityTest pc1_test = GRIM::Diagnostics::computePC1CausalityTest(
                ctx.model->getLmHeadLayer(), ts, flat_targets, cfg.d_model, cfg.vocab_size, g_collapse_token_id,
                cfg.lm_head_center_hidden_states,
                stream);
            ctx.logging.logger->log(GRIM::Diagnostics::formatPC1CausalityTest(pc1_test, batch_idx));

            // Structured equation logging to CSV
            EQ_LOG("PC1_CAUSALITY_TEST", GRIM::Diagnostics::formatPC1CausalityTest(pc1_test, batch_idx),
                   static_cast<int>(batch_idx), 0, 0, GRIM::EquationPhase::GRADIENT_CLIP);
        }

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
        {
            const float* weights_ptr = ctx.model->getLmHeadLayer()->weights().data;
            const float* grads_ptr = ctx.model->getEmbeddingLayer()->tokenWeights().grad_data();

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
                    content_norm_sum += std::sqrt(sq);
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
                    const float w_norm = static_cast<float>(std::sqrt(w_sq));
                    const float w_mean = static_cast<float>(w_sum / cfg.d_model);

                    // Gradient row (may be null if not yet computed)
                    float g_norm = 0.0f, g_sum = 0.0f;
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
                        g_norm = static_cast<float>(std::sqrt(g_sq));
                        g_sum = static_cast<float>(gs);
                        has_grad = any_nonzero;
                    }

                    // Count appearances as INPUT token in this batch
                    // (special tokens only appear as input if BOS is prepended, etc.)
                    // We don't have input_ids readily available here, so skip input count.

                    diag << "  " << SPECIAL_NAMES[s] << "(id=" << tok_id << "): "
                         << "||W||=" << w_norm
                         << " w_mean=" << w_mean;
                    if (grads_ptr) {
                        diag << " ||grad||=" << g_norm
                             << " grad_sum=" << g_sum
                             << (has_grad ? "" : " [ZERO_GRAD]");
                    } else {
                        diag << " [NO_GRAD_BUFFER]";
                    }

                    // Anomaly: special token weight norm diverging from content tokens
                    if (content_norm_mean > 0.0 && w_norm > 3.0f * content_norm_mean) {
                        diag << " [ANOMALY] ||W||=" << w_norm
                             << " >> content_mean=" << Internal::formatScalar(static_cast<float>(content_norm_mean), 6);
                    }
                    if (w_norm < 1e-6f) {
                        diag << " [ANOMALY] NEAR_ZERO_WEIGHT";
                    }
                    if (!std::isfinite(w_norm) || !std::isfinite(g_norm)) {
                        throw std::runtime_error(
                            "[SPECIAL_TOKEN_EQUATION] Non-finite special token weight/grad: "
                            + std::string(SPECIAL_NAMES[s]) + " ||W||=" + std::to_string(w_norm)
                            + " ||grad||=" + std::to_string(g_norm)
                            + " at batch " + std::to_string(batch_idx + 1));
                    }
                    diag << "\n";
                }
                diag << "  content_baseline: ||W||_mean=" << Internal::formatScalar(static_cast<float>(content_norm_mean), 6)
                     << " (sampled " << content_norm_count << " tokens)";

                ctx.logging.logger->log(diag.str());
                EQ_LOG("SPECIAL_TOKEN_EQUATION", diag.str(),
                       static_cast<int>(batch_idx), 0, 0, GRIM::EquationPhase::GRADIENT_CLIP);
            }
        }
    }
    
    // ========================================================================
    // DIAGNOSTIC: Export gradients for PyTorch comparison (EVERY batch)
    // ========================================================================
    const bool export_grads = false;
    static bool exported_grads = false;
    if (batch_idx < 10 && export_grads ) {  // Export first 10 batches for gradient tracking
        exported_grads = true;
        
        const auto& ts = ctx.model->getTrainingState();
        const auto& cfg = ctx.model->getConfig();
        cudaStream_t stream = ts.stream_ctrl.getPrimaryStream();
        
        auto exportBuffer = [&](const char* name, const float* d_ptr, size_t count) {
            if (!d_ptr) {
                printf("[GradExport] SKIP %s: nullptr\n", name);
                return;
            }
            std::vector<float> h_data(count);
            cudaMemcpyAsync(h_data.data(), d_ptr, count * sizeof(float), 
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            
            // Get GRIM root for gradient dumps directory
            fs::path grim_root = GRIM::Config::resolveGrimRoot();
            fs::path dump_dir = grim_root / "gradient_dumps";
            fs::create_directories(dump_dir);
            std::string path = (dump_dir / (std::string(name) + ".bin")).string();
            FILE* f = fopen(path.c_str(), "wb");
            if (f) {
                fwrite(&count, sizeof(size_t), 1, f);
                fwrite(h_data.data(), sizeof(float), count, f);
                fclose(f);
                
                // Compute norm for logging
                float sq_sum = 0.0f;
                for (size_t i = 0; i < count; ++i) sq_sum += h_data[i] * h_data[i];
                printf("[GradExport] %s: %zu elements, norm=%.6f\n", name, count, sqrtf(sq_sum));
            }
        };
        
        printf("\n========================================\n");
        printf("GRADIENT EXPORT BATCH %d FOR PYTORCH COMPARISON\n", batch_idx);
        printf("========================================\n");
        
        // Embedding grads
        exportBuffer("embedding_grads", ctx.model->getEmbeddingLayer()->tokenWeights().grad_data(), 
                     static_cast<size_t>(cfg.vocab_size) * cfg.d_model);
        
        // Per-layer gradients (via encoder's Tensor.grad per AUTOGRAD MIGRATION)
        auto* gpu_encoder = &ctx.model->getGpuEncoder();
        for (int layer = 0; layer < cfg.num_layers; ++layer) {
            std::string prefix = "layer" + std::to_string(layer) + "_";
            auto* enc = gpu_encoder ? gpu_encoder->getLayer(layer) : nullptr;
            
            if (enc) {
                exportBuffer((prefix + "qkv_grads").c_str(), enc->attnWqkv().grad_data(), enc->attnWqkv().numel());
                exportBuffer((prefix + "wo_grads").c_str(), enc->attnWo().grad_data(), enc->attnWo().numel());
                exportBuffer((prefix + "ffn_w1_grads").c_str(), enc->ffnW1().grad_data(), enc->ffnW1().numel());
                exportBuffer((prefix + "ffn_w2_grads").c_str(), enc->ffnW2().grad_data(), enc->ffnW2().numel());
            } else {
                printf("[GradExport] SKIP layer %d: encoder layer is null\n", layer);
            }
        }
        
        printf("========================================\n");
        printf("Export complete. Run: python compare_gradients_pytorch.py\n");
        printf("========================================\n\n");
        
        // ========================================================================
        // TEXT DUMP: Export gradient values for comparison with PyTorch
        // ========================================================================
        if (batch_idx < 2) {
            fs::path grim_root = GRIM::Config::resolveGrimRoot();
            std::string grad_txt_path = (grim_root / "grim_gradients.txt").string();
            ctx.model->dumpGradientValues(batch_idx + 1, grad_txt_path);
        }
    }
    
    // ========================================================================
    // NUMERICAL GRADIENT CHECK (Finite Difference Verification)
    // ========================================================================
    // Compares analytical gradients from backward pass against numerical gradients
    // computed via central differences: numerical_grad ≈ (L(θ+ε) - L(θ-ε)) / (2ε)
    // Enable via: GRIM_GRAD_CHECK_ENABLED=1 environment variable
    // Control batch count via: GRIM_GRAD_CHECK_MAX_BATCHES=N (default: 3)
    // WARNING: This is EXPENSIVE - adds 2 extra forward passes per checked batch!
    //
    // IMPORTANT: Only run on LAST micro-batch when gradients are fully accumulated
    // ========================================================================
    // Logic: After this batch's backward, increment micro_step. If it reaches accum_steps,
    // this was the last micro-batch and we should run numerical checks.
    const int micro_step_after_backward = ctx.optimizer.current_micro_step + 1;
    const bool is_last_micro_batch = (micro_step_after_backward >= accum_steps);
    
    // DEBUG: Log gradient check decision
    ctx.logging.logger->log("[GradCheckDebug] batch=" + std::to_string(batch_idx + 1) + 
                            " micro_step=" + std::to_string(ctx.optimizer.current_micro_step) + 
                            "/" + std::to_string(accum_steps) + 
                            " is_last=" + (is_last_micro_batch ? "yes" : "no"));
    
    Internal::NumericalGradCheckResult grad_check_result{};
    if (is_last_micro_batch) {
        grad_check_result = Internal::performNumericalGradientCheck(
            ctx, payload, batch_idx);
        
        // DEBUG: Log whether check was performed
        ctx.logging.logger->log("[GradCheckDebug] performed=" + 
                                std::string(grad_check_result.performed ? "yes" : "no"));
    }
    
    if (grad_check_result.performed && !grad_check_result.passed) {
        // Log a warning but don't abort - gradient check failures could be due to
        // numerical precision issues, especially with mixed-precision training
        ctx.logging.logger->log("[NumGradCheck] WARNING: Gradient check failed with " +
                                Internal::formatScalar(grad_check_result.relative_error * 100.0f, 2) +
                                "% relative error. This may indicate a bug in the backward pass.");
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
                            " cached_preclip=" + Internal::formatScalar(state.last_grad_norm));
    
    // === TIMING GUARD: Track expensive operations between POST-BACKWARD logs ===
    auto grad_ops_start = std::chrono::steady_clock::now();
    
    // Compute gradient norm on GPU (syncs every batch for correct clipping)
    auto norm_start = std::chrono::steady_clock::now();
    result.grad_norm = ctx.model->computeGradNorm(true);
    result.normalized_grad_norm = result.grad_norm;  // No longer normalized differently
    const float preclip_grad_norm = result.grad_norm;
    auto norm_elapsed_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - norm_start).count();
    
    // Log gradient component breakdown (always synced)
    // Issue #138: Show wall_time (includes backward drain) AND gpu_kernel_time (actual norm computation)
    // This eliminates the misleading 3ms-53ms variance in the log that is actually backward pipeline drain.
    const float gpu_kernel_ms = ctx.model->lastGpuNormMs();
    const float drain_ms = norm_elapsed_ms - gpu_kernel_ms;
    ctx.logging.logger->log("[GradTrace] POST-BACKWARD synced computeGradNorm in " +
                                Internal::formatScalar(norm_elapsed_ms, 2) + "ms (kernel=" +
                                Internal::formatScalar(gpu_kernel_ms, 2) + "ms drain=" +
                                Internal::formatScalar(drain_ms, 2) + "ms)");

        const auto& ts = ctx.model->getTrainingState();
        if (ts.gradnorm_ctrl.isInitialized()) {
            const auto& gn = ts.gradnorm_ctrl.getHostMetrics();
            if (gn.has_nan || gn.has_inf) {
                std::ostringstream nf_log;
                nf_log << "[GradTrace] NON-FINITE grads detected"
                       << " nan=" << (gn.has_nan ? "true" : "false")
                       << " inf=" << (gn.has_inf ? "true" : "false")
                       << " first_nan_group=" << gn.first_nan_group
                       << " first_nan_value=" << gn.first_nan_value
                       << " first_inf_group=" << gn.first_inf_group
                       << " first_inf_value=" << gn.first_inf_value
                       << " groups_processed=" << gn.groups_processed;
                ctx.logging.logger->log(nf_log.str());
                
                // RULE 20: Fail loud! NaN/Inf in gradients is a critical bug - crash immediately
                throw std::runtime_error("[FATAL] NaN/Inf detected in gradients at batch " + 
                                        std::to_string(batch_idx + 1) + 
                                        " first_nan_group=" + std::to_string(gn.first_nan_group) +
                                        " - investigate the backward pass!");
            }
        }

    std::string comp_log = Internal::formatGradientComponents(ctx.model.get());
    if (!comp_log.empty()) {
        ctx.logging.logger->log(comp_log);
        
        // Sanity check: sum of squares should match total (use 5% relative threshold)
        const auto& gm = ctx.model->gradientMetrics();
        const bool tied = ctx.model->getConfig().tie_embeddings;
        // When tied: only lm_head_norm is populated (embedding_norm=0)
        float computed_total_sq = (tied ? 0.0f : gm.embedding_norm * gm.embedding_norm) +
                                       gm.lm_head_norm * gm.lm_head_norm +
                                       gm.attention_norm * gm.attention_norm +
                                       gm.ffn_norm * gm.ffn_norm +
                                   gm.rmsnorm_norm * gm.rmsnorm_norm +
                                   gm.scratchblock_norm * gm.scratchblock_norm;
        float computed_total = std::sqrt(computed_total_sq);
        float rel_diff = std::abs(computed_total - gm.total_norm);
        float threshold = 0.05f * gm.total_norm;  // 5% relative tolerance
        if (rel_diff > threshold && rel_diff > 0.1f) {  // Also allow tiny absolute diff
            ctx.logging.logger->log("[GradTrace] WARNING: component sum=" + Internal::formatScalar(computed_total) +
                                    " != total=" + Internal::formatScalar(gm.total_norm) +
                                    " diff=" + Internal::formatScalar(rel_diff) +
                                    " threshold=" + Internal::formatScalar(threshold));
        }

        if (logit_trace_enabled) {
            std::ostringstream trace_msg;
            trace_msg << "[LogitTrace][PostBackward] source=grad_metrics"
                      << " tied=" << (tied ? "yes" : "no")
                      << " batch=" << (batch_idx + 1)
                      << " preclip_grad_norm=" << Internal::formatScalar(preclip_grad_norm, 6)
                      << " total_norm=" << Internal::formatScalar(gm.total_norm, 6)
                      << " lm_head_norm=" << Internal::formatScalar(gm.lm_head_norm, 6)
                      << " embedding_norm=" << Internal::formatScalar(gm.embedding_norm, 6)
                      << " attention_norm=" << Internal::formatScalar(gm.attention_norm, 6)
                      << " ffn_norm=" << Internal::formatScalar(gm.ffn_norm, 6)
                      << " rmsnorm_norm=" << Internal::formatScalar(gm.rmsnorm_norm, 6)
                      << " scratchblock_norm=" << Internal::formatScalar(gm.scratchblock_norm, 6);
            ctx.logging.logger->log(trace_msg.str());
        }
    } else {
        ctx.logging.logger->log("[GradTrace] WARNING: grad_metrics not ready after computeGradNorm!");
    }
    
    ctx.logging.logger->log("[GradTrace] POST-GRADNORM preclip=" + Internal::formatScalar(preclip_grad_norm));
    
    // ========================================================================
    // DIAGNOSTIC: [EMB_GRAD_EQUATION] Embedding gradient spike analysis (Issue #141)
    // Rule 21 equation-based logging for tied-weight gradient decomposition.
    // Runs every diag_interval batches (same cadence as Token277 diagnostics).
    // Identifies which token rows concentrate gradient mass and whether
    // atomicAdd scatter density correlates with spike magnitude.
    // ========================================================================
    {
        static float prev_emb_norm_for_spike_diag = 0.0f;
        
        // Gate to same interval as other expensive diagnostics
        static int emb_grad_diag_interval = 10;
        const bool kEmbGradDiagEnabled = GRIM::getEquationLogger().isEnabled() &&
            (batch_idx == 0 || (batch_idx + 1) % std::max(emb_grad_diag_interval, 1) == 0);
        
        const auto& gm = ctx.model->gradientMetrics();
        const float curr_emb_norm = gm.lm_head_norm;  // tied weights: lm_head_norm IS emb grad norm
        
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
                    prev_emb_norm_for_spike_diag, curr_emb_norm,
                    stream);
                
                std::string emb_eq_str = GRIM::Diagnostics::formatEmbGradEquation(emb_diag, batch_idx);
                ctx.logging.logger->log(emb_eq_str);
                
                EQ_LOG("EMB_GRAD_EQUATION", emb_eq_str,
                       static_cast<int>(batch_idx), 0, 0, GRIM::EquationPhase::GRADIENT_CLIP);
            }
        }
        
        prev_emb_norm_for_spike_diag = curr_emb_norm;
    }
    
    // === TIMING GUARD: Track operations between POST-BACKWARD and PRE-OPTIMIZER ===
    auto pre_optimizer_start = std::chrono::steady_clock::now();
    
    // Handle gradient spikes
    auto spike_start = std::chrono::steady_clock::now();
    if (Internal::handleGradientSpike(ctx, state, batch, preclip_grad_norm, preclip_grad_norm, 
                                      result.loss, clip_selection, batch_idx)) {
        // CRITICAL: Zero gradients before reset - bad gradients must not persist
        ctx.model->zeroGrad();
        ctx.optimizer.current_micro_step = 0;  // Reset accumulation window
        ctx.model->getTrainingState().autograd_intermediates.clear();
        result.skipped = true;
        result.skip_reason = "gradient_spike";
        ctx.global_step++;
        return result;
    }
    auto spike_elapsed_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - spike_start).count();
    
    // === TELEMETRY CONTROL (GPU-NATIVE) ===
    auto telemetry_start = std::chrono::steady_clock::now();
    
    // PERFORMANCE: Only evaluate telemetry every 10 batches (requires sync for D2H transfer)
    // Use cached decision for intermediate batches to avoid sync stalls
    static GRIM::Telemetry::ControlDecision cached_telemetry_decision = []() {
        GRIM::Telemetry::ControlDecision d;
        d.grad_scale_factor = 1.0f;  // Neutral scale (no intervention)
        d.action = GRIM::Telemetry::ControlAction::Continue;
        return d;
    }();
    static int telemetry_counter = 0;
    const bool telemetry_control_enabled = ctx.config.hyperparameters.telemetry_control_enabled;
    const bool telemetry_control_active = telemetry_control_enabled &&
        ctx.telemetry.enabled &&
        ctx.telemetry.controller &&
        ctx.telemetry.controller->isInitialized();
    const bool should_eval_telemetry = telemetry_control_active && (telemetry_counter == 0);
    telemetry_counter = (telemetry_counter + 1) % 10;
    
    GRIM::Telemetry::ControlDecision telemetry_decision = cached_telemetry_decision;  // Use cached by default
    if (!telemetry_control_active) {
        // Monitoring-only mode: disable all control actions/scaling regardless of cached state.
        telemetry_decision = GRIM::Telemetry::ControlDecision{};
        telemetry_decision.grad_scale_factor = 1.0f;
        telemetry_decision.action = GRIM::Telemetry::ControlAction::Continue;
        cached_telemetry_decision = telemetry_decision;
    }
    
    if (telemetry_control_active && should_eval_telemetry) {
        // Compute average sequence length for this batch
        float avg_seq_len = token_stats.batch_size > 0 
            ? static_cast<float>(token_stats.total_tokens) / static_cast<float>(token_stats.batch_size)
            : 0.0f;
        
        // GPU-native decision: single kernel reads telemetry + computes action
        // Only 48-byte D2H transfer at end (requires sync - that's why we only do this every 10 batches)
        telemetry_decision = ctx.telemetry.controller->evaluate(
            ctx.telemetry.lattice,
            result.grad_norm,             // raw gradient norm (computed after backward)
            result.loss,                  // current batch loss
            static_cast<int>(token_stats.total_tokens),  // valid tokens in batch
            avg_seq_len,                  // average sequence length
            ctx.global_step,              // training step
            ctx.model->getTrainingState().stream_ctrl.getPrimaryStream()  // Rule 22: no cached streams
        );
        
        cached_telemetry_decision = telemetry_decision;  // Cache for next 9 batches
        
        // Log decision (every 10 steps or on important actions)
        if (ctx.global_step % 10 == 0 || 
            telemetry_decision.action != GRIM::Telemetry::ControlAction::Continue ||
            telemetry_decision.spike_severity != GRIM::Telemetry::SpikeSeverity::None) {
            std::string desc = ctx.telemetry.controller->describeDecision(telemetry_decision);
            ctx.logging.logger->log("[TelemetryControl] batch=" + std::to_string(batch_idx + 1) + " " + desc);
        }
        
        // Check for fatal conditions
        if (telemetry_decision.action == GRIM::Telemetry::ControlAction::FatalError) {
            throw std::runtime_error("FATAL: Telemetry control detected unrecoverable state (accumulation bug or state corruption)");
        }
    }
    
    auto telemetry_elapsed_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - telemetry_start).count();
    const bool allow_telemetry_actions = telemetry_control_active && should_eval_telemetry;
    GRIM::Telemetry::ControlAction telemetry_action = telemetry_decision.action;
    if (!allow_telemetry_actions && telemetry_action != GRIM::Telemetry::ControlAction::Continue) {
        // Cached decisions are for scaling only; one-shot actions must execute once per eval.
        telemetry_action = GRIM::Telemetry::ControlAction::Continue;
        telemetry_decision.cooldown_extension = 0;
    }
    
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
    
    // Compute clip parameters now (telemetry scale, enabled flag) but defer actual clipping
    const float telemetry_scale = telemetry_control_active ? telemetry_decision.grad_scale_factor : 1.0f;
    const float safe_scale_factor = std::max(telemetry_scale, 0.01f);
    const float effective_per_token_limit = clip_selection.per_token_limit * safe_scale_factor;
    const bool clipping_enabled = (hp.grad_clip_norm > 0.0f);
    
    // No clipping here — deferred to post-accumulation inside should_step
    auto clipping_elapsed_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - clipping_start).count();
    
    // Learning rate computation (accum_steps already computed above)
    auto lr_start = std::chrono::steady_clock::now();
    const float scheduled_lr = Internal::getScheduledLearningRate(
        ctx.global_step, hp.learning_rate, hp.warmup_steps, 
        ctx.config.stability.enabled);
    
    result.learning_rate = scheduled_lr;
    
    // Dynamic LR adjustment - pass scheduled LR as ceiling to respect cosine decay
    if (hp.dynamic_lr_enabled && !ctx.config.stability.enabled) {
        ctx.optimizer.dynamic_lr_controller.setBaseLearningRate(scheduled_lr);
        if (ctx.global_step >= hp.dynamic_lr_warmup_steps) {
            result.learning_rate = ctx.optimizer.dynamic_lr_controller.update(
                result.normalized_grad_norm, result.loss, scheduled_lr);
        } else {
            ctx.optimizer.dynamic_lr_controller.update(
                result.normalized_grad_norm, result.loss, scheduled_lr);
        }
    }
    
    // Spike cooldown handling
    if (state.grad_spike_cooldown > 0 && !ctx.config.stability.enabled) {
        state.grad_spike_cooldown--;
        const float floor_lr = std::max(hp.dynamic_lr_min, ctx.config.stability.lr_min);
        const float spike_cap_lr = std::max(floor_lr, scheduled_lr * kGradSpikeLrFraction);
        result.learning_rate = std::min(result.learning_rate, spike_cap_lr);
    }
    
    // === TELEMETRY CONTROL ACTIONS ===
    // GPU kernel already made all decisions - just execute them
    switch (telemetry_action) {
        case GRIM::Telemetry::ControlAction::SkipStep:
            // Skip optimizer step entirely - reset accumulation window
            ctx.model->zeroGrad();
            ctx.optimizer.current_micro_step = 0;
            ctx.model->getTrainingState().autograd_intermediates.clear();
            
            result.skipped = true;
            result.skip_reason = "telemetry_control_skip";
            ctx.logging.logger->log("[TelemetryControl] SKIP_STEP batch=" + std::to_string(batch_idx + 1) + " (reset accumulation window)");
            return result;
            
        case GRIM::Telemetry::ControlAction::ExtendCooldown:
            state.grad_spike_cooldown += telemetry_decision.cooldown_extension;
            ctx.logging.logger->log("[TelemetryControl] EXTEND_COOLDOWN +" + std::to_string(telemetry_decision.cooldown_extension) + " steps");
            break;
            
        case GRIM::Telemetry::ControlAction::TriggerSoftRestart:
            // Scale momentum instead of zeroing - smoother adaptation to regime changes
            // Telemetry already computed optimal damping factor (volatility_damping * decay_boost)
            ctx.logging.logger->log("[TelemetryControl] MOMENTUM_DAMPING: scaling by " + 
                                   std::to_string(telemetry_decision.grad_scale_factor));
            GRIM::SoftRestart::scaleOptimizerMoments(ctx.model.get(), telemetry_decision.grad_scale_factor);
            
            // Reset telemetry anchors to prevent repeated drift triggers (Rule 22: explicit API call)
            if (ctx.telemetry.enabled && ctx.telemetry.lattice) {
                GRIM::Telemetry::TelemetryError err = GRIM::Telemetry::resetTelemetryAnchors(
                    ctx.telemetry.lattice,
                    ctx.model->getTrainingState().stream_ctrl.getPrimaryStream()
                );
                if (err != GRIM::Telemetry::TelemetryError::OK) {
                    ctx.logging.logger->log("[TelemetryControl] WARNING: Failed to reset anchors after momentum damping");
                }
            }
            break;
            
        case GRIM::Telemetry::ControlAction::InjectPlateauNoise:
            {
                // Inject Gaussian noise into weights to escape plateau
                ctx.logging.logger->log("[TelemetryControl] ═════════════════════════════════════════════════");
                ctx.logging.logger->log("[TelemetryControl] PLATEAU DETECTED - Injecting noise to escape local minimum");
                
                auto& training_state = ctx.model->getTrainingState();
                cudaStream_t stream = training_state.stream_ctrl.getPrimaryStream();
                const auto& cfg = ctx.model->getConfig();
                const auto& tc_cfg = ctx.telemetry.control_config;
                
                // Generate reproducible seed from global step
                uint64_t noise_seed = static_cast<uint64_t>(ctx.global_step) * 1099511628211ull;
                
                size_t total_params_perturbed = 0;
                
                // Inject into LM head weights (always accessible via LMHeadLayer)
                if (ctx.model->getLmHeadLayer()->weights().data) {
                    size_t lm_size = static_cast<size_t>(cfg.vocab_size) * cfg.d_model;
                    GRIM::Telemetry::launchPlateauNoiseInjection(
                        ctx.model->getLmHeadLayer()->weights().data,
                        lm_size,
                        tc_cfg.plateau_noise_std,
                        tc_cfg.plateau_noise_proportional,
                        noise_seed,
                        stream
                    );
                    total_params_perturbed += lm_size;
                    ctx.logging.logger->log("[TelemetryControl] Injected noise into lm_head_weights (" + 
                                           std::to_string(lm_size) + " params)");
                }
                
                // Inject into LM head bias if present
                if (ctx.model->getLmHeadLayer()->bias().data) {
                    GRIM::Telemetry::launchPlateauNoiseInjection(
                        ctx.model->getLmHeadLayer()->bias().data,
                        static_cast<size_t>(cfg.vocab_size),
                        tc_cfg.plateau_noise_std,
                        tc_cfg.plateau_noise_proportional,
                        noise_seed + 1,
                        stream
                    );
                    total_params_perturbed += cfg.vocab_size;
                }
                
                // Sync to ensure noise injection completes
                cudaStreamSynchronize(stream);
                
                ctx.logging.logger->log("[TelemetryControl] Total parameters perturbed: " + std::to_string(total_params_perturbed));
                ctx.logging.logger->log("[TelemetryControl] noise_std=" + std::to_string(tc_cfg.plateau_noise_std) +
                                       " proportional=" + std::string(tc_cfg.plateau_noise_proportional ? "true" : "false"));
                ctx.logging.logger->log("[TelemetryControl] ═════════════════════════════════════════════════");
                
                // Also scale momentum to prevent optimizer from immediately undoing the noise
                GRIM::SoftRestart::scaleOptimizerMoments(ctx.model.get(), 0.5f);
                ctx.logging.logger->log("[TelemetryControl] Scaled optimizer momentum by 0.5 to preserve perturbation");
                
                // Reset telemetry anchors
                if (ctx.telemetry.enabled && ctx.telemetry.lattice) {
                    GRIM::Telemetry::resetTelemetryAnchors(
                        ctx.telemetry.lattice,
                        stream
                    );
                }
            }
            break;
            
        default:
            break;
    }
    
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
                                "ms (spike=" + Internal::formatScalar(spike_elapsed_ms, 2) + 
                                "ms, telemetry=" + Internal::formatScalar(telemetry_elapsed_ms, 2) + 
                                "ms, clipping=" + Internal::formatScalar(clipping_elapsed_ms, 2) + 
                                "ms, lr=" + Internal::formatScalar(lr_elapsed_ms, 2) + 
                                "ms, sample=" + Internal::formatScalar(sample_elapsed_ms, 2) + "ms)");
    }
    
    ctx.logging.logger->log("[GradTrace] PRE-OPTIMIZER batch=" + std::to_string(batch_idx + 1) + 
                            " lr=" + Internal::formatScalar(result.learning_rate, 8) +
                            " grad_norm=" + Internal::formatScalar(result.grad_norm) +
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
        // Issue #135: POST-ACCUMULATION gradient clipping
        //
        // Clipping runs ONCE on the fully accumulated + scaled gradients.
        // Recompute grad norm since accum_scale changed magnitudes.
        // Per-component clipping (Issue #139):
        //   1. emb_clip  — LM_HEAD (+ EMBEDDING if untied)
        //   2. enc_clip  — ATTENTION + FFN + RMSNORM + SCRATCHBLOCK (encoder params)
        // ========================================================================
        if (clipping_enabled) {
            clipping_start = std::chrono::steady_clock::now();
            
            // Recompute norm on the accumulated+scaled gradients
            result.grad_norm = ctx.model->computeGradNorm(true);
            result.normalized_grad_norm = result.grad_norm;
            const float post_accum_norm = result.grad_norm;
            
            {
                const auto& gm = ctx.model->gradientMetrics();
                
                // emb_norm: LM_HEAD norm (includes tied embedding when tie_embeddings=true)
                float emb_norm_sq = gm.lm_head_norm * gm.lm_head_norm;
                if (!ctx.model->getConfig().tie_embeddings) {
                    emb_norm_sq += gm.embedding_norm * gm.embedding_norm;
                }
                const float emb_norm = std::sqrt(emb_norm_sq);
                
                // encoder_norm: everything that's NOT emb/lm_head
                const float enc_norm_sq = gm.attention_norm * gm.attention_norm
                                        + gm.ffn_norm * gm.ffn_norm
                                        + gm.rmsnorm_norm * gm.rmsnorm_norm
                                        + gm.scratchblock_norm * gm.scratchblock_norm;
                const float enc_norm = std::sqrt(enc_norm_sq);
                
                bool any_clipped = false;
                
                // Clip embedding/LM head independently
                if (emb_norm > effective_per_token_limit) {
                    const float emb_clip_coef = effective_per_token_limit / (emb_norm + 1e-8f);
                    ctx.model->scaleGradientsByType(emb_clip_coef, GRIM::ParamGroupType::LM_HEAD);
                    if (!ctx.model->getConfig().tie_embeddings) {
                        ctx.model->scaleGradientsByType(emb_clip_coef, GRIM::ParamGroupType::EMBEDDING);
                    }
                    any_clipped = true;
                }
                
                // Clip encoder components independently
                if (enc_norm > effective_per_token_limit) {
                    const float enc_clip_coef = effective_per_token_limit / (enc_norm + 1e-8f);
                    ctx.model->scaleGradientsByType(enc_clip_coef, GRIM::ParamGroupType::ATTENTION);
                    ctx.model->scaleGradientsByType(enc_clip_coef, GRIM::ParamGroupType::FFN);
                    ctx.model->scaleGradientsByType(enc_clip_coef, GRIM::ParamGroupType::RMSNORM);
                    ctx.model->scaleGradientsByType(enc_clip_coef, GRIM::ParamGroupType::SCRATCHBLOCK);
                    any_clipped = true;
                }
                
                // Recompute post-clip total norm
                const float clipped_emb = std::min(emb_norm, effective_per_token_limit);
                const float clipped_enc = std::min(enc_norm, effective_per_token_limit);
                result.grad_norm = std::sqrt(clipped_emb * clipped_emb 
                                           + clipped_enc * clipped_enc);
                result.normalized_grad_norm = result.grad_norm;
                result.gradient_clipped = any_clipped;
                
                ctx.logging.logger->log("[PostAccumClip] batch=" + std::to_string(batch_idx + 1) +
                                        " post_accum_norm=" + Internal::formatScalar(post_accum_norm, 6) +
                                        " emb_norm=" + Internal::formatScalar(emb_norm, 6) +
                                        " enc_norm=" + Internal::formatScalar(enc_norm, 6) +
                                        " emb_clipped=" + (emb_norm > effective_per_token_limit ? "YES" : "NO") +
                                        " enc_clipped=" + (enc_norm > effective_per_token_limit ? "YES" : "NO") +
                                        " post_clip_total=" + Internal::formatScalar(result.grad_norm, 6));
            }
            
            ctx.model->recordGradientClip(effective_per_token_limit, result.gradient_clipped);
            clipping_elapsed_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - clipping_start).count();
        }
        
        // ========================================================================
        // PRE-OPTIMIZER Token 277 Weight Snapshot (Issue #36.5)
        // Log weight row 277 BEFORE optimizer step to track delta
        // ========================================================================
        float pre_w277_norm = 0.0f;
        float pre_w277_mean = 0.0f;
        {
            const auto& ts = ctx.model->getTrainingState();
            const auto& cfg = ctx.model->getConfig();
            cudaStream_t stream = ts.stream_ctrl.getPrimaryStream();
            
            if (ctx.model->getLmHeadLayer()->weights().data && g_collapse_token_id < cfg.vocab_size) {
                std::vector<float> w277_row(static_cast<size_t>(cfg.d_model));
                const float* w277_ptr = ctx.model->getLmHeadLayer()->weights().data + static_cast<size_t>(g_collapse_token_id) * cfg.d_model;
                cudaMemcpyAsync(w277_row.data(), w277_ptr, 
                                static_cast<size_t>(cfg.d_model) * sizeof(float),
                                cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                
                float sum = 0.0f, sq_sum = 0.0f;
                for (size_t i = 0; i < w277_row.size(); ++i) {
                    sum += w277_row[i];
                    sq_sum += w277_row[i] * w277_row[i];
                }
                pre_w277_norm = std::sqrt(sq_sum);
                pre_w277_mean = sum / static_cast<float>(cfg.d_model);
            }
        }
        
        ctx.model->updateWeights(result.learning_rate,
                                 &ctx.optimizer.optimizer_state,
                                 ctx.config.hyperparameters.weight_decay);
        // Reset micro_step counter after optimizer step completes
        ctx.optimizer.current_micro_step = 0;

        // Periodic equation log flush (every 10 optimizer steps)
        // ISSUE FIX: Must call flushAsync() FIRST to copy device buffer to host,
        // THEN flushSync() to process and write the data
        if (ctx.optimizer.optimizer_state.step % 10 == 0) {
            GRIM::getEquationLogger().flushAsync();  // Device→Host async copy
            GRIM::getEquationLogger().flushSync();   // Wait, process, write to file
        }

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
        
        // ========================================================================
        // POST-OPTIMIZER Token 277 Weight Delta (Issue #36.5)
        // Track how much weight row 277 changed after optimizer step
        // ========================================================================
        {
            const auto& ts = ctx.model->getTrainingState();
            const auto& cfg = ctx.model->getConfig();
            cudaStream_t stream = ts.stream_ctrl.getPrimaryStream();
            
            if (ctx.model->getLmHeadLayer()->weights().data && g_collapse_token_id < cfg.vocab_size) {
                std::vector<float> w277_row(static_cast<size_t>(cfg.d_model));
                const float* w277_ptr = ctx.model->getLmHeadLayer()->weights().data + static_cast<size_t>(g_collapse_token_id) * cfg.d_model;
                cudaMemcpyAsync(w277_row.data(), w277_ptr, 
                                static_cast<size_t>(cfg.d_model) * sizeof(float),
                                cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                
                float sum = 0.0f, sq_sum = 0.0f;
                for (size_t i = 0; i < w277_row.size(); ++i) {
                    sum += w277_row[i];
                    sq_sum += w277_row[i] * w277_row[i];
                }
                float post_w277_norm = std::sqrt(sq_sum);
                float post_w277_mean = sum / static_cast<float>(cfg.d_model);
                
                float delta_norm = post_w277_norm - pre_w277_norm;
                float delta_mean = post_w277_mean - pre_w277_mean;
                
                std::ostringstream msg;
                msg << std::fixed << std::setprecision(8);
                msg << "[Token277] POST-OPT batch=" << (batch_idx + 1);
                msg << " pre_norm=" << pre_w277_norm;
                msg << " post_norm=" << post_w277_norm;
                msg << " delta_norm=" << delta_norm;
                msg << " delta_mean=" << delta_mean;
                
                // WARNING FLAGS
                if (delta_norm > 0.0001f) {
                    msg << " ⚠️ NORM_INCREASED";  // Weight row getting larger!
                }
                if (delta_mean > 0.0001f) {
                    msg << " ⚠️ MEAN_INCREASED";  // Weight values drifting positive!
                }
                
                ctx.logging.logger->log(msg.str());
            }
        }
        
        if (pre_sample.valid && post_sample.valid) {
            const float update_rms = GRIM::Diagnostics::computeUpdateRms(pre_sample, post_sample);
            const std::string update_msg = "[UpdateMag] batch=" + std::to_string(batch_idx + 1) +
                                           " update_rms=" + Internal::formatScalar(update_rms, 8) +
                                           " param_rms=" + Internal::formatScalar(pre_sample.rms, 8);
            ctx.logging.logger->log(update_msg);
            EmitModuleInfo(ModuleId::Optimizer, update_msg, ctx.global_step);
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
        result.grad_norm,                   // primary: gradient norm
        result.normalized_grad_norm,        // secondary: per-token norm
        "grad_norm",
        "post_backward");
    
    // Update telemetry lattice (GPU-resident, fail loud)
    // YAGNI: Only track grad_norm (μ magnitude) to avoid memory bloat
    // Single stream with 10 internal dimensions (μ, σ_tilde, v_σ, etc.)
    if (ctx.telemetry.lattice && ctx.telemetry.enabled) {
        // ISSUE #14 FIX: Use PRE-CLIP gradient norm for telemetry baseline.
        // Before: used result.grad_norm (post-clip, always ~1.0 due to clipping)
        // This caused spike detection to think normal gradients (3-12) were 10x+ spikes
        // relative to the clamped 1.0 baseline, skipping 62% of batches!
        // After: use preclip_grad_norm so baseline reflects actual gradient magnitude
        
        // Guard against NaN/Inf BEFORE passing to telemetry (fail loud with diagnostic)
        if (!std::isfinite(preclip_grad_norm)) {
            throw std::runtime_error(
                "FATAL: Gradient norm is " + std::string(std::isnan(preclip_grad_norm) ? "NaN" : "Inf") +
                " at batch " + std::to_string(batch_idx + 1) + " step " + std::to_string(ctx.global_step) +
                " loss=" + std::to_string(result.loss) +
                " - indicates gradient explosion or numerical instability in backward pass");
        }
        
        ctx.logging.logger->log("[TelemetryLattice] PRE-UPDATE batch=" + std::to_string(batch_idx + 1) + 
                                " step=" + std::to_string(ctx.global_step) + 
                                " grad_norm=" + Internal::formatScalar(preclip_grad_norm, 6));
        
        float observations[5] = {
            result.loss,            // Stream 0: LOSS
            preclip_grad_norm,      // Stream 1: GRAD_NORM_MEAN (PRE-CLIP total norm - Issue #14 fix)
            preclip_grad_norm,      // Stream 2: GRAD_NORM_MAX (use same as mean for now)
            result.learning_rate,   // Stream 3: LEARNING_RATE
            static_cast<float>(result.tokens_processed)  // Stream 4: TOKENS_PER_BATCH
        };
        
        GRIM::Telemetry::TelemetryError tel_err = GRIM::Telemetry::updateTelemetryLattice(
            ctx.telemetry.lattice, observations, ctx.global_step);
        
        ctx.logging.logger->log("[TelemetryLattice] POST-UPDATE batch=" + std::to_string(batch_idx + 1) + 
                                " step=" + std::to_string(ctx.global_step) + 
                                " error_code=" + std::to_string(static_cast<int>(tel_err)));
        
        if (tel_err != GRIM::Telemetry::TelemetryError::OK) {
            // FATAL - telemetry detected NaN/Inf
            throw std::runtime_error(
                std::string("FATAL Telemetry: ") + 
                GRIM::Telemetry::getTelemetryErrorMessage(tel_err));
        }
    }
    
    ctx.global_step++;
    state.last_grad_norm = result.grad_norm;
    
    // Log update probe for QKV weights (if configured)
    if (ctx.model->hasUpdateProbe()) {
        const auto& probe = ctx.model->updateProbe();
        const bool should_log_probe = (ctx.global_step <= 5) || (ctx.global_step % 10 == 0);
        if (should_log_probe) {
            // Log aggregate metrics
            std::ostringstream probe_msg;
            probe_msg << "[UpdateProbe] group=" << probe.group_name
                      << " upd_rms=" << std::fixed << std::setprecision(6) << probe.update_rms
                      << " grad_rms=" << std::fixed << std::setprecision(6) << probe.grad_rms
                      << " param_rms=" << std::fixed << std::setprecision(6) << probe.parameter_rms
                      << " rel=" << std::fixed << std::setprecision(5) << probe.relative_update
                      << " max_abs=" << std::scientific << std::setprecision(3) << probe.max_abs_update
                      << " lr=" << std::scientific << std::setprecision(6) << probe.learning_rate
                      << " step=" << probe.optimizer_step
                      << " n=" << probe.sample_size;
            ctx.logging.logger->log(probe_msg.str());
            
            // Log first 10 weight values (before/after/gradient)
            const auto& w_before = ctx.model->updateProbeWeightsBefore();
            const auto& w_after = ctx.model->updateProbeWeightsAfter();
            const auto& w_grad = ctx.model->updateProbeGradSample();
            
            if (!w_before.empty() && !w_after.empty() && !w_grad.empty()) {
                const size_t n = std::min<size_t>(10, w_before.size());
                
                std::ostringstream before_msg, after_msg, grad_msg, delta_msg;
                before_msg << "[UpdateProbe] weights_before[0:" << n << "]=[";
                after_msg << "[UpdateProbe] weights_after[0:" << n << "]=[";
                grad_msg << "[UpdateProbe] gradients[0:" << n << "]=[";
                delta_msg << "[UpdateProbe] deltas[0:" << n << "]=[";
                
                for (size_t i = 0; i < n; ++i) {
                    if (i > 0) {
                        before_msg << ",";
                        after_msg << ",";
                        grad_msg << ",";
                        delta_msg << ",";
                    }
                    before_msg << std::fixed << std::setprecision(6) << w_before[i];
                    after_msg << std::fixed << std::setprecision(6) << w_after[i];
                    grad_msg << std::scientific << std::setprecision(3) << w_grad[i];
                    delta_msg << std::scientific << std::setprecision(3) << (w_after[i] - w_before[i]);
                }
                
                before_msg << "]";
                after_msg << "]";
                grad_msg << "]";
                delta_msg << "]";
                
                ctx.logging.logger->log(before_msg.str());
                ctx.logging.logger->log(after_msg.str());
                ctx.logging.logger->log(grad_msg.str());
                ctx.logging.logger->log(delta_msg.str());
            }
        }
        if (logit_trace_enabled) {
            std::ostringstream trace_msg;
            trace_msg << "[LogitTrace][PostOptimizer] source=update_probe"
                      << " batch=" << (batch_idx + 1)
                      << " opt_step=" << ctx.optimizer.optimizer_state.step
                      << " group=" << probe.group_name
                      << " upd_rms=" << std::fixed << std::setprecision(6) << probe.update_rms
                      << " grad_rms=" << std::fixed << std::setprecision(6) << probe.grad_rms
                      << " param_rms=" << std::fixed << std::setprecision(6) << probe.parameter_rms
                      << " rel=" << std::fixed << std::setprecision(5) << probe.relative_update
                      << " max_abs=" << std::scientific << std::setprecision(3) << probe.max_abs_update
                      << " lr=" << std::scientific << std::setprecision(6) << probe.learning_rate
                      << " n=" << probe.sample_size;
            ctx.logging.logger->log(trace_msg.str());
        }
        ctx.model->clearUpdateProbeFlag();
    }
    
    // Free autograd intermediate tensors (logits, encoder outputs, etc.)
    // Must happen AFTER all diagnostics that read from intermediates
    ctx.model->getTrainingState().autograd_intermediates.clear();
    
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
    // BUG FIX: Must pass stream to ResetGuessCache - nullptr causes cudaMemsetAsync crash!
    if (ctx.config.hyperparameters.guess_aux_enabled &&
        state.guess_cache_ready && !state.guess_cache_faulted) {
        cudaStream_t primary_stream = ctx.model->getTrainingState().stream_ctrl.getPrimaryStream();
        GRIMTS::ResetGuessCache(primary_stream);
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
    
    // Process batches
    for (int batch_idx = 0; batch_idx < total_batches_to_run; ++batch_idx) {
        const auto& batch = schedule.batches[single_batch_overfit ? 0 : batch_idx];
        
        // Log progress periodically
        if (batch_idx % 5 == 0) {
            std::ostringstream msg;
            msg << "[Batch " << (batch_idx + 1) << "/" << total_batches_to_run << "] "
                << "size=" << batch.seq_ids.size()
                << " len=" << batch.min_seq_len << "-" << batch.max_seq_len
                << " eff=" << static_cast<int>(batch.packing_efficiency * 100) << "%"
                << " accum_steps=" << accum_steps;
            ctx.logging.logger->log(msg.str());
        }
        
        // BUG FIX Issue #28 (Jan 11, 2026): DOUBLE GRADIENT BUG - SAME BATCH PROCESSED TWICE!
        // The previous code had an inner loop: for (int micro_idx = 0; micro_idx < accum_steps; ++micro_idx)
        // that called processBatch() accum_steps times with THE SAME BATCH!
        // This caused each batch to be processed twice, doubling gradients and leading to loss explosion.
        //
        // CORRECT BEHAVIOR: Each batch is processed ONCE. The current_micro_step counter
        // tracks how many batches have been processed and decides when to run the optimizer 
        // step after accum_steps DIFFERENT batches have been processed.
        //
        // The inner loop was WRONG because:
        //   - With accum_steps=2, batch 0 was processed twice → gradients for batch 0 doubled
        //   - Then batch 1 was processed twice → gradients for batch 1 doubled
        //   - Each "optimizer step" used gradients from ONE batch (counted twice), not TWO batches
        //
        // NOW: Process each batch once. After accum_steps consecutive batches (e.g., batch 0 then batch 1),
        // the optimizer step runs with gradients accumulated from DIFFERENT batches as intended.
        
        BatchResult batch_result = processBatch(ctx, state, batch, batch_idx, total_batches_to_run, epoch_idx);
        
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
            if (ctx.config.hyperparameters.guess_aux_enabled &&
                state.guess_cache_ready && !state.guess_cache_faulted) {
                const GRIMTS::GuessCacheTelemetry telemetry = GRIMTS::GetCacheTelemetry(false);
                std::ostringstream cache_msg;
                cache_msg << "Telemetry: fill=" << (telemetry.fill_ratio * 100.0f) << "%"
                          << " records=" << telemetry.total_records
                          << " hits=" << telemetry.trends.total_hits
                          << " misses=" << telemetry.trends.total_misses
                          << " health=" << (telemetry.is_healthy ? "OK" : "DEGRADED");
                EmitModuleInfo(ModuleId::GuessCache, cache_msg.str(), ctx.global_step);
            }
        }

        logInferenceSample(ctx, state);
        
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
        
        // Micro-validation
        Internal::maybeRunMicroValidation(ctx, state, batch_result.loss, batch_result.learning_rate);
    }
    
    result.avg_loss = epoch_loss / std::max(result.batches_processed, 1);
    result.duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - epoch_start);
    
    ctx.logging.logger->log("[Epoch " + std::to_string(epoch_idx + 1) + "] " +
                            Internal::formatMetric("avg_loss", result.avg_loss));
    
    // Log telemetry vectors (multi-scale summary)
    if (ctx.telemetry.lattice && ctx.telemetry.enabled) {
        ctx.logging.logger->log("========== TELEMETRY SUMMARY ==========");
        
        // Log level 0 (fast, every step)
        GRIM::Telemetry::TelemetryVector vec0_loss, vec0_grad;
        GRIM::Telemetry::readTelemetryVector(ctx.telemetry.lattice, 0, 
                                             (int)GRIM::Telemetry::MetricStream::LOSS, &vec0_loss);
        GRIM::Telemetry::readTelemetryVector(ctx.telemetry.lattice, 0, 
                                             (int)GRIM::Telemetry::MetricStream::GRAD_NORM_MEAN, &vec0_grad);
        
        std::ostringstream oss;
        oss << "[Tel-L0] LOSS: μ=" << vec0_loss.mu << " σ̃=" << vec0_loss.sigma_tilde 
            << " v_σ=" << vec0_loss.v_sigma << " Δ̄=" << vec0_loss.delta_bar 
            << " p=" << vec0_loss.p << " r_out=" << vec0_loss.r_out;
        ctx.logging.logger->log(oss.str());
        
        oss.str("");
        oss << "[Tel-L0] GRAD: μ=" << vec0_grad.mu << " σ̃=" << vec0_grad.sigma_tilde 
            << " v_σ=" << vec0_grad.v_sigma << " δμ=" << vec0_grad.delta_mu 
            << " δσ=" << vec0_grad.delta_sigma;
        ctx.logging.logger->log(oss.str());
        
        // Log level 2 (medium scale, stride=4)
        GRIM::Telemetry::TelemetryVector vec2_loss;
        GRIM::Telemetry::readTelemetryVector(ctx.telemetry.lattice, 2, 
                                             (int)GRIM::Telemetry::MetricStream::LOSS, &vec2_loss);
        
        oss.str("");
        oss << "[Tel-L2] LOSS (s=4): μ=" << vec2_loss.mu << " σ̃=" << vec2_loss.sigma_tilde 
            << " Δ̄=" << vec2_loss.delta_bar << " δμ=" << vec2_loss.delta_mu;
        ctx.logging.logger->log(oss.str());
        
        ctx.logging.logger->log("========================================");
    }
    
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
    std::unique_ptr<GuessCacheScope> guess_cache_scope;
    PHASE2_DEBUG_STDERR("[DEBUG] guess_aux_enabled=%d\n", ctx.config.hyperparameters.guess_aux_enabled ? 1 : 0);
    if (ctx.config.hyperparameters.guess_aux_enabled) {
        PHASE2_DEBUG_STDERR("[DEBUG] Attempting to create GuessCacheScope...\n");
        auto& training_state = ctx.model->getTrainingState();
        guess_cache_scope = std::make_unique<GuessCacheScope>(
            training_state,
            kDefaultGuessCacheCapacity,
            !ctx.config.cuda_exec.single_stream_mode);
        state.guess_cache_ready = guess_cache_scope->active();
        PHASE2_DEBUG_STDERR("[DEBUG] GuessCacheScope created, active=%d\n", state.guess_cache_ready ? 1 : 0);
    } else {
        state.guess_cache_ready = false;
    }
    
    if (state.guess_cache_ready) {
        PHASE2_DEBUG_STDERR("[DEBUG] About to call EmitModuleInfo for GuessCache...\n");
        EmitModuleInfo(ModuleId::GuessCache, 
            std::string("GPU cache ready (capacity=") + std::to_string(kDefaultGuessCacheCapacity) + ")", ctx.global_step);
        state.guess_cache_buffers = std::make_unique<GuessCacheBatchBuffers>();
        PHASE2_DEBUG_STDERR("[DEBUG] GuessCacheBatchBuffers created successfully\n");
    } else if (!ctx.config.hyperparameters.guess_aux_enabled) {
        EmitModuleInfo(ModuleId::GuessCache, "Guess cache disabled (guess_aux.enabled=false)", ctx.global_step);
    }
    
    PHASE2_DEBUG_STDERR("[DEBUG] About to initialize training log...");

    // Log configuration
    EmitModuleInfo(ModuleId::Training, "Starting training...", ctx.global_step);
    EmitModuleInfo(ModuleId::Training, std::string("  Warmup steps: ") + std::to_string(hp.warmup_steps), ctx.global_step);
    EmitModuleInfo(ModuleId::Training, std::string("  Target learning rate: ") + std::to_string(hp.learning_rate), ctx.global_step);
    EmitModuleInfo(ModuleId::Training, 
        std::string("  Dynamic LR: ") + (hp.dynamic_lr_enabled ? "enabled" : "disabled"), ctx.global_step);
    EmitModuleInfo(ModuleId::Training, 
        std::string("  Soft restart: ") + (hp.soft_restart_enabled ? "enabled" : "disabled"), ctx.global_step);
    EmitModuleInfo(ModuleId::Training, 
        std::string("  Auto-stop: ") + (hp.auto_stop_enabled ? "enabled" : "disabled"), ctx.global_step);
    
    try {
        for (int epoch = 0; epoch < hp.epochs; ++epoch) {
            EpochResult epoch_result = runEpoch(ctx, state, epoch);
            
            if (epoch_result.auto_stop_triggered) {
                EmitModuleInfo(ModuleId::Training, 
                    std::string("Auto-stop engaged after epoch ") + std::to_string(epoch + 1), ctx.global_step);
                break;
            }
        }
    } catch (const std::exception& e) {
        EmitModuleError(ModuleId::Training, 
            std::string("TRAINING ERROR: ") + e.what(), ctx.global_step);
        
        ctx.logging.status_writer->writeStatus(
            GRIMText::Control::TrainingState_Error,
            0, hp.epochs, 0, 0,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            "Training error", std::string(e.what()));
        
        throw;
    }
    return true;
}

} // namespace GRIMText::Training
