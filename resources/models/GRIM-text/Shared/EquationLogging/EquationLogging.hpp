#pragma once
// ============================================================================
// EQUATION LOGGING SYSTEM - Host-side CSV logger (Rule 21 format)
// ============================================================================
// Design:
//   - All logging is HOST-SIDE. No device kernels, no ring buffers, no CUDA.
//   - Callers build a fully-formatted Rule 21 body string and pass it to EQ_LOG.
//   - Entries are batched in memory and flushed to CSV periodically.
//   - Phase enum is retained for sort ordering within a flush window.
//
// Rule 21 body format (built by caller):
//   [TAG] equation_name: formula
//     INPUT_A (description): shape=[dims] min=X max=Y rms=Z
//     PARAMETERS: param1=value1, param2=value2
//     EXPECTED result = formula_with_symbols = values = computed
//     ACTUAL result: min=X max=Y rms=Z
//     [ANOMALY] if actual differs from expected
//
// Usage:
//   std::ostringstream eq;
//   eq << "[RMSNORM_EQUATION] y = x * gamma / sqrt(mean(x^2) + eps)\n";
//   eq << "  INPUT x: shape=[" << tokens << "," << d << "] rms=" << x_rms << "\n";
//   eq << "  EXPECTED output_rms = " << expected << "\n";
//   eq << "  ACTUAL output_rms = " << actual << "\n";
//   EQ_LOG("RMSNORM_EQUATION", eq.str(), batch, layer, step, GRIM::EquationPhase::RMSNORM);
// ============================================================================

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string> 
#include <chrono>
#include <vector>
#include <mutex>
#include <algorithm>

namespace GRIM {

// Phase identifiers for ordered logging
enum class EquationPhase : int {
    EMBEDDING_LOOKUP     = 0,
    POSITION_ENCODING    = 1,
    RMSNORM_PRE_ATTN     = 2,
    QKV_PROJECTION       = 3,
    ROPE_ROTATION        = 4,
    ALIBI_BIAS           = 5,
    FLASH_ATTENTION_FWD  = 6,
    ATTENTION_OUTPUT      = 7,
    RMSNORM_PRE_FFN      = 8,
    FFN_LAYER1           = 9,
    FFN_GELU             = 10,
    FFN_LAYER2           = 11,
    LM_HEAD_PROJECTION   = 12,
    LOSS_COMPUTATION     = 13,
    LOSS_BACKWARD        = 14,
    LM_HEAD_BACKWARD     = 15,
    FFN_BACKWARD         = 16,
    ATTENTION_BACKWARD   = 17,
    RMSNORM_BACKWARD     = 18,
    EMBEDDING_BACKWARD   = 19,
    GRADIENT_CLIP        = 20,
    ADAMW_UPDATE         = 21,
    WEIGHT_DECAY         = 22,
    RMSNORM              = 23,
    RESIDUAL_ADD         = 24,
    ATTENTION_SCORE      = 25,
    PHASE_COUNT          = 26
};

inline const char* phaseToString(EquationPhase phase) {
    switch (phase) {
        case EquationPhase::EMBEDDING_LOOKUP:    return "EMBEDDING_LOOKUP";
        case EquationPhase::POSITION_ENCODING:   return "POSITION_ENCODING";
        case EquationPhase::RMSNORM_PRE_ATTN:    return "RMSNORM_PRE_ATTN";
        case EquationPhase::QKV_PROJECTION:      return "QKV_PROJECTION";
        case EquationPhase::ROPE_ROTATION:       return "ROPE_ROTATION";
        case EquationPhase::ALIBI_BIAS:          return "ALIBI_BIAS";
        case EquationPhase::FLASH_ATTENTION_FWD: return "FLASH_ATTENTION_FWD";
        case EquationPhase::ATTENTION_OUTPUT:     return "ATTENTION_OUTPUT";
        case EquationPhase::RMSNORM_PRE_FFN:     return "RMSNORM_PRE_FFN";
        case EquationPhase::FFN_LAYER1:          return "FFN_LAYER1";
        case EquationPhase::FFN_GELU:            return "FFN_GELU";
        case EquationPhase::FFN_LAYER2:          return "FFN_LAYER2";
        case EquationPhase::LM_HEAD_PROJECTION:  return "LM_HEAD_PROJECTION";
        case EquationPhase::LOSS_COMPUTATION:    return "LOSS_COMPUTATION";
        case EquationPhase::LOSS_BACKWARD:       return "LOSS_BACKWARD";
        case EquationPhase::LM_HEAD_BACKWARD:    return "LM_HEAD_BACKWARD";
        case EquationPhase::FFN_BACKWARD:        return "FFN_BACKWARD";
        case EquationPhase::ATTENTION_BACKWARD:  return "ATTENTION_BACKWARD";
        case EquationPhase::RMSNORM_BACKWARD:    return "RMSNORM_BACKWARD";
        case EquationPhase::EMBEDDING_BACKWARD:  return "EMBEDDING_BACKWARD";
        case EquationPhase::GRADIENT_CLIP:       return "GRADIENT_CLIP";
        case EquationPhase::ADAMW_UPDATE:        return "ADAMW_UPDATE";
        case EquationPhase::WEIGHT_DECAY:        return "WEIGHT_DECAY";
        case EquationPhase::RMSNORM:             return "RMSNORM";
        case EquationPhase::RESIDUAL_ADD:        return "RESIDUAL_ADD";
        case EquationPhase::ATTENTION_SCORE:     return "ATTENTION_SCORE";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// EquationLogger - Host-only CSV writer for Rule 21 equation diagnostics
// ============================================================================

class EquationLogger {
public:
    EquationLogger() = default;
    ~EquationLogger() { shutdown(); }

    // Non-copyable
    EquationLogger(const EquationLogger&) = delete;
    EquationLogger& operator=(const EquationLogger&) = delete;

    // ========================================================================
    // Lifecycle
    // ========================================================================

    bool initialize(const std::string& log_path, bool enable = true) {
        if (!enable) {
            enabled_ = false;
            return true;
        }

        start_time_ = std::chrono::steady_clock::now();

        log_file_.open(log_path, std::ios::out | std::ios::app);
        if (!log_file_.is_open()) {
            std::cerr << "[EquationLogger] FAILED to open: " << log_path << std::endl;
            return false;
        }

        log_file_ << "# GRIM-text Equation Log (Rule 21 format)\n";
        log_file_ << "# Started: " << getCurrentTimestamp() << "\n\n";

        pending_.reserve(256);
        enabled_ = true;

        std::cout << "[EquationLogger] Initialized: " << log_path << std::endl;
        return true;
    }

    void shutdown() {
        if (!enabled_) return;

        flushSync();

        if (log_file_.is_open()) {
            log_file_ << "\n# === END OF LOG ===\n";
            log_file_ << "# Total entries: " << total_logged_ << "\n";
            log_file_.close();
        }

        enabled_ = false;
        std::cout << "[EquationLogger] Shutdown. Logged " << total_logged_ << " entries." << std::endl;
    }

    // ========================================================================
    // Main API — accepts pre-formatted Rule 21 body string
    // ========================================================================

    void logEquation(const std::string& tag,
                     const std::string& body,
                     int batch, int layer, int step,
                     EquationPhase phase) {
        if (!enabled_) return;

        auto now = std::chrono::steady_clock::now();
        float ts = std::chrono::duration<float, std::milli>(now - start_time_).count();

        std::lock_guard<std::mutex> lock(pending_mutex_);
        pending_.push_back({tag, body, batch, layer, step, phase, ts});
        total_logged_++;
    }

    // ========================================================================
    // Flush — sort by (step, batch, phase) then write
    // ========================================================================

    void flushSync() {
        if (!enabled_) return;

        std::lock_guard<std::mutex> lock(pending_mutex_);
        if (pending_.empty()) return;

        std::sort(pending_.begin(), pending_.end(),
            [](const LogEntry& a, const LogEntry& b) {
                if (a.step_idx != b.step_idx) return a.step_idx < b.step_idx;
                if (a.batch_idx != b.batch_idx) return a.batch_idx < b.batch_idx;
                return static_cast<int>(a.phase) < static_cast<int>(b.phase);
            });

        writeEntries();
        pending_.clear();
    }

    // flushAsync removed — device ring buffer deleted.
    // Stub kept so callers compile while updating.
    void flushAsync() { /* no-op */ }

    // ========================================================================
    // Getters
    // ========================================================================

    bool isEnabled() const { return enabled_; }

    void printStats() const {
        std::cout << "[EquationLogger] logged=" << total_logged_ << std::endl;
    }

private:
    struct LogEntry {
        std::string tag;
        std::string body;
        int batch_idx  = 0;
        int layer_idx  = 0;
        int step_idx   = 0;
        EquationPhase phase = EquationPhase::LOSS_COMPUTATION;
        float timestamp_ms = 0.0f;
    };

    bool enabled_ = false;
    std::ofstream log_file_;
    std::mutex file_mutex_;
    std::vector<LogEntry> pending_;
    std::mutex pending_mutex_;
    uint64_t total_logged_ = 0;
    std::chrono::steady_clock::time_point start_time_;

    // ========================================================================
    // File output — body written verbatim (caller owns Rule 21 formatting)
    // ========================================================================

    void writeEntries() {
        if (!log_file_.is_open()) return;

        std::lock_guard<std::mutex> lock(file_mutex_);

        for (const auto& e : pending_) {
            log_file_ << e.body;
            if (!e.body.empty() && e.body.back() != '\n') {
                log_file_ << '\n';
            }
            log_file_ << '\n';
        }

        log_file_.flush();
    }

    std::string getCurrentTimestamp() const {
        auto now = std::chrono::system_clock::now();
        auto t   = std::chrono::system_clock::to_time_t(now);
        char buf[64];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
        return std::string(buf);
    }
};

// ============================================================================
// Global singleton
// ============================================================================

inline EquationLogger& getEquationLogger() {
    static EquationLogger instance;
    return instance;
}

// Skip flag for gradient accumulation: when true, layers omit expensive D2H + fprintf
// equation diagnostics for this forward pass. Set by executeAutogradForward when
// accumulate=true (micro-batches 1..N-1), so we log only once per optimizer step.
inline bool& getEquationLoggingSkipThisPassRef() {
    static bool s_skip = false;
    return s_skip;
}

// ============================================================================
// EQ_LOG macro — the ONLY logging macro. Rule 20: no legacy macros kept.
//
// tag   — short identifier, e.g. "RMSNORM_EQUATION"
// body  — pre-formatted Rule 21 multi-line string (caller builds via ostringstream)
// batch — batch index
// layer — layer index (-1 if N/A)
// step  — global optimizer step
// phase — EquationPhase enum value (used for sort ordering)
// ============================================================================

#define EQ_LOG(tag, body, batch, layer, step, phase) \
    do { \
        auto& _eq_logger_ = GRIM::getEquationLogger(); \
        if (_eq_logger_.isEnabled()) { \
            _eq_logger_.logEquation(tag, body, batch, layer, step, phase); \
        } \
    } while(0)

} // namespace GRIM
