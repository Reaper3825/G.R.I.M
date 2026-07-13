#pragma once
//======================================================//
//  LogTypes.hpp — Unified logging type system
//======================================================//
//
//  Single source of truth for all log levels, groups,
//  phases, and entry structures across GRIM-text training.
//
//  Host-only. No CUDA dependency.
//
//  Author: Austin Wadkins
//  Date: April 2026
//======================================================//

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace GRIM::Logging {

//======================================================//
//  Log Level — severity + filtering gate
//======================================================//
//
//  Ordered by severity. The tape records only entries at
//  or above the configured threshold level.
//
//  Trace:   Per-element tensor stats, kernel launch params.
//           ONLY enable when debugging a specific subsystem.
//  Debug:   Per-layer forward/backward shapes, cache writes,
//           equation diagnostics (Rule 21).
//  Info:    Per-batch loss, grad norm, LR, validation results.
//           Default production level.
//  Warning: Anomaly flags, gradient clipping triggered,
//           high ρ spread, gamma drift.
//  Error:   NaN/Inf detected, kernel launch failure,
//           buffer mismatch, assertion about to fire.
//  Fatal:   Unrecoverable state. Logged then throws.
//
enum class LogLevel : uint8_t {
    Trace   = 0,
    Debug   = 1,
    Info    = 2,
    Warning = 3,
    Error   = 4,
    Fatal   = 5,
    COUNT   = 6
};

inline const char* logLevelToString(LogLevel level) {
    switch (level) {
        case LogLevel::Trace:   return "TRACE";
        case LogLevel::Debug:   return "DEBUG";
        case LogLevel::Info:    return "INFO";
        case LogLevel::Warning: return "WARN";
        case LogLevel::Error:   return "ERROR";
        case LogLevel::Fatal:   return "FATAL";
        default:                return "???";
    }
}

inline LogLevel logLevelFromString(const char* str) {
    if (!str) return LogLevel::Info;
    // Case-insensitive first-char match (fast path for config parsing)
    switch (str[0]) {
        case 't': case 'T': return LogLevel::Trace;
        case 'd': case 'D': return LogLevel::Debug;
        case 'i': case 'I': return LogLevel::Info;
        case 'w': case 'W': return LogLevel::Warning;
        case 'e': case 'E': return LogLevel::Error;
        case 'f': case 'F': return LogLevel::Fatal;
        default:            return LogLevel::Info;
    }
}

//======================================================//
//  Log Group — which subsystem produced the entry
//======================================================//
//
//  Mirrors ParamGroupType for model components, plus
//  system groups for infrastructure. Every log entry
//  belongs to exactly one group.
//
enum class LogGroup : uint8_t {
    // Model components (match ParamGroupType values)
    Embedding      = 0,
    LMHead         = 1,
    Attention      = 2,
    FFN            = 3,
    RMSNorm        = 4,
    ScratchBlock   = 5,
    Reserved6      = 6,
    ExecutionBlock = 7,
    SlotSelector   = 8,
    // System groups
    Loss           = 9,
    Optimizer      = 10,
    Scheduler      = 11,
    DataLoader     = 12,
    Checkpoint     = 13,
    Telemetry      = 14,
    Stream         = 15,
    Validation     = 16,
    System         = 17,
    COUNT          = 18
};

inline const char* logGroupToString(LogGroup group) {
    switch (group) {
        case LogGroup::Embedding:      return "EMB";
        case LogGroup::LMHead:         return "LM";
        case LogGroup::Attention:      return "ATTN";
        case LogGroup::FFN:            return "FFN";
        case LogGroup::RMSNorm:        return "NORM";
        case LogGroup::ScratchBlock:   return "SB";
        case LogGroup::Reserved6:      return "RESERVED6";
        case LogGroup::ExecutionBlock: return "EB";
        case LogGroup::SlotSelector:   return "SLOT";
        case LogGroup::Loss:           return "LOSS";
        case LogGroup::Optimizer:      return "OPT";
        case LogGroup::Scheduler:      return "SCHED";
        case LogGroup::DataLoader:     return "DATA";
        case LogGroup::Checkpoint:     return "CKPT";
        case LogGroup::Telemetry:      return "TEL";
        case LogGroup::Stream:         return "STRM";
        case LogGroup::Validation:     return "VAL";
        case LogGroup::System:         return "SYS";
        default:                       return "???";
    }
}

inline LogGroup logGroupFromString(const char* str) {
    if (!str) return LogGroup::System;
    // Match against short names
    if (std::strcmp(str, "EMB")   == 0 || std::strcmp(str, "Embedding")      == 0) return LogGroup::Embedding;
    if (std::strcmp(str, "LM")    == 0 || std::strcmp(str, "LMHead")         == 0) return LogGroup::LMHead;
    if (std::strcmp(str, "ATTN")  == 0 || std::strcmp(str, "Attention")      == 0) return LogGroup::Attention;
    if (std::strcmp(str, "FFN")   == 0 || std::strcmp(str, "FFN")            == 0) return LogGroup::FFN;
    if (std::strcmp(str, "NORM")  == 0 || std::strcmp(str, "RMSNorm")        == 0) return LogGroup::RMSNorm;
    if (std::strcmp(str, "SB")    == 0 || std::strcmp(str, "ScratchBlock")   == 0) return LogGroup::ScratchBlock;
    if (std::strcmp(str, "EB")    == 0 || std::strcmp(str, "ExecutionBlock") == 0) return LogGroup::ExecutionBlock;
    if (std::strcmp(str, "SLOT")  == 0 || std::strcmp(str, "SlotSelector")   == 0) return LogGroup::SlotSelector;
    if (std::strcmp(str, "LOSS")  == 0 || std::strcmp(str, "Loss")           == 0) return LogGroup::Loss;
    if (std::strcmp(str, "OPT")   == 0 || std::strcmp(str, "Optimizer")      == 0) return LogGroup::Optimizer;
    if (std::strcmp(str, "SCHED") == 0 || std::strcmp(str, "Scheduler")      == 0) return LogGroup::Scheduler;
    if (std::strcmp(str, "DATA")  == 0 || std::strcmp(str, "DataLoader")     == 0) return LogGroup::DataLoader;
    if (std::strcmp(str, "CKPT")  == 0 || std::strcmp(str, "Checkpoint")     == 0) return LogGroup::Checkpoint;
    if (std::strcmp(str, "TEL")   == 0 || std::strcmp(str, "Telemetry")      == 0) return LogGroup::Telemetry;
    if (std::strcmp(str, "STRM")  == 0 || std::strcmp(str, "Stream")         == 0) return LogGroup::Stream;
    if (std::strcmp(str, "VAL")   == 0 || std::strcmp(str, "Validation")     == 0) return LogGroup::Validation;
    return LogGroup::System;
}

//======================================================//
//  Log Phase — operation ordering within a batch
//======================================================//
//
//  Defines the canonical execution order of operations
//  within a single training step. When the tape flushes,
//  entries are sorted by phase so the log reads like
//  a step-by-step trace of the forward-backward pass.
//
//  Values are spaced for future insertions without
//  reordering existing entries.
//
//  Forward pass phases start at 0, backward at 200,
//  optimizer/system at 400.
//
enum class LogPhase : uint16_t {
    // ---- Forward pass ----
    BATCH_START            = 0,
    DATA_LOAD              = 10,
    EMBEDDING_LOOKUP       = 20,
    POSITION_ENCODING      = 30,
    SCRATCHBLOCK_FWD       = 40,
    RMSNORM_PRE_ATTN       = 100,
    QKV_PROJECTION         = 110,
    ROPE_ROTATION          = 120,
    ALIBI_BIAS             = 130,
    FLASH_ATTENTION_FWD    = 140,
    ATTENTION_OUTPUT        = 150,
    RESIDUAL_POST_ATTN     = 160,
    RMSNORM_PRE_FFN        = 170,
    FFN_LAYER1             = 180,
    FFN_GELU               = 190,
    FFN_LAYER2             = 195,
    RESIDUAL_POST_FFN      = 198,
    ENCODER_LAYER_DONE     = 199,
    LM_HEAD_PROJECTION     = 200,
    LOSS_COMPUTATION       = 210,

    // ---- Backward pass ----
    LOSS_BACKWARD          = 300,
    LM_HEAD_BACKWARD       = 310,
    FFN_BACKWARD           = 320,
    ATTENTION_BACKWARD     = 330,
    RMSNORM_BACKWARD       = 340,
    EMBEDDING_BACKWARD     = 350,
    SCRATCHBLOCK_BWD       = 360,

    // ---- Gradient processing ----
    GRADIENT_CLIP          = 400,
    GRADIENT_NORM          = 410,

    // ---- Optimizer ----
    ADAMW_UPDATE           = 500,
    WEIGHT_DECAY           = 510,
    LR_SCHEDULE            = 520,

    // ---- Post-step ----
    TELEMETRY_UPDATE       = 600,
    VALIDATION             = 700,
    CHECKPOINT             = 800,
    DIAGNOSTICS            = 900,
    BATCH_END              = 999,

    // ---- Lifecycle (non-batch, used for startup/shutdown) ----
    LIFECYCLE              = 1000,

    COUNT // sentinel
};

inline const char* logPhaseToString(LogPhase phase) {
    switch (phase) {
        case LogPhase::BATCH_START:         return "BATCH_START";
        case LogPhase::DATA_LOAD:           return "DATA_LOAD";
        case LogPhase::EMBEDDING_LOOKUP:    return "EMB_LOOKUP";
        case LogPhase::POSITION_ENCODING:   return "POS_ENC";
        case LogPhase::SCRATCHBLOCK_FWD:    return "SB_FWD";
        case LogPhase::RMSNORM_PRE_ATTN:    return "NORM_PRE_ATTN";
        case LogPhase::QKV_PROJECTION:      return "QKV_PROJ";
        case LogPhase::ROPE_ROTATION:       return "ROPE";
        case LogPhase::ALIBI_BIAS:          return "ALIBI";
        case LogPhase::FLASH_ATTENTION_FWD: return "FA_FWD";
        case LogPhase::ATTENTION_OUTPUT:    return "ATTN_OUT";
        case LogPhase::RESIDUAL_POST_ATTN:  return "RES_ATTN";
        case LogPhase::RMSNORM_PRE_FFN:     return "NORM_PRE_FFN";
        case LogPhase::FFN_LAYER1:          return "FFN_W1";
        case LogPhase::FFN_GELU:            return "FFN_GELU";
        case LogPhase::FFN_LAYER2:          return "FFN_W2";
        case LogPhase::RESIDUAL_POST_FFN:   return "RES_FFN";
        case LogPhase::ENCODER_LAYER_DONE:  return "ENC_DONE";
        case LogPhase::LM_HEAD_PROJECTION:  return "LM_PROJ";
        case LogPhase::LOSS_COMPUTATION:    return "LOSS_FWD";
        case LogPhase::LOSS_BACKWARD:       return "LOSS_BWD";
        case LogPhase::LM_HEAD_BACKWARD:    return "LM_BWD";
        case LogPhase::FFN_BACKWARD:        return "FFN_BWD";
        case LogPhase::ATTENTION_BACKWARD:  return "ATTN_BWD";
        case LogPhase::RMSNORM_BACKWARD:    return "NORM_BWD";
        case LogPhase::EMBEDDING_BACKWARD:  return "EMB_BWD";
        case LogPhase::SCRATCHBLOCK_BWD:    return "SB_BWD";
        case LogPhase::GRADIENT_CLIP:       return "GRAD_CLIP";
        case LogPhase::GRADIENT_NORM:       return "GRAD_NORM";
        case LogPhase::ADAMW_UPDATE:        return "ADAM";
        case LogPhase::WEIGHT_DECAY:        return "WD";
        case LogPhase::LR_SCHEDULE:         return "LR";
        case LogPhase::TELEMETRY_UPDATE:    return "TEL";
        case LogPhase::VALIDATION:          return "VAL";
        case LogPhase::CHECKPOINT:          return "CKPT";
        case LogPhase::DIAGNOSTICS:         return "DIAG";
        case LogPhase::BATCH_END:           return "BATCH_END";
        case LogPhase::LIFECYCLE:           return "LIFECYCLE";
        default:                            return "???";
    }
}

//======================================================//
//  Log Entry — single record on the tape
//======================================================//
//
//  Host-owned entry. Tags stay fixed-size for compactness;
//  messages use std::string so sinks receive the full text
//  without truncating long Rule 21 diagnostics.
//
struct LogEntry {
    // Classification
    LogLevel  level;           // 1 byte
    LogGroup  group;           // 1 byte
    LogPhase  phase;           // 2 bytes
    int16_t   layer_idx;       // -1 for global / N/A
    uint16_t  _pad0;           // alignment

    // Identity
    int32_t   global_step;
    int32_t   batch_idx;

    // Tag — short identifier (e.g. "RMSNORM_EQUATION", "grad_clip")
    char tag[48];

    // Message — pre-formatted text. Full payload is preserved.
    std::string message;

    // Optional scalar metrics (NaN = not set)
    float primary;
    float secondary;

    //--------------------------------------------------
    // Convenience: set tag safely
    //--------------------------------------------------
    void setTag(const char* t) {
        if (t) {
            std::strncpy(tag, t, sizeof(tag) - 1);
            tag[sizeof(tag) - 1] = '\0';
        } else {
            tag[0] = '\0';
        }
    }

    //--------------------------------------------------
    // Convenience: store message text exactly as given
    //--------------------------------------------------
    void setMessageText(const char* text) {
        message = text ? text : "";
    }

    void setMessageView(std::string_view text) {
        message.assign(text.data(), text.size());
    }

    //--------------------------------------------------
    // Convenience: snprintf into message
    //--------------------------------------------------
    template<typename... Args>
    void setMessage(const char* fmt, Args... args) {
        if (!fmt) {
            message.clear();
            return;
        }

        std::array<char, 512> stack_buf{};
        const int written = std::snprintf(stack_buf.data(), stack_buf.size(), fmt, args...);
        if (written < 0) {
            throw std::runtime_error("LogEntry::setMessage failed to format message");
        }

        if (static_cast<size_t>(written) < stack_buf.size()) {
            message.assign(stack_buf.data(), static_cast<size_t>(written));
            return;
        }

        std::vector<char> heap_buf(static_cast<size_t>(written) + 1, '\0');
        const int rewritten = std::snprintf(heap_buf.data(), heap_buf.size(), fmt, args...);
        if (rewritten != written) {
            throw std::runtime_error("LogEntry::setMessage failed to finalize message");
        }
        message.assign(heap_buf.data(), static_cast<size_t>(rewritten));
    }

    //--------------------------------------------------
    // Comparison for sort — phase order, then layer
    //--------------------------------------------------
    bool operator<(const LogEntry& other) const {
        if (phase != other.phase) return static_cast<uint16_t>(phase) < static_cast<uint16_t>(other.phase);
        return layer_idx < other.layer_idx;
    }
};

//======================================================//
//  BATCH_LOG macro — canonical recording point
//======================================================//
//
//  Usage:
//    BATCH_LOG(tape, LogLevel::Info, LogGroup::Loss, LogPhase::LOSS_COMPUTATION,
//              -1, "BATCH_LOSS", "loss=%.6f tokens=%d", loss, tokens);
//
//  The macro gates on level threshold before touching the tape,
//  so sub-threshold entries have zero cost.
//
#define BATCH_LOG(tape_ptr, lvl, grp, phs, layer, tag_str, fmt, ...)                     \
    do {                                                                                   \
        if ((tape_ptr) && (tape_ptr)->accepts(lvl)) {                                      \
            GRIM::Logging::LogEntry _entry{};                                              \
            _entry.level     = (lvl);                                                      \
            _entry.group     = (grp);                                                      \
            _entry.phase     = (phs);                                                      \
            _entry.layer_idx = static_cast<int16_t>(layer);                                \
            _entry.global_step = (tape_ptr)->currentStep();                                \
            _entry.batch_idx   = (tape_ptr)->currentBatch();                               \
            _entry.setTag(tag_str);                                                        \
            _entry.setMessage(fmt, ##__VA_ARGS__);                                         \
            _entry.primary   = __builtin_nanf("");                                         \
            _entry.secondary = __builtin_nanf("");                                         \
            (tape_ptr)->record(_entry);                                                    \
        }                                                                                  \
    } while (0)

//------------------------------------------------------
//  BATCH_LOG_V — same but with primary/secondary values
//------------------------------------------------------
#define BATCH_LOG_V(tape_ptr, lvl, grp, phs, layer, tag_str, p, s, fmt, ...)             \
    do {                                                                                   \
        if ((tape_ptr) && (tape_ptr)->accepts(lvl)) {                                      \
            GRIM::Logging::LogEntry _entry{};                                              \
            _entry.level     = (lvl);                                                      \
            _entry.group     = (grp);                                                      \
            _entry.phase     = (phs);                                                      \
            _entry.layer_idx = static_cast<int16_t>(layer);                                \
            _entry.global_step = (tape_ptr)->currentStep();                                \
            _entry.batch_idx   = (tape_ptr)->currentBatch();                               \
            _entry.setTag(tag_str);                                                        \
            _entry.setMessage(fmt, ##__VA_ARGS__);                                         \
            _entry.primary   = static_cast<float>(p);                                      \
            _entry.secondary = static_cast<float>(s);                                      \
            (tape_ptr)->record(_entry);                                                    \
        }                                                                                  \
    } while (0)

//------------------------------------------------------
//  EQ_LOG — equation diagnostic entry (Rule 21)
//  Replaces the old EquationLogger macro.
//  Level defaults to Debug (equation tracing is detailed).
//------------------------------------------------------
#define EQ_LOG(tape_ptr, grp, phs, layer, tag_str, body_str)                              \
    do {                                                                                   \
        if ((tape_ptr) && (tape_ptr)->accepts(GRIM::Logging::LogLevel::Debug)) {           \
            GRIM::Logging::LogEntry _entry{};                                              \
            _entry.level     = GRIM::Logging::LogLevel::Debug;                             \
            _entry.group     = (grp);                                                      \
            _entry.phase     = (phs);                                                      \
            _entry.layer_idx = static_cast<int16_t>(layer);                                \
            _entry.global_step = (tape_ptr)->currentStep();                                \
            _entry.batch_idx   = (tape_ptr)->currentBatch();                               \
            _entry.setTag(tag_str);                                                        \
            _entry.setMessageText(body_str);                                               \
            _entry.primary   = __builtin_nanf("");                                         \
            _entry.secondary = __builtin_nanf("");                                         \
            (tape_ptr)->record(_entry);                                                    \
        }                                                                                  \
    } while (0)

//======================================================//
//  Legacy ModuleId / ModuleLogLevel → LogGroup / LogLevel
//  mapping (used by EmitModuleLog bridge)
//======================================================//

/// Forward-declared enum from LogRecorder.hpp (avoids circular include).
/// The actual enum lives in GRIM::Logging — these are integer-level converters.
/// Callers pass the raw int value; no need to include LogRecorder.hpp.
inline LogGroup moduleIdToLogGroup(int module_id) {
    // ModuleId values (LogRecorder.hpp):
    //   ForwardPass=0, BackwardPass=1, Optimizer=2, Scheduler=3,
    //   Activations=4, Validation=5, Checkpoint=6, DataLoader=7,
    //   Inference=8, LogRecorder=9, Training=10,
    //   TrainingOrchestrator=11, StreamController=12, Loss=13,
    //   Attention=14, Custom=15, Autograd=16, ExecutionBlock=17
    switch (module_id) {
        case 0:  return LogGroup::System;          // ForwardPass
        case 1:  return LogGroup::System;          // BackwardPass
        case 2:  return LogGroup::Optimizer;       // Optimizer
        case 3:  return LogGroup::Scheduler;       // Scheduler
        case 4:  return LogGroup::Attention;       // Activations
        case 5:  return LogGroup::Validation;      // Validation
        case 6:  return LogGroup::Checkpoint;      // Checkpoint
        case 7:  return LogGroup::DataLoader;      // DataLoader
        case 8:  return LogGroup::System;          // Inference
        case 9:  return LogGroup::System;          // LogRecorder
        case 10: return LogGroup::System;          // Training
        case 11: return LogGroup::System;          // TrainingOrchestrator
        case 12: return LogGroup::Stream;          // StreamController
        case 13: return LogGroup::Loss;            // Loss
        case 14: return LogGroup::Attention;       // Attention
        case 15: return LogGroup::System;          // Custom
        case 16: return LogGroup::System;          // Autograd
        case 17: return LogGroup::ExecutionBlock;  // ExecutionBlock
        default: return LogGroup::System;
    }
}

/// Map legacy 3-level ModuleLogLevel (Info=0, Warning=1, Error=2) to LogLevel.
inline LogLevel moduleLogLevelToLogLevel(int module_level) {
    switch (module_level) {
        case 0:  return LogLevel::Info;     // ModuleLogLevel::Info
        case 1:  return LogLevel::Warning;  // ModuleLogLevel::Warning
        case 2:  return LogLevel::Error;    // ModuleLogLevel::Error
        default: return LogLevel::Info;
    }
}

} // namespace GRIM::Logging
