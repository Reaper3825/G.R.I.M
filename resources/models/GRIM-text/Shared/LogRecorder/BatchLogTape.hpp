#pragma once
//======================================================//
//  BatchLogTape.hpp — Batch-accumulation logging system
//======================================================//
//
//  Modeled after the autograd tape: operations record
//  entries during a training step, then flush() sorts
//  and dispatches to registered sinks at the batch boundary.
//
//  Key properties:
//    - Pre-allocated buffer (no heap alloc in hot path)
//    - Level gate at record() time (sub-threshold = zero cost)
//    - Phase-sorted output for readable step traces
//    - Per-group level overrides for targeted debugging
//    - Lifecycle messages bypass the tape (direct to sinks)
//
//  Thread safety:
//    - record() is mutex-protected for multi-thread safety
//      (e.g. async backward callbacks). The mutex is cheap
//      because contention is near-zero in practice.
//    - flush() must be called from a single thread (the
//      training loop thread, after backward completes).
//
//  Author: Austin Wadkins
//  Date: April 2026
//======================================================//

#include "LogTypes.hpp"
#include "Sinks/ILogSink.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace GRIM::Logging {

//======================================================//
//  Tape Configuration
//======================================================//

struct TapeConfig {
    LogLevel default_level = LogLevel::Info;   // Global threshold
    size_t   initial_capacity = 2048;          // Pre-allocated entry count
    bool     equation_csv_enabled = true;      // Whether CsvEquationSink is active
    bool     stderr_enabled = true;            // Whether StderrSink is active

    // Per-group level overrides (indexed by LogGroup).
    // COUNT = not-set (use default_level).
    LogLevel group_levels[static_cast<size_t>(LogGroup::COUNT)] = {};

    TapeConfig() {
        // Initialize all group overrides to COUNT (= "use default")
        for (size_t i = 0; i < static_cast<size_t>(LogGroup::COUNT); ++i) {
            group_levels[i] = LogLevel::COUNT;
        }
    }
};

//======================================================//
//  BatchLogTape
//======================================================//

class BatchLogTape {
public:
    explicit BatchLogTape(const TapeConfig& config)
        : config_(config)
    {
        entries_.reserve(config.initial_capacity);
    }

    ~BatchLogTape() = default;

    // Non-copyable, non-movable (contains mutex)
    BatchLogTape(const BatchLogTape&) = delete;
    BatchLogTape& operator=(const BatchLogTape&) = delete;
    BatchLogTape(BatchLogTape&&) = delete;
    BatchLogTape& operator=(BatchLogTape&&) = delete;

    //--------------------------------------------------
    //  Batch lifecycle
    //--------------------------------------------------

    /// Call at the start of each training step.
    /// Clears the buffer and sets step/batch identifiers.
    void beginBatch(int global_step, int batch_idx) {
        std::lock_guard<std::mutex> lock(mutex_);
        entries_.clear();
        current_step_  = global_step;
        current_batch_ = batch_idx;
    }

    /// Flush: sort by phase, dispatch to all sinks, clear buffer.
    /// Call at the end of each training step (after backward + optimizer).
    void flush() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (entries_.empty() && lifecycle_entries_.empty()) return;

        // Sort batch entries by phase order, then layer
        std::sort(entries_.begin(), entries_.end());

        // Dispatch batch entries to all sinks
        for (auto* sink : sinks_) {
            if (sink) {
                sink->receive(entries_.data(), entries_.size(), current_step_);
            }
        }

        // Dispatch lifecycle entries (unsorted — order of emission)
        if (!lifecycle_entries_.empty()) {
            for (auto* sink : sinks_) {
                if (sink) {
                    sink->receive(lifecycle_entries_.data(), lifecycle_entries_.size(), current_step_);
                }
            }
            lifecycle_entries_.clear();
        }

        entries_.clear();
    }

    //--------------------------------------------------
    //  Recording
    //--------------------------------------------------

    /// Check if a level would be recorded (for macro gating).
    bool accepts(LogLevel level) const {
        return static_cast<uint8_t>(level) >= static_cast<uint8_t>(config_.default_level);
    }

    /// Check if a level would be recorded for a specific group.
    bool accepts(LogLevel level, LogGroup group) const {
        LogLevel threshold = config_.group_levels[static_cast<size_t>(group)];
        if (threshold == LogLevel::COUNT) {
            threshold = config_.default_level;
        }
        return static_cast<uint8_t>(level) >= static_cast<uint8_t>(threshold);
    }

    /// Record an entry onto the tape. Thread-safe.
    void record(const LogEntry& entry) {
        std::lock_guard<std::mutex> lock(mutex_);
        entries_.push_back(entry);
    }

    /// Record a lifecycle entry (startup/shutdown, not batch-scoped).
    /// These are dispatched alongside the next flush, unsorted.
    void recordLifecycle(const LogEntry& entry) {
        std::lock_guard<std::mutex> lock(mutex_);
        lifecycle_entries_.push_back(entry);
    }

    /// Emit a lifecycle entry immediately to all sinks (no buffering).
    /// Use for critical startup/shutdown messages that must appear now.
    void emitImmediate(const LogEntry& entry) {
        // No lock needed — sinks handle their own thread safety
        for (auto* sink : sinks_) {
            if (sink) {
                sink->receive(&entry, 1, entry.global_step);
            }
        }
    }

    //--------------------------------------------------
    //  Sink management
    //--------------------------------------------------

    /// Register a sink. The tape does NOT own the sink (caller manages lifetime).
    /// Sinks must outlive the tape or be removed before destruction.
    void addSink(ILogSink* sink) {
        if (!sink) throw std::runtime_error("BatchLogTape::addSink — sink is NULL");
        sinks_.push_back(sink);
    }

    /// Remove a sink by pointer. Returns true if found and removed.
    bool removeSink(ILogSink* sink) {
        auto it = std::find(sinks_.begin(), sinks_.end(), sink);
        if (it != sinks_.end()) {
            sinks_.erase(it);
            return true;
        }
        return false;
    }

    /// Flush all registered sinks (call during shutdown).
    void flushSinks() {
        for (auto* sink : sinks_) {
            if (sink) sink->flush();
        }
    }

    //--------------------------------------------------
    //  Accessors
    //--------------------------------------------------

    int currentStep()  const { return current_step_; }
    int currentBatch() const { return current_batch_; }
    size_t entryCount() const { return entries_.size(); }
    const TapeConfig& config() const { return config_; }

    /// Skip flag for gradient-accumulation micro-batches.
    /// When true, expensive D2H equation diagnostics are suppressed
    /// (same weights → duplicate output on micro-batches 1..N-1).
    bool skipThisPass() const { return skip_this_pass_; }
    void setSkipThisPass(bool skip) { skip_this_pass_ = skip; }

    /// Update the global threshold at runtime.
    void setDefaultLevel(LogLevel level) { config_.default_level = level; }

    /// Update a per-group threshold at runtime.
    void setGroupLevel(LogGroup group, LogLevel level) {
        config_.group_levels[static_cast<size_t>(group)] = level;
    }

    /// Clear a per-group override (revert to default).
    void clearGroupLevel(LogGroup group) {
        config_.group_levels[static_cast<size_t>(group)] = LogLevel::COUNT;
    }

private:
    TapeConfig config_;
    std::vector<LogEntry> entries_;
    std::vector<LogEntry> lifecycle_entries_;
    std::vector<ILogSink*> sinks_;
    std::mutex mutex_;
    int current_step_  = 0;
    int current_batch_ = 0;
    bool skip_this_pass_ = false;
};

//======================================================//
//  Free functions (defined in BatchLogTape.cu)
//======================================================//

/// Parse a TapeConfig from string/bool/int parameters (for JSON config loading).
TapeConfig parseTapeConfig(
    const char* default_level_str,
    bool equation_csv_enabled,
    bool stderr_enabled,
    size_t initial_capacity = 2048);

/// Apply a per-group level override from string names.
void applyGroupOverride(TapeConfig& config, const char* group_str, const char* level_str);

/// Dump TapeConfig as a human-readable string (for diagnostics).
std::string dumpTapeConfig(const TapeConfig& config);

//======================================================//
//  Global tape accessor (set once at startup, cleared at shutdown)
//  This allows layer-level code (Encoding_GPU, AutogradLoss, etc.)
//  to access the tape without threading ctx through every call.
//  Thread-safe: record() is mutex-protected.
//======================================================//

/// Set the global tape pointer (call in Phase1 after construction).
void setGlobalTape(BatchLogTape* tape);

/// Get the global tape pointer (returns nullptr before setGlobalTape).
BatchLogTape* getGlobalTape();

} // namespace GRIM::Logging
