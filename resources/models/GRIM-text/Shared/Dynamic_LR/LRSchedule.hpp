//======================================================//
//  LRSchedule.hpp
//  Deterministic learning rate schedule (warmup + cosine decay)
//
//  Exposed struct — the full LR curve is queryable at any step
//  without training loop state. Usable for:
//    - Training loop step LR
//    - Checkpoint resume reconstruction (cumulative displacement)
//    - Offline visualization / analysis
//======================================================//

#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace GRIM::LR {

/// Configuration for a deterministic LR schedule.
/// Constructed once from config, then queried per step.
struct LRScheduleConfig {
    float base_lr = 0.0f;          ///< Peak learning rate (after warmup)
    float cosine_decay_min_lr = 0.0f; ///< Floor LR at end of cosine decay
    int warmup_steps = 0;          ///< Linear warmup from 0 → base_lr
    int total_steps = 0;           ///< Total optimizer steps for full schedule
    bool cosine_decay_enabled = false;
};

/// Result of querying the schedule at a given step.
struct LRScheduleResult {
    float lr = 0.0f;               ///< Learning rate at this step
    float progress = 0.0f;         ///< Overall progress [0, 1]
    float warmup_progress = 0.0f;  ///< Warmup progress [0, 1] (1.0 if past warmup)
    float decay_progress = 0.0f;   ///< Decay progress [0, 1] (0.0 if no decay or in warmup)
    bool in_warmup = false;
    bool in_decay = false;
};

/// Deterministic LR schedule: linear warmup → constant → cosine decay.
/// Pure function of step — no mutable state.
class LRSchedule {
public:
    /// Construct from config. Throws if base_lr <= 0.
    explicit LRSchedule(const LRScheduleConfig& config)
        : config_(config)
    {
        if (config_.base_lr <= 0.0f) {
            throw std::runtime_error("[LRSchedule] base_lr must be > 0, got " +
                                     std::to_string(config_.base_lr));
        }
    }

    /// Query the schedule at a given optimizer step.
    LRScheduleResult query(int step) const {
        LRScheduleResult r;
        r.progress = (config_.total_steps > 0)
            ? static_cast<float>(step) / static_cast<float>(config_.total_steps)
            : 0.0f;

        // Phase 1: Linear warmup
        if (step < config_.warmup_steps) {
            r.in_warmup = true;
            r.warmup_progress = static_cast<float>(step + 1) /
                                static_cast<float>(std::max(1, config_.warmup_steps));
            r.lr = config_.base_lr * r.warmup_progress;
            return r;
        }

        r.warmup_progress = 1.0f;

        // Phase 2: Cosine decay (or constant if disabled)
        if (!config_.cosine_decay_enabled || config_.total_steps <= config_.warmup_steps) {
            r.lr = config_.base_lr;
            return r;
        }

        r.in_decay = true;
        const int decay_steps = config_.total_steps - config_.warmup_steps;
        const int current_decay_step = step - config_.warmup_steps;
        r.decay_progress = static_cast<float>(current_decay_step) /
                           static_cast<float>(std::max(1, decay_steps));
        r.decay_progress = std::clamp(r.decay_progress, 0.0f, 1.0f);

        const float cosine = 0.5f * (1.0f + std::cos(3.14159265f * r.decay_progress));
        r.lr = config_.cosine_decay_min_lr +
               (config_.base_lr - config_.cosine_decay_min_lr) * cosine;
        return r;
    }

    /// Shorthand: just the LR value at a step.
    float lr(int step) const { return query(step).lr; }

    /// Reconstruct cumulative displacement: Σ lr(0..step-1).
    /// Used for checkpoint resume (Adam causation telemetry).
    float cumulativeDisplacement(int steps) const {
        float sum = 0.0f;
        for (int t = 0; t < steps; ++t) {
            sum += lr(t);
        }
        return sum;
    }

    const LRScheduleConfig& config() const { return config_; }

private:
    LRScheduleConfig config_;
};

} // namespace GRIM::LR
