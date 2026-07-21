#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace GRIM::ExecutionTransition {

// Optimizer-step schedule for handing complete execution trajectories from the
// teacher policy to the model policy. Alpha is never used to blend actions,
// scalars, or register states: it is only the probability that one trajectory
// is wholly model-authored.
struct ExecutionTransitionScheduleConfig {
    float initial_student_alpha = 0.0f;
    float final_student_alpha = 1.0f;
    int ramp_steps = 0;
};

struct ExecutionTransitionScheduleResult {
    float student_alpha = 1.0f;
    float ramp_progress = 1.0f;
    bool in_ramp = false;
};

class ExecutionTransitionSchedule {
public:
    explicit ExecutionTransitionSchedule(
        const ExecutionTransitionScheduleConfig& config)
        : config_(config)
    {
        validateAlpha(config_.initial_student_alpha, "initial_student_alpha");
        validateAlpha(config_.final_student_alpha, "final_student_alpha");
        if (config_.final_student_alpha < config_.initial_student_alpha) {
            throw std::runtime_error(
                "[ExecutionTransitionSchedule] final_student_alpha must be >= "
                "initial_student_alpha");
        }
        if (config_.ramp_steps < 0) {
            throw std::runtime_error(
                "[ExecutionTransitionSchedule] ramp_steps must be >= 0, got " +
                std::to_string(config_.ramp_steps));
        }
    }

    // The +1 mirrors LR warmup semantics: update zero already receives a small
    // amount of student exposure instead of creating a teacher-only plateau.
    ExecutionTransitionScheduleResult query(int optimizer_step) const {
        if (optimizer_step < 0) {
            throw std::runtime_error(
                "[ExecutionTransitionSchedule] optimizer_step must be >= 0, got " +
                std::to_string(optimizer_step));
        }

        ExecutionTransitionScheduleResult result;
        if (config_.ramp_steps == 0) {
            result.student_alpha = config_.final_student_alpha;
            return result;
        }

        result.ramp_progress = std::clamp(
            static_cast<float>(optimizer_step + 1) /
                static_cast<float>(config_.ramp_steps),
            0.0f,
            1.0f);
        result.in_ramp = result.ramp_progress < 1.0f;

        // Smoothstep avoids a derivative jump at either end of the handoff.
        const float x = result.ramp_progress;
        const float eased = x * x * (3.0f - 2.0f * x);
        result.student_alpha = config_.initial_student_alpha +
            (config_.final_student_alpha - config_.initial_student_alpha) * eased;
        result.student_alpha = std::clamp(result.student_alpha, 0.0f, 1.0f);
        return result;
    }

    float studentAlpha(int optimizer_step) const {
        return query(optimizer_step).student_alpha;
    }

    const ExecutionTransitionScheduleConfig& config() const { return config_; }

private:
    static void validateAlpha(float alpha, const char* name) {
        if (!std::isfinite(alpha) || alpha < 0.0f || alpha > 1.0f) {
            throw std::runtime_error(
                std::string("[ExecutionTransitionSchedule] ") + name +
                " must be finite and in [0, 1], got " + std::to_string(alpha));
        }
    }

    ExecutionTransitionScheduleConfig config_;
};

// SplitMix64 finalizer: stable across hosts and standard-library versions.
inline std::uint64_t mixTrajectoryKey(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

inline bool useModelTrajectory(
    float student_alpha,
    std::uint64_t optimizer_step,
    std::uint64_t batch_idx,
    int batch_row)
{
    if (!std::isfinite(student_alpha) || student_alpha < 0.0f || student_alpha > 1.0f) {
        throw std::runtime_error(
            "[ExecutionTransitionSchedule] student_alpha must be finite and in [0, 1]");
    }
    if (batch_row < 0) {
        throw std::runtime_error(
            "[ExecutionTransitionSchedule] batch_row must be >= 0");
    }
    if (student_alpha <= 0.0f) return false;
    if (student_alpha >= 1.0f) return true;

    std::uint64_t key = optimizer_step;
    key ^= mixTrajectoryKey(batch_idx + 0x632be59bd9b4e019ULL);
    key ^= mixTrajectoryKey(static_cast<std::uint64_t>(batch_row) +
                            0x8cb92baa3f3d8dd7ULL);
    const std::uint64_t sample = mixTrajectoryKey(key);

    // Convert the high 53 bits to the exact [0,1) domain representable by a
    // double, then compare against alpha. Endpoint cases above remain exact.
    constexpr double kInvTwoTo53 = 1.0 / 9007199254740992.0;
    const double unit = static_cast<double>(sample >> 11U) * kInvTwoTo53;
    return unit < static_cast<double>(student_alpha);
}

}  // namespace GRIM::ExecutionTransition
