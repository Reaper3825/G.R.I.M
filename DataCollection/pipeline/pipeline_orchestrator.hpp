#pragma once

#include "pipeline_types.hpp"
#include "pipeline_context.hpp"
#include "pipeline_stage.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace GRIM {
namespace Pipeline {

class PipelineOrchestrator {
public:
    PipelineOrchestrator();
    ~PipelineOrchestrator();

    void startPipeline(PipelineMode mode);
    void stopPipeline();
    PipelineState currentState() const;
    float overallProgress() const;

    struct Status {
        PipelineState state;
        float progress;
        std::string phase;
        std::string message;
        PipelineStats stats;
        int64_t elapsedSeconds;
        bool isRunning;
    };
    Status getStatus() const;

    using ProgressCallback = std::function<void(float, const std::string&)>;
    void setProgressCallback(ProgressCallback cb);

private:
    void executionThread();
    std::vector<PipelineState> stagesForMode(PipelineMode mode) const;

    std::unordered_map<PipelineState, std::unique_ptr<IPipelineStage>> stages_;
    PipelineContext context_;
    std::atomic<PipelineState> currentState_{PipelineState::Idle};
    std::unique_ptr<std::thread> thread_;
    mutable std::mutex mutex_;

    float overallProgress_ = 0.0f;
    std::string currentMessage_;
    ProgressCallback progressCallback_;
    std::chrono::steady_clock::time_point runStartTime_;
};

} // namespace Pipeline
} // namespace GRIM
