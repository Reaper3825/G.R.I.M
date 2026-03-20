#include "pipeline_orchestrator.hpp"
#include "chunk_spool.hpp"
#include "DataCollection/collection_state.hpp"
#include "stages/stage_initialize.hpp"
#include "stages/stage_collect.hpp"
#include "stages/stage_ingest.hpp"
#include "stages/stage_verify.hpp"
#include "stages/stage_dedup.hpp"
#include "stages/stage_preprocess.hpp"
#include "stages/stage_tag.hpp"
#include "stages/stage_write.hpp"

#include <chrono>
#include <iostream>

namespace GRIM {
namespace Pipeline {

PipelineOrchestrator::PipelineOrchestrator() {
    stages_[PipelineState::Initialize]   = std::make_unique<StageInitialize>();
    stages_[PipelineState::Collect]      = std::make_unique<StageCollect>();
    stages_[PipelineState::Ingest]       = std::make_unique<StageIngest>();
    stages_[PipelineState::Verify]       = std::make_unique<StageVerify>();
    stages_[PipelineState::Deduplicate]  = std::make_unique<StageDedup>();
    stages_[PipelineState::Preprocess]   = std::make_unique<StagePreprocess>();
    stages_[PipelineState::Tag]          = std::make_unique<StageTag>();
    stages_[PipelineState::Write]        = std::make_unique<StageWrite>();
}

PipelineOrchestrator::~PipelineOrchestrator() {
    stopPipeline();
    if (thread_ && thread_->joinable()) {
        thread_->join();
    }
}

std::vector<PipelineState> PipelineOrchestrator::stagesForMode(PipelineMode mode) const {
    switch (mode) {
        case PipelineMode::Full:
            return {PipelineState::Initialize, PipelineState::Collect, PipelineState::Ingest,
                    PipelineState::Verify, PipelineState::Deduplicate, PipelineState::Preprocess,
                    PipelineState::Tag, PipelineState::Write};
        case PipelineMode::CollectOnly:
            return {PipelineState::Initialize, PipelineState::Collect};
        case PipelineMode::VerifyOnly:
            return {PipelineState::Initialize, PipelineState::Verify};
        case PipelineMode::MergeOnly:
        case PipelineMode::MergeRebuild:
            return {PipelineState::Initialize, PipelineState::Ingest,
                    PipelineState::Verify, PipelineState::Deduplicate, PipelineState::Preprocess,
                    PipelineState::Tag, PipelineState::Write};
    }
    return {};
}

void PipelineOrchestrator::startPipeline(PipelineMode mode) {
    if (currentState_.load() != PipelineState::Idle &&
        currentState_.load() != PipelineState::Complete &&
        currentState_.load() != PipelineState::Error) {
        return;
    }

    if (thread_ && thread_->joinable()) {
        thread_->join();
    }

    // Reset context for new run (no copy/move assignment; PipelineContext has unique_ptr)
    context_.config = PipelineConfig{};
    context_.config.mode = mode;
    context_.stats = PipelineStats{};
    context_.stateManager.reset();
    context_.run = PipelineRunLayout{};
    context_.ingestCursor = ChunkCursor{};
    context_.verifyCursor = ChunkCursor{};
    context_.dedupCursor = ChunkCursor{};
    context_.preprocessCursor = ChunkCursor{};
    context_.tagCursor = ChunkCursor{};
    context_.cleanedChunk.clear();
    context_.taggedChunk.clear();
    context_.chunkSize = 5000;
    context_.datasetIO.reset();
    context_.stopRequested.store(false);

    context_.onProgress = [this](PipelineState state, float progress, const std::string& msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        overallProgress_ = progress;
        currentMessage_ = msg;
        if (progressCallback_) {
            progressCallback_(progress, msg);
        }
    };

    runStartTime_ = std::chrono::steady_clock::now();
    currentState_.store(PipelineState::Initialize);

    thread_ = std::make_unique<std::thread>(&PipelineOrchestrator::executionThread, this);
}

void PipelineOrchestrator::stopPipeline() {
    context_.stopRequested.store(true);
}

PipelineState PipelineOrchestrator::currentState() const {
    return currentState_.load();
}

float PipelineOrchestrator::overallProgress() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return overallProgress_;
}

PipelineOrchestrator::Status PipelineOrchestrator::getStatus() const {
    std::lock_guard<std::mutex> lock(mutex_);
    Status s;
    s.state    = currentState_.load();
    s.progress = overallProgress_;
    s.phase    = pipelineStateToString(s.state);
    s.message  = currentMessage_;
    s.stats    = context_.stats;
    s.isRunning = (s.state != PipelineState::Idle &&
                   s.state != PipelineState::Complete &&
                   s.state != PipelineState::Error);

    if (s.isRunning) {
        auto now = std::chrono::steady_clock::now();
        s.elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(
            now - runStartTime_).count();
    } else {
        s.elapsedSeconds = 0;
    }
    return s;
}

void PipelineOrchestrator::setProgressCallback(ProgressCallback cb) {
    std::lock_guard<std::mutex> lock(mutex_);
    progressCallback_ = std::move(cb);
}

void PipelineOrchestrator::executionThread() {
    auto stageList = stagesForMode(context_.config.mode);

    float totalWeight = 0.0f;
    for (auto stateId : stageList) {
        auto it = stages_.find(stateId);
        if (it != stages_.end()) totalWeight += it->second->progressWeight();
    }

    float cumulativeWeight = 0.0f;

    for (auto stateId : stageList) {
        if (context_.stopRequested.load()) {
            currentState_.store(PipelineState::Error);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                currentMessage_ = "Pipeline stopped by user";
            }
            return;
        }

        auto it = stages_.find(stateId);
        if (it == stages_.end()) {
            currentState_.store(PipelineState::Error);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                currentMessage_ = "Missing stage: " + std::string(pipelineStateToString(stateId));
            }
            return;
        }

        currentState_.store(stateId);
        auto& stage = it->second;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            currentMessage_ = std::string(stage->stageName()) + "...";
            overallProgress_ = (totalWeight > 0.0f)
                ? (cumulativeWeight / totalWeight) * 100.0f
                : 0.0f;
        }

        std::cout << "─── Stage: " << stage->stageName() << " ───\n";
        StageResult result = stage->execute(context_);

        if (!result.success) {
            currentState_.store(PipelineState::Error);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                currentMessage_ = "Error in " + std::string(stage->stageName())
                                + ": " + result.errorMessage;
            }
            std::cerr << "[Pipeline] ERROR in " << stage->stageName()
                      << ": " << result.errorMessage << "\n";
            return;
        }

        cumulativeWeight += stage->progressWeight();
        std::cout << "  Stage " << stage->stageName() << " completed in "
                  << result.durationSeconds << "s\n\n";
    }

    // Cleanup spool
    {
        ChunkSpool spool(context_.run.spoolRoot);
        spool.cleanupRun();
    }

    currentState_.store(PipelineState::Complete);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        overallProgress_ = 100.0f;
        currentMessage_ = "Pipeline complete";
    }

    std::cout << "=== PIPELINE COMPLETE ===\n";
    std::cout << "  Collected:  " << context_.stats.entriesCollected << "\n";
    std::cout << "  Ingested:   " << context_.stats.entriesIngested << "\n";
    std::cout << "  Verified:   " << context_.stats.entriesVerified << "\n";
    std::cout << "  Dedup removed: " << context_.stats.duplicatesRemoved << "\n";
    std::cout << "  Cleaned:    " << context_.stats.entriesCleaned << "\n";
    std::cout << "  Tagged:     " << context_.stats.entriesTagged << "\n";
    std::cout << "  Written:    " << context_.stats.entriesWritten << "\n\n";
}

} // namespace Pipeline
} // namespace GRIM
