#pragma once

#include <cstdint>
#include <string>

namespace GRIM {
namespace Pipeline {

enum class PipelineState : uint8_t {
    Idle,
    Initialize,
    Collect,
    Ingest,
    Verify,
    Deduplicate,
    Preprocess,
    Tag,
    Write,
    Complete,
    Error
};

enum class PipelineMode : uint8_t {
    Full,
    CollectOnly,
    VerifyOnly,
    MergeOnly,
    MergeRebuild
};

struct StageResult {
    bool success = true;
    std::string errorMessage;
    float durationSeconds = 0.0f;
};

inline const char* pipelineStateToString(PipelineState s) {
    switch (s) {
        case PipelineState::Idle:         return "idle";
        case PipelineState::Initialize:   return "initializing";
        case PipelineState::Collect:      return "collecting";
        case PipelineState::Ingest:       return "ingesting";
        case PipelineState::Verify:       return "verifying";
        case PipelineState::Deduplicate:  return "deduplicating";
        case PipelineState::Preprocess:   return "preprocessing";
        case PipelineState::Tag:          return "tagging";
        case PipelineState::Write:        return "writing";
        case PipelineState::Complete:     return "complete";
        case PipelineState::Error:        return "error";
    }
    return "unknown";
}

inline const char* pipelineModeToString(PipelineMode m) {
    switch (m) {
        case PipelineMode::Full:         return "full";
        case PipelineMode::CollectOnly:  return "collect";
        case PipelineMode::VerifyOnly:   return "verify";
        case PipelineMode::MergeOnly:    return "merge";
        case PipelineMode::MergeRebuild: return "merge-rebuild";
    }
    return "unknown";
}

} // namespace Pipeline
} // namespace GRIM
