#pragma once

#include "../pipeline_stage.hpp"

namespace GRIM {
namespace Pipeline {

class StageDedup : public IPipelineStage {
public:
    PipelineState stageId() const override { return PipelineState::Deduplicate; }
    const char* stageName() const override { return "deduplicate"; }
    float progressWeight() const override { return 0.10f; }
    StageResult execute(PipelineContext& ctx) override;
};

} // namespace Pipeline
} // namespace GRIM
