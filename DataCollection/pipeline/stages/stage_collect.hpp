#pragma once

#include "../pipeline_stage.hpp"

namespace GRIM {
namespace Pipeline {

class StageCollect : public IPipelineStage {
public:
    PipelineState stageId() const override { return PipelineState::Collect; }
    const char* stageName() const override { return "collect"; }
    float progressWeight() const override { return 0.15f; }
    StageResult execute(PipelineContext& ctx) override;
};

} // namespace Pipeline
} // namespace GRIM
