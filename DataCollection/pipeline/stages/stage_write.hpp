#pragma once

#include "../pipeline_stage.hpp"

namespace GRIM {
namespace Pipeline {

class StageWrite : public IPipelineStage {
public:
    PipelineState stageId() const override { return PipelineState::Write; }
    const char* stageName() const override { return "write"; }
    float progressWeight() const override { return 0.10f; }
    StageResult execute(PipelineContext& ctx) override;
};

} // namespace Pipeline
} // namespace GRIM
