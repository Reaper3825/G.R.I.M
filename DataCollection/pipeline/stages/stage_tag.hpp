#pragma once

#include "../pipeline_stage.hpp"

namespace GRIM {
namespace Pipeline {

class StageTag : public IPipelineStage {
public:
    PipelineState stageId() const override { return PipelineState::Tag; }
    const char* stageName() const override { return "tag"; }
    float progressWeight() const override { return 0.10f; }
    StageResult execute(PipelineContext& ctx) override;
};

} // namespace Pipeline
} // namespace GRIM
