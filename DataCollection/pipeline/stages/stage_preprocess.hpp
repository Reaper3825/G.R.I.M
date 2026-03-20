#pragma once

#include "../pipeline_stage.hpp"

namespace GRIM {
namespace Pipeline {

class StagePreprocess : public IPipelineStage {
public:
    PipelineState stageId() const override { return PipelineState::Preprocess; }
    const char* stageName() const override { return "preprocess"; }
    float progressWeight() const override { return 0.15f; }
    StageResult execute(PipelineContext& ctx) override;
};

} // namespace Pipeline
} // namespace GRIM
