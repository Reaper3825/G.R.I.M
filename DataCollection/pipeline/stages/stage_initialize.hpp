#pragma once

#include "../pipeline_stage.hpp"

namespace GRIM {
namespace Pipeline {

class StageInitialize : public IPipelineStage {
public:
    PipelineState stageId() const override { return PipelineState::Initialize; }
    const char* stageName() const override { return "initialize"; }
    float progressWeight() const override { return 0.05f; }
    StageResult execute(PipelineContext& ctx) override;
};

} // namespace Pipeline
} // namespace GRIM
