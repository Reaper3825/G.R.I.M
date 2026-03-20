#pragma once

#include "../pipeline_stage.hpp"

namespace GRIM {
namespace Pipeline {

class StageVerify : public IPipelineStage {
public:
    PipelineState stageId() const override { return PipelineState::Verify; }
    const char* stageName() const override { return "verify"; }
    float progressWeight() const override { return 0.15f; }
    StageResult execute(PipelineContext& ctx) override;
};

} // namespace Pipeline
} // namespace GRIM
