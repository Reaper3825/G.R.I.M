#pragma once

#include "../pipeline_stage.hpp"

namespace GRIM {
namespace Pipeline {

class StageIngest : public IPipelineStage {
public:
    PipelineState stageId() const override { return PipelineState::Ingest; }
    const char* stageName() const override { return "ingest"; }
    float progressWeight() const override { return 0.20f; }
    StageResult execute(PipelineContext& ctx) override;
};

} // namespace Pipeline
} // namespace GRIM
