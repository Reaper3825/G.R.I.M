#pragma once

#include "pipeline_types.hpp"

namespace GRIM {
namespace Pipeline {

struct PipelineContext;

class IPipelineStage {
public:
    virtual ~IPipelineStage() = default;
    virtual PipelineState stageId() const = 0;
    virtual const char* stageName() const = 0;
    virtual float progressWeight() const = 0;
    virtual StageResult execute(PipelineContext& ctx) = 0;
};

} // namespace Pipeline
} // namespace GRIM
