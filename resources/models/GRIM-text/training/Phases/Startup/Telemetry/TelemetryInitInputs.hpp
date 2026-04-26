#pragma once

#include "../Capacity/RunCapacity.hpp"

namespace GRIMText::Training {

struct TrainingContext;

struct TelemetryInitInputs {
    RunCapacity capacity;
};

TelemetryInitInputs makeTelemetryInitInputs(const TrainingContext& ctx);
void TelemetryReady(TrainingContext& ctx);

} // namespace GRIMText::Training

