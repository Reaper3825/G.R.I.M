#include "TelemetryInitInputs.hpp"

#include "../../Phase1_Startup.hpp"

namespace GRIMText::Training {

TelemetryInitInputs makeTelemetryInitInputs(const TrainingContext& ctx) {
    TelemetryInitInputs inputs;
    inputs.capacity = ctx.run_capacity;
    return inputs;
}

} // namespace GRIMText::Training

