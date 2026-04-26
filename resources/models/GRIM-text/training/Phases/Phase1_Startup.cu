#include "Phase1_Startup.hpp"

#include "Startup/Logging.hpp"
#include "Startup/Capacity/MemorySnapshot.hpp"
#include "Startup/Capacity/CapacityStem.hpp"
#include "Startup/Data/DataInfo.hpp"
#include "Startup/Model/ModelAllocationState.hpp"
#include "Startup/Resume/ResumeState.hpp"
#include "Startup/Telemetry/TelemetryInitInputs.hpp"
#include "Startup/Scheduling/SchedulerPreflight.hpp"
#include "Startup/Epoch/EpochPlan.hpp"
#include "Startup/Validation/StartupValidation.hpp"
#include "Startup/Validation/Phase2Handoff.hpp"

namespace GRIMText::Training {

std::unique_ptr<TrainingContext> executePhase1(int argc, char** argv) {
    auto ctx = std::make_unique<TrainingContext>();
    LoggingReady(*ctx, argc, argv);
    MemorySnapshotReady(*ctx);
    HyperparametersReady(*ctx);
    CapacityStemReady(*ctx);
    DataInfoReady(*ctx);
    ModelAllocated(*ctx);
    ResumeStateReady(*ctx);
    TelemetryReady(*ctx);
    SchedulerPreflightReady(*ctx);
    EpochPlanReady(*ctx);
    StartupValidated(*ctx);
    Phase2HandoffReady(*ctx);
    return ctx;
}

} // namespace GRIMText::Training
