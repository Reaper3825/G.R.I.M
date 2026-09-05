#include "bootstrap.hpp"

#include "../logger.hpp"
#include "../perception/perception.hpp"
#include "../perception/perception_context.hpp"
#include "../perception/physical/PhysicalGestureControlConfigIO.hpp"
#include "../resources.hpp"

#include <stdexcept>
#include <string>

void bootstrapPerceptionSubsystem()
{
    std::string gestureConfigError;
    if (!GRIM::Perception::Physical::ApplyPhysicalGestureControlConfigFromRuntime(
            aiConfig, gestureConfigError)) {
        throw std::runtime_error(
            "bootstrapPerceptionSubsystem: gesture-control config rejected: " +
            gestureConfigError);
    }

    GRIM::Perception::init();
    LOG_PHASE("Perception system initialized", true);

    GRIM::Perception::initContextManager();
    LOG_PHASE("Perception context manager initialized", true);
}