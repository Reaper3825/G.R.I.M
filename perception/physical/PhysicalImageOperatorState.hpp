#pragma once

#include <cstdint>
#include <string>

namespace GRIM { namespace Perception { namespace Physical {

// Lifecycle state shared by every Stage-2 image operator. Surfacing this
// explicitly is mandatory under Rule 20: a UI consumer or downstream router
// MUST be able to distinguish "operator never configured" from "configured
// but inference failed", so it can decide whether to show a hint, an error,
// or a result. There is intentionally no "Unknown" / "Default" — uninit'd
// operators report NoModelConfigured.
enum class PhysicalImageOperatorState : uint8_t {
    NoModelConfigured = 0,   // model path is empty; operator is intentionally idle
    ModelLoaded       = 1,   // weights are in memory and ready for inference
    ModelLoadFailed   = 2,   // load was attempted and threw — see last_error_reason
    InferenceFailed   = 3    // last RouteFrameToOperator threw — see last_error_reason
};

inline const char* DescribePhysicalImageOperatorState(PhysicalImageOperatorState s) {
    switch (s) {
        case PhysicalImageOperatorState::NoModelConfigured: return "NoModelConfigured";
        case PhysicalImageOperatorState::ModelLoaded:       return "ModelLoaded";
        case PhysicalImageOperatorState::ModelLoadFailed:   return "ModelLoadFailed";
        case PhysicalImageOperatorState::InferenceFailed:   return "InferenceFailed";
    }
    return "InvalidPhysicalImageOperatorState";
}

}}} // namespace GRIM::Perception::Physical
