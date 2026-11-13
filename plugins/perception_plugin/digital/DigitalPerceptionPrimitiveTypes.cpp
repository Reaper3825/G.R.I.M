#include "DigitalPerceptionPrimitiveTypes.hpp"

namespace GRIM { namespace Perception { namespace Digital {

const char* ToString(DigitalPrimitiveStatus status) noexcept {
    switch (status) {
        case DigitalPrimitiveStatus::Ok: return "ok";
        case DigitalPrimitiveStatus::Disabled: return "disabled";
        case DigitalPrimitiveStatus::Unsupported: return "unsupported";
        case DigitalPrimitiveStatus::Unavailable: return "unavailable";
        case DigitalPrimitiveStatus::Failed: return "failed";
    }
    return "unknown";
}

}}} // namespace GRIM::Perception::Digital
