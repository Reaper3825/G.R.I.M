#pragma once

#include <cstddef>
#include <memory>

#include "DigitalPerceptionPrimitiveTypes.hpp"

namespace GRIM { namespace Perception { namespace Digital {

// Optional platform enrichment. OCR remains the baseline digital text source;
// automation providers add exact native controls where an OS exposes them.
class DigitalAutomationProvider {
public:
    virtual ~DigitalAutomationProvider() = default;
    virtual const char* ProviderName() const noexcept = 0;
    virtual DigitalAutomationResult InspectForeground(
        const DigitalCaptureMetadata& source_metadata,
        std::size_t max_elements) = 0;
};

std::unique_ptr<DigitalAutomationProvider> CreatePlatformDigitalAutomationProvider();

}}} // namespace GRIM::Perception::Digital
