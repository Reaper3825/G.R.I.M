#pragma once

namespace GRIM { namespace Perception { namespace Digital {

// Pulls the latest capture attempt from DigitalFrameBus and projects compact,
// language-readable digital state into SessionContextManager's default session.
void TickDigitalContextProjector();
void ShutdownDigitalContextProjector();

}}} // namespace GRIM::Perception::Digital
