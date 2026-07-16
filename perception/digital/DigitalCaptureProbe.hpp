#pragma once

namespace GRIM { namespace Perception { namespace Digital {

// Returns true when a digital-capture probe argument was present. `exit_code`
// contains the process result and the caller should return immediately.
bool TryRunDigitalCaptureProbe(int argc, char* argv[], int& exit_code);

}}} // namespace GRIM::Perception::Digital
